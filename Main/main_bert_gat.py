# -*- coding: utf-8 -*-
# Main script for training ResGAT with BERT embeddings for Chinese text
import sys
import os
import os.path as osp
import warnings

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
dirname = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(dirname, '..'))

import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from Main.pargs import pargs
from Main.bert_dataset import BertTreeDataset
from Main.gat_model import ResGAT_graphcl, BiGAT_graphcl
from Main.utils import create_log_dict_sup, write_log, write_json
from Main.augmentation import augment
from Main.sort import sort_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def sup_train(train_loader, aug1, aug2, model, optimizer, device, lamda, use_unsup_loss):
    model.train()
    total_loss = 0

    augs1 = aug1.split('||')
    augs2 = aug2.split('||')

    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)

        out = model(data)
        sup_loss = F.nll_loss(out, data.y.long().view(-1))

        if use_unsup_loss:
            aug_data1 = augment(data, augs1)
            aug_data2 = augment(data, augs2)

            out1 = model.forward_graphcl(aug_data1)
            out2 = model.forward_graphcl(aug_data2)
            unsup_loss = model.loss_graphcl(out1, out2)

            loss = sup_loss + lamda * unsup_loss
        else:
            loss = sup_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)

def test(model, dataloader, num_classes, device):
    model.eval()
    error = 0

    y_true = []
    y_pred = []

    for data in dataloader:
        data = data.to(device)
        pred = model(data)
        error += F.nll_loss(pred, data.y.long().view(-1)).item() * data.num_graphs
        y_true += data.y.tolist()
        y_pred += pred.max(1).indices.tolist()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = round(accuracy_score(y_true, y_pred), 4)
    precs = []
    recs = []
    f1s = []
    for label in range(num_classes):
        precs.append(round(precision_score(y_true == label, y_pred == label, labels=[True]), 4))
        recs.append(round(recall_score(y_true == label, y_pred == label, labels=[True]), 4))
        f1s.append(round(f1_score(y_true == label, y_pred == label, labels=[True]), 4))
    micro_p = round(precision_score(y_true, y_pred, labels=range(num_classes), average='micro'), 4)
    micro_r = round(recall_score(y_true, y_pred, labels=range(num_classes), average='micro'), 4)
    micro_f1 = round(f1_score(y_true, y_pred, labels=range(num_classes), average='micro'), 4)

    macro_p = round(precision_score(y_true, y_pred, labels=range(num_classes), average='macro'), 4)
    macro_r = round(recall_score(y_true, y_pred, labels=range(num_classes), average='macro'), 4)
    macro_f1 = round(f1_score(y_true, y_pred, labels=range(num_classes), average='macro'), 4)
    return error / len(dataloader.dataset), acc, precs, recs, f1s, \
           [micro_p, micro_r, micro_f1], [macro_p, macro_r, macro_f1]

def test_and_log(model, val_loader, test_loader, num_classes, device, epoch, lr, loss, train_acc, log_record):
    val_error, val_acc, val_precs, val_recs, val_f1s, val_micro_metric, val_macro_metric = \
        test(model, val_loader, num_classes, device)
    test_error, test_acc, test_precs, test_recs, test_f1s, test_micro_metric, test_macro_metric = \
        test(model, test_loader, num_classes, device)
    log_info = 'Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Val ERROR: {:.7f}, Test ERROR: {:.7f}\n  Train ACC: {:.4f}, Validation ACC: {:.4f}, Test ACC: {:.4f}\n' \
                   .format(epoch, lr, loss, val_error, test_error, train_acc, val_acc, test_acc) \
               + f'  Test PREC: {test_precs}, Test REC: {test_recs}, Test F1: {test_f1s}\n' \
               + f'  Test Micro Metric(PREC, REC, F1):{test_micro_metric}, Test Macro Metric(PREC, REC, F1):{test_macro_metric}'

    log_record['val accs'].append(val_acc)
    log_record['test accs'].append(test_acc)
    log_record['test precs'].append(test_precs)
    log_record['test recs'].append(test_recs)
    log_record['test f1s'].append(test_f1s)
    log_record['test micro metric'].append(test_micro_metric)
    log_record['test macro metric'].append(test_macro_metric)
    return val_error, log_info, log_record

if __name__ == '__main__':
    # Parse arguments
    args = pargs()

    # Parameters
    dataset = args.dataset
    unsup_dataset = args.unsup_dataset
    device = args.gpu if args.cuda else 'cpu'
    runs = args.runs
    k = args.k

    # Check if this is a Chinese dataset
    is_chinese = 'Weibo' in dataset
    if not is_chinese:
        print("Warning: BERT-Chinese is optimized for Chinese datasets. Consider using a different model for non-Chinese datasets.")

    # Paths
    label_source_path = osp.join(dirname, '..', 'data', dataset, 'source')
    label_dataset_path = osp.join(dirname, '..', 'data', dataset, 'dataset')
    train_path = osp.join(label_dataset_path, 'train')
    val_path = osp.join(label_dataset_path, 'val')
    test_path = osp.join(label_dataset_path, 'test')
    
    # BERT cache directory
    bert_cache_dir = osp.join(dirname, '..', 'cache', 'bert_embeddings')
    os.makedirs(bert_cache_dir, exist_ok=True)

    # Logging
    log_name = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(time.time()))
    log_path = osp.join(dirname, '..', 'Log', f'bert_gat_{log_name}.log')
    log_json_path = osp.join(dirname, '..', 'Log', f'bert_gat_{log_name}.json')

    log = open(log_path, 'w')
    log_dict = create_log_dict_sup(args)
    log_dict['model'] = 'ResGAT with BERT-Chinese'

    # Training parameters
    split = args.split
    batch_size = args.batch_size
    undirected = args.undirected
    centrality = args.centrality
    weight_decay = args.weight_decay
    lamda = args.lamda
    epochs = args.epochs
    use_unsup_loss = args.use_unsup_loss

    # Print configuration
    print(f"Starting training with BERT + ResGAT for {dataset}")
    print(f"Device: {device}")
    print(f"BERT model: bert-base-chinese")
    print(f"Graph model: ResGAT")
    print(f"Centrality: {centrality}")
    print(f"Undirected: {undirected}")
    print(f"Batch size: {batch_size}")
    print(f"Runs: {runs}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {weight_decay}")
    print(f"Lambda: {lamda}")
    print(f"Use unsupervised loss: {use_unsup_loss}")

    # Main training loop
    for run in range(runs):
        print(f"Starting run {run+1}/{runs}")
        write_log(log, f'run:{run}')
        log_record = {'run': run, 'val accs': [], 'test accs': [], 'test precs': [], 'test recs': [], 'test f1s': [],
                      'test micro metric': [], 'test macro metric': []}

        # Sort dataset for this run
        sort_dataset(label_source_path, label_dataset_path, k_shot=k, split=split)

        # Load datasets with BERT embeddings
        print("Loading datasets with BERT embeddings...")
        train_dataset = BertTreeDataset(
            train_path, 
            bert_model_name='bert-base-chinese', 
            centrality_metric=centrality, 
            undirected=undirected,
            cache_dir=bert_cache_dir,
            device=device,
            dataset=dataset
        )
        
        val_dataset = BertTreeDataset(
            val_path, 
            bert_model_name='bert-base-chinese', 
            centrality_metric=centrality, 
            undirected=undirected,
            cache_dir=bert_cache_dir,
            device=device,
            dataset=dataset
        )
        
        test_dataset = BertTreeDataset(
            test_path, 
            bert_model_name='bert-base-chinese', 
            centrality_metric=centrality, 
            undirected=undirected,
            cache_dir=bert_cache_dir,
            device=device,
            dataset=dataset

        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Determine number of classes from dataset
        num_classes = 0
        for data in train_dataset:
            if hasattr(data, 'y') and data.y is not None:
                num_classes = max(num_classes, data.y.item() + 1)
        
        # Default to binary classification if no labels found
        if num_classes == 0:
            num_classes = 2
            print("Warning: No labels found in dataset. Defaulting to binary classification.")
        
        print(f"Number of classes: {num_classes}")
        print(f"Feature dimension: {train_dataset.num_features}")

        # Initialize ResGAT model
        model = ResGAT_graphcl(
            dataset=train_dataset, 
            num_classes=num_classes, 
            hidden=args.hidden,
            num_feat_layers=args.n_layers_feat, 
            num_conv_layers=args.n_layers_conv,
            num_fc_layers=args.n_layers_fc, 
            gfn=False, 
            collapse=False,
            residual=args.skip_connection, 
            res_branch=args.res_branch,
            global_pool=args.global_pool, 
            dropout=args.dropout,
            edge_norm=args.edge_norm, 
            heads=args.heads
        ).to(device)

        # Initialize optimizer and scheduler
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.000001)

        # Initial evaluation
        val_error, log_info, log_record = test_and_log(
            model, val_loader, test_loader, num_classes, device, 0, args.lr, 0, 0, log_record
        )
        write_log(log, log_info)
        print(log_info)

        # Training loop
        for epoch in range(1, epochs + 1):
            # Get current learning rate
            lr = scheduler.optimizer.param_groups[0]['lr']
            
            # Train one epoch
            loss = sup_train(train_loader, args.aug1, args.aug2, model, optimizer, device, lamda, use_unsup_loss)
            
            # Evaluate
            train_error, train_acc, _, _, _, _, _ = test(model, train_loader, num_classes, device)
            val_error, log_info, log_record = test_and_log(
                model, val_loader, test_loader, num_classes, device, epoch, lr, train_error, train_acc, log_record
            )
            write_log(log, log_info)
            print(log_info)
            
            # Update learning rate
            if split == '622':
                scheduler.step(val_error)

        # Calculate final metrics
        log_record['mean acc'] = round(np.mean(log_record['test accs'][-10:]), 3)
        write_log(log, '')

        # Update log dictionary
        log_dict['record'].append(log_record)
        write_json(log_dict, log_json_path)
        
        print(f"Run {run+1} completed. Mean accuracy: {log_record['mean acc']}")
    
    print("Training completed.")