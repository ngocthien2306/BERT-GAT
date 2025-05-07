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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.loader import DataLoader
from Main.pargs import pargs
from Main.dataset import TreeDataset
from Main.word2vec import Embedding, collect_sentences, train_word2vec
from Main.sort import sort_dataset
from Main.utils import create_log_dict_sup, write_log, write_json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from gat_enhance_model import ResGAT_graphcl, BiGAT_graphcl
import copy
import random


# Enhanced augmentation function
def enhanced_augment(data, augs):
    data_aug = copy.deepcopy(data)
    
    # Process each augmentation type
    for aug in augs:
        if aug == 'dropN':
            # Drop nodes with adaptive rate based on graph size
            node_drop_rate = min(0.2, 5 / data_aug.x.size(0)) if data_aug.x.size(0) > 10 else 0.1
            node_num = data_aug.x.size(0)
            perm = torch.randperm(node_num)
            preserve_num = int(node_num * (1 - node_drop_rate))
            preserved = perm[:preserve_num]
            data_aug.x = data_aug.x[preserved]
            data_aug.batch = data_aug.batch[preserved]
            
            # Update edge_index to maintain valid graph structure
            row, col = data_aug.edge_index
            mask = (row < preserve_num) & (col < preserve_num)
            data_aug.edge_index = data_aug.edge_index[:, mask]
            
        elif aug == 'dropE':
            # Drop edges with adaptive rate
            edge_num = data_aug.edge_index.size(1)
            edge_drop_rate = min(0.2, 10 / edge_num) if edge_num > 20 else 0.1
            perm = torch.randperm(edge_num)
            preserve_num = int(edge_num * (1 - edge_drop_rate))
            preserved = perm[:preserve_num]
            data_aug.edge_index = data_aug.edge_index[:, preserved]
            
        elif aug == 'maskN':
            # Mask node features with adaptive rate
            node_num = data_aug.x.size(0)
            feature_num = data_aug.x.size(1)
            mask_rate = min(0.3, 20 / feature_num) if feature_num > 50 else 0.1
            
            # Create mask matrix (1 = keep, 0 = mask)
            masks = torch.FloatTensor(node_num, feature_num).uniform_() > mask_rate
            masks = masks.to(data_aug.x.device)
            
            # Apply masking (replace masked values with zeros)
            data_aug.x = data_aug.x * masks
            
        elif aug == 'permE':
            # Permute a small fraction of edges while preserving connectivity
            edge_num = data_aug.edge_index.size(1)
            permute_num = int(edge_num * 0.1)
            if permute_num > 0:
                perm = torch.randperm(edge_num)
                preserved = perm[permute_num:]
                permuted = perm[:permute_num]
                
                # Shuffle the permuted edges
                permuted_shuffled = permuted[torch.randperm(permute_num)]
                
                # Combine preserved and permuted edges
                new_edge_index = torch.cat([
                    data_aug.edge_index[:, preserved],
                    torch.stack([
                        data_aug.edge_index[0, permuted],
                        data_aug.edge_index[1, permuted_shuffled]
                    ])
                ], dim=1)
                
                data_aug.edge_index = new_edge_index
                
    return data_aug


# Enhanced training function with tracking of both supervised and unsupervised losses
def enhanced_sup_train(train_loader, aug1, aug2, model, optimizer, device, lamda, use_unsup_loss):
    model.train()
    total_loss = 0
    total_sup_loss = 0
    total_unsup_loss = 0

    augs1 = aug1.split('||')
    augs2 = aug2.split('||')

    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)

        # Forward pass for supervised loss
        out = model(data)
        sup_loss = F.nll_loss(out, data.y.long().view(-1))
        total_sup_loss += sup_loss.item() * data.num_graphs

        # Apply contrastive learning if enabled
        if use_unsup_loss:
            # Create two augmented views
            aug_data1 = enhanced_augment(data, augs1)
            aug_data2 = enhanced_augment(data, augs2)

            # Get representations
            out1 = model.forward_graphcl(aug_data1)
            out2 = model.forward_graphcl(aug_data2)
            
            # Compute contrastive loss
            unsup_loss = model.loss_graphcl(out1, out2)
            total_unsup_loss += unsup_loss.item() * data.num_graphs

            # Combined loss with lambda weighting
            loss = sup_loss + lamda * unsup_loss
        else:
            loss = sup_loss

        # Backward pass and optimization
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    avg_loss = total_loss / len(train_loader.dataset)
    avg_sup_loss = total_sup_loss / len(train_loader.dataset)
    avg_unsup_loss = total_unsup_loss / len(train_loader.dataset) if use_unsup_loss else 0
    
    return avg_loss, avg_sup_loss, avg_unsup_loss


# Enhanced test function with more detailed metrics
def enhanced_test(model, dataloader, num_classes, device):
    model.eval()
    error = 0

    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            pred = model(data)
            error += F.nll_loss(pred, data.y.long().view(-1)).item() * data.num_graphs
            y_true += data.y.tolist()
            y_pred += pred.max(1).indices.tolist()
            y_score += torch.exp(pred).tolist()  # Convert log_softmax to probabilities

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    # Calculate standard metrics
    acc = round(accuracy_score(y_true, y_pred), 4)
    
    # Class-specific metrics
    precs = []
    recs = []
    f1s = []
    for label in range(num_classes):
        precs.append(round(precision_score(y_true == label, y_pred == label, labels=[True]), 4))
        recs.append(round(recall_score(y_true == label, y_pred == label, labels=[True]), 4))
        f1s.append(round(f1_score(y_true == label, y_pred == label, labels=[True]), 4))
    
    # Micro and macro metrics
    micro_p = round(precision_score(y_true, y_pred, labels=range(num_classes), average='micro'), 4)
    micro_r = round(recall_score(y_true, y_pred, labels=range(num_classes), average='micro'), 4)
    micro_f1 = round(f1_score(y_true, y_pred, labels=range(num_classes), average='micro'), 4)

    macro_p = round(precision_score(y_true, y_pred, labels=range(num_classes), average='macro'), 4)
    macro_r = round(recall_score(y_true, y_pred, labels=range(num_classes), average='macro'), 4)
    macro_f1 = round(f1_score(y_true, y_pred, labels=range(num_classes), average='macro'), 4)
    
    # Additional metrics for imbalanced scenarios
    weighted_f1 = round(f1_score(y_true, y_pred, labels=range(num_classes), average='weighted'), 4)
    
    # Calculate confusion matrix
    conf_matrix = None
    try:
        conf_matrix = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    except:
        pass
    
    return {
        'error': error / len(dataloader.dataset),
        'acc': acc,
        'precs': precs,
        'recs': recs,
        'f1s': f1s,
        'micro_metrics': [micro_p, micro_r, micro_f1],
        'macro_metrics': [macro_p, macro_r, macro_f1],
        'weighted_f1': weighted_f1,
        'conf_matrix': conf_matrix
    }


# Enhanced test and log function with more comprehensive logging
def enhanced_test_and_log(model, val_loader, test_loader, num_classes, device, epoch, lr, sup_loss, unsup_loss, train_acc, log_record):
    # Get validation results
    val_results = enhanced_test(model, val_loader, num_classes, device)
    val_error = val_results['error']
    val_acc = val_results['acc']
    
    # Get test results
    test_results = enhanced_test(model, test_loader, num_classes, device)
    test_error = test_results['error']
    test_acc = test_results['acc']
    test_precs = test_results['precs']
    test_recs = test_results['recs']
    test_f1s = test_results['f1s']
    test_micro_metric = test_results['micro_metrics']
    test_macro_metric = test_results['macro_metrics']
    
    # Create detailed log information
    log_info = 'Epoch: {:03d}, LR: {:7f}, Sup Loss: {:.7f}, Unsup Loss: {:.7f}, Val ERROR: {:.7f}, Test ERROR: {:.7f}\n  Train ACC: {:.4f}, Validation ACC: {:.4f}, Test ACC: {:.4f}\n' \
                   .format(epoch, lr, sup_loss, unsup_loss, val_error, test_error, train_acc, val_acc, test_acc) \
               + f'  Test PREC: {test_precs}, Test REC: {test_recs}, Test F1: {test_f1s}\n' \
               + f'  Test Micro Metric(PREC, REC, F1):{test_micro_metric}, Test Macro Metric(PREC, REC, F1):{test_macro_metric}\n' \
               + f'  Test Weighted F1: {test_results["weighted_f1"]}'
               
    # Update log record
    log_record['val accs'].append(val_acc)
    log_record['test accs'].append(test_acc)
    log_record['test precs'].append(test_precs)
    log_record['test recs'].append(test_recs)
    log_record['test f1s'].append(test_f1s)
    log_record['test micro metric'].append(test_micro_metric)
    log_record['test macro metric'].append(test_macro_metric)
    log_record['weighted_f1'] = test_results["weighted_f1"]
    
    if test_results['conf_matrix'] is not None:
        log_record['confusion_matrix'] = test_results['conf_matrix'].tolist()
    
    return val_error, log_info, log_record


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # Set seed for reproducibility
    set_seed()
    
    args = pargs()

    # Get parameters from arguments
    unsup_train_size = args.unsup_train_size
    dataset = args.dataset
    unsup_dataset = args.unsup_dataset
    vector_size = args.vector_size
    device = args.gpu if args.cuda else 'cpu'
    runs = args.runs
    k = args.k

    # Set word embedding and language options
    word_embedding = 'tfidf' if 'tfidf' in dataset else 'word2vec'
    lang = 'ch' if 'Weibo' in dataset else 'en'
    tokenize_mode = args.tokenize_mode

    # Set dataset options
    split = args.split
    batch_size = args.batch_size
    undirected = args.undirected
    centrality = args.centrality

    # Set training hyperparameters
    weight_decay = args.weight_decay
    lamda = args.lamda
    epochs = args.epochs 
    use_unsup_loss = args.use_unsup_loss
    
    # Dynamic lambda scheduling
    lamda_min = lamda * 0.1
    lamda_max = lamda
    
    # Define paths
    label_source_path = osp.join(dirname, '..', 'new_data', dataset, 'source')
    label_dataset_path = osp.join(dirname, '..', 'new_data', dataset, 'dataset')
    train_path = osp.join(label_dataset_path, 'train')
    val_path = osp.join(label_dataset_path, 'val')
    test_path = osp.join(label_dataset_path, 'test')
    unlabel_dataset_path = osp.join(dirname, '..', 'new_data', unsup_dataset, 'dataset', 'test')
    model_path = osp.join(dirname, '..', 'checkpoints',
                          f'w2v_{dataset}_{tokenize_mode}_{unsup_train_size}_{vector_size}.model')
    
    # Create log files
    log_name = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(time.time()))
    log_path = osp.join(dirname, '..', 'Log', f'{log_name}.log')
    log_json_path = osp.join(dirname, '..', 'Log', f'{log_name}.json')
    
    log = open(log_path, 'w')
    log_dict = create_log_dict_sup(args)
    
    # Print model configurations
    config_info = (
        f"Model: {args.model}, Hidden: {args.hidden}, Heads: {args.heads}\n"
        f"Learning Rate: {args.lr}, Weight Decay: {weight_decay}, Lambda: {lamda}\n"
        f"Batch Size: {batch_size}, Epochs: {epochs}, Unsupervised Loss: {use_unsup_loss}\n"
        f"Augmentations: Aug1={args.aug1}, Aug2={args.aug2}\n"
        f"Dataset: {dataset}, Word Embedding: {word_embedding}, Vector Size: {vector_size}\n"
    )
    print(config_info)
    write_log(log, config_info)
    
    # Train word2vec if needed
    if not osp.exists(model_path) and word_embedding == 'word2vec':
        print("Starting train word2vec...")
        sentences = collect_sentences(label_source_path, unlabel_dataset_path, unsup_train_size, lang, tokenize_mode)
        w2v_model = train_word2vec(sentences, vector_size)
        w2v_model.save(model_path)
        print("Word2vec training completed.")
    
    # Store best results across runs
    all_runs_best_acc = []
    all_runs_best_f1 = []
    
    # Run training for multiple runs
    for run in range(runs):
        print(f"\n==== Starting run {run+1}/{runs} ====")
        write_log(log, f'\n==== Run: {run} ====')
        
        # Initialize log record
        log_record = {
            'run': run, 
            'val accs': [], 
            'test accs': [], 
            'test precs': [], 
            'test recs': [], 
            'test f1s': [],
            'test micro metric': [], 
            'test macro metric': [],
            'best_epoch': 0,
            'best_acc': 0,
            'best_f1': 0
        }
        
        # Load word embeddings
        word2vec = Embedding(model_path, lang, tokenize_mode) if word_embedding == 'word2vec' else None
        
        # Prepare dataset
        print("Sorting dataset...")
        sort_dataset(label_source_path, label_dataset_path, k_shot=k, split=split)
        
        # Load datasets
        print("Loading datasets...")
        train_dataset = TreeDataset(train_path, word_embedding, word2vec, centrality, undirected)
        val_dataset = TreeDataset(val_path, word_embedding, word2vec, centrality, undirected)
        test_dataset = TreeDataset(test_path, word_embedding, word2vec, centrality, undirected)
        
        print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        num_classes = train_dataset.num_classes
        print(f"Number of classes: {num_classes}")
        
        # Initialize the model based on args
        if args.model == 'ResGAT':
            print("Initializing ResGAT model...")
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
        elif args.model == 'BiGAT':
            print("Initializing BiGAT model...")
            model = BiGAT_graphcl(
                train_dataset.num_features, 
                args.hidden, 
                args.hidden, 
                num_classes, 
                tddroprate=0.1, 
                budroprate=0.1, 
                heads=args.heads
            ).to(device)
        else:
            raise ValueError(f"Unknown model: {args.model}")
        
        # Print model structure
        print(f"Model parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        # Initialize optimizer
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
        
        # Learning rate scheduler with warmup
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # Restart every 10 epochs
            T_mult=1,  # Multiply T_0 by 1 after each restart
            eta_min=1e-6  # Minimum learning rate
        )
        
        # Early stopping parameters
        patience = 15
        best_val_error = float('inf')
        counter = 0
        best_epoch = 0
        
        # Initialize model checkpoint
        best_model_path = osp.join(dirname, '..', 'checkpoints', f'best_model_run_{run}.pt')
        
        # Initial evaluation
        val_results = enhanced_test(model, val_loader, num_classes, device)
        test_results = enhanced_test(model, test_loader, num_classes, device)
        
        val_error = val_results['error']
        log_info = f"Initial - Val ERROR: {val_error:.7f}, Test ACC: {test_results['acc']:.4f}"
        write_log(log, log_info)
        print(log_info)
        
        # Training loop
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Get current learning rate
            lr = scheduler.optimizer.param_groups[0]['lr']
            
            # Calculate current lambda based on cosine annealing
            progress = epoch / epochs
            lamda_current = lamda_min + 0.5 * (lamda_max - lamda_min) * (1 + np.cos(progress * np.pi))
            
            # Train the model
            train_loss, sup_loss, unsup_loss = enhanced_sup_train(
                train_loader, args.aug1, args.aug2, model, optimizer, device, lamda_current, use_unsup_loss
            )
            
            # Evaluate on training set
            train_results = enhanced_test(model, train_loader, num_classes, device)
            train_error = train_results['error']
            train_acc = train_results['acc']
            
            # Evaluate and log results
            val_error, log_info, log_record = enhanced_test_and_log(
                model, val_loader, test_loader, num_classes, device, epoch,
                lr, sup_loss, unsup_loss if use_unsup_loss else 0, train_acc, log_record
            )
            
            # Add epoch time to log
            epoch_time = time.time() - start_time
            log_info += f"\n  Epoch time: {epoch_time:.2f} seconds"
            
            write_log(log, log_info)
            print(log_info)
            
            # Update scheduler
            scheduler.step()
            
            # Early stopping check
            if val_error < best_val_error:
                best_val_error = val_error
                counter = 0
                best_epoch = epoch
                
                # Save best model
                torch.save(model.state_dict(), best_model_path)
                
                # Update log record for best model
                log_record['best_epoch'] = best_epoch
                log_record['best_acc'] = log_record['test accs'][-1]
                log_record['best_f1'] = log_record['test macro metric'][-1][2]  # Macro F1
                
                print(f"  New best model saved at epoch {epoch}!")
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    write_log(log, f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model for final evaluation
        model.load_state_dict(torch.load(best_model_path))
        
        # Final evaluation
        final_test_results = enhanced_test(model, test_loader, num_classes, device)
        
        # Calculate per-class performance
        per_class_performance = []
        for c in range(num_classes):
            per_class_performance.append({
                'Class': c,
                'Precision': log_record['test precs'][-1][c],
                'Recall': log_record['test recs'][-1][c],
                'F1': log_record['test f1s'][-1][c]
            })
        
        # Store best results for this run
        all_runs_best_acc.append(log_record['best_acc'])
        all_runs_best_f1.append(log_record['best_f1'])
        
        # Log final results
        final_log = (
            f"\n==== Run {run+1} Summary ====\n"
            f"Best epoch: {best_epoch}, Best test accuracy: {log_record['best_acc']:.4f}, "
            f"Best macro F1: {log_record['best_f1']:.4f}\n"
            f"Final test accuracy: {final_test_results['acc']:.4f}, "
            f"Final weighted F1: {final_test_results['weighted_f1']:.4f}\n"
            f"Per-class performance at best epoch:\n"
        )
        
        # Add per-class performance
        for cls_perf in per_class_performance:
            final_log += f"  Class {cls_perf['Class']}: P={cls_perf['Precision']:.4f}, R={cls_perf['Recall']:.4f}, F1={cls_perf['F1']:.4f}\n"
        
        write_log(log, final_log)
        print(final_log)
        
        # Update log record with mean accuracy
        log_record['mean acc'] = round(np.mean(log_record['test accs'][-10:]), 4)
        write_log(log, '')
        
        # Update log dictionary
        log_dict['record'].append(log_record)
        write_json(log_dict, log_json_path)
    
    # Final summary across all runs
    mean_acc = np.mean(all_runs_best_acc)
    std_acc = np.std(all_runs_best_acc)
    mean_f1 = np.mean(all_runs_best_f1)
    std_f1 = np.std(all_runs_best_f1)
    
    summary = (
        f"\n==== Overall Results ({runs} runs) ====\n"
        f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n"
        f"Mean macro F1: {mean_f1:.4f} ± {std_f1:.4f}\n"
        f"Best run accuracy: {max(all_runs_best_acc):.4f} (Run {all_runs_best_acc.index(max(all_runs_best_acc))+1})\n"
        f"Best run F1: {max(all_runs_best_f1):.4f} (Run {all_runs_best_f1.index(max(all_runs_best_f1))+1})"
    )
    
    write_log(log, summary)
    print(summary)
    
    # Close log file
    log.close()
    print(f"Training completed. Results saved to {log_path} and {log_json_path}")