# Graph Neural Networks for Rumor Detection

This repository contains implementation of various Graph Neural Network (GNN) architectures for rumor detection in social media, specifically focusing on the Weibo dataset.

## Overview

Rumor detection in social media is a challenging task due to the complex nature of information propagation. This project explores different GNN-based approaches to leverage the structural information in social media posts and their propagation patterns for effective rumor detection.

## Models

The repository implements several GNN architectures:

- **ResGCN**: Residual Graph Convolutional Network
- **BiGCN**: Bi-directional Graph Convolutional Network
- **ResGAT**: Residual Graph Attention Network
- **BiGAT**: Bi-directional Graph Attention Network
- **ResGAT with BERT**: Graph Attention Network enhanced with BERT embeddings

## Performance

### Overall Performance on Weibo Dataset

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| PLAN | 0.915 | 0.915 | 0.915 | 0.914 |
| BiGCN | 0.942 | 0.943 | 0.943 | 0.942 |
| UDGCN | 0.940 | 0.941 | 0.940 | 0.940 |
| GACL | 0.938 | 0.938 | 0.938 | 0.938 |
| DDGCN | 0.948 | 0.950 | 0.948 | 0.948 |
| RAGCL/ResGCN (Baseline) | 0.9582 | 0.9585 | 0.9583 | 0.9582 |
| ResGAT (Ours) | 0.9625 | 0.9633 | 0.9624 | 0.9625 |
| ResGAT+BERT (Ours) | 0.9646 | 0.9654 | 0.9648 | 0.9647 |

### Class-Specific Performance

| Model | Precision (Class 0) | Precision (Class 1) | Recall (Class 0) | Recall (Class 1) | F1 (Class 0) | F1 (Class 1) |
|-------|---------------------|---------------------|------------------|------------------|--------------|--------------|
| RAGCL/ResGCN (Baseline) | 0.9716 | 0.9454 | 0.9447 | 0.9719 | 0.9579 | 0.9585 |
| ResGAT (Ours) | 0.9446 | 0.9821 | 0.9829 | 0.9419 | 0.9634 | 0.9616 |
| ResGAT+BERT (Ours) | 0.9866 | 0.9442 | 0.9426 | 0.9870 | 0.9641 | 0.9652 |

Our experimental results demonstrate that:

1. **ResGAT outperforms ResGCN**: The attention mechanism in GAT allows the model to focus on important connections in the rumor propagation graph, leading to better performance compared to the GCN architecture (0.9625 vs 0.9582).

2. **BERT embeddings provide further improvement**: Incorporating BERT embeddings for Chinese text enhances the model's understanding of semantic content, resulting in the highest overall accuracy (0.9646) and F1 score (0.9647).

3. **Class-specific strengths**: ResGAT with BERT shows particularly strong precision for Class 0 (0.9866) and recall for Class 1 (0.9870), indicating balanced performance across rumor and non-rumor categories.

4. **Significant improvement over traditional baselines**: Our advanced models show substantial gains compared to earlier approaches like PLAN, BiGCN, and UDGCN.

## Features

- Multiple GNN architectures (ResGCN, BiGCN, ResGAT, BiGAT)
- BERT embeddings integration for Chinese text
- Support for different centrality metrics (PageRank, Degree, Eigenvector, Betweenness)
- Graph augmentation strategies (Edge dropping, Node dropping, Attribute masking)
- Semi-supervised and supervised training options
- Contrastive learning for improved representation

## Requirements

- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- Transformers (for BERT models)
- Gensim (for Word2Vec)
- NLTK and Jieba (for text processing)
- NetworkX (for graph centrality metrics)

## Project Structure

```
.
├── Data/                  # Dataset storage
├── Log/                   # Training logs
├── Main/                  # Main code
│   ├── augmentation.py    # Graph augmentation methods
│   ├── bert_dataset.py    # Dataset with BERT embeddings
│   ├── bert_embedding.py  # BERT embedding utilities
│   ├── dataset.py         # Dataset handling
│   ├── gat_model.py       # GAT model implementations
│   ├── main(sup).py       # Main script for supervised training
│   ├── main(semisup).py   # Main script for semi-supervised training
│   ├── main_bert_gat.py   # Main script for BERT+GAT model
│   ├── model.py           # GCN model implementations
│   ├── pargs.py           # Argument parsing
│   ├── sort.py            # Dataset sorting utilities
│   ├── utils.py           # Utility functions
│   └── word2vec.py        # Word2Vec embedding utilities
├── Model/                 # Saved models
├── centrality.py          # Node centrality calculation
└── process_data.py        # Data preprocessing script
```

## Usage

### Preprocessing

First, calculate node centrality for all data:

```bash
python centrality.py
```

Then preprocess the data:

```bash
python process_data.py
```

### Training

For supervised training with ResGAT:

```bash
python Main/main\(sup\).py --model ResGAT --dataset Weibo --centrality PageRank --batch_size 32 --epochs 100 --lr 0.001
```

For ResGAT with BERT embeddings:

```bash
python Main/main_bert_gat.py --dataset Weibo --centrality PageRank --batch_size 32 --epochs 100 --lr 0.001
```

### Configuration Options

- `--dataset`: Dataset name ('Weibo', 'Twitter15', etc.)
- `--model`: Model type ('ResGCN', 'BiGCN', 'ResGAT', 'BiGAT')
- `--centrality`: Node centrality metric ('PageRank', 'Degree', 'Eigenvector', 'Betweenness')
- `--aug1`, `--aug2`: Graph augmentation methods
- `--hidden`: Dimension of hidden layers
- `--heads`: Number of attention heads (for GAT models)
- `--use_unsup_loss`: Whether to use unsupervised contrastive loss

## Model Parameters

### ResGAT and ResGAT+BERT Configuration

```
Model parameters:
- Dataset: Weibo
- Train size: 20000
- Batch size: 32
- Undirected: True
- Hidden: 128
- Vector size: 200
- Layers: 1 feat, 3 conv, 2 fc
- Dropout: 0.3
- Edge norm: True
- Attention heads: 4
- Learning rate: 0.001
- Epochs: 100
- Weight decay: 0
- Centrality: PageRank
- Augmentation:
  - aug1: DropEdge,mean,0.2,0.7
  - aug2: NodeDrop,0.2,0.7
```

## Citation

If you use this code in your research, please cite our work:

```
@article{author2025graph,
  title={BERT-GAT: Enhancing Rumor Detection via Pre-trained Language Models and Graph Attention Networks for Social Media Propagation Trees},
  author={Nguyen Ngoc Thien, Bui Duc Nhan, Liu Yu-Tso},
  journal={},
  year={2025}
}
```

## Acknowledgements

This work builds upon previous research in graph-based rumor detection. We acknowledge the authors of the ResGCN and BiGCN models that served as our baselines.

## License

MIT