# -*- coding: utf-8 -*-
# Modified TreeDataset with BERT embeddings for Chinese text
import os
import json
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected
import numpy as np
from bert_embedding import BertEmbedding
import datetime

class BertTreeDataset(InMemoryDataset):
    def __init__(self, root, bert_model_name='bert-base-chinese', centrality_metric='PageRank', 
                 undirected=True, cache_dir=None, device=None, transform=None, pre_transform=None,
                 pre_filter=None):
        """
        TreeDataset with BERT embeddings for Chinese text
        
        Args:
            root: Root directory where the dataset should be saved
            bert_model_name: BERT model name ('bert-base-chinese' for Chinese)
            centrality_metric: Node centrality metric
            undirected: If True, convert graph to undirected
            cache_dir: Directory to cache BERT embeddings
            device: Device to run BERT model on
            transform, pre_transform, pre_filter: Dataset transforms
        """
        self.bert_model_name = bert_model_name
        self.centrality_metric = centrality_metric
        self.undirected = undirected
        self.cache_dir = cache_dir
        self.device = device
        
        # Initialize BERT embedder
        self.bert = BertEmbedding(model_name=bert_model_name, device=device)
        
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass
    
    def extract_temporal_info(self, post):
        """
        Extract temporal information from post
        
        Args:
            post: JSON post data
            
        Returns:
            Temporal info tensor
        """
        timestamps = []
        # Root post (source)
        timestamps.append(0.0)  # Source post is always at time 0
        
        # Comments
        for comment in post['comment']:
            # Ideally, extract from 'created_at' field if available
            # For the example, we use comment_id as a proxy for time
            comment_id = comment['comment id']
            timestamps.append(float(comment_id) / 1000.0)
        
        # Normalize timestamps to [0, 1]
        if len(timestamps) > 1:
            min_time = min(timestamps)
            max_time = max(timestamps)
            if max_time > min_time:
                timestamps = [(t - min_time) / (max_time - min_time) for t in timestamps]
        
        return torch.tensor(timestamps, dtype=torch.float32)

    def process(self):
        data_list = []
        raw_file_names = self.raw_file_names
        
        # Cache for BERT embeddings
        cache_file = None
        cached_embeddings = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(
                self.cache_dir, 
                f'bert_embeddings_{self.bert_model_name.replace("/", "_")}.pt'
            )
            if os.path.exists(cache_file):
                cached_data = torch.load(cache_file)
                cached_embeddings = cached_data.get('embeddings', {})
                print(f"Loaded {len(cached_embeddings)} cached embeddings")

        # Process each file
        for filename in raw_file_names:
            centrality = None
            y = []
            row = []
            col = []
            no_root_row = []
            no_root_col = []

            filepath = os.path.join(self.raw_dir, filename)
            post = json.load(open(filepath, 'r', encoding='utf-8'))
            
            # Extract all texts for batch processing
            texts = [post['source']['content']]
            for comment in post['comment']:
                texts.append(comment['content'])
            
            # Check cache first
            if cached_embeddings and filename in cached_embeddings:
                x = cached_embeddings[filename]
                print(f"Using cached embeddings for {filename}")
            else:
                # Get BERT embeddings for all texts in one batch
                x = self.bert.batch_encode(texts)
                
                # Cache embeddings
                if cache_file:
                    if cached_embeddings is None:
                        cached_embeddings = {}
                    cached_embeddings[filename] = x
                    torch.save({'embeddings': cached_embeddings}, cache_file)
                    print(f"Cached embeddings for {filename}")
            
            # Get label
            if 'label' in post['source'].keys():
                y.append(post['source']['label'])
            
            # Build graph structure
            for i, comment in enumerate(post['comment']):
                if comment['parent'] != -1:
                    no_root_row.append(comment['parent'] + 1)
                    no_root_col.append(comment['comment id'] + 1)
                row.append(comment['parent'] + 1)
                col.append(comment['comment id'] + 1)

            # Get centrality information
            if self.centrality_metric == "Degree":
                centrality = torch.tensor(post['centrality']['Degree'], dtype=torch.float32)
            elif self.centrality_metric == "PageRank":
                centrality = torch.tensor(post['centrality']['Pagerank'], dtype=torch.float32)
            elif self.centrality_metric == "Eigenvector":
                centrality = torch.tensor(post['centrality']['Eigenvector'], dtype=torch.float32)
            elif self.centrality_metric == "Betweenness":
                centrality = torch.tensor(post['centrality']['Betweenness'], dtype=torch.float32)
            
            # Build edge indices
            edge_index = [row, col]
            no_root_edge_index = [no_root_row, no_root_col]
            y = torch.LongTensor(y)
            edge_index = to_undirected(torch.LongTensor(edge_index)) if self.undirected else torch.LongTensor(edge_index)
            no_root_edge_index = torch.LongTensor(no_root_edge_index)
            
            # Extract temporal information
            time_info = self.extract_temporal_info(post)
            
            # Create data object
            one_data = Data(
                x=x, 
                y=y, 
                edge_index=edge_index, 
                no_root_edge_index=no_root_edge_index,
                centrality=centrality,
                time=time_info
            ) if 'label' in post['source'].keys() else \
                Data(
                    x=x, 
                    edge_index=edge_index, 
                    no_root_edge_index=no_root_edge_index, 
                    centrality=centrality,
                    time=time_info
                )
            
            data_list.append(one_data)

        # Apply filters and transforms
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        # Save processed data
        all_data, slices = self.collate(data_list)
        torch.save((all_data, slices), self.processed_paths[0])