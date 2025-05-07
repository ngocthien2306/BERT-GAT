# -*- coding: utf-8 -*-
# BERT embeddings for Chinese text
import torch
from transformers import BertModel, BertTokenizer
import os
import numpy as np

class BertEmbedding:
    def __init__(self, model_name='bert-base-chinese', max_length=128, device=None):
        """
        BERT embeddings for Chinese text
        
        Args:
            model_name: Pre-trained BERT model name ('bert-base-chinese' recommended for Chinese)
            max_length: Maximum sequence length for BERT tokenizer
            device: Device to run the model on
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        
        print(f"Loading BERT model: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        print(f"BERT embedding dimension: {self.embedding_dim}")
    
    def get_sentence_embedding(self, text):
        """
        Get BERT embedding for a sentence
        
        Args:
            text: Input text
        
        Returns:
            Embedding tensor
        """
        # Handle empty text
        if not text or text.isspace():
            return torch.zeros(self.embedding_dim)
            
        # Tokenize and convert to tensors
        encoded_input = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move tensors to the correct device
        input_ids = encoded_input['input_ids'].to(self.device)
        attention_mask = encoded_input['attention_mask'].to(self.device)
        
        # Get embeddings (without gradient computation)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get the [CLS] token embedding (sentence representation)
            # Alternatively, you can use the mean of all token embeddings
            # cls_embedding = outputs.last_hidden_state[:, 0, :]
            
            # Calculate mean of token embeddings (weighted by attention mask)
            token_embeddings = outputs.last_hidden_state[0]
            mask = attention_mask[0].unsqueeze(-1).expand(token_embeddings.size()).float()
            masked_embeddings = token_embeddings * mask
            sum_embeddings = torch.sum(masked_embeddings, dim=0)
            sum_mask = torch.sum(attention_mask[0]).float()
            mean_embedding = sum_embeddings / sum_mask
            
            # Return embedding
            return mean_embedding
    
    def batch_encode(self, texts, batch_size=32):
        """
        Encode a batch of texts
        
        Args:
            texts: List of texts
            batch_size: Batch size for processing
        
        Returns:
            Tensor of embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encoded_inputs = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move tensors to device
            input_ids = encoded_inputs['input_ids'].to(self.device)
            attention_mask = encoded_inputs['attention_mask'].to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Get mean of token embeddings for each sentence
                batch_embeddings = []
                for j in range(len(batch_texts)):
                    token_embeddings = outputs.last_hidden_state[j]
                    mask = attention_mask[j].unsqueeze(-1).expand(token_embeddings.size()).float()
                    masked_embeddings = token_embeddings * mask
                    sum_embeddings = torch.sum(masked_embeddings, dim=0)
                    sum_mask = torch.clamp(torch.sum(attention_mask[j]).float(), min=1e-9)
                    mean_embedding = sum_embeddings / sum_mask
                    batch_embeddings.append(mean_embedding)
                
                # Stack batch embeddings
                batch_embeddings = torch.stack(batch_embeddings)
                all_embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        if all_embeddings:
            return torch.cat(all_embeddings, dim=0)
        else:
            return torch.zeros((0, self.embedding_dim))
    
    def save_cache(self, texts, cache_file):
        """
        Save embeddings for a list of texts to a cache file
        
        Args:
            texts: List of texts
            cache_file: Path to save cache
        """
        embeddings = self.batch_encode(texts)
        torch.save(embeddings, cache_file)
        print(f"Saved {len(texts)} embeddings to {cache_file}")
    
    def load_cache(self, cache_file):
        """
        Load embeddings from a cache file
        
        Args:
            cache_file: Path to cache file
        
        Returns:
            Tensor of embeddings
        """
        if os.path.exists(cache_file):
            embeddings = torch.load(cache_file, map_location=self.device)
            print(f"Loaded {embeddings.size(0)} embeddings from {cache_file}")
            return embeddings
        else:
            print(f"Cache file {cache_file} not found")
            return None