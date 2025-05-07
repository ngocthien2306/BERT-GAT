import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Parameter, LayerNorm
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.inits import glorot, zeros
import numpy as np
import random

class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.3, edge_norm=True, residual=True):
        super(GATLayer, self).__init__()
        self.gat = GATConv(
            in_channels, 
            out_channels, 
            heads=heads, 
            dropout=dropout,
            concat=True,
            negative_slope=0.2
        )
        # Use both batch norm and layer norm for better stability
        self.bn = BatchNorm1d(out_channels * heads)
        self.ln = LayerNorm(out_channels * heads)
        
        self.residual = residual
        self.res_conv = None
        # Enhanced residual connection
        if residual:
            self.res_conv = Linear(in_channels, out_channels * heads)
            self.bn_res = BatchNorm1d(out_channels * heads)
            
    def forward(self, x, edge_index):
        # Store identity for residual
        identity = x
        
        # Apply GAT layer
        out = self.gat(x, edge_index)
        
        # Apply normalization
        out = self.bn(out)
        out = self.ln(out)
        
        # Apply residual connection if enabled
        if self.residual:
            if self.res_conv is not None:
                identity = self.res_conv(identity)
                identity = self.bn_res(identity)
            out = out + identity
            
        return F.relu(out)

class ResGAT(torch.nn.Module):
    """GAT with BN and residual connections."""

    def __init__(self, dataset=None, num_classes=2, hidden=128, num_feat_layers=1, num_conv_layers=3,
                 num_fc_layers=2, gfn=False, collapse=False, residual=True,
                 res_branch="BNConvReLU", global_pool="sum", dropout=0.3,
                 edge_norm=True, heads=4):
        super(ResGAT, self).__init__()
        assert num_feat_layers == 1, "more feat layers are not now supported"
        self.num_classes = num_classes
        self.conv_residual = residual
        self.fc_residual = residual  # Enable residual connections for fc layers
        self.res_branch = res_branch
        self.collapse = collapse
        assert "sum" in global_pool or "mean" in global_pool, global_pool
        if "sum" in global_pool:
            self.global_pool = global_add_pool
        else:
            self.global_pool = global_mean_pool
        self.dropout = dropout
        self.heads = heads
        self.edge_norm = edge_norm

        self.use_xg = False
        if hasattr(dataset, '0') and "xg" in dataset[0]:  # Utilize graph level features.
            self.use_xg = True
            self.bn1_xg = BatchNorm1d(dataset[0].xg.size(1))
            self.lin1_xg = Linear(dataset[0].xg.size(1), hidden)
            self.bn2_xg = BatchNorm1d(hidden)
            self.lin2_xg = Linear(hidden, hidden)

        hidden_in = dataset.num_features
        if collapse:
            self.bn_feat = BatchNorm1d(hidden_in)
            self.bns_fc = torch.nn.ModuleList()
            self.lins = torch.nn.ModuleList()
            
            # Improved gating mechanism
            if "gating" in global_pool:
                self.gating = torch.nn.Sequential(
                    Linear(hidden_in, hidden_in),
                    LayerNorm(hidden_in),
                    torch.nn.ReLU(),
                    Linear(hidden_in, 1),
                    torch.nn.Sigmoid())
            else:
                self.gating = None
                
            for i in range(num_fc_layers - 1):
                self.bns_fc.append(BatchNorm1d(hidden_in))
                self.lins.append(Linear(hidden_in, hidden))
                hidden_in = hidden
            self.lin_class = Linear(hidden_in, self.num_classes)
        else:
            self.bn_feat = BatchNorm1d(hidden_in)
            feat_gfn = True  # set true so GAT is feat transform
            
            # Enhanced initial feature transformation
            self.conv_feat = GATConv(hidden_in, hidden // heads, heads=heads, dropout=dropout, negative_slope=0.2)
            self.bn_conv_feat = BatchNorm1d(hidden)
            self.ln_conv_feat = LayerNorm(hidden)
            
            # Improved gating mechanism 
            if "gating" in global_pool:
                self.gating = torch.nn.Sequential(
                    Linear(hidden, hidden),
                    LayerNorm(hidden),
                    torch.nn.ReLU(),
                    Linear(hidden, hidden),
                    torch.nn.ReLU(),
                    Linear(hidden, 1),
                    torch.nn.Sigmoid())
            else:
                self.gating = None
                
            self.bns_conv = torch.nn.ModuleList()
            self.convs = torch.nn.ModuleList()
            
            # Build the convolutional layers with enhanced GAT
            for i in range(num_conv_layers):
                self.bns_conv.append(BatchNorm1d(hidden))
                self.convs.append(GATLayer(hidden, hidden // heads, heads, dropout, edge_norm, residual))
                
            self.bn_hidden = BatchNorm1d(hidden)
            self.ln_hidden = LayerNorm(hidden)
            self.bns_fc = torch.nn.ModuleList()
            self.lns_fc = torch.nn.ModuleList()
            self.lins = torch.nn.ModuleList()
            
            for i in range(num_fc_layers - 1):
                self.bns_fc.append(BatchNorm1d(hidden))
                self.lns_fc.append(LayerNorm(hidden))
                self.lins.append(Linear(hidden, hidden))
            self.lin_class = Linear(hidden, self.num_classes)

        # Enhanced initialization
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.LayerNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None
            
        x = self.bn_feat(x)
        x = self.conv_feat(x, edge_index)
        x = self.bn_conv_feat(x)
        x = self.ln_conv_feat(x)
        x = F.relu(x)
        
        # Apply enhanced conv layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
        
        # Improved gating mechanism
        if self.gating is not None:
            gate = self.gating(x)
        else:
            gate = 1
            
        # Apply global pooling with gating
        x = self.global_pool(x * gate, batch)
        
        # Add graph-level features if available
        if xg is not None:
            x = x + xg
        
        # Apply FC layers with residual connections
        for i, lin in enumerate(self.lins):
            x_identity = x
            x = self.bns_fc[i](x)
            if hasattr(self, 'lns_fc'):
                x = self.lns_fc[i](x)
            x = F.relu(lin(x))
            
            if self.fc_residual:
                x = x + x_identity
        
        # Final normalization
        x = self.bn_hidden(x)
        if hasattr(self, 'ln_hidden'):
            x = self.ln_hidden(x)
            
        # Apply dropout before classification
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Classification layer
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

class ResGAT_graphcl(ResGAT):
    def __init__(self, **kargs):
        super(ResGAT_graphcl, self).__init__(**kargs)
        hidden = kargs['hidden']
        
        # Enhanced projection head for better contrastive learning
        self.proj_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden)
        )
        
        # Add layer normalization for better stability
        self.proj_norm = nn.LayerNorm(hidden)

    def forward_graphcl(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        x = self.bn_feat(x)
        x = self.conv_feat(x, edge_index)
        x = self.bn_conv_feat(x)
        x = self.ln_conv_feat(x)
        x = F.relu(x)
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
        # Improved gating
        if self.gating is not None:
            gate = self.gating(x)
        else:
            gate = 1
            
        x = self.global_pool(x * gate, batch)
        if xg is not None:
            x = x + xg
            
        # Apply FC layers with improved residual
        for i, lin in enumerate(self.lins):
            x_identity = x
            x = self.bns_fc[i](x)
            if hasattr(self, 'lns_fc'):
                x = self.lns_fc[i](x)
            x = F.relu(lin(x))
            
            if self.fc_residual:
                x = x + x_identity
                
        x = self.bn_hidden(x)
        if hasattr(self, 'ln_hidden'):
            x = self.ln_hidden(x)
            
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Apply enhanced projection head
        x = self.proj_head(x)
        x = self.proj_norm(x)  # Add layer norm for stability
        return x

    def loss_graphcl(self, x1, x2, mean=True):
        T = 0.1  # Lower temperature for sharper distribution
        batch_size, _ = x1.size()

        # Normalize embeddings for cosine similarity
        x1_norm = F.normalize(x1, p=2, dim=1)
        x2_norm = F.normalize(x2, p=2, dim=1)

        # Compute cosine similarity
        sim_matrix = torch.mm(x1_norm, x2_norm.t()) / T
        
        # Compute InfoNCE loss
        sim_matrix = torch.exp(sim_matrix)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-8)  # Add small epsilon for numerical stability
        loss = - torch.log(loss)
        
        if mean:
            loss = loss.mean()
        return loss

class TDrumorGAT(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, tddroprate, heads=4):
        super(TDrumorGAT, self).__init__()
        self.tddroprate = tddroprate
        self.heads = heads
        
        # Enhanced initial GAT layer
        self.conv1 = GATConv(
            in_feats, 
            hid_feats // heads, 
            heads=heads, 
            dropout=0.2,
            negative_slope=0.2
        )
        self.bn1 = BatchNorm1d(hid_feats)
        self.ln1 = LayerNorm(hid_feats)
        
        # Enhanced second GAT layer
        self.conv2 = GATConv(
            hid_feats + in_feats, 
            out_feats // heads, 
            heads=heads,
            dropout=0.2,
            negative_slope=0.2
        )
        self.bn2 = BatchNorm1d(out_feats)
        self.ln2 = LayerNorm(out_feats)

    def forward(self, data):
        device = data.x.device
        x, edge_index = data.x, data.edge_index

        # Apply edge dropout for TD direction
        if self.tddroprate > 0:
            edge_index_list = edge_index.tolist()
            length = len(edge_index_list[0])
            # Use deterministic dropout with seed for reproducibility
            torch.manual_seed(42)  
            poslist = torch.randperm(length)[:int(length * (1 - self.tddroprate))]
            poslist = poslist.sort()[0]
            tdrow = list(np.array(edge_index_list[0])[poslist])
            tdcol = list(np.array(edge_index_list[1])[poslist])
            edge_index = torch.LongTensor([tdrow, tdcol]).to(device)

        # Store original features
        x1 = x.float().clone()
        
        # First layer with normalization
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.ln1(x)
        x = F.relu(x)
        
        # Store intermediate features
        x2 = x.clone()
        
        # Root extension mechanism
        batch_size = max(data.batch) + 1
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            if index.sum() > 0:  # Ensure the batch has at least one node
                root_extend[index] = x1[index][0]
        
        # Concatenate with root features
        x = torch.cat((x, root_extend), 1)
        
        # Apply dropout for regularization
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Second layer with normalization
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.ln2(x)
        x = F.relu(x)
        
        # Second root extension
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            if index.sum() > 0:
                root_extend[index] = x2[index][0]
        
        # Final concatenation
        x = torch.cat((x, root_extend), 1)
        
        # Mean pooling by batch
        x = scatter_mean(x, data.batch, dim=0)
        return x

class BUrumorGAT(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, budroprate, heads=4):
        super(BUrumorGAT, self).__init__()
        self.budroprate = budroprate
        self.heads = heads
        
        # Enhanced initial GAT layer
        self.conv1 = GATConv(
            in_feats, 
            hid_feats // heads, 
            heads=heads, 
            dropout=0.2,
            negative_slope=0.2
        )
        self.bn1 = BatchNorm1d(hid_feats)
        self.ln1 = LayerNorm(hid_feats)
        
        # Enhanced second GAT layer
        self.conv2 = GATConv(
            hid_feats + in_feats, 
            out_feats // heads, 
            heads=heads,
            dropout=0.2,
            negative_slope=0.2
        )
        self.bn2 = BatchNorm1d(out_feats)
        self.ln2 = LayerNorm(out_feats)

    def forward(self, data):
        device = data.x.device
        x = data.x
        
        # Reverse edge direction for bottom-up propagation
        edge_index = data.edge_index.clone()
        edge_index[0], edge_index[1] = data.edge_index[1], data.edge_index[0]

        # Apply edge dropout for BU direction
        if self.budroprate > 0:
            edge_index_list = edge_index.tolist()
            length = len(edge_index_list[0])
            # Use deterministic dropout with seed for reproducibility
            torch.manual_seed(84)  # Different seed from TD
            poslist = torch.randperm(length)[:int(length * (1 - self.budroprate))]
            poslist = poslist.sort()[0]
            burow = list(np.array(edge_index_list[0])[poslist])
            bucol = list(np.array(edge_index_list[1])[poslist])
            edge_index = torch.LongTensor([burow, bucol]).to(device)

        # Store original features
        x1 = x.float().clone()
        
        # First layer with normalization
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.ln1(x)
        x = F.relu(x)
        
        # Store intermediate features
        x2 = x.clone()
        
        # Root extension mechanism
        batch_size = max(data.batch) + 1
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            if index.sum() > 0:
                root_extend[index] = x1[index][0]
        
        # Concatenate with root features
        x = torch.cat((x, root_extend), 1)
        
        # Apply dropout for regularization
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Second layer with normalization
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.ln2(x)
        x = F.relu(x)
        
        # Second root extension
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            if index.sum() > 0:
                root_extend[index] = x2[index][0]
        
        # Final concatenation
        x = torch.cat((x, root_extend), 1)
        
        # Mean pooling by batch
        x = scatter_mean(x, data.batch, dim=0)
        return x

class BiGAT_graphcl(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_classes, tddroprate=0.0, budroprate=0.0, heads=4):
        super(BiGAT_graphcl, self).__init__()
        self.TDrumorGAT = TDrumorGAT(in_feats, hid_feats, out_feats, tddroprate, heads)
        self.BUrumorGAT = BUrumorGAT(in_feats, hid_feats, out_feats, budroprate, heads)
        
        combined_dim = (out_feats + hid_feats) * 2
        
        # Add attention mechanism to combine TD and BU
        self.attention = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )
        
        # Enhanced normalization and regularization
        self.bn_combined = BatchNorm1d(combined_dim)
        self.ln_combined = LayerNorm(combined_dim)
        
        # Project to lower dimension for contrastive learning
        self.proj_head = nn.Sequential(
            nn.Linear(combined_dim, out_feats),
            nn.BatchNorm1d(out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, out_feats)
        )
        
        # Classification layer with dropout
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(combined_dim, num_classes)

    def forward(self, data):
        # Get representations from both directions
        TD_x = self.TDrumorGAT(data)
        BU_x = self.BUrumorGAT(data)
        
        # Combine representations
        combined = torch.cat((BU_x, TD_x), 1)
        
        # Apply attention mechanism to weight TD and BU
        attn_weights = self.attention(combined)
        BU_weight = attn_weights[:, 0].unsqueeze(1)
        TD_weight = attn_weights[:, 1].unsqueeze(1)
        
        # Weighted sum
        BU_weighted = BU_x * BU_weight
        TD_weighted = TD_x * TD_weight
        combined_weighted = torch.cat((BU_weighted, TD_weighted), 1)
        
        # Apply normalization
        combined = self.bn_combined(combined_weighted)
        combined = self.ln_combined(combined)
        
        # Apply dropout and classification
        combined = self.dropout(combined)
        x = self.fc(combined)
        
        return F.log_softmax(x, dim=-1)

    def forward_graphcl(self, data):
        # Get representations from both directions
        TD_x = self.TDrumorGAT(data)
        BU_x = self.BUrumorGAT(data)
        
        # Combine representations
        combined = torch.cat((BU_x, TD_x), 1)
        
        # Apply normalization
        combined = self.bn_combined(combined)
        combined = self.ln_combined(combined)
        
        # Apply projection head
        x = self.proj_head(combined)
        
        return x

    def loss_graphcl(self, x1, x2, mean=True):
        T = 0.1  # Lower temperature for sharper distribution
        batch_size, _ = x1.size()

        # Normalize embeddings for cosine similarity
        x1_norm = F.normalize(x1, p=2, dim=1)
        x2_norm = F.normalize(x2, p=2, dim=1)

        # Compute cosine similarity
        sim_matrix = torch.mm(x1_norm, x2_norm.t()) / T
        
        # Compute InfoNCE loss with hard negative mining
        sim_matrix = torch.exp(sim_matrix)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        
        # Add small epsilon for numerical stability
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-8)
        loss = - torch.log(loss)
        
        if mean:
            loss = loss.mean()
        return loss
