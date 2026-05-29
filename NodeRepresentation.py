import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep,global_sort_pool

class GNNNet(nn.Module):
    def __init__(self, n_output=1, num_features_pro=54, num_features_mol=78, output_dim=160, dropout=0.2):
        super(GNNNet, self).__init__()

        print('GNNNet Loaded with GAT Encoders')
        self.n_output = n_output
        
        # --- Drug GAT Layers ---
        # Using 2 heads for the first layer to capture multiple structural patterns
        self.mol_conv1 = GATConv(num_features_mol, num_features_mol, heads=2, dropout=dropout)
        # Input to conv2 must account for multi-head output: num_features * heads
        self.mol_conv2 = GATConv(num_features_mol * 2, num_features_mol * 2, heads=1)
        self.mol_conv3 = GATConv(num_features_mol * 2, num_features_mol * 4, heads=1)
        
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        # --- Protein GAT Layers ---
        self.pro_conv1 = GATConv(num_features_pro, num_features_pro, heads=2, dropout=dropout)
        self.pro_conv2 = GATConv(num_features_pro * 2, num_features_pro * 2, heads=1)
        self.pro_conv3 = GATConv(num_features_pro * 2, num_features_pro * 4, heads=1)
        
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug_feature, drug_adj, ibatch, pro_feature, pro_adj, pro_ibatch):
        # --- Drug branch ---
        x = self.relu(self.mol_conv1(drug_feature, drug_adj))
        x = self.relu(self.mol_conv2(x, drug_adj))
        x = self.relu(self.mol_conv3(x, drug_adj))
        x = gep(x, ibatch) # Global pooling [cite: 146, 150]
        
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)

        # --- Protein branch ---
        xt = self.relu(self.pro_conv1(pro_feature, pro_adj))
        xt = self.relu(self.pro_conv2(xt, pro_adj))
        xt = self.relu(self.pro_conv3(xt, pro_adj))
        xt = gep(xt, pro_ibatch) 
        
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)
        
        # Concatenate drug and protein features for the high-level graph [cite: 135, 136]
        xc = torch.cat((x, xt), 0)
        return xc

class combined(nn.Module):
    def __init__(self,n_output=160,output_dim=160,dropout=0.2):
        super(combined, self).__init__()

        # combined layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.n_output = n_output
        self.fc1 = nn.Linear(output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self,drug_fea,pro_fea):
        # add some dense layers
        features = torch.cat((drug_fea, pro_fea), 0)
        xc = self.fc1(features)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)

        return out


