import os.path as osp
import json
from typing import Callable, Union, Optional
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, EdgeConv, DynamicEdgeConv, TransformerConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GINConv, GINEConv

from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch_geometric.nn import knn_graph


def smooth_loss(pred, target):
    eps = 0.2

    n_class = pred.size(1)
    target = target.long()
    one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)

    loss = -(one_hot * log_prb).sum(dim=1).mean()

    return loss

def MLP(channels, batch_norm=True):
    return nn.Sequential(*[
        nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.LeakyReLU(), nn.BatchNorm1d(channels[i]))
        for i in range(1, len(channels))
    ])

class DGCNNNet(torch.nn.Module):
    def __init__(self, out_channels, k=20, aggr='max'):
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.lin1 = MLP([128 + 64, 1024])

        self.mlp = nn.Sequential(
            MLP([1024, 512]), nn.Dropout(0.5), MLP([512, 256]), nn.Dropout(0.5),
            nn.Linear(256, out_channels))

    def forward(self, data):

        pos, batch = data.pos.float(), data.batch
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch) # [32, 1024]
        graph_emb = out
        out = self.mlp(out) # [32, 10]

        return out, graph_emb

# PointNet2
class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNet2(torch.nn.Module):
    def __init__(self, out_channels):
        super(PointNet2, self).__init__()

        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = Lin(1024, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, out_channels)

    def forward(self, data):

        sa0_out = (data.x, data.pos.float(), data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out # x: size([32, 1024])
        global_emb = x
        
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)

        return x, global_emb


class SimpleTransform(torch.nn.Module):
   def __init__(self, num_node_features = 1024):
        super().__init__()
        self.input_transform = MLP([3, num_node_features])


   def forward(self, input):   

        output = self.input_transform(input)


        return output

# Graph Embedding Network
class GraphNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GraphNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        
        self.lin = MLP([1024, 512, 256, num_classes])

        self.lin1 = MLP([2*hidden_channels, 1024])

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x1 = self.conv1(x, edge_index)
        x1 = F.leaky_relu(x1)
        x2 = self.conv2(x1, edge_index)
        x2 = F.leaky_relu(x2)

        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)  # [batch_size, hidden_channels]
        graph_emb = out

        out = F.dropout(out, p=0.5, training=self.training)
        out = self.lin(out)
        
        return out, graph_emb

