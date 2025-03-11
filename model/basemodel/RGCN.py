import torch
import torch.nn as nn
import torch.nn.functional as F

from model.basemodel.FCL import FCL
from torch_geometric.nn import RGCNConv

class Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_relations):
        super(Model, self).__init__()
        self._conv1 = RGCNConv(in_dim, hidden_dim, num_relations, is_sorted=True)
        self._conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations, is_sorted=True)
        self._conv3 = RGCNConv(hidden_dim, hidden_dim, num_relations, is_sorted=True)

    def forward(self, x, edge_index, edge_type):
        u = self._conv1(x, edge_index, edge_type)
        u = F.relu(u)
        u = self._conv2(u, edge_index, edge_type)
        u = F.relu(u)
        u = self._conv3(u, edge_index, edge_type)
        return torch.mean(u, dim=0).view(1, -1)             # [1, 256]

class RGCN(nn.Module):
    def __init__(self, config):
        super(RGCN, self).__init__()
        _num_relations = config['num_etypes']
        _in_dim = config['num_features']
        _hidden_dim = config['hidden_channels']

        _device = 'cuda:' + str(config['gpu_index']) if config['gpu_index'] >= 0 else 'cpu'
        self._model = Model(_in_dim, _hidden_dim, _num_relations).to(_device)
        self._fcl = FCL(_hidden_dim * 2).to(_device)

    def forward(self, data):
        x_1, edge_index_1, edge_type_1 = data['g1'].x, data['g1'].edge_index, data['g1'].edge_type
        x_2, edge_index_2, edge_type_2 = data['g2'].x, data['g2'].edge_index, data['g2'].edge_type
        u_1 = self._model(x_1, edge_index_1, edge_type_1)
        u_2 = self._model(x_2, edge_index_2, edge_type_2)
        score = self._fcl(u_1, u_2)
        return score
