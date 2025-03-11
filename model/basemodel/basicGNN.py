import torch
import torch.nn as nn

from model.basemodel.FCL import FCL
from torch_geometric.nn import GAT, GCN, GIN

class Model(nn.Module):
    r"""包含GCN, GIN, GAT模块"""
    def __init__(self, model_name, in_dim, hidden_dim, num_layers):
        super(Model, self).__init__()
        if model_name == 'GCN':
            self._GNN = GCN(in_dim, hidden_dim, num_layers)
        elif model_name == 'GIN':
            self._GNN = GIN(in_dim, hidden_dim, num_layers)
        elif model_name == 'GAT':
            self._GNN = GAT(in_dim, hidden_dim, num_layers)
        else:
            raise ValueError("Unknown model: {}".format(model_name))
        
    def forward(self, x, edge_index):
        u = self._GNN(x, edge_index)
        return torch.mean(u, dim=0).view(1, -1)             # [1, 256]

class basicGNN(nn.Module):
    def __init__(self, config):
        super(basicGNN, self).__init__()
        _model_name = config['model']
        _in_dim = config['num_features']
        _hidden_dim = config['hidden_channels']
        _num_layers = config['num_layers']

        _device = 'cuda:' + str(config['gpu_index']) if config['gpu_index'] >= 0 else 'cpu'
        
        self._model = Model(_model_name, _in_dim, _hidden_dim, _num_layers).to(_device)
        self._fcl = FCL(_hidden_dim * 2).to(_device)
    
    def forward(self, data):
        x_1, edge_index_1 = data['g1'].x, data['g1'].edge_index
        x_2, edge_index_2 = data['g2'].x, data['g2'].edge_index
        u_1 = self._model(x_1, edge_index_1)
        u_2 = self._model(x_2, edge_index_2)
        score = self._fcl(u_1, u_2)
        return score
