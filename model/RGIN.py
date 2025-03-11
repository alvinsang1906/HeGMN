import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional
from torch_geometric.typing import Adj
from torch_geometric.nn import MessagePassing, inits
from torch_geometric.utils import one_hot, scatter

class RGINConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, num_bases=4):
        super(RGINConv, self).__init__(aggr='add')
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._num_relations = num_relations
        self._num_bases = num_bases

        self._weight = nn.Parameter( torch.empty(num_bases, in_channels, in_channels) )
        self._comp = nn.Parameter( torch.empty(num_relations, num_bases) )
        self._root = nn.Parameter( torch.empty(in_channels, in_channels) )
        self._bias = nn.Parameter( torch.empty(in_channels) )
        self._eps = nn.Parameter(torch.Tensor([0]))

        self._mlp = nn.Sequential(  nn.Linear(in_channels, out_channels), 
                                    nn.LayerNorm(out_channels), 
                                    nn.ReLU(), 
                                    nn.Linear(out_channels, out_channels) )
        
        self._reset_parameters()

    def _reset_parameters(self):
        super().reset_parameters()
        inits.glorot(self._weight)
        inits.glorot(self._comp)
        inits.glorot(self._root)
        inits.zeros(self._bias)

    def message(self, x_j: Tensor, edge_type: Tensor, edge_index_j: Tensor) -> Tensor:
        weight = (self._comp @ self._weight.view(self._num_bases, -1)).view( self._num_relations, self._in_channels, self._in_channels ) # [R, I, O]
        if not torch.is_floating_point(x_j):
            weight_index = edge_type * weight.size(1) + edge_index_j
            return weight.view(-1, self.out_channels)[weight_index]

        return torch.bmm(x_j.unsqueeze(-2), weight[edge_type]).squeeze(-2)

    def aggregate(self, inputs: Tensor, edge_type: Tensor, index: Tensor, dim_size: Optional[int] = None) -> Tensor:
        norm = one_hot(edge_type, self._num_relations, dtype=inputs.dtype)
        norm = scatter(norm, index, dim=0, dim_size=dim_size)[index]
        norm = torch.gather(norm, 1, edge_type.view(-1, 1))
        norm = 1. / norm.clamp_(1.)
        inputs = norm * inputs
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size)

    def forward(self, x: Tensor, edge_index: Adj, edge_type: Tensor):
        size = (x.size(0), x.size(0))
        out = self.propagate(edge_index, x=x, edge_type=edge_type, size=size)
        out += x @ self._root + self._bias
        out += (1 + self._eps) * x
        out = self._mlp(out)
        return out

class RGIN(nn.Module):
    def __init__(self, config):
        super(RGIN, self).__init__()
        self._dropout = config['dropout']
        self._conv1 = RGINConv(config['num_features'], config['hidden_dim'], config['num_etypes'])
        self._conv2 = RGINConv(config['hidden_dim'], config['hidden_dim'], config['num_etypes'])
        self._conv3 = RGINConv(config['hidden_dim'], config['hidden_dim'], config['num_etypes'])

    def forward(self, x, edge_index, edge_type):
        u_1 = self._conv1(x, edge_index, edge_type)
        u_1 = F.relu(u_1)

        u_2 = self._conv2(u_1, edge_index, edge_type)
        u_2 = F.relu(u_2)

        u = self._conv3(u_2, edge_index, edge_type)
        return u
