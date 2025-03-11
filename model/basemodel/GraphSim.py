import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch

class GraphSim(nn.Module):
    def __init__(self, config):
        super(GraphSim, self).__init__()
        self._gnn = GNNLayers(config)
        self._cls = CLS(config['cls_dim'])

    def forward(self, data):
        edge_index_1, edge_index_2 = data['g1'].edge_index, data['g2'].edge_index
        x_1, x_2 = data['g1'].x, data['g2'].x

        x_1, x_2, x_3 = self._gnn(x_1, edge_index_1, x_2, edge_index_2)

        scores = self._cls(x_1, x_2, x_3)
        return scores.view(-1)

class GNNLayers(nn.Module):
    def __init__(self, config):
        super(GNNLayers, self).__init__()
        self._gcn1 = GCNConv(config['num_features'], config['gcn_first_dim'])
        self._gcn2 = GCNConv(config['gcn_first_dim'], config['gcn_second_dim'])
        self._gcn3 = GCNConv(config['gcn_second_dim'], config['gcn_third_dim'])
        self._reshape = config['reshape']

    def _Maxpadding_and_Resizing(self, x_t, x_s, max_num_nodes):
        x_t = to_dense_batch(x_t, max_num_nodes=max_num_nodes)[0]
        x_s = to_dense_batch(x_s, max_num_nodes=max_num_nodes)[0]
        x = torch.bmm(x_s, x_t.transpose(1, 2))
        x = F.interpolate(x.unsqueeze(1), size=(self._reshape, self._reshape), mode='bilinear', align_corners=False)
        return x

    def _node_embedding(self, x, edge_index):
        x = self._gcn1(x, edge_index).relu()
        x1 = x.clone()
        x = self._gcn2(x, edge_index).relu()
        x2 = x.clone()
        x = self._gcn3(x, edge_index).relu()
        x3 = x.clone()

        return x1, x2, x3

    def forward(self, x_1, edge_index_1, x_2, edge_index_2):
        x_s_1, x_s_2, x_s_3 = self._node_embedding(x_1, edge_index_1)
        x_t_1, x_t_2, x_t_3 = self._node_embedding(x_2, edge_index_2)

        num_nodes_1, num_nodes_2 = x_1.size(0), x_2.size(0)
        max_num_nodes = max(num_nodes_1, num_nodes_2)
        x_1 = self._Maxpadding_and_Resizing(x_t_1, x_s_1, max_num_nodes)
        x_2 = self._Maxpadding_and_Resizing(x_t_2, x_s_2, max_num_nodes)
        x_3 = self._Maxpadding_and_Resizing(x_t_3, x_s_3, max_num_nodes)
        return x_1, x_2, x_3

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self._cnn = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=8,kernel_size=6,stride=1),
            # nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=6,stride=1),
            # nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1),
            # nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1),
            # nn.MaxPool2d(3)
            # nn.ReLU()
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        x = self._cnn(x)
        return x

class CLS(nn.Module):
    def __init__(self, hidden_size):
        super(CLS, self).__init__()
        self._cnn = nn.ModuleList([CNN(),CNN(),CNN()])
        self._linear = nn.Sequential(
                # nn.Flatten(),
                nn.Linear(32 * 3, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
        )

    def forward(self, x_1, x_2, x_3):
        x_1 = self._cnn[0](x_1)
        x_2 = self._cnn[1](x_2)
        x_3 = self._cnn[2](x_3)
        x = torch.cat([x_1, x_2, x_3],dim=1)
        x = self._linear(x)
        return x.squeeze(1)
