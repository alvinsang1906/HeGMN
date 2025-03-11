import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.backend
import torch_geometric.typing
from typing import Optional
from torch_geometric.nn import RGCNConv, GINConv, MessagePassing, inits
from torch_geometric.utils import scatter
from dgl.nn.pytorch import RelGraphConv
from torch_geometric.typing import torch_sparse

class TensorNetworkModule(nn.Module):
    def __init__(self, config):
        super(TensorNetworkModule, self).__init__()
        self._W = nn.Parameter(torch.Tensor(config['NTN_out_dim'], config['hidden_dim'], config['hidden_dim'])) # K * D * D
        self._V = nn.Parameter(torch.Tensor(config['NTN_out_dim'], 2 * config['hidden_dim']))                  # K * 2D
        self._b = nn.Parameter(torch.Tensor(1, config['NTN_out_dim']))                                          # 1 * K
        nn.init.xavier_uniform_(self._W)
        nn.init.xavier_uniform_(self._V)
        nn.init.xavier_uniform_(self._b)

    def forward(self, hi, hj):
        """
        hi: 1 * D
        hj: 1 * D
        W3: D * D * K
        """
        term_1 = []
        for W_0 in self._W:
            term_1.append(torch.mm(torch.mm(hi, W_0), hj.T))
        term_1 = torch.cat(term_1, dim=1) # 1 * K
        term_2 = torch.mm(self._V, torch.cat((hi, hj),dim = 1).T).T # 1 * K

        scores = F.relu(term_1 + term_2 + self._b)
        return scores # 1 * K


def masked_edge_index(edge_index, edge_mask):
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    return torch_sparse.masked_select_nnz(edge_index, edge_mask, layout='coo')

class RGINConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, num_bases=4):
        super(RGINConv, self).__init__(aggr='mean')
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._num_relations = num_relations
        self._num_bases = num_bases

        self._weight = nn.Parameter( torch.empty(num_bases, in_channels, out_channels) )
        self._comp = nn.Parameter( torch.empty(num_relations, num_bases) )
        self._root = nn.Parameter( torch.empty(in_channels, out_channels) )
        self._bias = nn.Parameter( torch.empty(out_channels) )

        self._use_segment_matmul_heuristic_output: Optional[bool] = None
        self._mlp = nn.Sequential( nn.Linear(in_channels, out_channels), 
                                    nn.LayerNorm(out_channels), 
                                    nn.ReLU(), 
                                    nn.Linear(out_channels, out_channels) )
        self._eps = nn.Parameter(torch.Tensor([0]))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        inits.glorot(self._weight)
        inits.glorot(self._comp)
        inits.glorot(self._root)
        inits.zeros(self._bias)

    def message(self, x_j, edge_type_ptr):
        return x_j
        weight = self._weights[edge_type]
        return F.relu( torch.mm(x_j, weight) )
    
    def update(self, aggr_out):
        return aggr_out
    
    def forward(self, x, edge_index, edge_type):
        weight = ( self._comp @ self._weight.view(self._num_bases, -1) ).view( self._num_relations, self._in_channels, self._out_channels )
        out = torch.zeros(x.size(0), self._out_channels, device=x.device)
        size = (x.size(0), x.size(0))

        use_segment_matmul = torch_geometric.backend.use_segment_matmul

        if use_segment_matmul is None:
                segment_count = scatter(torch.ones_like(edge_type), edge_type, dim_size=self._num_relations)

                self._use_segment_matmul_heuristic_output = (
                    torch_geometric.backend.use_segment_matmul_heuristic(
                        num_segments=self._num_relations,
                        max_segment_size=int(segment_count.max()),
                        in_channels=self._weight.size(1),
                        out_channels=self._weight.size(2),
                    ))

                assert self._use_segment_matmul_heuristic_output is not None
                use_segment_matmul = self._use_segment_matmul_heuristic_output

        for i in range(self._num_relations):
            tmp = masked_edge_index(edge_index, edge_type == i)

            h = self.propagate(tmp, x=x, edge_type_ptr=None,
                                size=size)
            out = out + (h @ weight[i])
        
        out = out + x @ self._root + self._bias #自身节点的偏置 --sangs? 是否需要？
        out = self._mlp((1 + self._eps) * x) + out
        return out

class RGIN(nn.Module):
    def __init__(self, config):
        super(RGIN, self).__init__()
        self._dropout = config['dropout']
        self._conv1 = RGINConv(config['num_features'], config['RGCN_hidden_dim'], config['num_etypes'])
        self._conv2 = RGINConv(config['RGCN_hidden_dim'], config['RGCN_hidden_dim'], config['num_etypes'])
        self._conv3 = RGINConv(config['RGCN_hidden_dim'], config['RGCN_hidden_dim'], config['num_etypes'])

    def forward(self, x, edge_index, edge_type):
        u_1 = self._conv1(x, edge_index, edge_type)
        u_1 = F.relu(u_1)

        u_2 = self._conv2(u_1, edge_index, edge_type)
        u_2 = F.relu(u_2)

        u = self._conv3(u_2, edge_index, edge_type)
        return u


class RGCN(nn.Module):
    def __init__(self, config, edges=False):
        super(RGCN, self).__init__()
        self._dropout = config['dropout']
        self._conv1 = RGCNConv(config['num_features'], config['RGCN_hidden_dim'], config['num_etypes'], is_sorted=True)
        self._conv2 = RGCNConv(config['RGCN_hidden_dim'], config['RGCN_hidden_dim'], config['num_etypes'], is_sorted=True)
        self._conv3 = RGCNConv(config['RGCN_hidden_dim'], config['RGCN_hidden_dim'], config['num_etypes'], is_sorted=True)

    def forward(self, x, edge_index, edge_type):
        u_1 = self._conv1(x, edge_index, edge_type)
        u_1 = F.relu(u_1)
        # u_1 = F.dropout(u_1, p=self._dropout, training=self.training)

        u_2 = self._conv2(u_1, edge_index, edge_type)
        u_2 = F.relu(u_2)
        # u_2 = F.dropout(u_2, self._dropout, training=self.training)

        u = self._conv3(u_2, edge_index, edge_type)

        return u # [N, 128]

class CrossAttention(nn.Module):
    def __init__(self, config):
        super(CrossAttention, self).__init__()
        self._in_dim = config['RGCN_hidden_dim']
        self._num_heads = config['num_heads']
        self._scale = self._in_dim ** -0.5

        self._set_up_layers()

    def _set_up_layers(self):
        self._q = nn.Linear(self._in_dim, self._num_heads * self._in_dim)
        self._k = nn.Linear(self._in_dim, self._num_heads * self._in_dim)

    def forward(self, emb_1, emb_2, node_type_1, node_type_2):
        q_1 = self._q(emb_1).view(-1, self._num_heads, self._in_dim).transpose(-2, -3)
        q_2 = self._q(emb_2).view(-1, self._num_heads, self._in_dim).transpose(-2, -3)
        k_1 = self._k(emb_1).view(-1, self._num_heads, self._in_dim).transpose(-2, -3).transpose(-1, -2)
        k_2 = self._k(emb_2).view(-1, self._num_heads, self._in_dim).transpose(-2, -3).transpose(-1, -2)
        # print("q_1.shape expected [8, max_nums, 128] is {}, k_1.shape expected [8, 128, max_nums] is {}".format(q_1.shape, k_1.shape))
        a_1 = torch.matmul(q_1, k_2) * self._scale
        a_2 = torch.matmul(q_2, k_1).transpose(-1, -2) * self._scale
        a = torch.cat([a_1, a_2]) # [16, max_nums, max_nums]

        # mask
        mask_1, mask_2 = torch.zeros_like(a_1), torch.zeros_like(a_2)
        for i in range(self._num_heads):
            for j in range(len(node_type_1)):
                for k in range(len(node_type_2)):
                    mask_1[i, j, k] = node_type_1[j] == node_type_2[k]

        for i in range(self._num_heads):
            for j in range(len(node_type_2)):
                for k in range(len(node_type_1)):
                    mask_2[i, j, k] = node_type_2[j] == node_type_1[k]
        a_1 *= mask_1
        a_2 *= mask_2
        a = torch.cat([a_1, a_2]) # [16, max_nums, max_nums]

        return a

class MatchAttention(nn.Module):
    def __init__(self, config):
        super(MatchAttention, self).__init__()
        self._config = config
        self._device = 'cuda:' + str(config['gpu_index']) if config['gpu_index'] >= 0 else 'cpu'
        self._set_up_layers()
    
    def _set_up_layers(self):
        self._align = Alignment(self._config).to(self._device)

        self._pooling = Pooling(self._config).to(self._device)

    def forward(self, mat):
        mat = self._align(mat)
        y = self._pooling(mat)
        return y

class Alignment(nn.Module):
    def __init__(self, config):
        super(Alignment, self).__init__()
        self._config = config
        self._max_nodes = config['max_nums']
        self._device = 'cuda:' + str(config['gpu_index']) if config['gpu_index'] >= 0 else 'cpu'
        self._set_up_layers()
    
    def _set_up_layers(self):
        self._norm = nn.LayerNorm(self._max_nodes ** 2)
        self._dropout = nn.Dropout(self._config['dropout'])
        self._self_attention = MultiHeadAttention(self._config).to(self._device)
        self._ffn = FeedForwardNetwork(self._config).to(self._device)

    def forward(self, x):
        H = x.shape[0]
        x = x.view(H, -1)

        y = self._norm(x)
        y = self._self_attention(y)
        y = self._dropout(y)
        x = x + y

        y = self._norm(x)
        y = self._ffn(y)
        y = self._dropout(y)
        x = x + y

        x = x.view(H, self._max_nodes, self._max_nodes)

        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self._in_dim = config['max_nums'] ** 2
        self._num_heads = 4
        self._scale = self._in_dim ** -0.5
        self._dropout = config['dropout']

        self._set_up_layers()
    
    def _set_up_layers(self):
        self._q = nn.Linear(self._in_dim, self._num_heads * self._in_dim)
        self._k = nn.Linear(self._in_dim, self._num_heads * self._in_dim)
        self._v = nn.Linear(self._in_dim, self._num_heads * self._in_dim)
        self._dropout = nn.Dropout(self._dropout)
        self._output = nn.Linear(self._num_heads * self._in_dim, self._in_dim)

        nn.init.xavier_uniform_(self._q.weight)
        nn.init.xavier_uniform_(self._k.weight)
        nn.init.xavier_uniform_(self._v.weight)
        nn.init.xavier_uniform_(self._output.weight)

    def forward(self, x):
        # print("x.shape expected [16, max_nums ** 2] is {}".format(x.shape))
        q = self._q(x).view(-1, self._num_heads, self._in_dim).transpose(0, 1) * self._scale        # [4, 16, max_nums ** 2]
        k = self._k(x).view(-1, self._num_heads, self._in_dim).transpose(0, 1).transpose(-1, -2)    # [4, max_nums ** 2, 16]
        v = self._v(x).view(-1, self._num_heads, self._in_dim).transpose(0, 1)                      # [4, 16, max_nums ** 2]
        # print("q.shape [4, 16, max_nums ** 2] {}".format(q.shape))
        # print("k.shape [4, max_nums ** 2, 16] {}".format(k.shape))
        # print("v.shape [4, 16, max_nums ** 2] {}".format(v.shape))

        a = torch.matmul(q, k)
        a = torch.softmax(a, dim=2) # [4, 16, 16]
        a = self._dropout(a)
        # print("a.shape expected [4, 16, 16] is {}".format(a.shape))
        y = a.matmul(v).transpose(-2, -3).contiguous().view(-1, self._num_heads * self._in_dim)
        y = self._output(y)
        return y

class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super(FeedForwardNetwork, self).__init__()
        self._in_dim = config['max_nums'] ** 2
        self._hidden_dim = config['RGCN_hidden_dim']
        self._set_up_layers()
    
    def _set_up_layers(self):
        self._l1 = nn.Linear(self._in_dim, self._hidden_dim)
        self._gelu = nn.GELU()
        self._l2 = nn.Linear(self._hidden_dim, self._in_dim)

    def forward(self, x):
        x = self._l1(x)
        x = self._gelu(x)
        x = self._l2(x)
        return x
    
class Pooling(nn.Module):
    def __init__(self, config):
        super(Pooling, self).__init__()
        self._config = config

        self._in_dim = config['num_heads'] * 2
        self._set_up_layers()
    
    def _set_up_layers(self):
        # 16, 32
        self._cnn1 = nn.Conv2d(self._in_dim, self._config['pool_first_dim'], (3, 3))
        self._pooling = nn.AdaptiveAvgPool2d((7, 7))
        # 32, 64
        self._cnn2 = nn.Conv2d(self._config['pool_first_dim'], self._config['pool_second_dim'], (3, 3))
        # 64, 1
        self._cnn3 = nn.Conv2d(self._config['pool_second_dim'], self._config['pool_third_dim'], (3, 3))
        # 1, 256
        self._cnn4 = nn.Conv2d(self._config['pool_third_dim'], self._config['pool_forth_dim'], (3, 3))

        nn.init.xavier_uniform_(self._cnn1.weight)
        nn.init.xavier_uniform_(self._cnn2.weight)
        nn.init.xavier_uniform_(self._cnn3.weight)
        nn.init.xavier_uniform_(self._cnn4.weight)

    def forward(self, sim_mat):
        out = F.leaky_relu(self._cnn1(sim_mat), 0.3, True)
        out = self._pooling(out)
        out = F.leaky_relu(self._cnn2(out), 0.3, True)
        out = F.leaky_relu(self._cnn3(out), 0.3, True)
        out = F.leaky_relu(self._cnn4(out), 0.3, True).squeeze().view(1, -1) # [1, 256]
        # print("out.shape expected [1, 256] is {}".format(out.shape))
        return out

class FCL(nn.Module):
    r"""四层全连接层"""
    def __init__(self, in_dim, factor):
        super(FCL, self).__init__()
        self._fc1 = nn.Linear(in_dim, in_dim // factor)
        self._fc2 = nn.Linear(in_dim // factor, in_dim // (factor * 2))
        self._fc3 = nn.Linear(in_dim // (factor * 2), in_dim // (factor * 4))
        self._score = nn.Linear(in_dim // (factor * 4), 1)
    
    def forward(self, h):
        score = F.relu(self._fc1(h))
        score = F.relu(self._fc2(score))
        score = F.relu(self._fc3(score))
        score = F.sigmoid(self._score(score))
        return score.view(-1)
