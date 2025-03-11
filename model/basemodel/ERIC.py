import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv

class ERIC(nn.Module):
    def __init__(self, config):
        super(ERIC, self).__init__()
        self._config = config
        self._device = 'cuda:' + str(config['gpu_index']) if config['gpu_index'] >= 0 else 'cpu'
        self._setup_layers()

    def _setup_layers(self):
        self._gnn_list = nn.ModuleList()
        self._mlp_list_inner = nn.ModuleList()
        self._mlp_list_outer = nn.ModuleList()
        self._NTN_list = nn.ModuleList()

        self._gnn_list.append( GINConv(nn.Sequential(
            nn.Linear(self._config['num_features'], self._config['filters'][0]),
            nn.ReLU(),
            nn.Linear(self._config['filters'][0], self._config['filters'][0]),
            nn.BatchNorm1d(self._config['filters'][0]) ), eps=True))
        for i in range(len(self._config['filters']) - 1):
            self._gnn_list.append( GINConv(nn.Sequential(
                nn.Linear(self._config['filters'][i], self._config['filters'][i + 1]),
                nn.ReLU(),
                nn.Linear(self._config['filters'][i + 1], self._config['filters'][i + 1]),
                nn.LayerNorm(self._config['filters'][i + 1])), eps=True))
        
        for i in range(len(self._config['filters'])):
            self._mlp_list_inner.append(MLPLayers(self._config['filters'][i], self._config['filters'][i], None, num_layers=1, use_bn=False))
            self._mlp_list_outer.append(MLPLayers(self._config['filters'][i], self._config['filters'][i], None, num_layers=1, use_bn=False))
        
        # self._NTN = TensorNetworkModule(self._config, self._config['filters'][-1])
        self._NTN = TensorNetworkModule(self._config)

        self._channel_dim = sum(self._config['filters'])
        self._reduction = self._config['reduction']
        self._conv_stack = nn.Sequential(
            nn.Linear(self._channel_dim, self._channel_dim // self._reduction),
            nn.ReLU(),
            nn.Dropout(p = self._config['dropout']),
            nn.Linear(self._channel_dim // self._reduction, (self._channel_dim // self._reduction) ),
            nn.Dropout(p = self._config['dropout']),
            nn.Tanh())
        self._GCL_model = GCL(self._config, sum(self._config['filters']))
        self._gamma = nn.Parameter(torch.Tensor(1))

        self._score_layer = nn.Sequential(
            nn.Linear((self._channel_dim // self._reduction) , 16),
            nn.ReLU(),
            nn.Linear(16 , 1))
        self._score_sim_layer = nn.Sequential(
            nn.Linear(self._config['tensor_neurons'], self._config['tensor_neurons']),
            nn.ReLU(),
            nn.Linear(self._config['tensor_neurons'], 1))
        self._alpha = nn.Parameter(torch.Tensor(1))
        self._beta = nn.Parameter(torch.Tensor(1))
    
    def _convolutional_pass_level(self, enc, edge_index, x):
        feat = enc(x, edge_index)
        feat = F.relu(feat)
        feat = F.dropout(feat, p = self._config['dropout'], training=self.training)
        return feat
    
    def _deepsets_outer(self, feat, filter_idx):
        pool = torch.sum(feat, dim=0).view(1, -1) # [1, D]
        temp = F.relu(self._mlp_list_outer[filter_idx](pool))
        return temp

    def forward(self, data):
        edge_index_1 = data['g1'].edge_index
        edge_index_2 = data['g2'].edge_index
        features_1 = data["g1"].x # [N, D]
        features_2 = data["g2"].x
        conv_source_1 = torch.clone(features_1)
        conv_source_2 = torch.clone(features_2)

        for i in range( len(self._config['filters']) ):  # 分层contrast
            conv_source_1 = self._convolutional_pass_level(self._gnn_list[i], edge_index_1, conv_source_1)
            conv_source_2 = self._convolutional_pass_level(self._gnn_list[i], edge_index_2, conv_source_2)
            deepsets_inner_1 = F.relu(self._mlp_list_inner[i](conv_source_1))
            deepsets_inner_2 = F.relu(self._mlp_list_inner[i](conv_source_2))
            deepsets_outer_1 = self._deepsets_outer(deepsets_inner_1, i)
            deepsets_outer_2 = self._deepsets_outer(deepsets_inner_2, i)
            diff_rep = torch.exp(- torch.pow(deepsets_outer_1 - deepsets_outer_2, 2)) if i == 0 else torch.cat((diff_rep, torch.exp(-torch.pow(deepsets_outer_1 - deepsets_outer_2, 2))), dim=1)
            cat_node_embeddings_1 = conv_source_1 if i == 0 else torch.cat((cat_node_embeddings_1, conv_source_1), dim = 1) # [N, 64 + 64 + 32 + 16]
            cat_node_embeddings_2 = conv_source_2 if i == 0 else torch.cat((cat_node_embeddings_2, conv_source_2), dim = 1)
            cat_global_embedding_1 = deepsets_outer_1 if i == 0 else torch.cat((cat_global_embedding_1, deepsets_outer_1), dim = 1)
            cat_global_embedding_2 = deepsets_outer_2 if i == 0 else torch.cat((cat_global_embedding_2, deepsets_outer_2), dim = 1)
        L_cl = 0
        if self.training:
            L_cl = self._GCL_model(cat_node_embeddings_1, cat_node_embeddings_2) * self._gamma
            L_cl = torch.tanh(L_cl)
        score_rep = self._conv_stack(diff_rep).view(1, -1)

        sim_rep = self._NTN(deepsets_outer_1, deepsets_outer_2)
        sim_score = torch.sigmoid(self._score_sim_layer(sim_rep).squeeze())
        
        score = torch.sigmoid(self._score_layer(score_rep)).view(-1)
            
        comb_score = self._alpha * score + self._beta * sim_score
        return comb_score, L_cl

class MLPLayers(nn.Module):
    def __init__(self, n_in, n_hid, n_out, num_layers = 2 ,use_bn=True, act = 'relu'):
        super(MLPLayers, self).__init__()
        modules = []
        modules.append(nn.Linear(n_in, n_hid))
        out = n_hid
        use_act = True
        for i in range(num_layers-1):  # num_layers = 3  i=0,1
            if i == num_layers - 2:
                use_bn = False
                use_act = False
                out = n_out
            modules.append(nn.Linear(n_hid, out))
            if use_bn:
                modules.append(nn.BatchNorm1d(out)) 
            if use_act:
                modules.append(nn.ReLU())
        self._mlp_list = nn.Sequential(*modules)

    def forward(self, x):
        x = self._mlp_list(x)
        return x

class TensorNetworkModule(nn.Module):
    def __init__(self, config):
        super(TensorNetworkModule, self).__init__()
        self._config = config
        self.W3 = nn.Parameter(torch.Tensor(self._config['tensor_neurons'], self._config['filters'][-1], self._config['filters'][-1])) # K * D * D 特定跟论文中不同，为了更好实现代码
        self.V  = nn.Parameter(torch.Tensor(self._config['tensor_neurons'], 2 * self._config['filters'][-1]))                          # K * 2D
        self.b3 = nn.Parameter(torch.Tensor(1, self._config['tensor_neurons']))                                                        # 1 * K
        nn.init.xavier_uniform_(self.W3)
        nn.init.xavier_uniform_(self.V)
        nn.init.xavier_uniform_(self.b3)

    def forward(self, hi, hj):
        """
        hi: 1 * D
        hj: 1 * D
        W3: D * D * K
        """
        term_1 = []
        for W_0 in self.W3:
            term_1.append(torch.mm(torch.mm(hi, W_0), hj.T))
        term_1 = torch.cat(term_1, dim=1) # 1 * K
        term_2 = torch.mm(self.V, torch.cat((hi, hj),dim = 1).T).T # 1 * K

        scores = F.relu(term_1 + term_2 + self.b3) # SimGNN公式(3)
        return scores # 1 * K


# class TensorNetworkModule(nn.Module):
#     def __init__(self, config, filters):
#         super(TensorNetworkModule, self).__init__()
#         self._config = config
#         self._filters = filters
#         self._setup_weights()
#         self._init_parameters()

#     def _setup_weights(self):
#         self._weight_matrix = nn.Parameter( torch.Tensor(self._filters, self._filters, self._config['tensor_neurons']) )
#         self._weight_matrix_block = nn.Parameter( torch.Tensor(self._config['tensor_neurons'], 2 * self._filters))
#         self._bias = nn.Parameter( torch.Tensor(self._config['tensor_neurons'], 1) )

#     def _init_parameters(self):
#         nn.init.xavier_uniform_(self._weight_matrix)
#         nn.init.xavier_uniform_(self._weight_matrix_block)
#         nn.init.xavier_uniform_(self._bias)

#     def forward(self, embedding_1, embedding_2):
#         scoring = torch.matmul(embedding_1, self._weight_matrix.view(self._filters, -1))
#         scoring = scoring.view(self._filters, -1).permute(1, 0)
#         scoring = torch.matmul(scoring, embedding_2.view(self._filters, 1))
#         combined_representation = torch.cat((embedding_1, embedding_2), 1)
#         block_scoring = torch.t(torch.mm(self._weight_matrix_block, torch.t(combined_representation)))
#         scores = F.relu(scoring + block_scoring + self._bias.view(-1))
#         print(scores.shape)
#         return scores

class GCL(nn.Module):
    def __init__(self, config, embedding_dim):
        super(GCL, self).__init__()
        self._config = config
        self._embedding_dim = embedding_dim
        self._init_emb()

    def _init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, z1, z2, g1 = None, g2 = None):
        g1 = torch.sum(z1, dim=0).view(1, -1) # [1, 176]
        g2 = torch.sum(z2, dim=0).view(1, -1) # [1, 176]
        self_sim_1   = torch.mm(z1,g1.t()) # [N, 1]
        self_sim_2   = torch.mm(z2,g2.t()) # [M, 1]
        cross_sim_12 = torch.mm(z1,g2.t()) # [N, 1]
        cross_sim_21 = torch.mm(z2,g1.t()) # [M, 1]

        L_1 = get_positive_expectation(self_sim_1, average=False).sum() - get_positive_expectation(cross_sim_12, average=False).sum()
        L_2 = get_positive_expectation(self_sim_2, average=False).sum() - get_positive_expectation(cross_sim_21, average=False).sum()
        return L_1 - L_2
    
def get_positive_expectation(p_samples, average=True):
    """Computes the positive part of a divergence / difference.

    Args:
        p_samples: Positive samples.    一个矩阵 [n_nodes, n_graphs] 每个节点和它所在的图的相似度， 其他位置为0
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor
    """
    log_2 = math.log(2.)
    Ep = log_2 - F.softplus(- p_samples)
    if average:
        return Ep.mean()
    else:
        return Ep
