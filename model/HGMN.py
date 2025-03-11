import torch
import torch.nn as nn
import torch.nn.functional as F

# from model.layers import RGCN, AttnMatch, FCL, CNN
from model.RGIN import RGIN
from model.layers import CrossAttention, MatchAttention, FCL
from utils.utils import profiled_function, print_evals

class HGMN(nn.Module):
    r"""Heterogeneous Graph Matching Networks"""
    def __init__(self, config):
        super(HGMN, self).__init__()
        self._config = config
        self._device = 'cuda:' + str(config['gpu_index']) if config['gpu_index'] >= 0 else 'cpu'
        self._setup_layers()

    def _setup_layers(self):
        self._nconv = RGIN(self._config).to(self._device)

        if self._config['node_match'] is False:
            # 只有节点生成的图级别表示
            self._fcl = FCL(self._config['hidden_dim'] * 2, 2).to(self._device)
        else:
            # 有节点表示生成的图表示，以及节点匹配信息
            self._nca = CrossAttention(self._config).to(self._device)    # 跨图注意力层
            self._nmatch = MatchAttention(self._config).to(self._device) # 对齐注意力层
            
            self._fcl = FCL(self._config['hidden_dim'] * 4, 4).to(self._device)

    def _pad(self, u_1, u_2):
        r"""
            生成两张图的mask矩阵，并且将两张图扩充到一样的大小
        """
        n_1, n_2, d = u_1.shape[0], u_2.shape[0], u_2.shape[1]
        max_nums = self._config['max_nums']
        u_1 = torch.cat([u_1, torch.zeros(max_nums - n_1, d).to(self._device)])
        u_2 = torch.cat([u_2, torch.zeros(max_nums - n_2, d).to(self._device)])
        mask_1 = torch.cat([torch.ones(n_1), torch.zeros(max_nums - n_1)]).to(self._device)
        mask_2 = torch.cat([torch.ones(n_2), torch.zeros(max_nums - n_2)]).to(self._device)
        return u_1, mask_1, u_2, mask_2

    def forward(self, data):
        # 得到节点表示 u_1: [N, 128] u_2: [M, 128]
        x_1, edge_index_1, node_type_1, edge_type_1 = data['g1'].x, data['g1'].edge_index, data['g1'].node_type, data['g1'].edge_type
        x_2, edge_index_2, node_type_2, edge_type_2 = data['g2'].x, data['g2'].edge_index, data['g2'].node_type, data['g2'].edge_type
        u_1, u_2 = self._nconv(x_1, edge_index_1, edge_type_1), self._nconv(x_2, edge_index_2, edge_type_2)

        # 得到图级别表示 h_1: [1, 128] h_2: [1, 128]
        h_1 = torch.mean(u_1, dim=0).view(1, -1)
        h_2 = torch.mean(u_2, dim=0).view(1, -1)

        h = torch.cat([h_1, h_2], 1)
        # graph_score = self._fcl_graph(h)
        # u_1, _, u_2, _ = self._pad(u_1, u_2)
        # simmat = self._ca(u_1, u_2, node_type_1, node_type_2)
        # node_match = self._match(simmat) # [1, 256]
        # node_score = self._fcl(node_match)
        # return graph_score, node_score



        if self._config['node_match'] is False:
            # 只有节点级嵌入生成的图表示
            score = self._fcl(h)
            return score
        
        # 否则一定有节点匹配矩阵
        u_1, _, u_2, _ = self._pad(u_1, u_2) # [max_nums, 128] [1, max_nums] [max_nums, 128] [1, max_nums]
        simmat = self._nca(u_1, u_2, node_type_1, node_type_2)
        node_match = self._nmatch(simmat) # [1, 256]
        nh = torch.cat([h, node_match], 1) # [1, 512]

        score = self._fcl(nh)
        return score
