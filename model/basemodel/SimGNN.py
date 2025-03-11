import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

class SimGNN(nn.Module):
    def __init__(self, config):
        super(SimGNN, self).__init__()
        self._config = config
        self._num_features = config['num_features']
        self._device = 'cuda:' + str(config['gpu_index']) if config['gpu_index'] >= 0 else 'cpu'
        self._setup_layers()

    def _calculate_bottleneck_features(self):
        """
        送入全连接层之前的维度大小，根据是否使用直方图而改变，如果使用直方图就是32 * 1， 否则就是16 * 1\\
        最终的维度大小存入feature_count中
        """
        if self._config['histogram']:
            self.feature_count = self._config['tensor_neurons'] + self._config['bins']  # 32
        else:
            self.feature_count = self._config['tensor_neurons'] # 16

    def _setup_layers(self):
        self._calculate_bottleneck_features()
        # 三层GCN，原文中设置为：64 --> 32 --> 16
        self.conv1 = GCNConv(self._num_features, self._config['filters_1'])        #    --> 64
        self.conv2 = GCNConv(self._config['filters_1'], self._config['filters_2']) # 64 --> 32
        self.conv3 = GCNConv(self._config['filters_2'], self._config['filters_3']) # 32 --> 16

        self.attention = AttentionModule(self._config).to(self._device) # 1 * D

        self.tensor_network = TensorNetworkModule(self._config).to(self._device) # 1 * K

        self.fully_connected_first = nn.Linear(self.feature_count, self._config['bottle_neck_neurons_1'])                       #  -> 16
        self.fully_connected_second = nn.Linear(self._config['bottle_neck_neurons_1'], self._config['bottle_neck_neurons_2'])   # 16 -> 8
        self.fully_connected_third = nn.Linear(self._config['bottle_neck_neurons_2'], self._config['bottle_neck_neurons_3'])    # 8 -> 4
        self.scoring_layer = nn.Linear(self._config['bottle_neck_neurons_3'], 1)                                                # 4 -> 1

    def _calculate_histogram(self, Ui, Uj):
        """
        计算直方图
        Ui: 图i的特征矩阵 维度 Ni * D
        Uj: 图j的特征矩阵 维度 Nj * D
        返回直方图: 直方图的相似度分数 B
        """
        Ni, Di = Ui.shape
        Nj, Dj = Uj.shape

        N = max(Ni, Nj)

        S1 = torch.sigmoid(torch.mm(Ui, Uj.T)) # 维度应该是N * N, 实际上现在暂时得到的是Ni * Nj 因为文中说需要填充
        S = torch.zeros(N, N)
        S[:Ni, :Nj] = S1
        S = S.view(-1, 1)
        hist = torch.histc(S, bins=self._config['bins'])
        hist = hist / torch.sum(hist)        
        return hist.view(1, -1).to(self._device) # 1 * B

    def _convolutional_pass(self, A, X):
        """
        三层图卷积，得到节点表示\\
        A: 邻接矩阵\\
        X: 节点初始特征 维度 N * num_features\\
        返回 features: 一张图的节点表示 维度 N * D
        """
        features = self.conv1(X, A)
        features = F.relu(features)
        features = F.dropout(features, p=self._config['dropout'], training=self.training)
        features = self.conv2(features, A)
        features = F.relu(features)
        features = F.dropout(features, p=self._config['dropout'], training=self.training)
        features = self.conv3(features, A)
        return features

    def forward(self, data):
        edge_index_1 = data["g1"].edge_index
        edge_index_2 = data["g2"].edge_index
        features_1 = data["g1"].x
        features_2 = data["g2"].x

        # U1, U2分别是图1和图2的节点级表示
        U1 = self._convolutional_pass(edge_index_1, features_1) # N1 * D
        U2 = self._convolutional_pass(edge_index_2, features_2) # N2 * D

        if self._config['histogram']:
            hist = self._calculate_histogram(U1, U2) # 1 * B

        # h1, h2 是图1和图2的图级表示
        h1 = self.attention(U1) # 1 * D
        h2 = self.attention(U2) # 1 * D

        scores = self.tensor_network(h1, h2) # 1 * K

        if self._config['histogram']:
            scores = torch.cat((scores, hist), dim= 1) # 1 * (B + K)

        scores = F.relu(self.fully_connected_first(scores))
        scores = F.relu(self.fully_connected_second(scores))
        scores = F.relu(self.fully_connected_third(scores))

        score = torch.sigmoid(self.scoring_layer(scores)).view(-1)
        return score

class AttentionModule(nn.Module):
    """
    通过GCN得到节点嵌入以后，应该输入该模块得到图嵌入。\\
    输入的节点嵌入维度为 N * D，N是节点个数，D是嵌入维度，也就是filters_3.也就是16
    """
    def __init__(self, config):
        super(AttentionModule, self).__init__()
        self.config = config
        # 创建权重W2, 维度是D * D, 用于计算全局上下文c
        self.W2 = nn.Parameter(torch.Tensor(self.config['filters_3'], self.config['filters_3']))
        # 使用 Xavier 方法初始化权重矩阵W2, 据说可以缓解梯度下降问题
        nn.init.xavier_uniform_(self.W2)

    def forward(self, x): 
        """
        返回图级表示矩阵 根据后面的NTN模块，这里输出的h应该是D * 1格式 \\
        x 维度：N * D 整张图的节点级表示\\
        w2 维度：D * D \\
        c == transformed_global 的维度应该是 N * D \times D * D = N * D.mean() = 1 * D\\
        SimGNN公式(2)
        """
        c = torch.tanh(torch.mm(x, self.W2).mean(dim=0)).view(1, -1) # 1 * D
        h = torch.mm(x.T, torch.sigmoid(torch.mm(x, c.T))).T # 1 * D = (D * N \times (N * D \times D * 1)).T
        return h # 1 * D

class TensorNetworkModule(nn.Module):
    def __init__(self, config):
        super(TensorNetworkModule, self).__init__()
        self.config = config
        self.W3 = nn.Parameter(torch.Tensor(self.config['tensor_neurons'], self.config['filters_3'], self.config['filters_3'])) # K * D * D 特定跟论文中不同，为了更好实现代码
        self.V  = nn.Parameter(torch.Tensor(self.config['tensor_neurons'], 2 * self.config['filters_3']))                       # K * 2D
        self.b3 = nn.Parameter(torch.Tensor(1, self.config['tensor_neurons']))                                                  # 1 * K
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
