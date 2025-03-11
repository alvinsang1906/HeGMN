import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

class NAGSL(nn.Module):
    def __init__(self, config):
        super(NAGSL, self).__init__()
        self._config = config
        self._max_num_nodes = config['max_nums']
        self._device = 'cuda:' + str(config['gpu_index']) if config['gpu_index'] >= 0 else 'cpu'

        if config['share_qk']:
            q = nn.Linear(config['embedding_size'], config['embedding_size'] * config['n_heads'], bias=config['msa_bias'])
            k = nn.Linear(config['embedding_size'], config['embedding_size'] * config['n_heads'], bias=config['msa_bias'])
        else:
            q = k = None
        
        self.embedding_learning = GCNTransformerEncoder(config, q, k)

        self.embedding_interaction = CrossAttention(config, q, k)

        self.sim_mat_learning = SimMatLearning(config)

    def forward(self, data):
        x_0, x_1 = data['g1'].x, data['g2'].x
        edge_index_0, edge_index_1 = data['g1'].edge_index, data['g2'].edge_index

        embeddings_0 = self.embedding_learning(x_0, edge_index_0) # [Max, 128]
        embeddings_1 = self.embedding_learning(x_1, edge_index_1) # [Max, 128]
        
        sim_mat = self.embedding_interaction(embeddings_0, embeddings_1) # [16, Max, Max]
        score = self.sim_mat_learning(sim_mat).view(-1)

        return score

class GCNTransformerEncoder(nn.Module):
    def __init__(self, config, q=None, k=None):
        super(GCNTransformerEncoder, self).__init__()
        self._config = config
        self._device = 'cuda:' + str(config['gpu_index']) if config['gpu_index'] >= 0 else 'cpu'
        self.GCN_first = GCNConv(config['num_features'], config['embedding_size'])
        self.GCN_second = GCNConv(config['embedding_size'], config['embedding_size'])
        self.GCN_third = GCNConv(config['embedding_size'], config['embedding_size'])

        self.self_attention_norm = nn.LayerNorm(config['embedding_size'])
        self.self_attention = MultiHeadAttention(config, q, k)
        self.self_attention_dropout = nn.Dropout(config['dropout'])

        self.ffn_norm = nn.LayerNorm(config['embedding_size'])
        self.ffn = FeedForwardNetwork(config)
        self.ffn_dropout = nn.Dropout(config['dropout'])

    def forward(self, x, edge_index):
        first_gcn_result = F.relu( self.GCN_first(x, edge_index) )
        second_gcn_result = F.relu( self.GCN_second(first_gcn_result, edge_index) )
        gcn_result = F.relu( self.GCN_third(second_gcn_result, edge_index) )

        gcn_result = gcn_result + first_gcn_result + second_gcn_result if self._config['GT_res'] else gcn_result
        max_num = self._config['max_nums']
        diff_rows = max_num - gcn_result.shape[0]
        gcn_result = F.pad(gcn_result, (0, 0, 0, diff_rows)).to(self._device) # [Max, 128]

        self_att_result = self.self_attention_norm(gcn_result)
        self_att_result = self.self_attention(self_att_result)
        self_att_result = self.self_attention_dropout(self_att_result)
        self_att_result = gcn_result + self_att_result

        ffn_result = self.ffn_norm(self_att_result)
        ffn_result = self.ffn(ffn_result)
        ffn_result = self.ffn_dropout(ffn_result)
        self_att_result = self_att_result + ffn_result

        encoder_result = gcn_result + self_att_result if self._config['GT_res'] else self_att_result

        return encoder_result

class MultiHeadAttention(nn.Module):
    def __init__(self, config, q=None, k=None):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads = config['n_heads'] # 8
        embedding_size = config['embedding_size'] # 128

        self.att_size = embedding_size
        self.scale = embedding_size ** -0.5

        self.linear_q = q if q else nn.Linear(embedding_size, num_heads * embedding_size, bias=config['msa_bias'])
        self.linear_k = k if k else nn.Linear(embedding_size, num_heads * embedding_size, bias=config['msa_bias'])
        self.linear_v = nn.Linear(embedding_size, num_heads * embedding_size, bias=config['msa_bias'])
        self.att_dropout = nn.Dropout(config['dropout'])

        self.output_layer = nn.Linear(num_heads * embedding_size, embedding_size)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.xavier_uniform_(self.linear_v.weight)

    def forward(self, x):
        d_k = self.att_size
        d_v = self.att_size

        q = self.linear_q(x).view(self.num_heads, -1, d_k) # [8, Max, 128]
        k = self.linear_k(x).view(self.num_heads, -1, d_k).transpose(-1, -2) # [8, 128, Max]
        v = self.linear_v(x).view(self.num_heads, -1, d_v) # [8, Max, 128]
        
        q = q * self.scale
        a = torch.matmul(q, k) 
        a = torch.softmax(a, dim=2)
        a = self.att_dropout(a) # [8, Max, Max]

        y = a.matmul(v).transpose(-2, -3).contiguous().view(-1, self.num_heads * d_v)
        y = self.output_layer(y) # [Max, 128]

        return y

class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super(FeedForwardNetwork, self).__init__()
        hidden_size = config['embedding_size']
        ffn_size = config['encoder_ffn_size'] # 128
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, config, q=None, k=None):
        super(CrossAttention, self).__init__()
        self.n_heads = n_heads = config['n_heads']
        self.embedding_size = embedding_size = config['embedding_size']
        self.scale = embedding_size ** -0.5

        self.linear_q = q if q else nn.Linear(embedding_size, n_heads * embedding_size, bias=config['msa_bias'])
        self.linear_k = k if k else nn.Linear(embedding_size, n_heads * embedding_size, bias=config['msa_bias'])

    def forward(self, embeddings_i, embeddings_j):
        q_i = self.linear_q(embeddings_i).view(-1, self.n_heads, self.embedding_size).transpose(-2, -3) # [8, Max, 128]
        k_i = self.linear_k(embeddings_i).view(-1, self.n_heads, self.embedding_size).transpose(-2, -3).transpose(-1, -2) # [8, 128, Max]
        q_j = self.linear_q(embeddings_j).view(-1, self.n_heads, self.embedding_size).transpose(-2, -3) # [8, Max, 128]
        k_j = self.linear_k(embeddings_j).view(-1, self.n_heads, self.embedding_size).transpose(-2, -3).transpose(-1, -2) # [8, 128, Max]
        a_i = torch.matmul(q_i, k_j) 
        a_i *= self.scale # [8, Max, Max]

        a_j = torch.matmul(q_j, k_i).transpose(-1, -2) 
        a_j *= self.scale # [8, Max, Max]

        return torch.cat([a_i, a_j], dim=0) # [16, Max, Max]

class SimMatLearning(nn.Module):
    def __init__(self, config):
        super(SimMatLearning, self).__init__()
        self._config = config

        if config['channel_align']:
            self.channel_alignment = ChannelAlignment(config)

        if config['sim_mat_learning_ablation']:
            self.sim_mat_pooling = SimMatPooling(config)
        else:
            self.sim_CNN = SimCNN(config)

    def forward(self, mat):
        if self._config['channel_align']:
            mat = self.channel_alignment(mat)

        if self._config['sim_mat_learning_ablation']:
            score = self.sim_mat_pooling(mat)
        else:
            score = self.sim_CNN(mat)

        return score

class ChannelAlignment(nn.Module):
    def __init__(self, config):
        super(ChannelAlignment, self).__init__()
        self._max_num = config['max_nums']

        self.self_attention_norm = nn.LayerNorm(self._max_num ** 2)
        self.self_attention = MultiHeadAttentionCA(config)
        self.self_attention_dropout = nn.Dropout(config['dropout'])

        self.ffn_norm = nn.LayerNorm(self._max_num ** 2)
        self.ffn = FeedForwardNetworkCA(config)
        self.ffn_dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        H = x.size(0)
        x = x.view(H, -1)

        y = self.self_attention_norm(x)
        y = self.self_attention(x)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x.view(H, self._max_num, self._max_num)

class MultiHeadAttentionCA(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttentionCA, self).__init__()
        self.num_heads = num_heads = config['n_channel_transformer_heads']
        embedding_size = config['max_nums'] ** 2

        self.att_size = embedding_size
        self.scale = embedding_size ** -0.5

        self.linear_q = nn.Linear(embedding_size, num_heads * embedding_size, bias=config['msa_bias'])
        self.linear_k = nn.Linear(embedding_size, num_heads * embedding_size, bias=config['msa_bias'])
        self.linear_v = nn.Linear(embedding_size, num_heads * embedding_size, bias=config['msa_bias'])
        self.att_dropout = nn.Dropout(config['dropout'])
        self.output_layer = nn.Linear(num_heads * embedding_size, embedding_size)

        nn.init.xavier_uniform_(self.linear_q.weight)
        nn.init.xavier_uniform_(self.linear_k.weight)
        nn.init.xavier_uniform_(self.linear_v.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        d_k = self.att_size
        d_v = self.att_size
        q = self.linear_q(x).view(-1, self.num_heads, d_k).transpose(0, 1) # [4, 16, 100]
        k = self.linear_k(x).view(-1, self.num_heads, d_k).transpose(0, 1).transpose(-1, -2) # [4, 100, 16]
        v = self.linear_v(x).view(-1, self.num_heads, d_v).transpose(0, 1) # [4, 16, 100]

        q = q * self.scale
        a = torch.matmul(q, k) 

        a = torch.softmax(a, dim=2)
        a = self.att_dropout(a) # [4, 16, 16]

        y = a.matmul(v).transpose(-2, -3).contiguous().view(-1, self.num_heads * d_v) # [16, 400]
        return self.output_layer(y) # [16, 100]

class FeedForwardNetworkCA(nn.Module):
    def __init__(self, config):
        super(FeedForwardNetworkCA, self).__init__()
        hidden_size = config['max_nums'] ** 2
        ffn_size = config['channel_ffn_size']
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

class SimMatPooling(nn.Module):
    def __init__(self, config):
        super(SimMatPooling, self).__init__()
        self._config = config
        in_channels = config['n_heads'] * 2

        self.pooling = nn.AdaptiveAvgPool2d((7, 7))

        self.cnn_1 = nn.Conv2d(in_channels=in_channels, out_channels=config['conv_channels_0'], kernel_size=(3, 3))
        self.cnn_2 = nn.Conv2d(in_channels=config['conv_channels_0'], out_channels=config['conv_channels_1'], kernel_size=(3, 3))
        self.cnn_3 = nn.Conv2d(in_channels=config['conv_channels_1'], out_channels=config['conv_channels_2'], kernel_size=(3, 3))
        self.cnn_4 = nn.Conv2d(in_channels=config['conv_channels_2'], out_channels=config['conv_channels_3'], kernel_size=(3, 3))

        self.fc_1 = nn.Linear(config['conv_channels_3'], config['conv_channels_3'] // 2)
        self.fc_2 = nn.Linear(config['conv_channels_3'] // 2, config['conv_channels_3'] // 4)
        self.fc_3 = nn.Linear(config['conv_channels_3'] // 4, config['conv_channels_3'] // 8)
        self.fc_4 = nn.Linear(config['conv_channels_3'] // 8, 1)

        nn.init.xavier_uniform_(self.cnn_1.weight)
        nn.init.xavier_uniform_(self.cnn_2.weight)
        nn.init.xavier_uniform_(self.cnn_3.weight)
        nn.init.xavier_uniform_(self.cnn_4.weight)

        nn.init.xavier_uniform_(self.fc_1.weight)
        nn.init.xavier_uniform_(self.fc_2.weight)
        nn.init.xavier_uniform_(self.fc_3.weight)
        nn.init.xavier_uniform_(self.fc_4.weight)

    def forward(self, sim_mat):
        out = F.leaky_relu(self.cnn_1(sim_mat), negative_slope=self._config['conv_l_relu_slope'], inplace=True)
        out = self.pooling(out)
        out = F.leaky_relu(self.cnn_2(out), negative_slope=self._config['conv_l_relu_slope'], inplace=True)
        out = F.leaky_relu(self.cnn_3(out), negative_slope=self._config['conv_l_relu_slope'], inplace=True)
        out = F.leaky_relu(self.cnn_4(out), negative_slope=self._config['conv_l_relu_slope'], inplace=True).squeeze()

        out = F.dropout(F.leaky_relu(self.fc_1(out), negative_slope=self._config['conv_l_relu_slope'], inplace=True), p=self._config['dropout'], training=self.training)
        out = F.dropout(F.leaky_relu(self.fc_2(out), negative_slope=self._config['conv_l_relu_slope'], inplace=True), p=self._config['dropout'], training=self.training)
        out = F.dropout(F.leaky_relu(self.fc_3(out), negative_slope=self._config['conv_l_relu_slope'], inplace=True), p=self._config['dropout'], training=self.training)
        out = F.leaky_relu(self.fc_4(out), negative_slope=self._config['conv_l_relu_slope'], inplace=True)
        out = torch.sigmoid(out).squeeze(-1)

        return out

class SimCNN(nn.Module):
    def __init__(self, config):
        super(SimCNN, self).__init__()
        self._config = config
        in_planes = config['n_heads'] * 2
        self.d = config['max_nums']

        self.e2econv1 = E2EBlock(config, in_channel=in_planes, out_channel=config['conv_channels_0'])
        self.e2econv2 = E2EBlock(config, in_channel=config['conv_channels_0'], out_channel=config['conv_channels_1'])
        self.E2N = nn.Conv2d(in_channels=config['conv_channels_1'], out_channels=config['conv_channels_2'], kernel_size=(1, self.d))
        self.N2G = nn.Conv2d(config['conv_channels_2'], config['conv_channels_3'], (self.d, 1))

        self.fc_1 = nn.Linear(config['conv_channels_3'], config['conv_channels_3'] // 2)
        self.fc_2 = nn.Linear(config['conv_channels_3'] // 2, config['conv_channels_3'] // 4)
        self.fc_3 = nn.Linear(config['conv_channels_3'] // 4, config['conv_channels_3'] // 8)
        self.fc_4 = nn.Linear(config['conv_channels_3'] // 8, 1)

        nn.init.xavier_uniform_(self.E2N.weight)
        nn.init.xavier_uniform_(self.N2G.weight)
        nn.init.xavier_uniform_(self.fc_1.weight)
        nn.init.xavier_uniform_(self.fc_2.weight)
        nn.init.xavier_uniform_(self.fc_3.weight)
        nn.init.xavier_uniform_(self.fc_4.weight)

    def forward(self, sim_mat, mask_ij):
        out = F.leaky_relu(self.e2econv1(sim_mat), negative_slope=self._config['conv_l_relu_slope'], inplace=True)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=self._config['conv_l_relu_slope'], inplace=True)

        out = F.leaky_relu(self.E2N(out), negative_slope=self._config['conv_l_relu_slope'], inplace=True)
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=self._config['conv_l_relu_slope'], inplace=True), p=self._config['dropout'], training=self.training).squeeze()

        out = F.dropout(F.leaky_relu(self.fc_1(out), negative_slope=self._config['conv_l_relu_slope'], inplace=True), p=self._config['dropout'], training=self.training)
        out = F.dropout(F.leaky_relu(self.fc_2(out), negative_slope=self._config['conv_l_relu_slope'], inplace=True), p=self._config['dropout'], training=self.training)
        out = F.dropout(F.leaky_relu(self.fc_3(out), negative_slope=self._config['conv_l_relu_slope'], inplace=True), p=self._config['dropout'], training=self.training)
        out = F.leaky_relu(self.fc_4(out), negative_slope=self._config['conv_l_relu_slope'], inplace=True)
        out = torch.sigmoid(out).squeeze(-1)

        return out

class E2EBlock(nn.Module):
    def __init__(self, config, in_channel, out_channel):
        super(E2EBlock, self).__init__()
        self._config = config
        self.d = config['max_nums']
        self.cnn1 = nn.Conv2d(in_channel, out_channel, (1, self.d))
        self.cnn2 = nn.Conv2d(in_channel, out_channel, (self.d, 1))
        nn.init.xavier_uniform_(self.cnn1.weight)
        nn.init.xavier_uniform_(self.cnn2.weight)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)
