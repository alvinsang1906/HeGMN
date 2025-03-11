import torch
import torch.nn as nn

from model.basemodel.FCL import FCL

def graph_prop_once(node_states, from_idx, to_idx, message_net):
    from_states = node_states[from_idx]
    to_states = node_states[to_idx]
    edge_inputs = [from_states, to_states]
    _device = node_states.device
    
    edge_inputs = torch.cat(edge_inputs, dim=-1) # [E, 64]
    messages = message_net(edge_inputs) # [E, 64]
    result = torch.zeros(node_states.shape[0], messages.shape[1], device=_device)
    result.scatter_add_(0, to_idx.view(-1, 1).expand_as(messages), messages).to(_device)
    return result  # [N, 64]

class GraphEncoder(nn.Module):
    def __init__(self, node_feature_dim, node_hidden_sizes=None):
        super(GraphEncoder, self).__init__()
        self._node_feature_dim = node_feature_dim
        self._node_hidden_sizes = node_hidden_sizes if node_hidden_sizes else None
        self._build_model()

    def _build_model(self):
        layer = []
        layer.append(nn.Linear(self._node_feature_dim, self._node_hidden_sizes[0]))
        for i in range(1, len(self._node_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append( nn.Linear(self._node_hidden_sizes[i - 1], self._node_hidden_sizes[i]) )
        self.MLP1 = nn.Sequential(*layer)

    def forward(self, node_features):
        # node_features: [N, D]
        if self._node_hidden_sizes is None:
            node_outputs = node_features
        else:
            node_outputs = self.MLP1(node_features)

        return node_outputs

class GraphAggregator(nn.Module):
    def __init__(self, node_hidden_sizes, graph_transform_sizes=None, input_size=None, gated=True):
        super(GraphAggregator, self).__init__()
        self._node_hidden_sizes = node_hidden_sizes   # [128]
        self._graph_transform_sizes = graph_transform_sizes  # [128]
        self._graph_state_dim = node_hidden_sizes[-1]  # 128
        self._input_size = input_size  # 32
        #  The last element is the size of the aggregated graph representation.
        self._gated = gated
        self._aggregation_op = None
        self.MLP1, self.MLP2 = self.build_model()

    def build_model(self):
        node_hidden_sizes = self._node_hidden_sizes
        if self._gated:
            node_hidden_sizes[-1] = self._graph_state_dim * 2

        layer = []
        layer.append(nn.Linear(self._input_size[0], node_hidden_sizes[0]))
        for i in range(1, len(node_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append(
                nn.Linear(node_hidden_sizes[i - 1], node_hidden_sizes[i]))
        MLP1 = nn.Sequential(*layer)

        if (self._graph_transform_sizes is not None and len(self._graph_transform_sizes) > 0):
            layer = []
            layer.append(nn.Linear(self._graph_state_dim, self._graph_transform_sizes[0]))
            for i in range(1, len(self._graph_transform_sizes)):
                layer.append(nn.ReLU())
                layer.append(nn.Linear(self._graph_transform_sizes[i - 1], self._graph_transform_sizes[i]))
            MLP2 = nn.Sequential(*layer)

        return MLP1, MLP2

    def forward(self, node_states, graph_idx, n_graphs):
        node_states_g = self.MLP1(node_states)

        if self._gated:
            gates = torch.sigmoid(node_states_g[:, :self._graph_state_dim])
            node_states_g = node_states_g[:, self._graph_state_dim:] * gates
        
        graph_states = torch.zeros(n_graphs, node_states_g.shape[1], device=node_states_g.device)
        graph_states.scatter_add_(0, graph_idx.view(-1, 1).expand_as(node_states_g).long(), node_states_g)

        if (self._graph_transform_sizes is not None and len(self._graph_transform_sizes) > 0):
            # self._graph_transform_sizes: [128]
            graph_states = self.MLP2(graph_states)

        return graph_states

class GraphEmbeddingNet(nn.Module):
    def __init__(self, config):
        super(GraphEmbeddingNet, self).__init__()
        self._config = config
        self._device = 'cuda:' + str(config['gpu_index']) if config['gpu_index'] >= 0 else 'cpu'
        self._encoder = GraphEncoder(config['num_features'], [config['node_state_dim']]).to(self._device)
        self._aggregator = GraphAggregator([config['graph_rep_dim']], [config['graph_rep_dim']], [config['node_state_dim']]).to(self._device)

        self._node_state_dim = config['node_state_dim']
        self._node_hidden_sizes = config['node_hidden_sizes']
        self._n_prop_layers = config['n_prop_layers']
        self._share_prop_params = config['share_prop_params']
        self._use_reverse_direction = config['use_reverse_direction']
        self._reverse_dir_param_different = config['reverse_dir_param_different']
        self._layer_norm = config['layer_norm']
        self._prop_layers = []
        self._prop_layers = nn.ModuleList()
        self._layer_class = GraphPropMatchingLayer
        self.build_model()

    def _build_layer(self, layer_id):
        return self._layer_class(self._config).to(self._device)

    def _apply_layer(self, layer, node_states, from_idx, to_idx, graph_idx, n_graphs):
        del graph_idx, n_graphs
        return layer(node_states, from_idx, to_idx)

    def build_model(self):
        if len(self._prop_layers) < self._n_prop_layers:
            for i in range(self._n_prop_layers):
                if i == 0 or not self._share_prop_params:
                    layer = self._build_layer(i)
                else:
                    layer = self._prop_layers[0]
                self._prop_layers.append(layer)

    def forward(self, node_features, from_idx, to_idx, graph_idx, n_graphs):
        node_features = self._encoder(node_features)
        node_states = node_features  # [N, 32]

        layer_outputs = [node_states]

        for layer in self._prop_layers:
            node_states = self._apply_layer(layer, node_states, from_idx, to_idx, graph_idx, n_graphs)
            layer_outputs.append(node_states)

        self._layer_outputs = layer_outputs
        return self._aggregator(node_states, graph_idx, n_graphs)

class GraphPropLayer(nn.Module):
    def __init__(self, config):
        super(GraphPropLayer, self).__init__()
        self._node_state_dim = config['node_state_dim']  # 32
        self._node_hidden_sizes = config['node_hidden_sizes'][:] + [config['node_state_dim']] # [64, 32]
        self._edge_hidden_sizes = config['edge_hidden_sizes']  # [64, 64]
        self._use_reverse_direction = config['use_reverse_direction']
        self._reverse_dir_param_different = config['reverse_dir_param_different']
        self._device = 'cuda:' + str(config['gpu_index']) if config['gpu_index'] >= 0 else 'cpu'
        self._layer_norm = config['layer_norm']
        self.build_model()

        if self._layer_norm:
            self.layer_norm1 = nn.LayerNorm()
            self.layer_norm2 = nn.LayerNorm()

    def build_model(self):
        layer = []
        layer.append(nn.Linear(self._node_state_dim * 2, self._edge_hidden_sizes[0]))
        for i in range(1, len(self._edge_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append( nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]) )
        self._message_net = nn.Sequential(*layer).to(self._device)

        if self._use_reverse_direction:
            if self._reverse_dir_param_different:
                layer = []
                layer.append(nn.Linear(self._node_state_dim * 2, self._edge_hidden_sizes[0]))
                for i in range(1, len(self._edge_hidden_sizes)):
                    layer.append(nn.ReLU())
                    layer.append( nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]) )
                self._reverse_message_net = nn.Sequential(*layer).to(self._device)
            else:
                self._reverse_message_net = self._message_net

        self.GRU = nn.GRU(self._node_state_dim * 3, self._node_state_dim)

    def _compute_aggregated_messages(self, node_states, from_idx, to_idx):
        aggregated_messages = graph_prop_once(node_states, from_idx, to_idx, self._message_net)

        # optionally compute message vectors in the reverse direction
        if self._use_reverse_direction:
            reverse_aggregated_messages = graph_prop_once(node_states, to_idx, from_idx, self._reverse_message_net)

            aggregated_messages += reverse_aggregated_messages

        if self._layer_norm:
            aggregated_messages = self.layer_norm1(aggregated_messages)

        return aggregated_messages  # [N, 64]

    def _compute_node_update(self, node_states, node_state_inputs, node_features=None):
        if node_features is not None:
            node_state_inputs.append(node_features)

        if len(node_state_inputs) == 1:
            node_state_inputs = node_state_inputs[0]
        else:
            node_state_inputs = torch.cat(node_state_inputs, dim=-1)

        node_state_inputs = torch.unsqueeze(node_state_inputs, 0)
        node_states = torch.unsqueeze(node_states, 0)
        _, new_node_states = self.GRU(node_state_inputs, node_states)
        new_node_states = torch.squeeze(new_node_states)
        return new_node_states  # [N, node_state_dim]

    def forward(self, node_states, from_idx, to_idx, node_features=None):
        aggregated_messages = self._compute_aggregated_messages(node_states, from_idx, to_idx)

        return self._compute_node_update(node_states, [aggregated_messages], node_features=node_features)

class GraphPropMatchingLayer(GraphPropLayer):
    def forward(self, node_states, from_idx, to_idx, graph_idx, n_graphs, node_features=None):
        aggregated_messages = self._compute_aggregated_messages(node_states, from_idx, to_idx)

        cross_graph_attention = batch_block_pair_attention(node_states, graph_idx, n_graphs)
        attention_input = node_states - cross_graph_attention

        return self._compute_node_update(node_states, [aggregated_messages, attention_input], node_features=node_features)

def batch_block_pair_attention(data, block_idx, n_blocks):
    if not isinstance(n_blocks, int):
        raise ValueError('n_blocks (%s) has to be an integer.' % str(n_blocks))

    if n_blocks % 2 != 0:
        raise ValueError('n_blocks (%d) must be a multiple of 2.' % n_blocks)

    sim = pairwise_dot_product_similarity

    results = []

    partitions = []
    for i in range(n_blocks):
        partitions.append(data[block_idx == i, :])

    for i in range(0, n_blocks, 2):
        x = partitions[i]
        y = partitions[i + 1]
        attention_x, attention_y = compute_cross_attention(x, y, sim)
        results.append(attention_x)
        results.append(attention_y)
    results = torch.cat(results, dim=0)

    return results

def pairwise_dot_product_similarity(x, y):
    return torch.mm(x, torch.transpose(y, 1, 0))

def compute_cross_attention(x, y, sim):
    a = sim(x, y)
    a_x = torch.softmax(a, dim=1)  # i->j
    a_y = torch.softmax(a, dim=0)  # j->i
    attention_x = torch.mm(a_x, y)
    attention_y = torch.mm(torch.transpose(a_y, 1, 0), x)
    return attention_x, attention_y

class GraphMatchingNetwork(GraphEmbeddingNet):
    def __init__(self, config):
        super(GraphMatchingNetwork, self).__init__(config)

    def _apply_layer(self, layer, node_states, from_idx, to_idx, graph_idx, n_graphs):
        return layer(node_states, from_idx, to_idx, graph_idx, n_graphs)

class GMN(nn.Module):
    def __init__(self, config):
        super(GMN, self).__init__()
        self._config = config
        self._device = 'cuda:' + str(config['gpu_index']) if config['gpu_index'] >= 0 else 'cpu'

        self._gmn = GraphMatchingNetwork(config).to(self._device)
        self._fcl = FCL(config['graph_rep_dim'] * 2).to(self._device)

    def forward(self, data):
        x_1, edge_index_1 = data['g1'].x, data['g1'].edge_index
        x_2, edge_index_2 = data['g2'].x, data['g2'].edge_index
        x = torch.concat([x_1, x_2], dim=0)  # [N + M, D]

        N, M = x_1.shape[0], x_2.shape[0]
        from_1 = edge_index_1[0]
        to_1 = edge_index_1[1]
        from_2 = edge_index_2[0] + N
        to_2 = edge_index_2[1] + N
        from_idx = torch.concat([from_1, from_2], dim=0)
        to_idx = torch.concat([to_1, to_2], dim=0)

        graph_idx_1 = torch.zeros(N)
        graph_idx_2 = torch.ones(M)

        graph_idx = torch.concat([graph_idx_1, graph_idx_2], dim=0).to(self._device)
        n_graphs = 2

        u_1 = self._gmn(x, from_idx, to_idx, graph_idx, n_graphs)
        u_2 = self._gmn(x, to_idx, from_idx, graph_idx, n_graphs)
        h_1, h_2 = torch.mean(u_1, dim=0).view(1, -1), torch.mean(u_2, dim=0).view(1, -1)
        score = self._fcl(h_1, h_2)
        return score
