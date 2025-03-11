import dgl
import yaml
import torch
import random
import cProfile
import numpy as np
import networkx as nx
import matplotlib.lines
import matplotlib.pyplot as plt

from texttable import Texttable

def nice_printer(config):
    r"""
        打印配置
        参数 config: 字典，键值分别为参数的名称和对应的参数值
    """
    tabel_data = [['Key', 'Value']] + [[k, v] for k, v in config.items()]
    t = Texttable().set_precision(4)
    t.add_rows(tabel_data)
    print(t.draw())

def set_seed(seed):
    r"""
        设置所有的随机数种子都为seed
        参数 seed: int， 将要设置的随机数种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # 启用确定性计算，降低性能，但是确保实验可重复，尤其是dropout等随机性操作
    torch.backends.cudnn.benchmark = False      # 禁用自动调整策略，提高稳定性

def get_config(args):
    r"""
        通过传入的args参数，得到完整的 config字典
        参数 args: 默认包含一些main函数中公用参数
        返回 config: 完整参数字典，键值分别为参数名和参数对应的值
    """
    config = get_part_config('model/config.yml')['ALL_MODEL']
    if args.model == 'HGMN':
        config.update( get_part_config('model/config.yml')['HGMN'] )
    elif args.model in ['GCN', 'GIN', 'GAT']:
        config.update( get_part_config()['basicGNN'] )
    else:
        config.update( get_part_config()[args.model] )

    config.update( get_part_config('model/config.yml')[args.dataset])

    config['model'] = args.model
    config['log_path'] = args.log_path + args.model + '/' + args.dataset + ('_hetero/' if config['HGED'] is True else '_homoro/')

    return config

def get_part_config(config_path="model/basemodel/config.yml"):
    r"""
        读取config.yml文件
        参数 config_path: 默认为model/basemodel/目录下的config.yml文件
        返回 config: 字典，键值分别为参数名和参数对应的值
    """
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config

def cal_p_at_k(k, prediction, target):
    r"""
        计算指标p@k
        参数 k: 计算指标的具体值，一般为10 或 20
        参数 prediction: 模型产生的预测值，一般为余弦相似度或者是经过FCL后的值
        参数 target: ground truth 真实值
        返回 p@k
    """
    best_k_pred = prediction.argsort()[::-1][:k] # 降序排序，选择前k个最佳值的 索引
    best_k_target = _cal_p_at_k(k, -target)
    return len(set(best_k_pred).intersection(set(best_k_target))) / k

def _cal_p_at_k(k, target):
    target_increase = np.sort(target)
    target_value_sel = (target_increase <= target_increase[k - 1]).sum()

    best_k_target = target.argsort()[:target_value_sel] if target_value_sel > k else target.argsort()[:k]
    return best_k_target

def print_evals(mse_error, rho, tau, p10, p20):
    r"""
        用于模型test阶段结束后，打印所有的结果
        参数 mse_error: float，模型预测的MSE
        参数 rho: 斯皮尔曼相关系数
        参数 tau: 肯德尔相关系数
        参数 p10: p@10
        参数 p20: p@20
    """
    print("mse(10^-3): " + str(round(mse_error * 1000, 5)) + '.')
    print("rho: " + str(round(rho, 5)) + '.')
    print("tau: " + str(round(tau, 5)) + '.')
    print("p@10: " + str(round(p10, 5)) + '.')
    print("p@20: " + str(round(p20, 5)) + '.')

def oldnid2newnid(nids, nlabels):
    r"""要求传入两个参数，分别是节点id，和节点types。两个参数均为列表形式
        返回字典，kv分别是原始节点id和新的节点id
    """
    unique_labels = np.unique(nlabels)

    # 初始化一个字典用于存储新旧节点编号的映射
    new_node_ids = {}

    # 对每个标签进行重新编号
    for label in unique_labels:
        nodes_with_label = [node_id for node_id, node_label in zip(nids, nlabels) if node_label == label]
        new_ids = list(range(len(nodes_with_label)))
        
        # 更新新旧节点编号的映射
        new_node_ids.update(dict(zip(nodes_with_label, new_ids)))
    return new_node_ids

def profiled_function(func):
    r"""
        用于测试函数性能
        直接在函数名之上添加装饰器 @profiled_function 即可
    """
    def wrapper(*args, **kwargs):
        profile = cProfile.Profile()
        result = profile.runcall(func, *args, **kwargs)
        profile.print_stats(sort='tottime')
        return result
    return wrapper

def load_data(dataset_name):
    r"""
        加载数据集
        参数 dataset: string, 数据集的名字
        返回 data: 字典，数据集包含的所有内容
        异常 ValueError: 没有对应的数据集
    """
    if dataset_name == 'AIDS700nef':                                     # 是经典的同质图相似度数据集
        from utils.data_loader.AIDS import data_loader
    elif dataset_name == 'MUTAG':                                        # 分子化合物数据集
        from utils.data_loader.MUTAG import data_loader
    elif dataset_name == 'IMDB':                                         # 电影-演员-导演数据集
        from utils.data_loader.IMDB import data_loader
    elif dataset_name == 'DBLP':                                         # 小引文网络
        from utils.data_loader.DBLP import data_loader
    elif dataset_name == 'ACM':                                          # 大引文网络
        from utils.data_loader.ACM import data_loader
    else:
        raise ValueError("Unknown dataset: {}".format(dataset_name))
    
    return data_loader()

def create_training_batches(training_len, batch_size):
    r"""
        生成训练的batch，将来需要在每个epoch中都进行调用
        参数 training_len: int, 训练集的长度，则图id范围为[0, training_len - 1]
        参数 batch_size: int, batch的大小
        返回 batches: 双层list，除了最后一个以外，其他内部list长度均为batch_size。
    """
    src_graph_id = list(range(training_len))
    dst_graph_id = list(range(training_len))
    random.shuffle(src_graph_id)
    random.shuffle(dst_graph_id)
    temp = list(zip(src_graph_id, dst_graph_id))
    batches = [temp[i : i + batch_size] for i in range(0, len(temp), batch_size)]
    return batches

def create_training_batches_all(training_len, batch_size):
    r"""
        生成训练的batch，将来需要在每个epoch中都进行调用
        参数 training_len: int, 训练集的长度，则图id范围为[0, training_len - 1]
        参数 batch_size: int, batch的大小
        返回 batches: 双层list，除了最后一个以外，其他内部list长度均为batch_size。
    """
    train_graph_list = list(range(training_len))
    combinations = [(i, j) for i in train_graph_list for j in train_graph_list if i < j]
    batches = [combinations[i:i + batch_size] for i in range(0, len(combinations), batch_size)]
    return batches

def create_train_validate_test_pairs_id(training_len, testing_len, batch_size, set_type=0, proportion=622):
    r"""
        生成验证或测试的图对，将来需要在每个epoch中都进行调用
        参数 training_len: int, 训练集的长度，则图id范围为[0, training_len - 1]
        参数 testing_len: int, 训练集的长度，则图id范围为[0, testing_len - 1]
        参数 batch_size: int, batch的大小
        参数 set_type: int, default: 0，表示返回的是训练集图对，测试集图对还是验证集图对。
                如果为0，表示返回训练集图对；
                如果为1，表示返回验证集图对；
                如果为2，表示返回测试集图对；
        参数 proportion: int, default: 622，表示训练集:验证集:测试集的比率，有两种选择，622 or 811
        返回 pairs: list, 其中包含格式为元组类型的图对id
        异常 ValueError: 不支持的proportion或不支持的validatie
    """
    if proportion == 811:
        if set_type == 0:
            return create_training_batches(training_len, batch_size)
        elif set_type == 1:
            return [(i, j) for i in range(training_len) for j in range(testing_len // 2)]
        elif set_type == 2:
            return [(i, j) for i in range(training_len) for j in range(testing_len // 2, testing_len)]
        else:
            raise ValueError("Unknown set_type: {}".format(set_type))
    elif proportion == 622:
        real_training_len = int(training_len * 0.75)
        if set_type == 0:
            return create_training_batches_all(real_training_len, batch_size)
        elif set_type == 1:
            return [(i, j) for i in range(real_training_len) for j in range(real_training_len, training_len)]
        elif set_type == 2:
            return [(i, j) for i in range(real_training_len) for j in range(testing_len)]
        else:
            raise ValueError("Unknown set_type: {}".format(set_type))
    else:
        raise ValueError("Unknown proportion: {}".format(proportion))

def torch2dgl(data):
    r"""
        将输入的pyg同质图格式转换为dgl图格式
        参数 data: dict, data['g1']是图1的pyg格式，data['g2']是图2的
        返回 G_1, G_2: dgl Graph
    """
    x_1, edge_index_1 = data['g1'].x, data['g1'].edge_index
    x_2, edge_index_2 = data['g2'].x, data['g2'].edge_index
    N_1, N_2 = x_1.shape[0], x_2.shape[0]
    G_1 = dgl.graph((edge_index_1[0], edge_index_1[1]), num_nodes=N_1)
    G_2 = dgl.graph((edge_index_2[0], edge_index_2[1]), num_nodes=N_2)
    G_1.ndata['features'] = x_1
    G_2.ndata['features'] = x_2
    return G_1, G_2

def view_dataset_metrics(graphs):
    r"""
        查看输入数据集的各项指标，例如最大/最小/平均 节点/边 数，数据集中图数
        参数 graphs: List, 其中包含该数据集的所有nx同质图
    """
    min_nodes, min_edges = 999, 999
    max_nodes, max_edges = 0, 0
    sum_nodes, sum_edges = 0, 0

    for g in graphs:
        min_nodes = min(min_nodes, g.number_of_nodes())
        max_nodes = max(max_nodes, g.number_of_nodes())
        sum_nodes += g.number_of_nodes()
        min_edges = min(min_edges, g.number_of_edges())
        max_edges = max(max_edges, g.number_of_edges())
        sum_edges += g.number_of_edges()
    num_graphs = len(graphs)
    avg_nodes = sum_nodes / num_graphs
    avg_edges = sum_edges / num_graphs
    print("num_graphs: {}".format(num_graphs))
    print("min_nodes: {}, max_nodes: {}, avg_nodes: {}".format(min_nodes, max_nodes, round(avg_nodes, 2)))
    print("min_edges: {}, max_edges: {}, avg_edges: {}".format(min_edges, max_edges, round(avg_edges, 2)))

def calculate_ranking_correlation(rank_corr_function, prediction, target):
    r"""
        计算相关系数
        参数 rank_corr_function: 函数，是scipy.stats中的函数
        参数 prediction: double，是模型的预测值
        参数 target: double，是数据集的准确值
        返回 相关系数
    """
    def ranking_func(data):
        sort_id_mat = np.argsort(-data)
        n = sort_id_mat.shape[0]
        rank = np.zeros(n)
        for i in range(n):
            finds = np.where(sort_id_mat == i)
            fid = finds[0][0]
            while fid > 0:
                cid = sort_id_mat[fid]
                pid = sort_id_mat[fid - 1]
                if data[pid] == data[cid]:
                    fid -= 1
                else:
                    break
            rank[i] = fid + 1
        return rank
    
    r_prediction = ranking_func(prediction)
    r_target = ranking_func(target)

    return rank_corr_function(r_prediction, r_target).correlation

def prec_at_ks(true_r, pred_r, ks, rm=0):
    r"""
        计算 p@k
        参数 true_r: double，数据集的真实值
        参数 pred_r: double，模型的预测值
        参数 ks: int，k的值
        返回 ps: int，最后的p@k
    """
    def top_k_ids(data, k, inclusive, rm):
        """
        :param data: input
        :param k:
        :param inclusive: whether to be tie inclusive or not.
            For example, the ranking may look like this:
            7 (sim_score=0.99), 5 (sim_score=0.99), 10 (sim_score=0.98), ...
            If tie inclusive, the top 1 results are [7, 9].
            Therefore, the number of returned results may be larger than k.
            In summary,
                len(rtn) == k if not tie inclusive;
                len(rtn) >= k if tie inclusive.
        :param rm: 0
        :return: for a query, the ids of the top k database graph
        ranked by this model.
        """
        sort_id_mat = np.argsort(-data)
        n = sort_id_mat.shape[0]
        if k < 0 or k >= n:
            raise RuntimeError('Invalid k {}'.format(k))
        if not inclusive:
            return sort_id_mat[:k]
        # Tie inclusive.
        dist_sim_mat = data
        while k < n:
            cid = sort_id_mat[k - 1]
            nid = sort_id_mat[k]
            if abs(dist_sim_mat[cid] - dist_sim_mat[nid]) <= rm:
                k += 1
            else:
                break
        return sort_id_mat[:k]
    true_ids = top_k_ids(true_r, ks, inclusive=True, rm=rm)
    pred_ids = top_k_ids(pred_r, ks, inclusive=True, rm=rm)
    ps = min( len(set(true_ids).intersection(set(pred_ids)) ), ks) / ks
    return ps

def draw_with_nx(graph, gid, mode):
    r"""
        绘制异质图
        参数 graph: networks格式的带标签图
    """
    plt.figure(figsize=(2, 2))
    plt.axis('off')
    node_types = list(set([x[1] for x in graph.nodes(data="node_type") if x[1] is not None]))
    edge_types = list(set([x[2] for x in graph.edges(data="edge_type") if x[2] is not None]))
    colors = ['#63B2EE', '#F8CB7F', '#76DA91', '#F89588', '#439F99', '##FF70C5', ] + ['#999999'] * 23
    nodes_colors = {node_types[i] : colors[node_types[i]] for i in range(len(node_types))}
    edges_colors = {edge_types[i] : colors[edge_types[i]] for i in range(len(edge_types))}

    nc, ec = [nodes_colors[node[1]] for node in graph.nodes(data='node_type')], [edges_colors[edge[2]] for edge in graph.edges(data='edge_type')]
    nx.draw_networkx(graph, node_size=100, with_labels=False, node_color=nc, edge_color=ec)
    plt.show()
    if mode == 1:
        plt.savefig(str(gid) + "pre.png", bbox_inches='tight', dpi=750)
    elif mode == 0:
        plt.savefig(str(gid) + 'tar.png', bbox_inches='tight', dpi=750)
        

def draw_result(nx_graphs, i_pre, i_tar):
    ### ---------------------- ground truth ----------------------- ###
        for i in range(3):
            draw_with_nx(nx_graphs[i_tar[i][0]], i, 0)
        mid = len(i_tar) // 2
        draw_with_nx(nx_graphs[i_tar[mid][0]], mid, 0)
        draw_with_nx(nx_graphs[i_tar[-1][0]], -1, 0)
    ### ---------------------- ground truth ----------------------- ###

    ### ----------------------- prediction ------------------------ ###
        for i in range(3):
            draw_with_nx(nx_graphs[i_pre[i][0]], i, 1)
        draw_with_nx(nx_graphs[i_pre[mid][0]], mid, 1)
        draw_with_nx(nx_graphs[i_pre[-1][0]], -1, 1)
    ### ----------------------- prediction ------------------------ ###
        ans = [i_tar[0][1],i_tar[1][1],i_tar[2][1],i_tar[mid][1], i_tar[-1][1], 
            i_pre[0][1], i_pre[1][1], i_pre[2][1],i_pre[mid][1], i_pre[-1][1]]
        print(ans)