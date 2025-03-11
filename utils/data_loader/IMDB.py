import os
import glob
import torch
import random
import pickle
import networkx as nx

from queue import Queue
from torch_geometric.data import Data
from cogdl.datasets.gtn_data import IMDB_GTNDataset
"""
    num_ntype: 3
        0
        1
        2
    num_etype: 3
        0, 0    etype: 0
        1, 0    etype: 1
        2, 0    etype: 2
"""

class data_loader():
    r"""对DBLP数据集进行处理，包括使用BFS进行子图采样。生成训练集，测试集"""
    def __init__(self, path=None):

        if path == None:
            _path = './datasets/gtn-imdb/'
        else:
            _path = path
        
        _pkl_files = glob.glob(_path + '*.pkl')
        _req_pkl_files = ['graphs.pkl', 'train_data.pkl', 'test_data.pkl', 'norm_ged_hetero.pkl']
        
        if all(os.path.basename(file) in _req_pkl_files for file in _pkl_files) and len(_pkl_files) == len(_req_pkl_files):
            print("exist all pickle files")
            self.graphs = self._load_pickle(_path + 'graphs.pkl')
            self.train_data = self._load_pickle(_path + 'train_data.pkl')
            self.test_data = self._load_pickle(_path + 'test_data.pkl')
            self.norm_ged_hetero = self._load_pickle(_path + 'norm_ged_hetero.pkl')
            self.norm_ged_homoro = None
        else:
            print("no pickle files")
            _hetero_data = IMDB_GTNDataset('./datasets')
            _homoro_graph = self._torch2nx(_hetero_data)
            self.graphs, hetero_list = self._generate_graphs(1200, _homoro_graph, _hetero_data.data)
            self.train_data, self.test_data = hetero_list[:960], hetero_list[960:]
            # self.norm_ged = self._get_ged(_path)
            self._save_files(_req_pkl_files)

    def _torch2nx(self, data):
        r"""
            将原始的 hetero_graph 格式的图转换为networkx的同质图格式，返回同质图g
            为每个节点和边赋值一个类型，放在属性中，命名为node_type和edge_type，分别都是从0开始的连续值，
            可以方便后续的BFS遍历，以及GED的计算
        """
        g = data.data.to_networkx()
        for nid in range(data.data.y.shape[0]):
            g.nodes[nid]['node_type'] = int(data.data.y[nid])
        etype = 0
        ntype2etype = {}    # 定义一个节点类型到边类型的映射关系
        edge_attr = {}      # 为所有的边定义类型和边id
        for i in range(data.data.edge_index[0].shape[0]):
            srcid, dstid = int(data.data.edge_index[0][i]), int(data.data.edge_index[1][i])
            srctype, dsttype = g.nodes[srcid]['node_type'], g.nodes[dstid]['node_type']
            if (srctype, dsttype) not in ntype2etype:
                ntype2etype[(srctype, dsttype)] = etype
                ntype2etype[(dsttype, srctype)] = etype
                etype += 1
            edge_attr[(srcid, dstid)] = {'edge_type': ntype2etype[(srctype, dsttype)], 'edge_id': i} # 这里的edge_id不重要
        nx.set_edge_attributes(g, edge_attr)
        return g

    def _generate_graphs(self, nums, homoro_data, hetero_data):
        r"""
            生成nums张pyg异质图和nums张nx同质图
            参数：nums 最终要生成的图的张数
            参数：homoro_data 上一步得到的完整nx同质图
            参数：hetero_data 原始的完整异质图
            返回：graphs, nx图
            返回：hetero_list, pyg图
        """
        exist_nid_list = set()  # 一张图的所有节点id，并且set中的每一个元素都是按照大小顺序排列的，保证不会出现重复的图
        hetero_list = []        # 存放生成的pyg异质图，每张图有独立编号i
        graphs = []             # 存放生成的nx 同质图, 用于计算GED

        gid = 0
        while(len(graphs) < nums):
            g = nx.Graph(i=gid) # nx 同质图
            # 通过BFS生成nx同质图
            start_node = random.randint(0, homoro_data.number_of_nodes() - 1) # 起点id
            max_num = random.randint(10, 12) # 最多包含的节点个数 [10, 12]
            self._BFS(g, homoro_data, start_node, max_num, 0.5)
            if g.number_of_nodes() < 8 or g.number_of_node_types < 3: # 节点个数太少 或者 没有包含全部的节点类型 就舍弃
                continue
            # 此时图g中，node_type edge_type都是正常的，从0开始的连续值。
            # 但是node_id, edge_id都还是原始图上的间断值，并且没有添加节点属性x

            # 将nx同质图转换为pyg异质图 存入hetero_list中返回
            """添加节点"""
            oldnid_list = sorted(list(g.nodes))                     # 原始节点id按照从小到大的顺序放进去，为了保证特征向量的对应关系
            exist_nid = tuple(oldnid_list)
            if exist_nid in exist_nid_list:                         # 确保已经生成过的图不会再生成一遍
                continue
            exist_nid_list.add(exist_nid)
            feat_list = [hetero_data.x[nid] for nid in oldnid_list] # 存放的是按照顺序排列的节点特征
            x = torch.stack(feat_list)

            node_id_map = {value: index for index, value in enumerate(oldnid_list)} # 将原本的不连续节点id转换为连续从零开始的节点id值
            g = nx.relabel_nodes(g, node_id_map)
            node_type = torch.tensor([g.nodes[nid]['node_type'] for nid in range(g.number_of_nodes())])
            src_list, dst_list, etype_list = [], [], []
            for srcid, dstid in g.edges:
                src_list.append(srcid)
                src_list.append(dstid)
                dst_list.append(dstid)
                dst_list.append(srcid)
                etype_list.append(g.edges[srcid, dstid]['edge_type'])
                etype_list.append(g.edges[dstid, srcid]['edge_type'])
            edge_type = torch.tensor(etype_list)
            edge_index = torch.tensor([src_list, dst_list])
            pg = Data(i=gid, x=x, edge_index=edge_index, node_type=node_type, edge_type=edge_type)
            graphs.append(g)
            hetero_list.append(pg)
            gid += 1
        return graphs, hetero_list
    
    def _BFS(self, g, homoro_data, start_node, max_num, choice_rate):
        r"""
            在nx同质图g中，以start_node为起点，至多选择max_num个节点，选择率为choice_rate，进行BFS，返回最终的nx同质图g
            参数：g 是nx带有图编号i的同质图g，最后修改这张图
            参数：homoro_data nx转换后的完整的同质图
            参数：start_node 遍历的起点
            参数：max_num 最多遍历的节点个数
            参数：choice_rate 每个节点的选择率
        """
        ntypes = set() # 集合，记录图g中已经出现的节点类型
        def exist(srcid, dstid):
            r"""srcid是new节点, dstid是cur节点, cur节点一定在图g中了，但是new节点有可能不存在于图中"""
            if not g.has_node(srcid): # 如果srcid不存在
                if random.random() < choice_rate: # 如果需要选择这个节点
                    q.put(srcid)
                    ntypes.add(homoro_data.nodes[srcid]['node_type'])
                    g.add_node(srcid, node_type=homoro_data.nodes[srcid]['node_type'])
                    g.add_edge(srcid, dstid, edge_type=homoro_data[srcid][dstid]['edge_type'], edge_id=homoro_data[srcid][dstid]['edge_id'])
                else:
                    return
        
            for nid in g.nodes:
                # 判断 srcid 和g中其他所有的节点之间是否存在边，如果存在边并且没有连接过，则连起来，避免形成BFS生成树
                if homoro_data.has_edge(srcid, nid) and not g.has_edge(srcid, nid):
                    # 如果原图中有这条边，并且概率符合要求，并且新图中没有这条边，就创建一条边
                    g.add_edge(srcid, nid, edge_type=homoro_data[srcid][nid]['edge_type'], edge_id=homoro_data[srcid][nid]['edge_id'])

        q = Queue()
        q.put(start_node)
        ntypes.add(homoro_data.nodes[start_node]['node_type'])
        while not q.empty():
            if g.number_of_nodes() >= max_num: # 满足节点个数的条件就结束BFS
                break
            cur = q.get()
            if not g.has_node(cur): # 如果队列顶端的节点不在nx同质图g中
                g.add_node(cur, node_type=homoro_data.nodes[cur]['node_type'])
            for edge in homoro_data.edges(cur):
                if g.number_of_nodes() >= max_num:
                    break
                srcid, dstid = edge[0], edge[1]
                # srcid, dstid两个节点至多有一个不存在于图中，至少有一个已经存在于图中了。
                # srcid 一定是edge中的第一个元素 所以srcid == cur
                exist(dstid, srcid) # dstid 与 cur连接形成边
        g.number_of_node_types = len(ntypes)
    
    def _get_ged(self, path):
        r"""
            将使用nx库计算得到的排好序的GED值导入self.norm_ged中，以便后续使用
            参数 path: str, 数据集MUTAG存储的目录
        """
        ged_hetero, ged_homoro = torch.full((1200, 1200), float('inf')), torch.full((1200, 1200), float('inf'))
        norm_ged_hetero, norm_ged_homoro = torch.full((1200, 1200), float('inf')), torch.full((1200, 1200), float('inf'))
        for i in range(960):
            ged_hetero[i, i], ged_homoro[i, i] = 0.0, 0.0
            norm_ged_hetero[i, i], norm_ged_homoro[i, i] = 0.0, 0.0
        with open(path + '/GED_hetero.txt', 'r') as f:
            for line in f:
                G1_id, G2_id, tempged = line.strip().split('\t')
                ged_hetero[int(G1_id), int(G2_id)], ged_hetero[int(G2_id), int(G1_id)] = float(tempged), float(tempged)
                norm = round(float(tempged) / ((self.graphs[int(G1_id)].number_of_nodes() + self.graphs[int(G2_id)].number_of_nodes()) / 2), 4)
                norm_ged_hetero[int(G1_id), int(G2_id)], norm_ged_hetero[int(G2_id), int(G1_id)] = norm, norm
        
        with open(path + '/GED_homoro.txt', 'r') as f:
            for line in f:
                G1_id, G2_id, tempged = line.strip().split('\t')
                ged_homoro[int(G1_id), int(G2_id)], ged_homoro[int(G2_id), int(G1_id)] = float(tempged), float(tempged)
                norm = round(float(tempged) / ((self.graphs[int(G1_id)].number_of_nodes() + self.graphs[int(G2_id)].number_of_nodes()) / 2), 4)
                norm_ged_homoro[int(G1_id), int(G2_id)], norm_ged_homoro[int(G2_id), int(G1_id)] = norm, norm
        return norm_ged_hetero, norm_ged_homoro

    def _load_pickle(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
        
    def _save_files(self, _req_pkl_files):
        for pkl_file in _req_pkl_files:
            with open ('./datasets/gtn-imdb/' + pkl_file, 'wb') as f:
                pickle.dump(getattr(self, pkl_file.split('.')[0]), f)
