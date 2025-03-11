import os
import glob
import torch
import pickle
import numpy as np
import networkx as nx

from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset

class data_loader:
    def __init__(self, path=None):
        if path == None:
            _path = './datasets/MUTAG/'
        else:
            _path = path
        _pkl_files = glob.glob(_path + '*.pkl')
        _req_pkl_files = ['graphs.pkl', 'train_data.pkl', 'test_data.pkl', 'norm_ged_hetero.pkl', 'norm_ged_homoro.pkl']

        if all(os.path.basename(file) in _req_pkl_files for file in _pkl_files) and len(_pkl_files) == len(_req_pkl_files):
            print("exist all pickle files")
            self.graphs = self._load_pickle(_path + 'graphs.pkl')
            self.train_data = self._load_pickle(_path + 'train_data.pkl')
            self.test_data = self._load_pickle(_path + 'test_data.pkl')
            self.norm_ged_hetero = self._load_pickle(_path + 'norm_ged_hetero.pkl')
            self.norm_ged_homoro = self._load_pickle(_path + 'norm_ged_homoro.pkl')
        else:
            print("no pickle files")
            _homoro_data = TUDataset('./datasets', 'MUTAG')
            self.graphs = self._torch2nx(_homoro_data)
            self.train_data, self.test_data = self._nx2torch(_homoro_data)
            self.norm_ged_hetero, self.norm_ged_homoro = self._get_ged(_path)
            self._save_files(_req_pkl_files)

    def _load_pickle(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def _torch2nx(self, data):
        r"""
            将pyg格式的同质图，转换为nx格式，用于计算HGED
            参数 data: Data，pyg格式的同质图，里面包含188张图
            返回 graphs: List, 存放了188张nx同质图，可以用于计算GED
        """
        graphs = []
        for i in range(len(data)):
            g = nx.Graph(i=i)
            N = data[i].x.shape[0]
            for j in range(N):
                g.add_node(j, node_type=int( np.argmax(data[i].x[j]) ))

            eid = 0
            for j in range(len(data[i].edge_attr)):
                srcid, dstid = int(data[i].edge_index[0][j]), int(data[i].edge_index[1][j])
                etype = int(np.argmax(data[i].edge_attr[j]))
                # [srcid, dstid]这条边一定不存在于g中，但是[dstid, srcid]可能已经添加过了。
                if g.has_edge(dstid, srcid) is False:
                    g.add_edge(srcid, dstid, edge_type=etype, edge_id=eid)
                    eid += 1
            graphs.append(g)
        return graphs
        
    def _nx2torch(self, data):
        r"""
            将nx同质图转换为pyg格式的同质图，并且分割为训练集train_data，以及测试集test_data
            参数 data: Data, pyg格式同质图
            返回 train_data: List, 其中包含了188 * 0.8张同质图训练集
            返回 test_data: List, 其中剩下的部分188 * 0.2就可以平分为验证集和测试集
        """
        train_data, test_data = [], []
        max_degree = 0
        for nxg in self.graphs:
            degree = dict(nxg.degree())
            max_value = max(degree.values())
            max_degree = max(max_value, max_degree)
        
        for i in range(len(data)):
            degree = dict(self.graphs[i].degree())
            degree = dict(sorted(degree.items(), key=lambda item: int(item[0]))) # 按 key 排序
            d = list(degree.values())       # 将 values 转换为 list 
            d = np.eye(max_degree + 1)[d]   # onehot编码
            x = torch.cat([data[i].x, torch.as_tensor(d).float()], dim=1)
            
            node_type = []
            for j in range(data[i].x.shape[0]):
                node_type.append( np.argmax(data[i].x[j]) )
            node_type = torch.tensor(node_type)

            edge_type = []
            for j in range(data[i].edge_attr.shape[0]):
                edge_type.append( np.argmax(data[i].edge_attr[j]))
            edge_type = torch.tensor(edge_type)
            g = Data(x=x, edge_index=data[i].edge_index, node_type=node_type, edge_type=edge_type, i=i)
            if i < int(len(data) * 0.8):
                train_data.append(g)
            else:
                test_data.append(g)
        return train_data, test_data

    def _get_ged(self, path):
        r"""
            将使用nx库计算得到的排好序的GED值导入self.norm_ged中，以便后续使用
            参数 path: str, 数据集MUTAG存储的目录
        """
        ged_hetero, ged_homoro = torch.full((188, 188), float('inf')), torch.full((188, 188), float('inf'))
        norm_ged_hetero, norm_ged_homoro = torch.full((188, 188), float('inf')), torch.full((188, 188), float('inf'))
        for i in range(150):
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
    
    def _save_files(self, _req_pkl_files):
        for pkl_file in _req_pkl_files:
            with open ('./datasets/MUTAG/' + pkl_file, 'wb') as f:
                pickle.dump(getattr(self, pkl_file.split('.')[0]), f)
