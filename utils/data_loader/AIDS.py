from torch_geometric.datasets import GEDDataset
"""
AIDS700nef:
    train_data: 560张图 Data(edge_index=[2, 16], i=[1], x=[9, 29], num_nodes=9)
    test_data: 140张图 二者均通过下标访问，train_data[0] test_data[0]
    ged: train_data中两两之间存在GED值 train_data和test_data之间存在GED值；test_data与test_data之间不存在GED
    train_data.ged[train_data[0].i, test_data[2].i] 返回train_data中第一张图与test_data中第三张图之间的GED值
    train_data.norm_ged[XXX.i, YYY.i] 返回两张图之间的nGED值，定义为nGED = (GED(G1, G2)) / ((|G1|+|G2|)/2)
"""

class data_loader:
    r"""
        使用AIDS700nef, LINUX, IMDBMulti, ALKANE四个不同的 GEDDataset 数据集输入数据
        train_data: 训练数据集，是pyg Data
        test_data: 测试数据集，是pyg Data
        train_pairs_id: 训练图对，每个元素是一个元组，元组中包含两个数，是两个不同的图的id。id来源 (训练集，训练集)
        validate_pairs_id: 验证图对，id来源 (训练集，测试集前半)
        test_pairs_id: 测试图对，id来源 (训练集，测试集后半)
            训练集：验证集：测试集 = 8 : 1 : 1
        num_features: 图特征维度
        norm_ged: 两张图nGED，但是还没有使用 e^-x 进行归一化
    """
    def __init__(self):
        path = './datasets/AIDS700nef'
        dataset = path.split('/')[-1]
        self.train_data = GEDDataset(path, dataset)
        self.test_data = GEDDataset(path, dataset, False)
        
        self.norm_ged_homoro = self.train_data.norm_ged
