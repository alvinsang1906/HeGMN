import argparse

r"""这个文件涉及main函数需要的参数解析
    --dataset_path: 数据集的文件路径    默认为./datasets/
    --dataset:      使用的数据集名称    默认为ACM
    --log_path:     日志文件存放路径    默认为./Logs/
    --model:        使用的模型         默认为HGMN
"""
class Parser:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(description="Heterogeneous Graph Matching Network (HGMN)")
        self.set_arguments()
        
    def set_arguments(self):
        # 开始添加参数 
        # 数据集文件的路径 默认datasets路径
        self.parser.add_argument('--dataset_path'
                            , type=str
                            , default='./datasets/'
                            , help='path to the datasets')

        # 接下来要使用的数据集 默认MUTAG
        self.parser.add_argument('--dataset'
                            , type=str
                            , default='ACM'
                            , choices=['ACM', 'DBLP', 'IMDB', 'MUTAG', 'AIDS700nef']
                            , help='the specific dataset which will be used next')

        # 设置log路径 默认就是Logs目录
        self.parser.add_argument('--log_path'
                            , type=str
                            , default='./Logs/'
                            , help='path to logs')

        # 设置模型名称，默认使用本文模型
        self.parser.add_argument('--model'
                            , type=str
                            , default='HGMN'
                            , choices=['HGMN', 'SimGNN', 'ERIC', 'GMN', 'SimpleHGN', 'GAT', 'GCN', 'GIN', 'RGCN', 'GraphSim', 'NAGSL']
                            , help='input which model will be train next')

    def parse(self):
        # 解析参数，args中就是上面的所有参数内容
        args, unparsed  = self.parser.parse_known_args()
        
        if len(unparsed) != 0:
            raise ValueError('Unknown argument: {}'.format(unparsed))
        return args
