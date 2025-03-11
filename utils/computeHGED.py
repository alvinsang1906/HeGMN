import torch
import pickle
import argparse
import networkx as nx
import multiprocessing

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='IMDB', choices=['IMDB', 'DBLP', 'ACM', 'MUTAG'], type=str, help='which dataset needs to calculate GED or HGED')
    parser.add_argument('--HGED', default=1, type=int, help='1 --- means calculate HGED, 0 --- means calculate GED')
    parser.add_argument('--multiprocess_number', default=32, type=int, help='change the number of multiprocess')
    parser.add_argument('--mode', default=0, type=int, help='\
                        0 --- means calculate all graph_pairs from gid_1 to gid_2;\
                        1 --- means calculate all graph_pairs just on special one gid from [gid, gid_1] to [gid, gid_2]; \
                        2 --- means check the file GED_hetero or GED_hetero has all the computed HGED or GED, if so ,save them; if not calculate them;\
                        3 --- means need to sort the para_ged.txt file')
    parser.add_argument('--gid1', default=-1, type=int, help='when mode == 1 or mode == 2, means the start graph_id')
    parser.add_argument('--gid2', default=-1, type=int, help='when mode == 1 or mode == 2, means the end graph_id')
    parser.add_argument('--gid', default=-1, type=int, help='when mode == 2, means the special graph_id')
    args = parser.parse_args()
    return args

def set_path_and_filename(args):
    path = '../datasets/'
    if args.dataset == 'IMDB':
        from data_loader.IMDB import data_loader
        path += 'gtn-imdb/'
        graph_nums = 1200
    elif args.dataset == 'DBLP':
        from data_loader.DBLP import data_loader
        path += 'gtn-dblp/'
        graph_nums = 700
    elif args.dataset == 'ACM':
        from data_loader.ACM import data_loader
        path += 'gtn-acm/'
        graph_nums = 1000
    elif args.dataset == 'MUTAG':
        from data_loader.MUTAG import data_loader
        path += 'MUTAG/'
        graph_nums = 188
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    filename = path + 'para_ged.txt'
    data = data_loader(path)

    if args.HGED == 1:
        GED_file = path + 'GED_hetero.txt'
        norm_ged_file = path + 'norm_ged_hetero.pkl'
    else:
        GED_file = path + 'GED_homoro.txt'
        norm_ged_file = path + 'norm_ged_homoro.pkl'
    return path, filename, graph_nums, data, GED_file, norm_ged_file

def set_node_and_edge_match(args):
    if args.HGED == 1:
        def node_match(n1, n2):
            if n1['node_type'] == n2['node_type']:
                return 0
            else:
                return 2
        
        def edge_match(e1, e2):
            if e1['edge_type'] == e2['edge_type']:
                return 0
            else:
                return 2
        return node_match, edge_match
    
    if args.HGED == 0:
        def node_match(n1, n2):
            if n1['node_type'] == n2['node_type']:
                return 0
            else:
                return 1
        
        def edge_match(e1, e2):
            if e1['edge_type'] == e2['edge_type']:
                return 0
            else:
                return 1
        return node_match, edge_match
    raise ValueError('Unknown HGED: {}'.format(args.HGED))

def check_error(args):
    if args.mode == 0:
        if args.gid1 == -1 or args.gid2 == -1:
            raise ValueError('when mode == 0, gid1 and gid2 can not be unassigned')
    if args.mode == 1:
        if args.gid1 == -1 or args.gid2 == -1 or args.gid == -1:
            raise ValueError('when mode == 1, gid1, gid2 and gid can not be unassigned')
    if args.mode == 2 or args.mode == 3:
        if args.gid1 != -1 or args.gid2 != -1 or args.gid != -1:
            raise ValueError('when mode == 2 or mode == 3, gid1, gid2 and gid need to be unassigned')

def generate_graph_pairs_list(gid1, gid2, graph_nums, gid=-1):
    if gid == -1:
        graph_pairs_list = [(i, j) for i in range(gid1, gid2) for j in range(i + 1, graph_nums)]
        return graph_pairs_list
    graph_pairs_list = [(gid, i) for i in range(gid1, gid2)]
    return graph_pairs_list

def separate_graph_pairs(graph_pairs_list, multiprocess_number):
    graph_pairs_list = [graph_pairs_list[i : i + multiprocess_number] for i in range(0, len(graph_pairs_list), multiprocess_number)]
    return graph_pairs_list

def calculateGED(filename, graph_pair, node_match, edge_match):
    graph1 = data.graphs[graph_pair[0]]
    graph2 = data.graphs[graph_pair[1]]
    ged = nx.graph_edit_distance(graph1, graph2, node_subst_cost=node_match, edge_subst_cost=edge_match, timeout=600)
    with open(filename, 'a') as f:
        f.write("{}\t{}\t{}\n".format(graph1.graph['i'], graph2.graph['i'], ged))

def multiProcess_tasks(func, filename, graph_pairs_list, node_match=None, edge_match=None):
    for graph_pairs in graph_pairs_list:
        processes = []
        for gp in graph_pairs:
            process = multiprocessing.Process(target=func, args=(filename, gp, node_match, edge_match))
            processes.append(process)
            process.start()
        
        for process in processes:
            process.join()

def compute_from_graph_pairs_list(graph_pairs_list, multiprocess_number, func, filename, node_match, edge_match):
    graph_pairs_list = separate_graph_pairs(graph_pairs_list, multiprocess_number)
    multiProcess_tasks(func, filename, graph_pairs_list, node_match, edge_match)

def sort_file(filename):
    with open(filename, "r") as f:
        data = [line.rstrip('\n').split('\t') for line in f]

    sdata = sorted(data, key=lambda x: (int(x[0]), int(x[1])))
    with open(filename, "w") as f:
        for d in sdata:
            f.write("{}\t{}\t{}\n".format(d[0], d[1], d[2]))

def save2arrs(graph_nums, GED_file):
    ged = torch.full((graph_nums, graph_nums), float('inf'))
    norm_ged = torch.full((graph_nums, graph_nums), float('inf'))
    train_nums = int(graph_nums * 0.8)
    for i in range(train_nums):
        ged[i, i] = 0.0
        norm_ged[i, i] = 0.0

    with open(GED_file, 'r') as f:
        for line in f:
            G1_id, G2_id, tempged = line.strip().split('\t')
            if int(G1_id) >= graph_nums or int(G2_id) >= graph_nums:
                continue

            ged[int(G1_id), int(G2_id)], ged[int(G2_id), int(G1_id)] = float(tempged), float(tempged)
            norm = round(float(tempged) / ((data.graphs[int(G1_id)].number_of_nodes() + data.graphs[int(G2_id)].number_of_nodes()) / 2), 4)
            norm_ged[int(G1_id), int(G2_id)], norm_ged[int(G2_id), int(G1_id)] = norm, norm

    graph_pairs_list = is_exist_all_ged(norm_ged, graph_nums, train_nums)
    return graph_pairs_list, norm_ged

def is_exist_all_ged(norm_ged, graph_nums, train_nums):
    graph_pairs_need2becaled = []
    for i in range(graph_nums):
        for j in range(graph_nums):
            if i < train_nums or j < train_nums:
                if norm_ged[i, j] == torch.inf and i < j:
                    graph_pairs_need2becaled.append((i, j))
            else:
                if norm_ged[i, j] != torch.inf:
                    norm_ged[i, j] = torch.inf
    return graph_pairs_need2becaled

def save2pkl(norm_ged, norm_ged_file):
    with open(norm_ged_file, 'wb') as f:
        pickle.dump(norm_ged, f)

if __name__ == "__main__":
    args = get_args()
    path, filename, graph_nums, data, GED_file, norm_ged_file = set_path_and_filename(args)
    node_match, edge_match = set_node_and_edge_match(args)

    check_error(args)
    if args.mode == 3:
        # 排序文件 para_ged.txt, 生成已经排好序的para_ged.txt文件
        sort_file(filename)
    elif args.mode == 2:
        graph_pairs_list, norm_ged = save2arrs(graph_nums, GED_file)
        if len(graph_pairs_list) != 0:
            compute_from_graph_pairs_list(graph_pairs_list, args.multiprocess_number, calculateGED, filename, node_match, edge_match)
            print("Need to check one more time!!!")
        else:
            save2pkl(norm_ged, norm_ged_file)
            print("Generate Finished!!!")
    else:
        graph_pairs_list = generate_graph_pairs_list(args.gid1, args.gid2, graph_nums, args.gid)
        compute_from_graph_pairs_list(graph_pairs_list, args.multiprocess_number, calculateGED, filename, node_match, edge_match)
        print("Compute Finished!!!")
