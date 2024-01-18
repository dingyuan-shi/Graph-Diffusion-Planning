import pickle
from torch.utils.data import Dataset
import h5py
from os.path import join, exists
import torch
import numpy as np
import networkx as nx
from loader.node2vec import get_node2vec


class TrajFastDataset(Dataset):
    def __init__(self, city, dates, path, device, is_pretrain):
        super().__init__()
        name = city
        self.device = device
        
        shrink_G_path = join(path, f"{name}_shrink_G.pkl")
        shrink_A_path = join(path, f"{name}_shrink_A.ts")
        shrink_NZ_path = join(path, f"{name}_shrink_NZ.pkl")
        
        if exists(shrink_G_path):
            print("loading")
            self.G = pickle.load(open(shrink_G_path, "rb"))
            self.A = pickle.load(open(shrink_A_path, "rb"))
            self.shrink_nonzero_dict = pickle.load(open(shrink_NZ_path, "rb"))
            print("finished")
        else:
            self.G = pickle.load(open(join(path, f"{name}_G.pkl"), "rb"))
            self.n_vertex = len(self.G.nodes)
            self.A_orig = torch.load(join(path, f"{name}_A.ts"), map_location=torch.device("cpu"))
            print("loading path...")
            self.v_paths = np.loadtxt(join(path, f"{name}_v_paths.csv"), delimiter=',') 
            print("finish loading")
            nonzeros = np.nonzero(self.v_paths.sum(0))[0]
            self.nonzeros = nonzeros
            print(f"shrink into {nonzeros.shape[0]} nodes")
            B = self.A_orig[nonzeros, :]
            self.A = B[:, nonzeros]
            self.v_paths = self.v_paths[:, nonzeros]
            self.length = self.v_paths.shape[0]
            self.shrink_nonzero_dict = dict()
            for k in range(nonzeros.shape[0]):
                self.shrink_nonzero_dict[nonzeros[k]] = k
        
            # shrink G
            G_shrink = nx.Graph()
            shrink_node_attrs = [(k, {"lat": self.G.nodes[nonzeros[k]]["lat"], "lng": self.G.nodes[nonzeros[k]]["lng"]}) for k in range(self.nonzeros.shape[0])]
            G_shrink.add_nodes_from(shrink_node_attrs)
            for i in range(self.A.shape[0]):
                for j in range(self.A.shape[0]):
                    if self.A[i, j] > 0.5:
                        G_shrink.add_edge(i, j)
            self.G = G_shrink
            self.A = self.A.to(self.device)
            print("finish shrink")
            pickle.dump(self.G, open(shrink_G_path, "wb"))
            pickle.dump(self.A, open(shrink_A_path, "wb"))
            pickle.dump(self.shrink_nonzero_dict, open(shrink_NZ_path, "wb"))
        
        
        self.n_vertex = len(self.G.nodes)
        self.dates = dates
        h5_file = join(path, f"{city}_h5_paths.h5")
        self.f = h5py.File(h5_file, "r")
        sample_len = [self.f[date]["state_prefix"].shape[0] - 1 for date in dates]
        accu_len = [0 for _ in range(len(sample_len) + 1)]
        for k, l in enumerate(sample_len):
            accu_len[k + 1] = accu_len[k] + l
        self.accu_len = accu_len
        self.total_len = accu_len[-1]
        # if pretrain
        if is_pretrain:
            embed_path = join(path, f"{city}_node2vec.pkl")
            path_path = join(path, f"{city}_path.pkl")
            get_node2vec(self.G, embed_path, path_path)
        
    def __upper_bound(self, num):
        l, r = 0, len(self.accu_len)
        while l < r:
            m = (l + r) // 2
            if self.accu_len[m] <= num:
                l = m + 1
            else:
                r = m
        return l
            
    def __getitem__(self, index):
        idx = self.__upper_bound(index) - 1
        date = self.dates[idx]
        offset = index - self.accu_len[idx]
        pleft, pright = self.f[date]["state_prefix"][offset], self.f[date]["state_prefix"][offset + 1]
        # return self.__filter(self.f[date]["states"][pleft: pright])
        return [self.shrink_nonzero_dict[node] for node in self.f[date]["states"][pleft: pright]]
        
    def __len__(self):
        return self.total_len
    
    def __filter(self, points):
        points_filtered = []
        
        showup = set()
        for k, node in enumerate(points):
            node = self.shrink_nonzero_dict[node]
            if node not in showup:
                showup.add(node)
                points_filtered.append(node)
            else:
                while points_filtered[-1] != node:
                    showup.discard(points_filtered[-1])
                    points_filtered.pop()
        return points_filtered
    
    def get_real_paths(self, num=500):
        choices = np.random.choice(a=self.total_len, size=num, replace=False).tolist()
        return [self.__getitem__(c) for c in choices]
    