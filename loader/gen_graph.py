import networkx as nx
import torch
from torch.utils.data import Dataset
from os.path import join, exists
import numpy as np
from tqdm import tqdm
import multiprocessing
import pickle
from loader.node2vec import get_node2vec

def gen_batch(pid, A, G, starts, lengths, beg, end):
    sub_num = end - beg
    v_paths = np.zeros([sub_num, A.shape[0]])
    for k in tqdm(range(sub_num)):
        start = starts[k].item()
        cur = start
        v_paths[k][start] = 1
        length = lengths[k]
        for l in range(length - 1):
            next_cand = A[cur] * (1 - v_paths[k])
            next_cand = np.nonzero(next_cand)[0]
            if next_cand.shape[0] == 0:
                break
            next_node = next_cand[torch.randint(0, next_cand.shape[0], (1,))[0]].item()
            assert next_node in G[cur]
            cur = next_node
            v_paths[k][cur] = 1
    return v_paths


def reverse(A, v_paths, num=-1):
    paths = []
    path_num = min(num, v_paths.shape[0]) if num != -1 else v_paths.shape[0]
    n_vertex = A.shape[0]
    device = A.device
    for k in range(path_num):
        start = torch.argmax(v_paths[k,:]).item()
        # input()
        marked = torch.ones(n_vertex, device=device)
        path = [start]
        marked[start] = 0
        while len(path) <= n_vertex:
            cur = path[-1]
            target = A[cur] * v_paths[k] * marked
            nex = torch.argmax(target).item()
            if target[nex].item() < 0.25:
                break
            path.append(nex)
            marked[nex] = 0
        paths.append(path)
    return paths
    
    
class DataGenerator(Dataset):
    def __init__(self, n_vertex, n_path, min_len, max_len, device, path="", name="") -> None:
        if len(path) == 0 or (not exists(join(path, f"{name}_G.pkl"))):
            self.n_vertex = n_vertex
            self.device = device
            self.G, self.A = self._gen_graph(n_vertex)
            print(f"generate a graph with {self.n_vertex} verteices and {len(self.G.edges)} edges")
            self.v_paths = self.gen_path(n_path, min_len, max_len)
            print(f"generate {self.v_paths.shape[0]} paths")
            self.length = self.v_paths.shape[0]
            if len(path):
                self._save_to_file(path, name)
        else:
            self.device = device
            self._load_from_file(path, name)
        embed_path = join(path, f"{name}_node2vec.pkl")
        path_path = join(path, f"{name}_path.pkl")
        self.node2emb = get_node2vec(self.G, embed_path, path_path, p=2, q=4)
        
    def _load_from_file(self, path, name):
        self.G = pickle.load(open(join(path, f"{name}_G.pkl"), "rb"))
        self.n_vertex = len(self.G.nodes)
        self.A = torch.load(join(path, f"{name}_A.ts"), map_location=self.device)
        v_paths = np.loadtxt(join(path, f"{name}_v_paths.csv"), delimiter=',') 
        self.v_paths = torch.from_numpy(v_paths).to(self.device).to(torch.float64)
        self.length = self.v_paths.shape[0]
    
    def _save_to_file(self, path, name):
        pickle.dump(self.G, open(join(path, f"{name}_G.pkl"), "wb"))
        torch.save(self.A, join(path, f"{name}_A.ts"))
        np.savetxt(join(path, f"{name}_v_paths.csv"), self.v_paths.cpu().detach().numpy(),delimiter=',', fmt='%d')
        
        
    def __getitem__(self, index):
        return self.v_paths[index]
    
    def __len__(self):
        return self.length
        
    def _gen_graph(self, n):
        G = nx.erdos_renyi_graph(n, 0.4, seed=123, directed=False)
        # build adjacent matrix
        A = torch.zeros([n, n], dtype=torch.float64).to(self.device)
        for a, b in G.edges:
            A[a, b] = 1.
            A[b,a] = 1.
        return G, A

    def gen_path(self, num_path, min_len, max_len):
        n = len(self.G.nodes)
        starts = torch.randint(0, n, [num_path], device=torch.device("cpu"))
        lengths = torch.randint(min_len, max_len + 1, [num_path], device=torch.device("cpu"))
        
        n_process = 10
        err = lambda err: print(err)
        batch_size = (num_path + n_process - 1) // n_process
        
        pool = multiprocessing.Pool(processes=n_process)
        res_mid = []
        A = self.A.clone().cpu().numpy()
        for i in range(0, num_path, batch_size):
            pid = i // batch_size
            beg, end = i, min(i + batch_size, num_path)
            res = pool.apply_async(gen_batch, (pid,A,self.G,starts,lengths,beg,end,), error_callback=err)
            res_mid.append(res)
        pool.close()
        pool.join()
        res_final = [torch.Tensor(each.get()).to(self.device) for each in res_mid]
        v_paths = torch.cat(res_final, dim=0).to(self.device)
        v_paths[torch.arange(num_path), starts] += 1.
        return v_paths
    
    def get_real_paths(self, num=-1):
        if hasattr(self, "paths"):
            return self.paths if num == -1 else self.paths[:num]
        self.paths = reverse(self.A, self.v_paths, num)
        return self.paths
    
    
class TrajDataset(Dataset):
    def __init__(self, path, name, device) -> None:
        super().__init__()
        self.device = device
        shrink_G_path = join(path, f"{name}_shrink_G.pkl")
        shrink_A_path = join(path, f"{name}_shrink_A.ts")
        shrink_p_path = join(path, f"{name}_shrink_v_paths.csv")
        if exists(shrink_G_path):
            print("loading")
            self.G = pickle.load(open(shrink_G_path, "rb"))
            self.A = pickle.load(open(shrink_A_path, "rb"))
            self.v_paths = np.loadtxt(shrink_p_path, delimiter=',')
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
        
            # shrink G
            G_shrink = nx.Graph()
            print(nonzeros)
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
            np.savetxt(shrink_p_path, self.v_paths, delimiter=',')
            
            
        embed_path = join(path, f"{name}_node2vec.pkl")
        path_path = join(path, f"{name}_rw_path.pkl")
        self.node2emb = get_node2vec(self.G, embed_path, path_path, p=2, q=4)
        self.length = self.v_paths.shape[0]
    
    def __getitem__(self, index):
            return self.v_paths[index]
    
    def __len__(self):
        return self.length
    
    def get_real_paths(self, num=-1):
        if hasattr(self, "paths"):
            return self.paths if num == -1 else self.paths[:num]
        self.paths = reverse(self.A, self.v_paths, num)
        return self.paths