from loader.dataset import TrajFastDataset
from torch.utils.data import Dataset, DataLoader, random_split
from planner.planner import Planner
from planner.trainer import Trainer
from models_seq.seq_models import Destroyer
import torch
import pickle
from os.path import join
from tqdm import tqdm
import random


def DTWDistance(G, s1, s2):
    DTW={}
 
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0
 
    for i in range(len(s1)):
        for j in range(len(s2)):
            dist= (s1[i]-s2[j])**2
            a_lng, a_lat = G.nodes[s1[i]]["lng"], G.nodes[s1[i]]["lat"]
            b_lng, b_lat = G.nodes[s2[j]]["lng"], G.nodes[s2[j]]["lat"]
            dist = (a_lng - b_lng) ** 2 + (a_lat - b_lat) ** 2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
 
    return DTW[len(s1) - 1, len(s2) - 1] ** 0.5


def LCSSDistance(a, b):
    lena = len(a)
    lenb = len(b)
    c = [[0 for i in range(lenb + 1)] for j in range(lena + 1)]
    # flag = [[0 for i in range(lenb + 1)] for j in range(lena + 1)]
    for i in range(lena):
        for j in range(lenb):
            if a[i] == b[j]:
                c[i + 1][j + 1] = c[i][j] + 1
                # flag[i + 1][j + 1] = 'ok'
            elif c[i + 1][j] > c[i][j + 1]:
                c[i + 1][j + 1] = c[i + 1][j]
                # flag[i + 1][j + 1] = 'left'
            else:
                c[i + 1][j + 1] = c[i][j + 1]
                # flag[i + 1][j + 1] = 'up'
    return c[lena - 1][lenb - 1]



def evaluate_all():
    # split train test
    city = "xian"
    suffix = "cd" if city == "chengdu" else "xa"
    device = torch.device("cuda")
    date = "20161101" if city == "chengdu" else "20161001"
    dataset = TrajFastDataset(city, [date], path="./sets_data/", device=device, is_pretrain=False)
    train_num = int(0.8 * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_num , len(dataset) - train_num])
    pretrain_path = join("sets_data/", f"{city}_node2vec.pkl")
    node2vec = pickle.load(open(pretrain_path, "rb"))
    
    planner_gpd = torch.load(f"sets_model/planner_gpd_{suffix}.pth")
    planner_gpd.eval()
    
    if city == "chengdu":
        map_path = join("./sets_data/real/map", "map_chengdu.pkl")
    else:
        map_path = join("./sets_data/real2/map", "map_xian.pkl")
    planners = [planner_gpd]
    planner_names = ["GPD"]
    test_segs = [test_dataset[k] for k in range(1000)]
    rec = dict()
    for planner, planner_name in zip(planners, planner_names):
        oris = [path[0] for path in test_segs]
        dests = [path[-1] for path in test_segs]
        # random.shuffle(dests)
        # plan batchly
        batch_size = 30
        paths_planned = []
        for b in tqdm(range((len(oris) + batch_size - 1) // batch_size)):
            left = b * batch_size
            right = min((b + 1) * batch_size, len(oris))
            paths_planned.extend(planner.plan(oris[left:right], dests[left:right]))
        mean_lcs, mean_dtw, max_lcs, max_dtw, min_lcs, min_dtw, hit = 0., 0., 0., 0., 1000000, 100000, []
        for k, (planned, ground) in enumerate(zip(paths_planned, test_segs)):
            lcs = LCSSDistance(planned, ground)
            dtw = DTWDistance(dataset.G, planned, ground)
            mean_lcs += lcs
            mean_dtw += dtw
            max_lcs = max(max_lcs, lcs)
            max_dtw = max(max_dtw, dtw)
            min_lcs = min(min_lcs, lcs)
            min_dtw = min(min_dtw, dtw)
            hit.append((len(ground), 1 if planned[-1] == dests[k] else 0))
        mean_lcs /= len(test_segs)
        mean_dtw /= len(test_segs)
        hit.sort()
        hit_lo = sum([each[1] for each in hit[:len(hit) // 3]])
        hit_md = sum([each[1] for each in hit[len(hit) // 3: 2 * len(hit) // 3]])
        hit_hi = sum([each[1] for each in hit[2 * len(hit) // 3:]])
        rec[planner_name] = {
            "mean_lcs": mean_lcs, 
            "mean_dtw": mean_dtw, 
            "max_lcs": max_lcs, 
            "max_dtw": max_dtw,
            "min_lcs": min_lcs, 
            "min_dtw": min_dtw, 
            "hit": [hit_lo / 333, hit_md / 333 , hit_hi / 334]
        }
        print(f"{planner_name}: {rec[planner_name]}")
        


if __name__ == "__main__":
    evaluate_all()