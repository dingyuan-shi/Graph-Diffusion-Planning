import torch
import numpy as np
from planner.planner import Planner
from loader.dataset import TrajFastDataset
from tqdm import tqdm
from utils.fetch_navi import fetch_routes
from utils.coors import gcj02_to_wgs84, wgs84_to_gcj02
from utils.visual import draw_gps
import random
import math


class Evaluator:
    def __init__(self, model: Planner, dataset: TrajFastDataset):
        self.model = model
        self.dataset = dataset
        self.device = model.device
        
    def _convert_from_id_to_lat_lng(self, paths):
        path_coors = []
        for path in paths:
            path_coors.append([[self.dataset.G.nodes[v]["lat"], self.dataset.G.nodes[v]["lng"]]  for v in path])
        return path_coors
    
    def eval(self, n_samples, suffix):
        planned_paths, orig_paths = self.eval_exsits(500)
        planned_paths_coors = self._convert_from_id_to_lat_lng(planned_paths)
        draw_gps(planned_paths_coors[:10], f"./figs/exist_plan_{suffix}.html", colors=["red"] * 10, no_points=False)
        orig_paths_coors = self._convert_from_id_to_lat_lng(orig_paths)
        draw_gps(orig_paths_coors[:10], f"./figs/exist_orig_{suffix}.html", colors=["blue"] * 10, no_points=False)
        low, mid, high = self.eval_hit_rate(planned_paths_coors, orig_paths_coors)
        print(low, mid, high)
        
    def eval_hit_rate(self, planned_paths, ground_truth):
        lens = [len(planned) for planned in planned_paths]
        lens.sort()
        a1 = lens[len(lens) // 3]
        a2 = lens[len(lens) // 3 * 2]
        recs = [[], [], []]
        for k, (planned, ground) in enumerate(zip(planned_paths, ground_truth)):
            if len(planned) <= a1:
                idx = 0
            elif a1 < len(planned) <= a2:
                idx = 1
            else:
                idx = 2
            hit_cnt = 0
            for v in planned:
                for u in ground:
                    if math.fabs(v[0] - u[0]) + math.fabs(v[1] - u[1]) < 0.01:
                        hit_cnt += 1
                        break
            recs[idx].append(hit_cnt / len(planned))
        returns = []
        for k, rec in enumerate(recs):
            returns.append(sum(rec) / len(rec))
        return returns
        
    def eval_exsits(self, n_samples):
        choices = np.random.choice(len(self.dataset), [n_samples], False).tolist()
        # evaluate batchly
        set_batch_size = 15
        n_batch = (n_samples + set_batch_size - 1) // set_batch_size
        planned_paths = []
        orig_paths = [self.dataset[choices[k]] for k in range(n_samples)]
        hits = [0 for i in range(n_samples)]
        for i in tqdm(range(n_batch)):
            left, right = i * set_batch_size, min((i + 1) * set_batch_size, n_samples)
            origs = [self.dataset[choices[k]][0] for k in range(left, right)]
            dests = [self.dataset[choices[k]][-1] for k in range(left, right)]
            xs_list = self.model.plan(origs, dests, eval_nll=False)
            planned_paths.extend(xs_list)
            hits[left: right] = [a[-1] == b for a, b in zip(xs_list, dests)] 
        return planned_paths, orig_paths
    
    def eval_nonexsits(self, n_samples):
        choices = np.random.choice(len(self.dataset), [n_samples], False).tolist()
        set_batch_size = 10
        n_batch = (n_samples + set_batch_size - 1) // set_batch_size
        hit_cnt = 0
        planned_paths_coors = []
        ground_paths_group = []
        for i in range(n_batch):
            left, right = i * set_batch_size, min((i + 1) * set_batch_size, n_samples)
            origs = [self.dataset[choices[k]][0] for k in range(left, right)]
            dests = [self.dataset[choices[k]][-1] for k in range(left, right)]
            random.shuffle(dests)
            xs_list = self.model.plan(origs, dests, eval_nll=False)
            for orig, dest in zip(origs, dests):
                orig_lng, orig_lat = self.dataset.G.nodes[orig]["lng"], self.dataset.G.nodes[orig]["lat"]
                dest_lng, dest_lat = self.dataset.G.nodes[dest]["lng"], self.dataset.G.nodes[dest]["lat"]
                
                orig_lng, orig_lat = wgs84_to_gcj02(orig_lng, orig_lat)
                dest_lng, dest_lat = wgs84_to_gcj02(dest_lng, dest_lat)
                paths = fetch_routes(orig_lng, orig_lat, dest_lng, dest_lat)
                ground_paths_group.append([])
                for path in paths:
                    path_converted = []
                    for lng, lat in path:
                        lng, lat = gcj02_to_wgs84(lng, lat)
                        path_converted.append([lat, lng])
                    ground_paths_group[-1].append(path_converted)
            
            for x_list in xs_list:
                path = [[self.dataset.G.nodes[v]["lat"], self.dataset.G.nodes[v]["lng"]] for v in x_list]
                planned_paths_coors.append(path)
        # planned_paths: list of paths, each path is a list of [lat, lng]
        # ground_paths_group: list of path groups, each group contains serveral paths, each path is a list of [lat, lng]
        hit_recs = [0 for k in range(n_samples)]
        hit_rec_idx = 0
        orig_paths_coors = []
        for k in range(n_samples):
            path = planned_paths_coors[k]
            groups = ground_paths_group[k]
            hit_rec = 0
            for i, ground_path in enumerate(groups):
                hit_cnt = 0
                # calculate two paths hit
                for v in path:
                    for u in ground_path:
                        if math.fabs(v[0] - u[0]) + math.fabs(v[1] - u[1]) < 0.01:
                            hit_cnt += 1
                            break
                if hit_cnt / len(path) > hit_rec:
                    hit_rec = hit_cnt / len(path)
                    hit_rec_idx = i
            orig_paths_coors.append(groups[hit_rec_idx])
        return planned_paths_coors, orig_paths_coors
        