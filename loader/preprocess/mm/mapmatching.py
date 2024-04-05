import matplotlib.pyplot as plt 
import torch
plt.switch_backend("agg")
from loader.preprocess.mm.refine_gps import get_trajectories
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap
import multiprocessing
from multiprocessing import cpu_count
from tqdm import tqdm
import os
from os.path import join
import pickle
import h5py
import numpy as np
import networkx as nx


def map_single(trajectory, map_con):
    path = [[each[1], each[2]] for each in trajectory]

    matcher = DistanceMatcher(map_con, 
                               max_dist=400, 
                               max_dist_init=400, 
                               min_prob_norm=0.2,
                                obs_noise=100, 
                                obs_noise_ne=100,
                              dist_noise=100,
                              max_lattice_width=20,
                              non_emitting_states=False)
    
    states, match_length = matcher.match(path)
    if len(states) < len(path):
        return None
    # shrink
    states_shrinked = [states[0]]
    link_points = [states[0][0], states[0][1]]
    states_to_point = [[trajectory[0]]]
    for i in range(1, len(states)):
        if states[i - 1][0] != states[i][0] or states[i - 1][1] != states[i][1]:
            assert states[i - 1][1] == states[i][0]
            link_points.append(states[i][1])
            states_shrinked.append(states[i])
            states_to_point.append([trajectory[i]])
        else:
            states_to_point[-1].append(trajectory[i])

    states_non_loop = []
    node_states = [a for a, b in states_shrinked] + [states_shrinked[-1][1]]
    # filter loops
    show_pos = dict()
    for a in node_states:    
        if a not in show_pos:
            show_pos[a] = len(states_non_loop)
            states_non_loop.append(a)
        else:
            for k in range(len(states_non_loop) - 1, show_pos[a], -1):
                last = states_non_loop.pop()
                show_pos.pop(last)
    if len(states_non_loop) < 5:
        return None
    return (link_points, states_shrinked, states_to_point, states_non_loop)


def map_batch(pid, trajectories, city, map_path):
    map_con = InMemMap.from_pickle(join(map_path, f"map_{city}.pkl"))
    trajectories_mapped = []
    for i in tqdm(range(len(trajectories)), ncols=80, position=pid):
        states_to_point_idx_states = map_single(trajectories[i], map_con)
        if states_to_point_idx_states:
            trajectories_mapped.append(states_to_point_idx_states)
    return trajectories_mapped


def mapmatching(date, city, raw_traj_path, map_path):
    trajectories = get_trajectories(date, raw_traj_path)
    trajectories_mapped = []
    
    n_process = min(int(cpu_count()) + 1, 20)
    trajectories_mapped_batch_mid = []
    with multiprocessing.Pool(processes=n_process) as pool:
        err = lambda err: print(err)
        batch_size = (len(trajectories) + n_process - 1) // n_process
        
        for i in range(0, len(trajectories), batch_size):
            pid = i // batch_size
            trajectory_mapped_batch = pool.apply_async(map_batch, (pid,trajectories[i: i + batch_size],city,map_path,), error_callback=err)
            trajectories_mapped_batch_mid.append(trajectory_mapped_batch)

        for each in trajectories_mapped_batch_mid:
            trajectories_mapped.extend(each.get())
        return trajectories_mapped


def get_matched_path(date, city, traj_path, map_path, raw_path):
    target_path = join(traj_path, f"traj_mapped_{city}_{date}.pkl")
    if os.path.exists(target_path):
        print("loading...")
        return pickle.load(open(target_path, "rb"))
    trajectories_mapped = mapmatching(date, city, raw_path, map_path)
    print("writing...")
    pickle.dump(trajectories_mapped, open(target_path, "wb"))
    print("write complete!")
    return trajectories_mapped


def process_gps_and_graph(city, map_path, data_path, raw_path, traj_path):
    name = city
    map_con = InMemMap.from_pickle(join(map_path, f"map_{city}.pkl"))
    # calculate G and A
    target_g_path = join(data_path, f"{name}_G.pkl")
    G = nx.Graph()
    node_attrs = [(cid, {"lat": lat, "lng": lng}) for cid, (lat, lng) in map_con.all_nodes()]
    G.add_nodes_from(node_attrs)
    G.add_edges_from([(a, b) for a, _, b, _ in map_con.all_edges()])
    n = G.number_of_nodes()
    A = torch.zeros([n, n], dtype=torch.float64)
    for a, b in G.edges:
        A[a, b] = 1.
        A[b,a] = 1.
    pickle.dump(G, open(target_g_path, "wb"))
    torch.save(A, join(data_path, f"{name}_A.ts"))
    gps_file_list = list(os.listdir(raw_path))
    gps_file_list.sort()
    gps_file_list = [each for each in gps_file_list if "gps" in each]
    h5_file = join(data_path, f"{name}_h5_paths.h5")
    with h5py.File(h5_file, "w") as f:
        for gps_file in gps_file_list[:1]:
            date = gps_file[4:]
            print("#####", date)
            f.create_group(date)
            trajectories_mapped = get_matched_path(date, city, traj_path, map_path, raw_path)
            # shrink 
            state_lengths, states = [], []
            
            for link_points, states_shrinked, states_to_point, states_non_loop in trajectories_mapped:
                state_lengths.append(len(states_non_loop))
                states.extend(states_non_loop)

            # calcluate prefix sum
            state_prefix = np.zeros(shape=len(state_lengths) + 1, dtype=np.int64)
            for k, L in enumerate(state_lengths):
                state_prefix[k + 1] = state_prefix[k] + L
            
            # length_info
            # pad all in one
            f[date].create_dataset("state_prefix", data=np.array(state_prefix))
            f[date].create_dataset("states", data=np.array(states))
    
    # calculate V
    target_v_path = join(data_path, f"{name}_v_paths.csv")
    vs = []
    for gps_file in gps_file_list[:1]:
        date = gps_file[4:]
        trajectories_mapped = get_matched_path(date, city, traj_path, map_path, raw_path)
        # (link_points, states_shrinked, states_to_point, states_non_loop)
        non_loops  = [each[-1] for each in trajectories_mapped]
        n_samples = len(non_loops)
        v_np = np.zeros([n_samples, n])
        for k, non_loop in enumerate(non_loops):
            v_np[k, non_loop] = 1.
            v_np[k, non_loop[0]] = 2.
        vs.append(v_np)
    # generate V
    v_data = np.concatenate(vs, axis=0)
    np.savetxt(target_v_path, v_data, delimiter=',', fmt='%d')