import numpy as np
import pickle
import os
from os.path import join
import pandas as pd
import multiprocessing
from tqdm import tqdm
import math
from datetime import datetime, timedelta, timezone
from functools import partial
from loader.preprocess.mm.utils import gcj02_to_wgs84


def convert_to_trajectory(group):
    trajectory = []
    for time, lng, lat in group.values:
        lng, lat = gcj02_to_wgs84(lng, lat)
        trajectory.append((int(time), lat, lng))
    return trajectory


def convert_single(group, time_zone):
    group = group.drop(columns=1, axis=1, inplace=False)
    group = group.sort_values(by=2).reset_index()
    group = group.drop(columns="index", axis=1, inplace=False)
    beg, end = group.index[0], group.index[-1]
    duration = group.at[end, 2] - group.at[beg, 2]
    if duration <= 300 or duration > 7200:
        return None
    init_timestamp = int(group.at[beg, 2])
    finish_timestamp = int(group.at[end, 2])
    init_dt = datetime.fromtimestamp(init_timestamp, time_zone)
    finish_dt = datetime.fromtimestamp(finish_timestamp, time_zone)
    if init_dt.day != finish_dt.day:
        return None
    return convert_to_trajectory(group)
    
    
def get_trajectories(date, raw_traj_path):
    print(f"processing date {date}...")
    data = pd.read_csv(open(join(raw_traj_path, f"gps_{date}"), "r"), header=None, sep=',')  
    data = data.drop(columns=[0], axis=1, inplace=False)
    print("read complete!")
    
    traj_grouped = data.groupby(by=[1], axis=0)
    n_process = min(int(os.cpu_count()) + 1, 30)
    
    trajectories = []
    time_zone = timezone(timedelta(hours=8))
    partialprocessParallel = partial(convert_single, time_zone=time_zone)
    with multiprocessing.Pool(n_process) as pool:
        results = list(tqdm(pool.imap(partialprocessParallel, [group for _, group in traj_grouped]), total=len(traj_grouped), ncols=80))
    trajectories = [each for each in results if each is not None]
    return trajectories
