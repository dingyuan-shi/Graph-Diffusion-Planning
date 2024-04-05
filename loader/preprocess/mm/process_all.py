from loader.preprocess.mm.fetch_rdnet import fetch_map, build_map
from loader.preprocess.mm.mapmatching import process_gps_and_graph


if __name__ == "__main__":
    
    data_path = "./sets_data/"
    
    # process real2
    city = "xian"
    bounds = [108.9, 34.20, 109.01, 34.28]
    map_path = "./sets_data/real2/map"
    
    fetch_map(city, bounds, map_path)
    map_con = build_map(city, map_path)
    
    raw_path = "./sets_data/real2/raw"
    traj_path = "./sets_data/real2/trajectories"
    process_gps_and_graph(city, map_path, data_path, raw_path, traj_path)
    
    # process real
    city = "chengdu"
    bounds = [104.0, 30.64, 104.15, 30.73]
    map_path = "./sets_data/real/map"
    fetch_map(city, bounds, map_path)
    map_con = build_map(city, map_path)
    
    raw_path = "./sets_data/real/raw"
    traj_path = "./sets_data/real/trajectories"
    process_gps_and_graph(city, map_path, data_path, raw_path, traj_path)
    