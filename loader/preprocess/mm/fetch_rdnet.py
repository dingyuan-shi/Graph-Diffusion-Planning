import matplotlib.pyplot as plt
plt.switch_backend("agg")
import os
from os.path import join
from leuvenmapmatching.map.inmem import InMemMap
import osmnx as ox
from typing import List


# fetch chengdu
def fetch_map(city: str, bounds: List[float], save_path: str):
    if os.path.exists(join(save_path, f"{city}.graphml")):
        return
    north, south, east, west = bounds[3], bounds[1], bounds[2], bounds[0]
    g = ox.graph_from_bbox(north, south, east, west, network_type='drive')
    ox.save_graphml(g, join(save_path, f"{city}.graphml"))
            
            
# build map
def build_map(city: str, map_path: str):
    g = ox.load_graphml(join(map_path, f"{city}.graphml"))
    nodes_p, edges_p = ox.graph_to_gdfs(g, nodes=True, edges=True)
    edges_p.plot()
    plt.savefig(join(map_path, "map.pdf"))
    plt.clf()
    map_con = InMemMap(name=f"map_{city}", use_latlon=True, use_rtree=True, index_edges=True, dir=map_path)
 
    # construct road network
    nid_to_cmpct = dict()
    cmpct_to_nid = []
    for node_id, row in nodes_p.iterrows():
        if node_id not in nid_to_cmpct:
            nid_to_cmpct[node_id] = len(cmpct_to_nid)
            cmpct_to_nid.append(node_id)
        cid = nid_to_cmpct[node_id]
        map_con.add_node(cid, (row['y'], row['x']))
    for node_id_1, node_id_2, _ in g.edges:
        if node_id_1 not in nid_to_cmpct:
            nid_to_cmpct[node_id_1] = len(cmpct_to_nid)
            cmpct_to_nid.append(node_id_1)
        if node_id_2 not in nid_to_cmpct:
            nid_to_cmpct[node_id_2] = len(cmpct_to_nid)
            cmpct_to_nid.append(node_id_2)
        cid1 = nid_to_cmpct[node_id_1]
        cid2 = nid_to_cmpct[node_id_2]
        map_con.add_edge(cid1, cid2)
        map_con.add_edge(cid2, cid1)
    map_con.dump()
    return map_con


if __name__ == "__main__":
    city="chengdu"
    bounds = [104.0, 30.64, 104.15, 30.73]
    save_path = "./sets_data/real/map"
    fetch_map(city, bounds, save_path)
    map_con = build_map(city, save_path)
