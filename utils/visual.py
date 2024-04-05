import matplotlib.pyplot as plt
plt.switch_backend("agg")
import torch
from os.path import join
from loader.dataset import TrajFastDataset
from models_seq.seq_models import Restorer
from utils.coors import wgs84_to_gcj02
import folium


def draw_gps(locations_series, html_path, colors=None, no_points=False):
    if type(locations_series[0]) is tuple:
        locations_series = [locations_series]
    
    # calculate center
    cen_lng, cen_lat, cnt = 0, 0, 0
    for series in locations_series:
        cnt += len(series)
        for y, x in series:
            cen_lat += y
            cen_lng += x
            
    m = folium.Map([cen_lat / cnt, cen_lng / cnt], zoom_start=13, attr='default',
                   tiles='https://tile.openstreetmap.org/{z}/{x}/{y}.png')
    for k, locations in enumerate(locations_series):
        color = "red" if colors is None else colors[k]
        folium.PolyLine(locations, weight=5, color=color, opacity=0.7).add_to(m)  
        if not no_points:
            folium.CircleMarker(locations[0], radius=5, fill=True, opacity=1., color="blue", fill_color="blue", fill_opacity=1., popup='<b>Starting Point</b>').add_to(m)
            folium.CircleMarker(locations[-1], radius=5, fill=True, opacity=1., color="green", fill_color="green", fill_opacity=1., popup='<b>End Point</b>').add_to(m)
    m.save(html_path)


def draw_paths(paths, G, html_path: str, colors=None, no_points=False):
    multiple_locs = []
    for path in paths:
        locs = [[G.nodes[v]["lat"], G.nodes[v]["lng"]] for v in path]
        multiple_locs.append(locs)
    draw_gps(multiple_locs, html_path=html_path, colors=colors, no_points=no_points)
    