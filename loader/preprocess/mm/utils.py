from leuvenmapmatching.map.inmem import InMemMap
import pickle
from os.path import join
import math
from tqdm import tqdm
import pandas as pd



pi = 3.1415926535897932384626  
ee = 0.00669342162296594323  
a = 6378245.0 

def _transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * pi) + 40.0 *
            math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 *
            math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret


def _transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * pi) + 40.0 *
            math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 *
            math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret


def gcj02_to_wgs84(lng, lat):
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [lng * 2 - mglng, lat * 2 - mglat]


def wgs84_to_gcj02(lng, lat):
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [mglng, mglat]


def compute_dist_basic(lng1, lat1, lng2, lat2):
    return ((lng1 - lng2) ** 2 + (lat1 - lat2) ** 2) ** 0.5


def filter_stayed(points):
    points_filtered = [points[0]]
    eps = 0.000000000000001
    for cur in points[1:]:
        prev = points_filtered[-1]
        if abs(prev[1] - cur[1]) >= eps or abs(prev[2] - cur[2]) >= eps:
            points_filtered.append(cur)
    return points_filtered


def filter_prefix(points, head_lat, head_lng):
    i = 0
    while i < len(points) - 1 and compute_dist_basic(head_lng, head_lat, points[i][2], points[i][1]) > compute_dist_basic(head_lng, head_lat, points[i + 1][2], points[i + 1][1]):
        i += 1
    return i


def time_convert(timestamp):
    tm = pd.Timestamp(timestamp, unit='s', tz='Asia/Shanghai')
    return int(tm.hour * 3600 + tm.minute * 60 + tm.second)


def get_locations_from_points(points, graph):
    locations = [graph[p][0] for p in points]
    return [(wgs84_to_gcj02(p[1], p[0])[1], wgs84_to_gcj02(p[1], p[0])[0]) for p in locations]
    
def get_locations_from_edges(edges, graph):
    points = [a for a, b in edges] + [edges[-1][1]]
    return get_locations_from_points(points, graph)


def calculate_edge_dis(edge_a, edge_b, graph):
    lat1, lng1 = graph[edge_a[0]][0]
    lat2, lng2 = graph[edge_a[1]][0]
    lat_a, lng_a = (lat1 + lat2) / 2, (lng1 + lng2) / 2
    
    lat1, lng1 = graph[edge_b[0]][0]
    lat2, lng2 = graph[edge_b[1]][0]
    lat_b, lng_b = (lat1 + lat2) / 2, (lng1 + lng2) / 2
    
    # return compute_dist_basic(lng_a, lat_a, lng_b, lat_b)
    return math.fabs(lat_a - lat_b) + math.fabs(lng_a - lng_b)
