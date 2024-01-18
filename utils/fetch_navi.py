import requests
import json

url = "https://restapi.amap.com/v5/direction/driving?parameters"


def fetch_routes(ori_lng, ori_lat, dst_lng, dst_lat):
    params = {
        "origin": f"{ori_lng:.6f},{ori_lat:.6f}",
        "destination": f"{dst_lng:.6f},{dst_lat:.6f}",
        "key": "YOUR OWN KEY",  # apply key for your own use
        "show_fields": "polyline",
    }
    res = json.loads(requests.get(url, params).content)
    count = int(res["count"])
    paths = []
    for k in range(count):
        path = res["route"]["paths"][k]["steps"]
        coors = []
        for seg in path:
            points_str = seg["polyline"].split(";")
            coors.extend([(float(each.split(",")[0]), float(each.split(",")[1])) for each in points_str])
        paths.append(coors)
    return paths

    
