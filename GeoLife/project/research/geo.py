"""坐标与距离工具。

改写自 Beijing-Trajectories-Project/Scripts/Cleaning/convertPos.py：
- 原文用 haversine + 椭球地球半径，把经纬度投影到以北京中心为原点的笛卡尔米坐标。
- 这里保留 haversine 点间距，并额外提供 equirectangular 局部投影
  (与 proposal 4.3 的 "1° lat ≈ 111320 m, lon 按 cos(lat) 缩放" 一致)，
  二者都只依赖 numpy。
"""
import numpy as np

from .config import M_PER_DEG_LAT

EARTH_R = 6371000.0  # 球面近似半径(米)


def haversine_meters(lat1, lon1, lat2, lon2):
    """两点球面距离(米)。输入为度。支持标量或等长数组。"""
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_R * np.arcsin(np.sqrt(a))


def to_local_xy(lat, lon, origin):
    """equirectangular 局部投影到米。origin=(lat, lon)。"""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    x = (lon - origin[1]) * M_PER_DEG_LAT * np.cos(np.radians(origin[0]))
    y = (lat - origin[0]) * M_PER_DEG_LAT
    return x, y


def xy_origin(lat, lon):
    """取数据集中位点作为投影原点。"""
    return (float(np.nanmedian(lat)), float(np.nanmedian(lon)))
