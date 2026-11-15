"""GeoLife 数据加载。

两条路径：
1. load_preprocessed_dir()  读取已预处理的 Example Data
   (Beijing-Trajectories-Project 附带: User/Trajectory/Labelled/*.csv，
    已含 datetime / x / y / Transportation Mode 列)。
2. load_raw_geolife()       读取官方原始发布
   (User/*/Trajectory/*.plt + User/*/labels.txt)。

改写自 Beijing-Trajectories-Project/Scripts/Cleaning/{stripHeader,labelTrajs,convertPos}.py，
只保留要点并向量化，去掉原脚本里对文件系统的原地重写。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .geo import to_local_xy, xy_origin
from .labels import normalize_series

# 原始 .plt 文件前 6 行为头；数据列顺序固定
_PLT_SKIP = 6
_PLT_COLS = ["lat", "lon", "alt_flag", "alt_ft", "day_no", "date", "time"]

_POINT_COLS = ["user_id", "traj_id", "datetime", "lat", "lon", "x", "y", "label"]


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in list(df.columns):
        if str(c).startswith("Unnamed"):
            df = df.drop(columns=c)
    return df


def load_preprocessed_dir(root: str | Path, min_points: int = 1) -> pd.DataFrame:
    """读取已预处理目录 (Example Data)。返回统一 points DataFrame。"""
    root = Path(root)
    frames = []
    for user_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        user_id = user_dir.name
        traj_root = user_dir / "Trajectory"
        if not traj_root.is_dir():
            continue
        # 已标注目录 (Windows 文件系统大小写不敏感，需去重)
        seen_dirs = set()
        for sub in sorted(p for p in traj_root.iterdir() if p.is_dir()):
            if sub.name.lower() != "labelled" or sub.name in seen_dirs:
                continue
            seen_dirs.add(sub.name)
            lab_dir = sub
            for f in sorted(lab_dir.glob("*.csv")):
                df = _clean_frame(pd.read_csv(f))
                df = df.rename(columns={"Lat": "lat", "Long": "lon", "Latitude": "lat", "Longitude": "lon"})
                if "datetime" not in df.columns:
                    # 兼容未合并 datetime 的中间态
                    df["datetime"] = pd.to_datetime(
                        df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce"
                    )
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
                df = df.dropna(subset=["datetime"])
                if len(df) < min_points:
                    continue
                df["user_id"] = user_id
                df["traj_id"] = f.stem
                df["label"] = normalize_series(
                    df["Transportation Mode"] if "Transportation Mode" in df.columns
                    else pd.Series([None] * len(df))
                )
                if "x" not in df.columns or "y" not in df.columns:
                    origin = xy_origin(df["lat"], df["lon"])
                    x, y = to_local_xy(df["lat"], df["lon"], origin)
                    df["x"], df["y"] = x, y
                frames.append(df[_POINT_COLS])
    if not frames:
        raise FileNotFoundError(f"no labelled trajectories found under {root}")
    return pd.concat(frames, ignore_index=True)


def load_raw_geolife(root: str | Path) -> pd.DataFrame:
    """读取官方原始 GeoLife (User/*/Trajectory/*.plt + User/*/labels.txt)。

    标签按时间区间与轨迹点匹配 (对应原 labelTrajs.py 的核心逻辑，向量化重写)。
    """
    root = Path(root)
    frames = []
    for user_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        user_id = user_dir.name
        traj_dir = user_dir / "Trajectory"
        labels_path = user_dir / "labels.txt"
        # 分类任务只需要有交通方式标签的用户 (69/182)，跳过无标签用户
        if not traj_dir.is_dir() or not labels_path.exists():
            continue

        # 读取标签区间 (GeoLife labels.txt 为 TAB 分隔)，按开始时间排序
        starts = ends = lab_vals = None
        if labels_path.exists():
            label_df = pd.read_csv(labels_path, sep="\t")
            label_df.columns = [c.strip() for c in label_df.columns]
            label_df["Start Time"] = pd.to_datetime(label_df["Start Time"], errors="coerce")
            label_df["End Time"] = pd.to_datetime(label_df["End Time"], errors="coerce")
            label_df["label"] = normalize_series(label_df["Transportation Mode"])
            label_df = label_df.dropna(subset=["Start Time", "End Time"]).sort_values("Start Time")
            starts = label_df["Start Time"].to_numpy()
            ends = label_df["End Time"].to_numpy()
            lab_vals = label_df["label"].to_numpy()

        origin = None
        for f in sorted(traj_dir.glob("*.plt")):
            raw = pd.read_csv(f, skiprows=_PLT_SKIP, header=None, names=_PLT_COLS)
            raw["datetime"] = pd.to_datetime(
                raw["date"].astype(str) + " " + raw["time"].astype(str), errors="coerce"
            )
            raw = raw.dropna(subset=["datetime", "lat", "lon"]).sort_values("datetime")
            raw["traj_id"] = f.stem

            if starts is not None and len(starts):
                # 向量化标签匹配: searchsorted 找最后一个 start <= t 的区间
                ts = raw["datetime"].to_numpy()
                idx = np.searchsorted(starts, ts, side="right") - 1
                idx_clip = np.clip(idx, 0, len(starts) - 1)
                valid = (idx >= 0) & (ts <= ends[idx_clip])
                raw["label"] = np.where(valid, lab_vals[idx_clip], None)
                raw = raw[valid].copy()
            else:
                raw["label"] = None

            if len(raw) == 0:
                continue
            if origin is None:
                origin = xy_origin(raw["lat"], raw["lon"])
            x, y = to_local_xy(raw["lat"], raw["lon"], origin)
            raw["x"], raw["y"] = x, y
            raw["user_id"] = user_id
            frames.append(raw[_POINT_COLS])

    if not frames:
        raise FileNotFoundError(f"no .plt trajectories found under {root}")
    return pd.concat(frames, ignore_index=True)
