"""特征提取。

Core A (统计特征, proposal 4.4)：每个 60s 窗口 -> 一个特征向量
  distance / speed / acceleration / heading change / stop ratio 的
  mean / std / max / median / 分位数。

Core B (序列特征, proposal 4.4)：每个窗口 -> 等长序列 (Δdist, speed, acc,
  Δheading, time_gap)，带 mask 用于不规则采样 (供 1D-CNN / GRU)。

速度/转角计算思路改写自 Beijing-Trajectories-Project/Scripts/trajAnalysis.py
(angles() / averageSpeed())，仅保留核心公式并向量化，去掉其 shapely/nolds 依赖。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STOP_SPEED_MS = 1.0  # 判"停留"的速度阈值 (m/s)


def _kinematics(w: pd.DataFrame) -> pd.DataFrame:
    """由窗口点列计算逐点运动学量 (全向量化)。"""
    w = w.sort_values("datetime").reset_index(drop=True)
    n = len(w)
    dt = w["datetime"].diff().dt.total_seconds().to_numpy(dtype=float)  # 秒, 首元素 NaN
    dx = w["x"].diff().to_numpy(dtype=float)
    dy = w["y"].diff().to_numpy(dtype=float)
    dist = np.sqrt(dx ** 2 + dy ** 2)

    # 速度 (m/s): dt<=0 或 NaN 记为 0，避免除零
    safe_dt = np.where(np.isfinite(dt) & (dt > 0), dt, 1.0)
    speed = np.where(np.isfinite(dt) & (dt > 0), dist / safe_dt, 0.0)
    speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)

    # 加速度: Δspeed / Δt
    dsp = np.diff(speed)
    dt_next = dt[1:]
    safe_dt_next = np.where(np.isfinite(dt_next) & (dt_next > 0), dt_next, 1.0)
    acc_step = np.where(np.isfinite(dt_next) & (dt_next > 0), dsp / safe_dt_next, 0.0)
    acc = np.concatenate([[0.0], acc_step])

    # 转角: 相邻位移向量夹角 (度)，全向量化
    heading = np.zeros(n, dtype=float)
    if n >= 3:
        d_x = dx[1:]  # 位移向量 (长度 n-1)
        d_y = dy[1:]
        n1 = np.sqrt(d_x[:-1] ** 2 + d_y[:-1] ** 2)
        n2 = np.sqrt(d_x[1:] ** 2 + d_y[1:] ** 2)
        dot = d_x[:-1] * d_x[1:] + d_y[:-1] * d_y[1:]
        denom = n1 * n2
        cosang = np.where(denom > 0, dot / np.where(denom > 0, denom, 1.0), 0.0)
        cosang = np.clip(cosang, -1.0, 1.0)
        heading[1:n - 1] = np.degrees(np.arccos(cosang))

    return pd.DataFrame({
        "dt": dt, "dist": dist, "speed": speed, "acc": acc, "heading": heading,
    })


def window_statistical_features(w: pd.DataFrame) -> dict:
    """Core A：单窗口统计特征。空窗口返回 NaN 向量。"""
    if len(w) == 0:
        return {k: float("nan") for k in _core_a_names()}
    k = _kinematics(w)
    speed = k["speed"].values
    acc = k["acc"].values
    heading = k["heading"].values
    feats = {
        "duration": float((w["datetime"].max() - w["datetime"].min()).total_seconds()),
        "n_points": len(w),
        "path_length": float(np.nansum(k["dist"])),
        "mean_speed": float(np.mean(speed)),
        "std_speed": float(np.std(speed)),
        "max_speed": float(np.max(speed)),
        "median_speed": float(np.median(speed)),
        "q25_speed": float(np.percentile(speed, 25)),
        "q75_speed": float(np.percentile(speed, 75)),
        "mean_acc": float(np.mean(acc)),
        "std_acc": float(np.std(acc)),
        "max_acc": float(np.max(np.abs(acc))),
        "mean_heading": float(np.mean(heading)),
        "sum_heading": float(np.sum(heading)),
        "std_heading": float(np.std(heading)),
        "stop_ratio": float(np.mean(speed < STOP_SPEED_MS)),
        "mean_gap": float(np.nanmean(k["dt"])) if np.any(np.isfinite(k["dt"])) else float("nan"),
    }
    return feats


def _core_a_names() -> list[str]:
    return [
        "duration", "n_points", "path_length",
        "mean_speed", "std_speed", "max_speed", "median_speed", "q25_speed", "q75_speed",
        "mean_acc", "std_acc", "max_acc",
        "mean_heading", "sum_heading", "std_heading",
        "stop_ratio", "mean_gap",
    ]


CORE_A_FEATURES = _core_a_names()


def build_core_a_matrix(wp_list: list[pd.DataFrame]) -> pd.DataFrame:
    """Core A 特征矩阵：行=窗口，列=特征。"""
    rows = [window_statistical_features(w) for w in wp_list]
    return pd.DataFrame(rows)


def window_sequence_features(w: pd.DataFrame) -> pd.DataFrame:
    """Core B：逐点序列特征 (Δdist, speed, acc, heading, dt)。"""
    k = _kinematics(w)
    return pd.DataFrame({
        "delta_dist": k["dist"].fillna(0.0),
        "speed": k["speed"],
        "acc": k["acc"],
        "heading": k["heading"],
        "time_gap": k["dt"].fillna(0.0),
    })


def build_core_b_sequences(wp_list: list[pd.DataFrame], seq_len: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Core B 序列：pad/截断到 seq_len。返回 (X, mask)。

    X 形状 (N, seq_len, 5)；mask 形状 (N, seq_len)，1 表示真实点。
    """
    n_feat = 5
    N = len(wp_list)
    X = np.zeros((N, seq_len, n_feat), dtype=np.float32)
    mask = np.zeros((N, seq_len), dtype=np.float32)
    for i, w in enumerate(wp_list):
        seq = window_sequence_features(w).to_numpy(dtype=np.float32)
        L = min(len(seq), seq_len)
        X[i, :L] = seq[:L]
        mask[i, :L] = 1.0
    return X, mask
