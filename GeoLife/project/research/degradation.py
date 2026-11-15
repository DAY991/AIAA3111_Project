"""数据退化：时间降采样 + 坐标高斯噪声 (proposal 4.3)。

- 降采样：每 N 秒保留 1 点 (N ∈ {10, 30, 60})。
- 噪声：对经纬度(换算成米)独立加高斯噪声，σ ∈ {5, 10, 20, 50} m。

约定：退化只作用在参考窗口的副本上，训练始终用干净参考窗口
(主实验 "test-only degradation"；"train+test" 为补充实验，见 proposal 4.4)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DOWNSAMPLE_SEC, NOISE_SIGMA_M


def downsample_window(w: pd.DataFrame, step_sec: int) -> pd.DataFrame:
    """每 step_sec 秒保留 1 点 (始终保留首点)。

    实现为"每个 step 秒桶内保留首点"，等价于按固定采样间隔抽稀，向量化无循环。
    """
    if step_sec <= 0 or len(w) <= 1:
        return w.copy()
    t_off = (w["datetime"] - w["datetime"].iloc[0]).dt.total_seconds()
    bucket = (t_off // step_sec).astype(int)
    keep = ~bucket.duplicated(keep="first")
    return w[keep].reset_index(drop=True)


def add_noise_window(w: pd.DataFrame, sigma_m: float, seed: int = 0) -> pd.DataFrame:
    """对局部坐标 x,y 加独立高斯噪声 (σ 单位: 米)。返回副本。"""
    w = w.copy()
    rng = np.random.default_rng(seed)
    if "x" in w.columns and "y" in w.columns:
        w["x"] = w["x"] + rng.normal(0.0, sigma_m, size=len(w))
        w["y"] = w["y"] + rng.normal(0.0, sigma_m, size=len(w))
    else:
        # 若无局部坐标则退化到经纬度：lat 直接换算，lon 按 cos(lat) 缩放
        from .config import M_PER_DEG_LAT
        lat_scale = M_PER_DEG_LAT
        w["lat"] = w["lat"] + rng.normal(0.0, sigma_m, size=len(w)) / lat_scale
        lon_scale = M_PER_DEG_LAT * np.cos(np.radians(w["lat"]))
        w["lon"] = w["lon"] + rng.normal(0.0, sigma_m, size=len(w)) / lon_scale
    return w


def degrade_window_list(wp_list: list[pd.DataFrame],
                        downsample_sec: int | None = None,
                        noise_sigma_m: float | None = None,
                        seed: int = 0) -> list[pd.DataFrame]:
    """对参考窗口副本列表统一施加退化 (先降采样后加噪)。"""
    out = []
    for i, w in enumerate(wp_list):
        d = w.copy()
        if downsample_sec is not None:
            d = downsample_window(d, downsample_sec)
        if noise_sigma_m is not None:
            d = add_noise_window(d, noise_sigma_m, seed=seed + i)
        out.append(d)
    return out


def condition_table() -> list[dict]:
    """生成所有退化条件 (proposal 4.3)。"""
    rows = [{"name": "clean", "downsample_sec": None, "noise_sigma_m": None}]
    for s in DOWNSAMPLE_SEC:
        rows.append({"name": f"downsample_{s}s", "downsample_sec": s, "noise_sigma_m": None})
    for sig in NOISE_SIGMA_M:
        rows.append({"name": f"noise_{sig}m", "downsample_sec": None, "noise_sigma_m": sig})
    return rows
