"""60s 固定非重叠窗口切分 (proposal 4.2)。

关键约定 (proposal "comparability rule")：
- 先在原始 (1–5s) 数据上建立参考窗口集；
- 退化(降采样/噪声)作用在参考窗口的副本上，之后重新判定可用性；
- 保留率 = 退化后仍可用的窗口数 / 参考窗口数。

实现：用"相对首点秒数 // window_sec"作为桶号做一次 groupby，避免逐窗口
布尔掩码的 O(points × windows) 开销，整体 O(points)。
"""
from __future__ import annotations

import pandas as pd

from .config import WINDOW_SEC, MIN_POINTS_PER_WINDOW, MAX_GAP_SEC


def _max_gap_sec(w: pd.DataFrame) -> float:
    """窗口内最大时间断档(秒)。窗口已按时间排序，无需再 sort。"""
    if len(w) <= 1:
        return 0.0
    dt = w["datetime"].to_numpy()
    gaps = (dt[1:] - dt[:-1]).astype("timedelta64[ms]").astype(float) / 1000.0
    return float(gaps.max())


def usability(w: pd.DataFrame, min_points: int = MIN_POINTS_PER_WINDOW,
              max_gap: float = MAX_GAP_SEC) -> tuple[bool, int, float]:
    """判定单个窗口是否可用：(点数 >= min_points) 且 (最大断档 <= max_gap)。"""
    n = len(w)
    if n < min_points:
        return False, n, 0.0
    gap = _max_gap_sec(w)
    return gap <= max_gap, n, gap


def segment_windows(points: pd.DataFrame,
                    window_sec: int = WINDOW_SEC,
                    min_points: int = MIN_POINTS_PER_WINDOW,
                    max_gap: float = MAX_GAP_SEC,
                    keep_unlabelled: bool = False) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """把 points 切成固定非重叠窗口。

    返回 (windows_df, window_points_list)：
      - windows_df 每行一个窗口，index 即 window_id；
      - window_points_list[i] 是窗口 i 的 points 子集 DataFrame。
    """
    metas = []
    wp_list: list[pd.DataFrame] = []
    wid = 0
    for (user_id, traj_id), g in points.groupby(["user_id", "traj_id"], sort=True):
        g = g.sort_values("datetime").reset_index(drop=True)
        # 先按"连续同标签段"切分：一条轨迹可能包含多种交通方式，必须分段再开窗
        lab = pd.Series(g["label"].to_numpy()).fillna("__unlabelled__")
        run = lab.ne(lab.shift()).cumsum()
        for run_id, s in g.groupby(run, sort=True):
            label = s["label"].iloc[0]
            if (not keep_unlabelled) and pd.isna(label):
                continue
            t0 = s["datetime"].iloc[0]
            t_off = (s["datetime"] - t0).dt.total_seconds()
            bucket = (t_off // window_sec).astype(int)
            for bi, w in s.groupby(bucket, sort=True):
                n = len(w)
                gap = _max_gap_sec(w)
                start = t0 + pd.Timedelta(seconds=float(bi) * window_sec)
                metas.append({
                    "window_id": wid,
                    "user_id": user_id,
                    "traj_id": traj_id,
                    "label": label,
                    "start_time": start,
                    "end_time": start + pd.Timedelta(seconds=window_sec),
                    "n_points": n,
                    "max_gap": gap,
                    "usable": (n >= min_points) and (gap <= max_gap),
                })
                wp_list.append(w.reset_index(drop=True))
                wid += 1
    windows_df = pd.DataFrame(metas).set_index("window_id") if metas else pd.DataFrame(
        columns=["user_id", "traj_id", "label", "start_time", "end_time",
                 "n_points", "max_gap", "usable"]
    )
    return windows_df, wp_list


def retention_after(wp_list: list[pd.DataFrame],
                    min_points: int = MIN_POINTS_PER_WINDOW,
                    max_gap: float = MAX_GAP_SEC) -> dict:
    """对(退化后)窗口列表重新判定可用性，返回保留率与统计。"""
    usable = sum(usability(w, min_points, max_gap)[0] for w in wp_list)
    total = len(wp_list)
    return {
        "total": total,
        "usable": usable,
        "retention": usable / total if total else float("nan"),
    }
