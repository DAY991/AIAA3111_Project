"""EDA 与结果汇总。只输出数据 (dict/DataFrame) 与 markdown 文本，不依赖 matplotlib。"""
from __future__ import annotations

import pandas as pd

from .config import DOWNSAMPLE_SEC, NOISE_SIGMA_M, WINDOW_SEC
from .labels import class_distribution
from .segmentation import retention_after, usability
from .degradation import downsample_window


def summarize_points(points: pd.DataFrame) -> dict:
    """points 级概览。"""
    n_users = points["user_id"].nunique()
    n_trajs = points.groupby(["user_id", "traj_id"]).ngroups
    gaps = []
    for _, g in points.groupby(["user_id", "traj_id"], sort=True):
        g = g.sort_values("datetime")
        d = g["datetime"].diff().dt.total_seconds().dropna()
        gaps.append(d)
    all_gaps = pd.concat(gaps)
    return {
        "n_points": len(points),
        "n_users": n_users,
        "n_trajectories": n_trajs,
        "class_distribution": class_distribution(points["label"]),
        "gap_sec_mean": float(all_gaps.mean()),
        "gap_sec_median": float(all_gaps.median()),
        "gap_sec_p90": float(all_gaps.quantile(0.9)),
    }


def degrade_retention_report(windows: pd.DataFrame,
                             wp_list: list[pd.DataFrame]) -> pd.DataFrame:
    """各退化条件下的保留率表 (proposal 4.2 comparability rule)。

    参考集 = 干净数据中"可用"的 60s 窗口；退化作用在其副本上再判可用性。
    注意：噪声只改坐标不改时间戳，因此噪声条件不改变可用性 (保留率恒为 1)，
    这里只为降采样条件做实际计算，其余条件直接标注。
    """
    ref_idx = [i for i, row in windows.iterrows() if row["usable"]]
    ref = [wp_list[i] for i in ref_idx]
    total = len(ref)

    rows = []
    for name in ["clean"] + [f"noise_{s}m" for s in NOISE_SIGMA_M]:
        rows.append({"condition": name, "total": total, "usable": total, "retention": 1.0})
    for s in DOWNSAMPLE_SEC:
        usable = 0
        for w in ref:
            d = downsample_window(w, s)
            if usability(d)[0]:
                usable += 1
        rows.append({
            "condition": f"downsample_{s}s",
            "total": total,
            "usable": usable,
            "retention": round(usable / total, 4) if total else float("nan"),
        })
    return pd.DataFrame(rows)


def render_markdown(points: pd.DataFrame, windows: pd.DataFrame,
                    retention: pd.DataFrame) -> str:
    """把 EDA 结果渲染为 markdown 文本。"""
    s = summarize_points(points)
    cd = s["class_distribution"]
    usable = int(windows["usable"].sum()) if len(windows) else 0
    total = len(windows)

    lines = [
        "# EDA 结果",
        "",
        "## 数据概览",
        f"- 点数: {s['n_points']:,}",
        f"- 用户数: {s['n_users']}",
        f"- 轨迹(文件)数: {s['n_trajectories']:,}",
        f"- 采样间隔(秒): mean={s['gap_sec_mean']:.2f}, median={s['gap_sec_median']:.2f}, p90={s['gap_sec_p90']:.2f}",
        "",
        "## 类别分布 (标签合并后, proposal 4.1)",
        "",
        "| 类别 | 点数 |",
        "|---|---|",
    ]
    for cls, cnt in cd.items():
        lines.append(f"| {cls} | {cnt:,} |")
    lines += [
        "",
        f"## {WINDOW_SEC}s 窗口切分 (proposal 4.2)",
        f"- 窗口总数: {total:,}",
        f"- 可用窗口: {usable:,}",
        f"- 可用率: {usable/total:.4f}" if total else "- 可用率: N/A",
        "",
        "## 退化后保留率 (comparability rule, 参考集=可用窗口)",
        "",
        "| condition | total | usable | retention |",
        "|---|---|---|---|",
    ]
    for _, row in retention.iterrows():
        lines.append(
            f"| {row['condition']} | {int(row['total'])} | {int(row['usable'])} | {row['retention']:.4f} |"
        )
    return "\n".join(lines)
