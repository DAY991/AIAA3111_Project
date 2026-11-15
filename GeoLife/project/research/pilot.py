"""小规模预实验 (proposal 4.7)。

go 判据：在 10s 降采样 且 σ=20m 噪声下，≥ 70% 参考窗口仍可用。
这里实现判据中的"保留率"部分；"基线 Macro-F1 出现可测量但非退化性下降"
需要训练基线分类器，属于 Week 2 (见 docs/PLAN.md 的 Iteration 2)。
"""
from __future__ import annotations

import pandas as pd

from .config import (
    PILOT_N_USERS, PILOT_GO_MIN_RETENTION,
    PILOT_GO_DOWNSAMPLE_SEC, PILOT_GO_NOISE_SIGMA,
)
from .degradation import downsample_window
from .segmentation import segment_windows, usability


def run_pilot(points: pd.DataFrame, n_users: int = PILOT_N_USERS) -> dict:
    """对前 n_users 个用户跑预实验，返回判定结果。

    参考集 = 干净数据中"可用"的 60s 窗口；噪声不改时间戳故不影响保留率，
    go 判据实质由降采样决定。
    """
    users = sorted(points["user_id"].unique())[:n_users]
    sub = points[points["user_id"].isin(users)]
    windows, wp_list = segment_windows(sub)

    ref_idx = [i for i, row in windows.iterrows() if row["usable"]]
    ref = [wp_list[i] for i in ref_idx]
    total = len(ref)

    usable_after = 0
    for w in ref:
        d = downsample_window(w, PILOT_GO_DOWNSAMPLE_SEC)
        if usability(d)[0]:
            usable_after += 1

    retention = usable_after / total if total else float("nan")
    go = (total > 0) and (retention >= PILOT_GO_MIN_RETENTION)
    return {
        "n_users": len(users),
        "users": users,
        "reference_windows": total,
        "clean_usable": total,
        "degraded_usable": usable_after,
        "retention": round(retention, 4),
        "threshold": PILOT_GO_MIN_RETENTION,
        "go": bool(go),
        "note": (
            "通过：窗口在降采样后仍足够可用，可进入主实验。"
            if go else
            "未通过：60s 窗口在 10s 降采样后点数 < 10，建议按 proposal 4.7 增大窗口长度"
            "(如 90–120s)后重跑并记录变更。"
        ),
    }


def render_markdown(result: dict) -> str:
    """预实验结果 -> markdown。"""
    lines = [
        "# 预实验结果 (pilot, proposal 4.7)",
        "",
        f"- 用户: {', '.join(result['users'])}",
        f"- 参考窗口数: {result['reference_windows']}",
        f"- 干净数据可用窗口: {result['clean_usable']}",
        f"- 退化后可用窗口 ({PILOT_GO_DOWNSAMPLE_SEC}s 降采样 + σ={PILOT_GO_NOISE_SIGMA}m 噪声): {result['degraded_usable']}",
        f"- 保留率: **{result['retention']:.4f}** (阈值 {result['threshold']})",
        f"- 判定: **{'PASS' if result['go'] else 'FAIL'}**",
        "",
        result["note"],
    ]
    return "\n".join(lines)
