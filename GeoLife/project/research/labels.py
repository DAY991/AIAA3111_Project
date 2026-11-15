"""标签合并规则 (proposal 4.1)。

GeoLife 原始标签 -> 五类：
  walk -> walk, bike -> bike, bus -> bus, car/taxi -> car, subway -> subway
剔除 train / boat / airplane / run / motorcycle / 无标签。
"""
import pandas as pd

from .config import LABEL_MAP, EXCLUDED_LABELS, CLASSES


def normalize_label(raw) -> "str | None":
    """把任意原始标签归一化为五类之一；剔除或无法识别的返回 None。"""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    key = str(raw).strip().lower()
    if key in EXCLUDED_LABELS:
        return None
    return LABEL_MAP.get(key)


def normalize_series(series: pd.Series) -> pd.Series:
    """对一列标签做归一化；无法识别/被剔除的置为 NaN。"""
    return series.map(normalize_label)


def class_distribution(labels: pd.Series) -> pd.Series:
    """五类样本计数(含缺失统计)，按固定 CLASSES 顺序输出。"""
    norm = normalize_series(labels)
    out = norm.value_counts(dropna=False).reindex(CLASSES + [None]).fillna(0).astype(int)
    out = out.rename(index={None: "(excluded/unlabelled)"})
    return out
