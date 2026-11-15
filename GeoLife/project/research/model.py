"""Core A 基线模型 (Iteration 2)。

统计特征 + 经典分类器 (Random Forest / XGBoost)，
采用 Leave-One-User-Out (LOUO) 跨用户评估 (proposal 4.5)。

类别不均衡处理 (Gate C)：支持 class_weight / 样本权重 / SMOTE 过采样，
且均在**每个训练折内部**施加，绝不触碰测试折 (避免数据泄漏)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CLASSES

_LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def _encode(y: pd.Series) -> np.ndarray:
    return np.array([_LABEL_TO_IDX.get(str(v), -1) for v in y], dtype=int)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro",
                        labels=list(range(len(CLASSES))), zero_division=0)
    per_class = f1_score(y_true, y_pred, average=None,
                         labels=list(range(len(CLASSES))), zero_division=0)
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": dict(zip(CLASSES, [float(v) for v in per_class])),
    }


def _sample_weight(y: np.ndarray) -> np.ndarray:
    """逆频率样本权重 (平衡)。"""
    classes, counts = np.unique(y, return_counts=True)
    w = np.ones(len(y), dtype=float)
    for c, cnt in zip(classes, counts):
        w[y == c] = len(y) / (len(classes) * cnt)
    return w


def _smote(X: pd.DataFrame, y: np.ndarray, random_state: int):
    """对训练折做 SMOTE 过采样 (平衡到多数类)。

    k_neighbors 按最小类样本数动态设置，避免小类样本不足报错。
    """
    from imblearn.over_sampling import SMOTE
    _, counts = np.unique(y, return_counts=True)
    k = min(5, int(counts.min()) - 1)
    if k < 1:
        return X, y  # 类样本过少，无法 SMOTE，退回原样
    sm = SMOTE(random_state=random_state, k_neighbors=k)
    Xr, yr = sm.fit_resample(X, y)
    return Xr, yr


def _prepare(X, y, users):
    y = pd.Series(np.asarray(y), index=X.index)
    users = pd.Series(np.asarray(users), index=X.index)
    y_int = _encode(y)
    return X.reset_index(drop=True), y_int, users.reset_index(drop=True)


def lo_uo_rf(X: pd.DataFrame, y: pd.Series, users: pd.Series,
             n_estimators: int = 300, random_state: int = 0,
             balancing: str = "class_weight", verbose: bool = False) -> dict:
    """LOUO 评估 (Random Forest)。balancing ∈ {none, class_weight, smote}。"""
    from sklearn.ensemble import RandomForestClassifier

    balancing = (balancing or "none").lower()
    X, y_int, users = _prepare(X, y, users)
    preds = np.empty(len(X), dtype=int)
    for k, u in enumerate(sorted(users.unique())):
        te = (users == u).to_numpy()
        tr = ~te
        Xtr, ytr = X[tr], y_int[tr]
        if balancing == "smote":
            Xtr, ytr = _smote(Xtr, ytr, random_state)
        clf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state,
            class_weight="balanced" if balancing == "class_weight" else None,
            n_jobs=-1,
        )
        clf.fit(Xtr, ytr)
        preds[te] = clf.predict(X[te])
        if verbose:
            print(f"  RF[{balancing}] fold {k+1}/{users.nunique()} user={u}", flush=True)

    res = _metrics(y_int, preds)
    res["n"] = int(len(X))
    res["n_folds"] = int(users.nunique())
    return res


def lo_uo_xgboost(X: pd.DataFrame, y: pd.Series, users: pd.Series,
                  n_estimators: int = 300, random_state: int = 0,
                  balancing: str = "none", verbose: bool = False) -> dict:
    """LOUO 评估 (XGBoost)。balancing ∈ {none, sample_weight, smote}。"""
    from xgboost import XGBClassifier

    balancing = (balancing or "none").lower()
    X, y_int, users = _prepare(X, y, users)
    preds = np.empty(len(X), dtype=int)
    for k, u in enumerate(sorted(users.unique())):
        te = (users == u).to_numpy()
        tr = ~te
        Xtr, ytr = X[tr], y_int[tr]
        sw = None
        if balancing == "smote":
            Xtr, ytr = _smote(Xtr, ytr, random_state)
        elif balancing == "sample_weight":
            sw = _sample_weight(ytr)
        clf = XGBClassifier(
            n_estimators=n_estimators, random_state=random_state,
            n_jobs=-1, eval_metric="mlogloss", tree_method="hist",
        )
        clf.fit(Xtr, ytr, sample_weight=sw)
        preds[te] = clf.predict(X[te]).astype(int)
        if verbose:
            print(f"  XGB[{balancing}] fold {k+1}/{users.nunique()} user={u}", flush=True)

    res = _metrics(y_int, preds)
    res["n"] = int(len(X))
    res["n_folds"] = int(users.nunique())
    return res
