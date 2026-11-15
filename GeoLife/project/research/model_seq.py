"""Core B 序列模型 (Iteration 3)：1D-CNN / GRU + 掩码处理不规则采样。

输入：`features.build_core_b_sequences` 产生的 (X, mask)，
  X 形状 (N, seq_len, 5)，5 通道 = (Δdist, speed, acc, heading, time_gap)。
评估：Leave-One-User-Out (LOUO)，与 Core A 同一协议。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import CLASSES
from .model import _encode, _metrics


def _class_weight(y: np.ndarray, n_class: int) -> np.ndarray:
    """逆频率类别权重 (长度 = n_class)，用于 CrossEntropyLoss。"""
    classes, counts = np.unique(y, return_counts=True)
    w = np.ones(n_class, dtype=float)
    for c, cnt in zip(classes, counts):
        w[c] = len(y) / (n_class * cnt)
    return w


class MaskedGlobalPooling(nn.Module):
    """对序列做掩码 mean+max 池化，忽略 padding。"""

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: (B, T, C), mask: (B, T)
        m = mask.unsqueeze(-1).float()               # (B, T, 1)
        cnt = m.sum(dim=1).clamp(min=1.0)            # (B, 1)
        avg = (x * m).sum(dim=1) / cnt               # (B, C)
        neg_inf = torch.finfo(x.dtype).min
        xmax = x.masked_fill(m == 0, neg_inf).max(dim=1).values  # (B, C)
        return torch.cat([avg, xmax], dim=-1)        # (B, 2C)


class CNNet(nn.Module):
    def __init__(self, n_feat: int = 5, n_class: int = 5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, 64, 7, padding=3), nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
        )
        self.pool = MaskedGlobalPooling()
        self.fc = nn.Linear(128 * 2, n_class)

    def forward(self, x, mask):
        h = self.conv(x.transpose(1, 2))   # (B, F, T) -> (B, 128, T)
        h = h.transpose(1, 2)              # (B, T, 128)
        p = self.pool(h, mask)
        return self.fc(p)


class GRUNet(nn.Module):
    def __init__(self, n_feat: int = 5, hidden: int = 64, n_class: int = 5):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True, bidirectional=True)
        self.pool = MaskedGlobalPooling()
        self.fc = nn.Linear(hidden * 2 * 2, n_class)

    def forward(self, x, mask):
        h, _ = self.gru(x)                 # (B, T, 2*hidden)
        p = self.pool(h, mask)
        return self.fc(p)


def _train_one_fold(model, Xtr, mtr, ytr, wtr, epochs, batch_size, lr, seed, device):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(wtr, dtype=torch.float32, device=device))
    n = len(Xtr)
    for _ in range(epochs):
        perm = torch.randperm(n)
        model.train()
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = model(Xtr[idx], mtr[idx])
            loss = ce(logits, ytr[idx])
            loss.backward()
            opt.step()
    return model


def _predict(model, X, mask, batch_size):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out.append(model(X[i:i + batch_size], mask[i:i + batch_size]).argmax(dim=1))
    return torch.cat(out).cpu().numpy()


def lo_uo_seq(X: np.ndarray, mask: np.ndarray, y, users,
              model_cls, epochs: int = 15, batch_size: int = 512,
              lr: float = 1e-3, seed: int = 0, verbose: bool = False) -> dict:
    """LOUO 评估序列模型 (1D-CNN / GRU)。"""
    y_int = _encode(y)
    users = np.asarray(users)
    n_class = len(CLASSES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.from_numpy(X.astype(np.float32)).to(device)
    mt = torch.from_numpy(mask.astype(np.float32)).to(device)
    yt = torch.from_numpy(y_int.astype(np.int64)).to(device)

    preds = np.empty(len(X), dtype=int)
    for k, u in enumerate(sorted(np.unique(users))):
        te = users == u
        tr = ~te
        model = model_cls(n_class=n_class).to(device)
        w = _class_weight(y_int[tr], n_class)
        model = _train_one_fold(
            model, Xt[tr], mt[tr], yt[tr], w, epochs, batch_size, lr, seed, device)
        preds[te] = _predict(model, Xt[te], mt[te], batch_size)
        if verbose:
            print(f"  {model_cls.__name__} fold {k+1}/{len(np.unique(users))} user={u}", flush=True)

    res = _metrics(y_int, preds)
    res["n"] = int(len(X))
    res["n_folds"] = int(len(np.unique(users)))
    return res
