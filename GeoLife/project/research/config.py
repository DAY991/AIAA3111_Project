"""项目集中配置：所有超参数与路径统一来自 proposal (Group16)。

数值与 proposal 4.1–4.7 一一对应，改动只改这里。
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"            # 原始 GeoLife (User/*/Trajectory/*.plt + labels.txt)
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"  # 中间产物 (points.parquet / windows.parquet / features)
DATA_REFERENCE = PROJECT_ROOT / "data" / "reference"  # 60s 参考窗口集
VENDOR_ROOT = PROJECT_ROOT / "vendor"
EXAMPLE_DATA = (
    VENDOR_ROOT / "Beijing-Trajectories-Project"
    / "Beijing-Trajectories-Project-master" / "Example Data"
)

# ---------------------------------------------------------------------------
# 切分协议 (proposal 4.2)
# ---------------------------------------------------------------------------
# 窗口长度：由 60s 改为 120s。
# 原因：60s 窗口在 10s 降采样后仅 ~6 点 < 10 点阈值，保留率=0% (预实验 FAIL)。
# 窗口扫描 (docs/WINDOW_SWEEP.md) 证实：60s/90s 保留率 0%，120s 保留率 83.3%，
# 通过 proposal 4.7 的 go 判据 (≥70%)。此变更已在 PLAN.md §3 Gate A 记录。
WINDOW_SEC = 120             # 固定非重叠时间窗
MIN_POINTS_PER_WINDOW = 10   # 每窗最少保留点数
MAX_GAP_SEC = 30.0           # 窗内最大允许断档 (否则判为不可用)

# ---------------------------------------------------------------------------
# 标签合并规则 (proposal 4.1)
# ---------------------------------------------------------------------------
# 原始标签 -> 五类；taxi 并入 car
LABEL_MAP = {
    "walk": "walk",
    "bike": "bike",
    "bus": "bus",
    "car": "car",
    "taxi": "car",
    "subway": "subway",
}
# 因样本不足而剔除的标签
EXCLUDED_LABELS = {"train", "boat", "airplane", "run", "motorcycle", "-", "", "unlabelled"}
CLASSES = ["walk", "bike", "bus", "car", "subway"]

# ---------------------------------------------------------------------------
# 退化设置 (proposal 4.3)
# ---------------------------------------------------------------------------
DOWNSAMPLE_SEC = [10, 30, 60]      # 每 N 秒保留 1 点 (60s 为压力测试)
NOISE_SIGMA_M = [5, 10, 20, 50]    # 坐标高斯噪声 σ(米)

# 坐标近似 (proposal 4.3): 1° 纬度 ≈ 111320 m；经度按 cos(lat) 缩放
M_PER_DEG_LAT = 111320.0

# ---------------------------------------------------------------------------
# 交叉验证 / 调参协议 (proposal 4.5)
# ---------------------------------------------------------------------------
LOUO_INNER_SPLITS = 3   # 内层按 user 分组的折数，仅用于训练用户内部的调参

# ---------------------------------------------------------------------------
# 小规模预实验 go 判据 (proposal 4.7)
# ---------------------------------------------------------------------------
PILOT_N_USERS = 3                 # 预实验用户数
PILOT_GO_MIN_RETENTION = 0.70     # 保留率阈值
PILOT_GO_DOWNSAMPLE_SEC = 10      # 判定所用的降采样强度
PILOT_GO_NOISE_SIGMA = 20         # 判定所用的噪声强度
