"""研究员部分 (research module) — GeoLife 交通方式识别项目的数据与研究基础层。

本包是项目的第一步落地：数据加载、标签合并、60s 窗口切分、
统计特征与序列特征提取、数据退化(降采样/噪声)与 EDA。
仅依赖 numpy + pandas，便于在任何环境复现。

参考(已删减/改写，非照搬):
- zpfbuaa/Beijing-Trajectories-Project (Scripts/Cleaning, Scripts/trajAnalysis.py)
- taspinar/GPSMachineLearning (notebooks)
- AmirSarrafzadeh/Transport-Mode-Detection (notebooks)
"""

from . import (  # noqa: F401
    config, geo, labels, loader, segmentation, features, degradation, eda, pilot, model, model_seq,
)

__all__ = [
    "config", "geo", "labels", "loader", "segmentation",
    "features", "degradation", "eda", "pilot", "model", "model_seq",
]
