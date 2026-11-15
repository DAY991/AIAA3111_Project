# 项目迭代规划 (Group 16)

> 依据：`proposal/Proposal.docx`（已按助教意见修订版）与课程评分标准拆分。
> 项目：**Robust Transportation Mode Classification from GPS Trajectories**（GeoLife，五分类 + 鲁棒性对比研究）。

---

## 1. 项目总览（从 proposal 拆出的"部件"）

| 部件 | 内容 | 对应 proposal |
|---|---|---|
| 任务 | 交通方式 5 分类：walk / bike / bus / car / subway | 4.1 |
| 数据 | Microsoft GeoLife（182 用户；本仓库已含 3 用户样例数据用于预实验） | 1.x |
| 切分 | 60s 非重叠窗口，≥10 点，最大断档 ≤30s | 4.2 |
| 退化 | 降采样 10/30/60s；坐标噪声 σ=5/10/20/50m | 4.3 |
| 模型 | **Core A** 统计特征+RF/XGBoost；**Core B** 序列特征+1D-CNN/GRU(mask) | 4.4 |
| 评估 | LOUO 用户不重叠；内层用户分组调参；Accuracy/Macro-F1/逐类/保留率 | 4.5 / 4.6 |
| 预实验 | 2–3 用户验证窗口在退化后仍可用 | 4.7 |
| 交付 | Report.pdf + 代码 + README + 数据；Poster.pdf | 2 / 3 / 5 |

---

## 2. 迭代划分（每轮一个可交付的 sprint）

### ✅ Iteration 0 — 数据与研究基础层「研究员部分」（已完成，本仓库）
**目标**：把 proposal 的协议落地成可运行的代码骨架，并用样例数据跑通。
**交付**：
- `research/` 包：`loader` / `labels` / `segmentation` / `features` / `degradation` / `eda` / `pilot`
- `data/processed/*.csv`：样例数据的点、窗口、Core A 特征矩阵
- `docs/EDA.md`、`docs/PILOT.md`：样例数据 EDA 与预实验结论
**出口判据**：样例数据端到端跑通 → 已达成。

### ✅ Iteration 1 — 完整数据落地 + EDA（Week 1，已完成）
**结果**（`docs/EDA_full.md`）：
- 官方 GeoLife 已下载并解压到 `data/raw/`（182 用户；69 有标签）。
- 全量加载：**5,434,096 点 / 4,476 条标注轨迹 / 64 用户**（69 有标签用户中 5 个标签与轨迹时间不匹配，属 GeoLife 已知数据质量问题，自动跳过）。
- 120s 窗口：**117,014 窗 / 92,378 可用（79%）**；10s 降采样保留率 **80.6%**（通过 go 判据 ✅）。
- 类别：walk 1.59M / bus 1.28M / bike 0.95M / car 0.76M / subway 0.29M 点。
- 产物：`data/processed/{points,windows,features_core_a}.csv`。
**出口判据**：全量 `points` 与参考窗口集已生成 ✅。

### 🔄 Iteration 2 — 预实验定稿 + Core A 基线（Week 2，进行中）
**目标**：敲定窗口/退化协议，训练统计特征基线。
**已完成**：
1. ✅ 窗口长度定稿 = **120s**（Gate A，见 §3）。
2. ✅ 统计特征提取（`features.build_core_a_matrix`）。
3. ✅ **Random Forest 基线（LOUO）已跑通**：`docs/BASELINE.md`
   - Accuracy **0.6949**，Macro-F1 **0.3388**
   - 逐类 F1：walk 0.83 / bus 0.70 / car 0.13 / subway 0.02 / bike 0.00
   - （bike/subway 极不均衡 + 跨用户，是最难两类，见 Gate C）
**待办**：
1. 加 **XGBoost** 与 SVM 对比（`research/model.py` 已含 XGBoost，需 `pip install xgboost`）。
2. 内层用户分组调参框架（LOUO 内 3 折）。
3. 类别不均衡处理（见 Gate C）。
**出口判据**：Core A 干净数据 Macro-F1 基线 + 每类 F1 + 混淆矩阵。

### ⏭ Iteration 3 — Core B 序列模型（Week 3–4）
**目标**：1D-CNN / GRU（mask 处理不规则采样），与 Core A 公平对比。
**任务**：
1. `features.build_core_b_sequences`（已实现，输出 (X, mask)）。
2. 实现 1D-CNN、GRU（需安装 `torch`；CPU 可跑，GPU 加速）。
3. 同一 LOUO 协议下训练；固定超参网格 ≤12 组/模型。
**出口判据**：Core B 在干净数据上的 Macro-F1，与 Core A 可并排比较。

### ⏭ Iteration 4 — 鲁棒性实验（Week 5）
**目标**：回答"哪种表示在哪种退化下更鲁棒"。
**任务**：
1. LOUO 跨用户评估（含逐折方差）。
2. 降采样 10/30/60s、噪声 σ=5/10/20/50m；test-only vs train+test。
3. 每条件报告保留率 + Macro-F1 + 逐类 P/R/F1。
**出口判据**：完整"模型 × 条件"结果矩阵。

### ⏭ Iteration 5 — 可视化与海报（Week 6）
**任务**：轨迹图、混淆矩阵、逐类指标、保留率曲线；Poster.pdf。

### ⏭ Iteration 6 — 报告（Week 7）
**任务**：按 Report 结构写 6–10 页；补 Disclosures（AIGC 使用声明）；打包 `Group16_Project.zip`。

---

## 3. 关键决策点（gates）

### Gate A：窗口长度 vs 降采样冲突 —— ✅ 已解决（选 120s）
样例数据窗口扫描（`docs/WINDOW_SWEEP.md`）结论：

| 窗口长度 | 10s 降采样保留率 | 通过 go(≥70%)? |
|---|---|---|
| 60s | 0.0000 | ❌ |
| 90s | 0.0000（仅 ~9 点 < 10） | ❌ |
| **120s** | **0.8332** | ✅ |
| 150s | 0.8529 | ✅ |

> **决策：窗口长度 = 120s**（最小可行长度；10s 降采样后 ~12 点 ≥ 10 点阈值，保留率 83.3% > 70%）。
> 已写入 `research/config.py`（`WINDOW_SEC = 120`），并在报告中记录此变更
> （proposal 4.7 预留了"改 90–120s"的口子）。

### Gate B：计算预算（proposal 4.4）
Core A ≈ 分钟级；Core B ≈ 1–2h/配置；总预算 ≤20 GPU-h，固定超参网格。若无 GPU，用 CPU 回退并缩小网格。

### Gate C：类别不均衡（样例已见）
walk/bus 占多数、bike/subway 极少 → 主指标用 **Macro-F1**，必要时加类别权重/采样。

---

## 4. 里程碑时间线（对齐课程节点）

| 节点 | 日期 | 交付 |
|---|---|---|
| Proposal 已提交 | 10/15 | `Group16_Proposal.pdf` |
| Poster | 11/27 | `Poster.pdf` |
| Poster Session | 12/04 | 现场答辩 |
| Final | 12/11 | `Group16_Project.zip` |
| Peer 评估 | 12/18 | 个人互评 |

---

## 5. 依赖与风险

- **依赖**：`research/` 仅需 numpy+pandas；模型阶段需 `scikit-learn`/`xgboost`/`torch`（README 有安装说明）。
- **数据**：完整 GeoLife ~1.8GB，需网络下载；若受限用样例数据跑通全流程并说明（评分允许）。
- **版权/规范**：不编造参考文献；Disclosures 声明 AIGC 使用。

---

## 6. AIGC / 大语言模型使用决策

**结论：核心分类流程不接入 LLM；AIGC 仅作开发/写作辅助（并在报告披露）。**

| 用法 | 是否采用 | 理由 |
|---|---|---|
| 核心管线用 LLM 直接分类 | ❌ 不用 | 任务是数值 GPS 轨迹→特征→分类器，LLM 无天然位置；硬塞会被答辩(30分)问倒，反而扣分 |
| AIGC 作开发/调试/报告辅助 | ✅ 用（披露） | 正是课程发 AIGC API 与要求 Disclosures 的意图 |
| LLM 嵌入作"第三种表示" | ⚠️ 可选（stretch） | 把 120s 窗口运动画像转文本描述→LLM embedding→分类器，与 Core A/B 对比；有学术价值但增加复杂度与依赖，仅在 Core A/B 提前完成后评估 |

> 关于"研究员"：本项目未搭建 LLM 智能体式的"研究员"；`research/` 包是**数据研究基础层**
> （加载/切分/特征/退化/EDA），是 Core A/B 与所有鲁棒性实验的前置，**予以保留**。
