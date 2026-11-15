# Core A 基线结果 (统计特征 + Random Forest, LOUO)

- 窗口长度: 120s
- 可用窗口数: 11219
- 评估协议: Leave-One-User-Out (跨用户, proposal 4.5)

| 指标 | 值 |
|---|---|
| Accuracy | 0.6949 |
| **Macro-F1** | **0.3388** |

## 逐类 F1

| 类别 | F1 |
|---|---|
| walk | 0.8317 |
| bike | 0.0000 |
| bus | 0.7039 |
| car | 0.1338 |
| subway | 0.0244 |