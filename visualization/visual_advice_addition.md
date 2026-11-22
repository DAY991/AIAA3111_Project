在你已完成的六项基础上，我建议最多再做一张、或两张“整合型”图：

7. **Coverage–F1 trade-off（优先级最高）**
   - 横轴：窗口保留率
   - 纵轴：RF/GRU Macro-F1
   - 点：native、5s、10s、噪声级别
   - 30s保留率为0时标注为“not evaluated”，不要画成F1=0
   - 数据：`grid_primary_condition.csv`和`retention_pilot.csv`
   - 它能把已有曲线和覆盖率信息整合成一个实用结论。

8. **F1交互效应图（有空间再做）**
   - 行：5/10/20m噪声
   - 列：5/10s抽样
   - 分别展示RF和GRU的交互量
   - 数据：`interaction_contrasts.csv`
   - 标题直接写“joint degradation interaction on bounded Macro-F1”，并说明这不是因果效应。

另外可以增加一个**9.关键词式结果条**，不占独立一张图：

- 90,120 clean windows
- 71,941 common windows
- 10s retention：79.83%
- 最大GRU优势：+14.28个百分点
- 30s retention：0%
- Clean speed importance：0.3572

**不建议再做：**

- 所有条件逐类热力图
- 混淆矩阵
- 用户级别图
- 17个特征的完整漂移图
- 全部交互效应与完整海报方法流程重复展示
- 复杂的T-SNE或运行时间图

最终海报最优结构是：流程图和关键指标作导入，中间放退化曲线与热力图，右侧放类别分布、速度重要性和特征漂移，逐类和交互效应作为一个或两个小补充面板。