还没有完全完成。

从当前 GitHub仓库 看，核心实验结果的可视化已经完成，但完整项目和最终海报层面的可视化大约完成了60%–70%。
已完成的三张核心图
1. Sampling Coverage
retention_pilot.png

已经展示：
- 原生、5秒、10秒和30秒采样的窗口保留率；
- 90,120个参考窗口；
- 71,941个共同支持窗口；
- 不同交通方式的保留率差异；
- 30秒采样下没有可用窗口。
完成度：可以直接放海报。
2. Combined Degradation Curves
deployment_f1_curves.png

已经展示：
- Native、5秒和10秒三种采样条件；
- 0、5、10和20米噪声；
- RF与GRU的Macro-F1；
- 组合退化下的非单调变化。
完成度：这是海报最重要的主图，可以直接使用。
3. GRU−RF Difference Heatmap
deployment_difference_heatmap.png

已经展示：
- 12个组合条件；
- GRU相对RF领先的百分点；
- 最大差距为5秒+10米条件下的14.28个百分点；
- GRU优势并不会随退化程度单调增加。
完成度：可以直接放海报。
还缺少的必要可视化
1. 项目流程图
目前没有一张完整展示项目方法的图。

建议内容：
GeoLife
→ Label alignment
→ Same-label segmentation
→ 120 s windows
→ RF statistical features / GRU sequences
→ LOUO
→ Sampling × Noise grid
→ Evaluation
优先级：最高。

这是海报Methods部分最需要补的图。
2. 数据集和类别分布图
目前覆盖率图显示了各类别的保留率，但没有展示原始窗口的类别数量和不平衡程度。

建议增加一张小型柱状图：
- walk：28,746
- bus：24,624
- car：18,125
- bike：13,840
- subway：4,785
用途：
- 解释为什么使用Macro-F1；
- 说明subway是少数类；
- 帮助观众理解逐类表现差异。
优先级：中高。

也可以把它压缩成一个小表，不一定单独占一张大图。
3. RF特征解释图
Member B已经完成了特征漂移和置换重要性分析，但目前只有CSV，没有海报友好的图。

建议做一张简化图，仅展示speed-group importance：
Condition	Importance
Clean	0.3572
10 s + 0 m	0.1844
Native + 20 m	0.0055
10 s + 20 m	0.0720


这张图能够解释：
- RF在干净数据上高度依赖速度特征；
- 20米噪声几乎破坏了这种依赖；
- 降采样后噪声对速度估计的影响有所变化。
优先级：高。

它能让海报从“展示结果”提升到“解释结果”。
4. 特征漂移对比图
可以展示20米噪声造成的平均速度变化：
Condition	Median absolute change in mean speed
Native + 20 m	12.10 m/s
10 s + 20 m	1.42 m/s


形式可以是两个柱子或一个简洁的示意图。

优先级：中。

如果海报空间不够，可以把这个数字放在RF特征解释图旁边，不必单独制作。
5. 逐类别性能图
仓库已有逐类别CSV，但没有图。

可以选择四个代表条件：
- Clean
- 10 s + 0 m
- Native + 20 m
- 10 s + 20 m
比较walk、bike、bus、car、subway的RF和GRU F1。

不过完整画出来会比较拥挤。海报中更建议只展示clean条件的逐类表现，或者突出：
- GRU在clean car类别领先约17.5个百分点；
- RF在clean subway类别领先约7.7个百分点。
优先级：可选。
6. 噪声重复或不确定性图
ds10_n10已经有三个噪声种子：
- RF：0.4024 ± 0.0015
- GRU：0.4283 ± 0.0012
目前没有误差棒图。

由于只有一个条件做了多种子，不值得单独做大图。建议直接在海报文字中写：
The GRU advantage at 10 s + 10 m remains consistent across three noise realizations.

优先级：低。
不必再制作的图
为了保持海报简洁，不建议继续制作：
- 12个条件的完整数字表；
- 每个条件一张混淆矩阵；
- 所有17个统计特征的漂移图；
- 每个用户的单独结果图；
- 30秒条件的模型曲线；
- 代码结构图；
- 运行时间或环境图；
- 复杂的t-SNE图。