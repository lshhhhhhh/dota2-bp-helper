# Dota 2 BP Helper 胜负导向 Benchmark

模型目标已经从“预测玩家会选什么”改为：

`P(获胜 | 当前公开阵容, 候选英雄, BP轮次)`

胜方的五个选择得到标签 1，败方的五个选择得到标签 0。Policy 仍用于预测玩家
行为，但不再决定推荐排序。

## 时间留出测试

所有模型只使用较早的 80% 比赛训练。下表使用各分段最新的 10% 比赛；这些比赛
没有参与模型训练或参数选择。

| 模型 | 测试比赛 | 第三轮 AUC | 只看英雄总体胜率 AUC | 公开阵容 AUC 增量 | LogLoss | 校准误差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 传奇及以上 | 2,867 | 0.561 | 0.522 | +0.039 | 0.688 | 0.014 |
| 统帅及以下 | 3,553 | 0.574 | 0.552 | +0.021 | 0.685 | 0.009 |
| 全段位 | 6,652 | 0.569 | 0.534 | +0.035 | 0.686 | 0.007 |

AUC 为 0.5 等于随机。三套模型都从公开 4v4 阵容中获得了英雄总体胜率之外的
胜负信息。LogLoss 和校准误差越低越好。

## 第三轮历史胜率关联

“前五内”表示真实玩家的最后一手落在模型推荐前五名。这仍不是随机实验，但比旧的
“预测玩家选择”指标更接近产品目标。

| 模型 | 前五内胜率 | 前五外胜率 | 观察差值 | 前五内样本 | 粗略 95% 区间 | 旧选择预测模型差值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 传奇及以上 | 56.7% | 49.4% | +7.3 点 | 471 | +2.6～+12.0 | +3.5 点 |
| 统帅及以下 | 60.0% | 48.9% | +11.1 点 | 702 | +7.2～+14.9 | +2.9 点 |
| 全段位 | 57.1% | 49.2% | +7.9 点 | 1,280 | +5.0～+10.7 | +1.8 点 |

## 能说什么，不能说什么

- 可以说：模型在未参与训练的新比赛上具有第三轮胜负预测能力；公开阵容带来了
  可测量的 AUC 增量。
- 可以说：历史上真实选择落在新模型前五时，观察到更高胜率。
- 不能说：App 已经被证明让用户胜率提高 7～11 个百分点。
- 原因：玩家、位置、熟练度和选择并非随机分配；没有选择的英雄也没有真实结果。

真正验证因果提升仍需要预先定义的在线随机对照实验。若检验胜率从 50% 提高到
53%，理想 1:1 分组、95% 显著性和 80% 功效下约需 8,700 场完成对局。

完整机器可读结果位于各模型目录的 `outcome_benchmark.json`。

## 推荐排序质量

上面的指标衡量的都是「这个阵容会不会赢」。它们无法衡量「下一手该选谁」，因为
Outcome 模型的状态项对所有候选加同一个数：它改变预测胜率，但不改变排序。一个
排序完全固定的模型可以在上面每一项上拿到相同的分数。

`ranking_benchmark.json` 专门衡量排序。**同状态配对**取同一场比赛双方的第三轮
选择，在同一个状态下打分：状态项精确抵消，只剩候选排序；获胜方的英雄和落败方的
英雄来自同一场比赛，比赛层面的混杂也一并抵消。随机等于 0.5。

| 模型 | Outcome 模型 | 静态英雄强度榜 | Policy | 随机 | 模型−静态榜 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 传奇及以上 | 0.524 | 0.524 | 0.503 | 0.505 | +0.03 点 |
| 统帅及以下 | 0.547 | 0.546 | 0.505 | 0.502 | +0.06 点 |
| 全段位 | 0.539 | 0.537 | 0.504 | 0.501 | +0.16 点 |

排序确实优于随机，但**全部优势都来自英雄的整体强度**。相对一张固定梯度榜的增量
是 0.03～0.16 个百分点，远在置信区间之内，等于零。

排序对阵容的响应程度（4000 个第三轮局面，已排除英雄被选走造成的变化）：

| 模型 | top-5 出现过的英雄数 | top-5 变化率 | 与固定排序的 Spearman |
| --- | ---: | ---: | ---: |
| Outcome 模型 | 5～7（共 127） | 0.000～0.011 | 0.999 |
| 静态英雄强度榜 | 5 | 0.000 | 1.000 |
| Policy | 71～92 | 0.998～1.000 | 0.74～0.80 |

Outcome 模型的推荐实质上是一张静态梯度榜。Policy 是真正随阵容变化的，但它的
排序在配对指标上只有 0.503～0.505，等于随机 —— 会看阵容并不等于排得准。

### 出厂配置：候选池

只按预测胜率排序，会得到一张几乎不随阵容变化的列表：第 2 轮有 81%～89% 的局面
拿到**同一组**五个英雄，整个留出集里只有 5～7 个英雄进过 top-5，而这些推荐玩家
实际只有 9%～12% 的时候会选。一个没人会采纳的推荐，排序再准也没有价值。

出厂配置因此先用 Policy 取出玩家最可能选的 `DEFAULT_CANDIDATE_POOL = 20` 个英雄，
在池内按预测胜率排序，池外英雄保持原有次序排在后面（不丢弃，列表长度不受影响）。
`ranking_benchmark.json` 里的 `shipped_recommender` 就是这个配置。

| 模型 | 同状态配对 | 95% 区间 | 命中差 | H@5 | top-5 英雄种类 | 第2轮最常见组合占比 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 传奇及以上（默认） | 0.5216 | [0.5087, 0.5346] | +3.07 点 | 0.185 | 51 | 10.5% |
| 统帅及以下 | 0.5208 | [0.5092, 0.5324] | +3.83 点 | 0.175 | 51 | 5.9% |
| 全段位 | 0.5198 | [0.5113, 0.5283] | +2.48 点 | 0.178 | 44 | 8.3% |

三个分段的区间下界都高于 0.5，排序仍显著优于随机。代价是同状态配对从未筛选的
0.5242 / 0.5466 / 0.5386 降到上表数值 —— 约 0.3～2.6 个点。命中差在传奇及以上和
统帅及以下都**变好**（2.20→3.07、3.55→3.83）。

**不要把池大小「凑整」到 40。**池大小对排序质量的影响不是单调的：40 在传奇及以上
（App 的默认模型）拿到 0.5113，区间 [0.4984, 0.5243] **包含随机**，而 20 在该分段
的每一项指标上都优于 40。改动前用
`python -m d2draft.ranking_benchmark --candidate-pool N` 重跑扫描。

### 胜败方 top-5 命中差

`outcome_split_hit` 衡量界面上真正显示的东西：玩家实际选的英雄有没有落在推荐
前五。胜方和败方都当作合理参考 —— 人类的选择基本都说得通 —— 只问模型能不能把
后来赢的那一方和输的那一方分开。按比赛配对统计。

| 方法 | 传奇及以上 | 统帅及以下 | 全段位 |
| --- | ---: | ---: | ---: |
| Outcome 模型 | +2.20 点 | +3.94 点 | +2.74 点 |
| 静态英雄强度榜 | +0.91 点 | +3.55 点 | +2.78 点 |
| Policy（模仿玩家选择） | +2.62 点 | +2.17 点 | +1.38 点 |
| 随机 | −0.66 点 | +0.31 点 | −0.20 点 |

**只能报告差值，不能报告两个命中率的加权和。**两项都会被「猜中玩家会选什么」
同时拉满：以 `w_胜=2, w_败=1` 计算，第三轮 Policy 得 0.755、Outcome 得 0.302，
纯模仿模型会以 2.5 倍排到第一，而它的同状态配对准确率只有 0.504，等于随机。

差值本身也不是完全干净的：Policy 在这一项上拿到 +1.38～+2.62，传奇及以上一档
甚至最高，但它的排序能力是随机。机制是获胜方倾向于选更主流、更好预测的英雄，
所以一个模仿模型不需要任何排序能力也能拿到正的差值。

因此这两项要一起看：`same_state_pairwise` 用上了完整排序、与英雄流行度无关，是
更稳健的排序判据；`outcome_split_hit` 更贴近产品实际展示，但会奖励与玩家行为
重合。两者在传奇及以上一档结论不同（配对指标上 Outcome 与静态榜打平，命中差上
Outcome 领先 1.3 点），原因是该分段玩家很少选静态榜推荐的英雄（H@5 仅 0.049），
命中率过低把差值压向了零 —— 那反映的是推荐与玩家行为的重合度，不是排序质量。

### 为什么不用准确率

`outcome_benchmark.json` 里仍然输出 `accuracy`，但它**不是主指标**，也不能拿去
和别的论文比。三个原因：

1. **阈值是拍脑袋的，而且影响结果。**把判定阈值从 0.48 扫到 0.52，全段位模型的
   准确率在 0.5474～0.5506 之间摆动，最优点落在 0.510 而不是 0.500。这个指标超出
   随机的总信号只有 5 个点，其中约 6% 是阈值造成的。
2. **它丢掉概率。**预测 0.99 和预测 0.51 都算「预测赢」。22.6% 的预测挤在
   0.48~0.52 区间内，对这些样本 ACC 把一次抛硬币记成一次明确判断。
3. **地板随数据集变。**我们每场比赛同时贡献一个赢的决策和一个输的决策，所以基准
   恰好是 0.5000；而 DraftRec 的 Dota2 数据里天辉胜率 51.8%，无脑猜天辉就有
   0.5180。拿他们的 0.5750 和我们的 0.5504 直接比是错的，扣掉各自地板后是
   +5.7 点对 +5.0 点。

Bootstrap 也显示，相对于超出随机的那部分信号，准确率的波动（0.084）大于
AUC（0.070）。判别力看 AUC，概率可信度看 LogLoss 和校准误差，排序看
`ranking_benchmark.json`。

### 有效样本量是比赛数，不是样本数

每场比赛的第三轮产生**一对镜像样本**：同一个 4v4 从双方视角各看一次，标签相反。
所以 13,304 个第三轮样本的有效样本量接近 6,652 场，而不是 13,304。

`same_state_pairwise` 和 `outcome_split_hit` 已经按比赛配对统计。但
`binary_metrics` 把样本当独立处理 —— 目前没有对 AUC 或准确率发布置信区间，所以
没有错误；**如果以后要加，必须按比赛而不是按样本计算**，否则区间会偏窄。

### 离线策略评估当前不可用

`off_policy_value` 使用 Policy 作为倾向性模型做自归一化 IPS。当前三套模型上它
都被标记为 `usable: false`：有效样本量只有 19%～21%，且把目标策略换成均匀分布
后估计值只变化 0.006～0.007，说明数字由 `1/μ` 分母主导，与被评估的模型无关。
在 Policy 的 Hit@1 提高之前，不要引用这个估计值。

## 已经测过并排除的方向

以下三个方向都曾看起来有希望，都在测量后被放弃。记录在这里是为了让后来者能
**复核数字，而不是把这些路再走一遍**。三个检查都可以重跑（需要私有采集数据库）：

```powershell
python -m d2draft.negative_results --check interactions
python -m d2draft.negative_results --check composition
python -m d2draft.negative_results --check margin-targets
```

### 1. 配合与克制的交互建模

模型里的 synergy/counter 嵌入学到的值，跨候选标准差只有 0.0021，而每英雄固定偏置
是 0.0892 —— **比值 0.024**。5000 个随机局面里只有 6 个英雄能进过 top-5，固定排名
20 名开外的英雄最好只能爬到第 16 名。排序相对静态梯度榜的增量是 +0.16 / +0.03 /
+0.06 点，全在置信区间内。

数据量差多少：8,001 个英雄对，平均每对只有 166 次队友同队观测，这个样本量只能
检出 15.4 个百分点的胜率差，而真实配合效应大约 2~5 个点。要检出 2 个点需要约
390 万场，我们有 6.6 万场。

文献同向：DraftRec（WWW 2022）在 5 万场 Dota 2 上比较逻辑回归、因子分解机、
图神经网络和显式配合建模，所有方法都在逻辑回归的 ±0.4% 内。JueWuDraft 显示
交互建模在 **AI 自对弈**数据上把 AUC 从 0.771 拉到 0.908，但在人类数据上只有
0.642 → 0.694 —— 人类阵容是自我平衡的，交互信号被玩家自己抹掉了。

### 2. 位置与阵容结构检查

从 35,503 场按队内经济排序推出实际位置。**英雄不是位置固定的**：位置分布熵中位数
0.811（1.0 = 五个位置均匀），最固定的英雄也只有 0.426。手写位置表在同一个版本内
就已经是错的，不必等到版本更新。

**畸形阵容不存在**：每队「天然一号位」期望个数均值 1.00、标准差 0.30，5%~95%
分位 [0.53, 1.51]。数据里没有 5 个大哥的阵容，包括统帅及以下。

**就算到极端也几乎没信号**：按组成分箱的胜率是 0.476 / 0.506 / 0.509 / 0.504 /
0.504 / 0.476，只有最极端的 10% 两端各掉 3 个点。保留：位置用赛后经济排名代理，
有噪声；且一号位分布是在同一批比赛上估的。

### 3. 用连续净胜差替代二元胜负作训练目标

同一个自变量（阵容强度差）回归不同目标，n=35,503：

| 训练目标 | t | R² | 相对效率 |
| --- | ---: | ---: | ---: |
| 二元胜负（在用） | 25.71 | 0.01827 | 1.00x |
| 击杀差 | 24.62 | 0.01679 | 0.96x |
| 兵营差 | 24.39 | 0.01648 | 0.95x |
| 防御塔差 | 24.13 | 0.01613 | 0.94x |
| 净资产差 | 12.71 | 0.00453 | 0.49x |
| 净资产差/分钟 | 8.33 | 0.00195 | 0.32x |

**每个连续目标都比二元胜负差。**净胜差的额外方差来自双方水平差和比赛长度，
不是阵容 —— 加进去等于往目标里掺噪声。「一场比赛只有 1 个 bit 太少」这个假设
是错的：那 1 个 bit 是这批数据里最干净的信号。

顺带得到一个上限数字：**阵容强度差只能解释比赛结果 1.8% 的方差**（R²=0.018，
n=35,503，t=25.7 高度显著）。这是这个问题本身的性质，不是模型的缺陷。

### 4. 加大模型（嵌入维度）

在完整 53,212 场训练池上扫描嵌入维度，固定最新 10% 测试集，每档 3 个随机种子：

| 维度 | epochs | 测试 AUC | 同状态配对 | 训练集 AUC | 交互项/固定偏置 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 12 | 0.5671 | 0.5361 | 0.5517 | 0.010 |
| 16 | 12 | 0.5671 | 0.5359 | 0.5518 | 0.022 |
| 32 | 12 | 0.5671 | 0.5369 | 0.5518 | 0.043 |
| 64 | 12 | 0.5672 | 0.5376 | 0.5519 | 0.090 |
| 128 | 30 | 0.5665 | 0.5372 | 0.5520 | 0.129 |

**容量提高 16 倍，测试 AUC 变化 0.0007，种子标准差本身就有 0.0003~0.0015。**
30 个 epoch 反而略差。

最关键的是**训练集 AUC 也不动**（0.5517→0.5520）：这不是过拟合，是模型连训练数据
都无法拟合得更好 —— 数据里已经没有更多可学的东西了。模型不是容量受限的。

交互项/固定偏置比值随维度从 0.010 涨到 0.129，但测试指标全部不动。**嵌入会把给它
的容量全部吸收成噪声**，所以这个比值单独上升不能作为「嵌入学到了配合克制」的证据。

### 数据量：AUC 还在涨，排序已经收敛

同一套划分，按训练规模扫描（每档 3 个种子）：

| 训练比赛数 | 第三轮 AUC | 同状态配对 |
| ---: | ---: | ---: |
| 2,000 | 0.5277 | 0.5083 |
| 5,000 | 0.5346 | 0.5128 |
| 10,000 | 0.5513 | 0.5180 |
| 20,000 | 0.5603 | 0.5287 |
| 35,000 | 0.5641 | **0.5373** |
| 53,212 | 0.5671 | **0.5359** |

**AUC 没有收敛**：从 1 万场起每翻倍稳定增加约 0.005，看不到平台期。
**排序已经收敛**：35,000 → 53,212（多 52% 数据）反而下降 0.0014，在噪声内。

两者分离的原因还是同一个结构事实：更多数据改善 `state_strength`，而该项对所有候选
加同一个数，永远不影响排序。排序由 `candidate_bias`（127 个参数）决定，早已收敛。

**预算含义**：继续购买 7.41 数据只会让「你这把 44%」更准，**不会改变推荐哪些英雄**。
按当前速率把 AUC 从 0.567 提到 0.60 约需 6.6 次翻倍（约 640 万场）。真正值得花钱的是
**下一个版本** —— 版本更替会让英雄强度漂移，而排序本质就是一张强度榜，这一点额外的
7.41 数据无法弥补。

复现：`python -m d2draft.negative_results --check capacity`（需要几分钟）。

### 逻辑回归与神经网络等价

在同一套划分上训练一个纯逻辑回归（每英雄一个队友系数、一个敌方系数、一个候选
系数，共 384 个参数）与当前神经网络（约 8,385 个参数）对比：

| 模型 | 参数 | 第3轮 AUC | LogLoss | 校准误差 | 同状态配对 | 命中差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 神经网络（当前） | 8,385 | 0.5694 | 0.6860 | 0.0071 | 0.5386 | 2.74 点 |
| 逻辑回归 | 384 | 0.5679 | 0.6864 | 0.0079 | 0.5356 | 3.19 点 |
| 静态强度榜 | 127 | 0.5259 | 5.0356 | 0.4766 | 0.5370 | 2.78 点 |

22 倍更少的参数，每项都在噪声内打平，训练耗时 14 秒。注意第三行：静态梯度榜
**排序一样好但概率完全没校准**（LogLoss 5.04、校准误差 0.477），所以不能直接用
梯度榜代替 —— 阵容评估功能依赖校准过的概率。逻辑回归等于一张校准过的梯度榜。

保留：这个逻辑回归是在临时脚本里训的，只试了一组 L2，没调参，只跑了全段位。

**决定：保留神经网络，不换逻辑回归。**理由是逻辑回归在结构上**永远**无法表达交互，
数据再多也没用；而神经网络的 synergy/counter 项已经存在，只是在当前数据量下被正则
压住了，数据增长后重调 L2 就能放开。今天两者打平，所以持有这个选择权不花任何成本，
而迁移要动 bundle 格式、推荐器和重训流程，换来的是零。

作为对照：RecSys 2018 在 300 万场人类对局上，逻辑回归与神经网络仍只差 0.03 AUC；
JueWuDraft 在 3000 万场上是 0.642 → 0.694。也就是说人类数据上的实质增益出现在
300 万～3000 万场之间，我们有 6.6 万场，差 45～450 倍。

**监控方式**：每次重训后跑 `python -m d2draft.negative_results --check interactions`，
看交互项与固定偏置的比值（当前 **0.024**）。**但这个比值单独上升不构成证据。**实测把
嵌入维度从 8 提到 128，该比值从 0.010 升到 0.129，而测试 AUC、同状态配对和命中差
全部不动 —— 嵌入会把给它的容量全部吸收成噪声。只有比值上升**并且**留出集指标同时
改善，才说明嵌入真的学到了东西。

同一组实验也说明模型**不是容量受限**：维度提高 16 倍，连训练集 AUC 都只从 0.5517
变到 0.5520。加大模型没有意义，瓶颈在数据本身。

**与重训策略的关联**：如果改为频繁重训、每次只用较短的近期窗口，8,385 个参数的过拟合
风险恰好在单次数据量变小时上升。因此重训应继续走 `d2draft/train_outcome.py` 的微调
路径（从上一版模型热启动），而不是在短窗口上从零训练。

### 已明确排除的产品方向

玩家熟练度与账号历史是文献里唯一被验证有效的额外信号（DraftRec 在 LoL 上把
准确率从 0.5255 提到 0.5535，其中绝大部分来自玩家历史而非英雄交互）。**用户已
明确决定不做**：需要绑定 Steam 账号，产品复杂度过高，而且玩家会自己跳过不会玩
的英雄。不要重提。

---

# Dota 2 BP Helper Outcome Benchmark

The recommendation objective is now:

`P(win | public lineup, candidate hero, draft round)`

All five picks on the winning side receive label 1, and all five picks on the losing
side receive label 0. Policy remains a player-behavior model but no longer determines
the recommendation ranking.

## Chronological holdout

Each model is trained only on the oldest 80% of its matches. The table below uses the
newest 10%, which is not used for training or parameter selection.

| Model | Test matches | Round-3 AUC | Hero win-rate-only AUC | Public-lineup AUC gain | LogLoss | Calibration error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legend and above | 2,867 | 0.561 | 0.522 | +0.039 | 0.688 | 0.014 |
| Archon and below | 3,553 | 0.574 | 0.552 | +0.021 | 0.685 | 0.009 |
| All ranks | 6,652 | 0.569 | 0.534 | +0.035 | 0.686 | 0.007 |

An AUC of 0.5 is random. All three models extract outcome information from the public
4v4 lineup beyond global hero win rate. Lower LogLoss and calibration error are better.

## Historical round-3 win-rate association

“Inside top five” means the actual last pick appeared in the model's first five
recommendations. This is still not a randomized experiment, but it is aligned more
closely with the product goal than the old pick-prediction metric.

| Model | Win rate inside top 5 | Outside top 5 | Observed difference | Inside samples | Approx. 95% interval | Old pick-model difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legend and above | 56.7% | 49.4% | +7.3 pts | 471 | +2.6 to +12.0 | +3.5 pts |
| Archon and below | 60.0% | 48.9% | +11.1 pts | 702 | +7.2 to +14.9 | +2.9 pts |
| All ranks | 57.1% | 49.2% | +7.9 pts | 1,280 | +5.0 to +10.7 | +1.8 pts |

## What these results do and do not establish

- Supported: the model predicts round-3 outcomes on newer, unseen matches, and public
  lineup information produces a measurable AUC gain.
- Supported: actual picks inside the new top five have a higher observed historical
  win rate.
- Not supported: the app has been proven to raise a user's win rate by 7–11 points.
- Why: players, roles, proficiency, and picks were not randomly assigned, and unchosen
  heroes have no observed outcome.

A pre-registered online randomized trial is still required for causal validation. To
test an increase from 50% to 53% with ideal 1:1 assignment, 95% significance, and 80%
power requires roughly 8,700 completed matches.

Full machine-readable results are stored in each model directory's
`outcome_benchmark.json`.

## Recommendation ranking quality

Everything above measures "will this lineup win". None of it measures "which hero
should I pick next", because the outcome model's state term adds the same amount
to every candidate: it moves the predicted win probability without moving the
ranking. A model with a completely fixed ranking scores identically on all of it.

`ranking_benchmark.json` measures the ranking directly. **Same-state pairwise**
takes both round-3 picks of one match and scores them at the same state: the state
term cancels exactly, leaving only the candidate ranking, and because the winning
and losing hero come from the same match, match-level confounders cancel too.
Chance is 0.5.

| Model | Outcome model | Static hero strength | Policy | Random | Model − static |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legend and above | 0.524 | 0.524 | 0.503 | 0.505 | +0.03 pts |
| Archon and below | 0.547 | 0.546 | 0.505 | 0.502 | +0.06 pts |
| All ranks | 0.539 | 0.537 | 0.504 | 0.501 | +0.16 pts |

The ranking does beat chance, but **all of that comes from per-hero global
strength**. The gain over a fixed tier list is 0.03–0.16 points, well inside the
confidence intervals, which is zero.

How much the ranking responds to the draft (4,000 round-3 states, with variation
caused by heroes being unavailable excluded):

| Model | Distinct heroes seen in top 5 | Top-5 change rate | Spearman vs fixed order |
| --- | ---: | ---: | ---: |
| Outcome model | 5–7 of 127 | 0.000–0.011 | 0.999 |
| Static hero strength | 5 | 0.000 | 1.000 |
| Policy | 71–92 | 0.998–1.000 | 0.74–0.80 |

The outcome model's recommendation is a static tier list in practice. Policy does
respond to the draft, but its ranking scores 0.503–0.505 on the pairwise test,
which is chance: reacting to the lineup is not the same as ranking well.

### The shipped configuration: a candidate pool

Ranking purely by predicted win probability produces a list that barely moves: at
round 2, between 81% and 89% of drafts receive the **same** five heroes, only 5–7
heroes ever reach the top five across the whole holdout, and players actually take
those suggestions 9–12% of the time. A recommendation nobody acts on has no value
regardless of how well it is ordered.

The shipped configuration therefore uses Policy to take the
`DEFAULT_CANDIDATE_POOL = 20` heroes players are most likely to pick, ranks those by
predicted win probability, and leaves the rest in their original order behind them —
nothing is dropped, so a long list still fills. `shipped_recommender` in
`ranking_benchmark.json` measures exactly this.

| Model | Same-state pairwise | 95% interval | Hit gap | H@5 | Distinct top-5 | Round-2 dominance |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Legend and above (default) | 0.5216 | [0.5087, 0.5346] | +3.07 pts | 0.185 | 51 | 10.5% |
| Archon and below | 0.5208 | [0.5092, 0.5324] | +3.83 pts | 0.175 | 51 | 5.9% |
| All ranks | 0.5198 | [0.5113, 0.5283] | +2.48 pts | 0.178 | 44 | 8.3% |

Every interval's lower bound clears 0.5, so the ranking still beats chance. The cost
is same-state pairwise falling from the unfiltered 0.5242, 0.5466, and 0.5386 to the
values above — between 0.3 and 2.6 points. The hit gap **improves** for legend and
above and for archon and below (2.20 → 3.07 and 3.55 → 3.83).

**Do not round the pool up to 40.** Pool size does not affect ranking quality
monotonically: 40 scores 0.5113 on legend-and-above — the app's default model — with
an interval of [0.4984, 0.5243] that includes chance, while 20 beats 40 there on every
metric. Re-run the sweep with
`python -m d2draft.ranking_benchmark --candidate-pool N` before changing it.

### Winner/loser top-5 hit gap

`outcome_split_hit` measures what the UI actually shows: whether the hero a player
really took landed inside the visible top five. Both sides count as reasonable
references — human picks are broadly sensible — so the only question is whether the
model separates the side that went on to win from the side that did not. Paired by
match.

| Method | Legend and above | Archon and below | All ranks |
| --- | ---: | ---: | ---: |
| Outcome model | +2.20 pts | +3.94 pts | +2.74 pts |
| Static hero strength | +0.91 pts | +3.55 pts | +2.78 pts |
| Policy (pick imitation) | +2.62 pts | +2.17 pts | +1.38 pts |
| Random | −0.66 pts | +0.31 pts | −0.20 pts |

**Report the difference, never a weighted sum of the two hit rates.** Both terms are
maximised by predicting what players pick: with `w_win=2, w_loss=1` at round 3,
Policy scores 0.755 and the outcome model 0.302, so a pure imitation model ranks
first by 2.5x while its same-state pairwise accuracy is 0.504, which is chance.

The difference is not fully clean either. Policy scores +1.38 to +2.62 here, highest
of all methods in the legend-and-above bracket, with no ranking ability at all. The
mechanism is that winning sides pick more mainstream, more predictable heroes, so an
imitator gets a positive gap without ranking anything.

Read the two together. `same_state_pairwise` uses the full ordering and is
independent of hero popularity, so it is the more robust ranking verdict.
`outcome_split_hit` is closer to what the product displays but rewards overlap with
player behaviour. They disagree for legend and above — a tie on the pairwise test,
a 1.3 point lead for the outcome model on the hit gap — because players in that
bracket rarely take the static list's heroes at all (H@5 of 0.049), and a hit rate
that low compresses the gap toward zero. That reflects overlap with player
behaviour, not ranking quality.

### Why accuracy is not used

`outcome_benchmark.json` still emits `accuracy`, but it is **not a headline metric**
and it cannot be compared against other papers. Three reasons:

1. **The threshold is arbitrary and it moves the result.** Sweeping the decision
   threshold from 0.48 to 0.52 moves the all-ranks model between 0.5474 and 0.5506,
   with the optimum at 0.510 rather than 0.500. The metric's entire signal above
   chance is 5 points, so roughly 6% of the score is a threshold artefact.
2. **It discards the probability.** A prediction of 0.99 and one of 0.51 both count
   as "predicts a win". 22.6% of predictions sit inside 0.48–0.52, where accuracy
   records a coin flip as a confident call.
3. **The floor changes with the dataset.** Every match here contributes one winning
   and one losing decision, so our base rate is exactly 0.5000. DraftRec's Dota2 set
   has a 51.8% Radiant win rate, so always guessing Radiant scores 0.5180. Comparing
   their 0.5750 with our 0.5504 directly is wrong; net of each floor it is +5.7
   points against +5.0.

Bootstrapping also shows accuracy is noisier than AUC relative to the signal above
chance (0.084 against 0.070). Use AUC for discrimination, log loss and calibration
error for probability quality, and `ranking_benchmark.json` for the ranking.

### Effective sample size is matches, not examples

Each match produces a **mirrored pair** of round-3 examples: the same 4v4 seen from
both sides, with opposite labels. The effective sample size behind 13,304 round-3
examples is therefore closer to 6,652 matches.

`same_state_pairwise` and `outcome_split_hit` already pair by match. `binary_metrics`
treats examples as independent — nothing published today is wrong, because no
confidence interval is reported for AUC or accuracy, but **if one is ever added it
must be computed across matches**, or it will be too narrow.

### Off-policy evaluation is currently unusable

`off_policy_value` runs self-normalised IPS with Policy as the propensity model.
It is currently marked `usable: false` for all three models: effective sample size
is only 19–21%, and swapping the target for a uniform policy moves the estimate by
0.006–0.007, showing the number is driven by the `1/mu` denominator rather than by
the model under evaluation. Do not quote it until Policy's Hit@1 improves.

## Directions that were measured and ruled out

Three directions looked promising and were each abandoned after measurement. They
are recorded here so a future maintainer can **check the numbers instead of redoing
the work**. All three checks re-run against the private collection database:

```powershell
python -m d2draft.negative_results --check interactions
python -m d2draft.negative_results --check composition
python -m d2draft.negative_results --check margin-targets
```

### 1. Modelling synergy and counter interactions

The synergy and counter embeddings do learn non-zero values, but their spread across
candidates is 0.0021 against 0.0892 for the fixed per-hero bias — a **ratio of
0.024**. Across 5,000 random states only 6 heroes ever reach the top 5, and a hero
ranked 20th or worse on the fixed order can never climb above 16th. The ranking gain
over a static tier list is +0.16, +0.03, and +0.06 points, all inside the confidence
intervals.

The data is far short of what pairwise effects need: 8,001 hero pairs with an average
of 166 same-team observations each, which can only detect a 15.4 point win-rate gap,
while real synergies are perhaps 2–5 points. Detecting 2 points would take roughly
3.9 million matches against the 66,515 available.

The literature agrees. DraftRec (WWW 2022) compared logistic regression,
factorisation machines, graph neural networks, and explicit synergy modelling on
50,000 Dota 2 matches and every method landed within ±0.4% of logistic regression.
JueWuDraft shows interaction modelling lifting AUC from 0.771 to 0.908 on **AI
self-play** data but only 0.642 to 0.694 on human data: human lineups are
self-balancing, so players erase the interaction signal themselves.

### 2. Role and lineup-composition checks

Positions were derived from 35,503 matches by ranking net worth within each team.
**Heroes are not locked to positions**: median position entropy is 0.811 where 1.0
is uniform across all five, and even the most fixed hero sits at 0.426. A
hand-authored role table is already wrong inside a single patch, never mind after
one.

**Unbalanced lineups do not occur**: the expected number of natural position-1
heroes per team has mean 1.00 and standard deviation 0.30, 5th to 95th percentile
[0.53, 1.51]. There are no five-carry drafts in the data, including in the
archon-and-below bracket.

**Even the extremes carry almost nothing**: win rate by composition bin runs 0.476,
0.506, 0.509, 0.504, 0.504, 0.476, so only the outer 10% on each side loses about 3
points. Caveats: position is proxied by end-of-game net worth rank, which is noisy,
and the position-1 distribution is estimated on the same matches.

### 3. Replacing the binary label with a continuous victory margin

The same predictor (draft strength difference) regressed on different targets,
n=35,503:

| Training target | t | R² | Relative efficiency |
| --- | ---: | ---: | ---: |
| Binary win/loss (in use) | 25.71 | 0.01827 | 1.00x |
| Kill difference | 24.62 | 0.01679 | 0.96x |
| Barracks difference | 24.39 | 0.01648 | 0.95x |
| Tower difference | 24.13 | 0.01613 | 0.94x |
| Net worth difference | 12.71 | 0.00453 | 0.49x |
| Net worth per minute | 8.33 | 0.00195 | 0.32x |

**Every continuous target is worse than the binary label.** The extra variance in a
margin comes from the skill gap between the teams and from match length, not from
the draft, so adding it dilutes the signal. The premise that "one bit per match is
too little" is wrong: that bit is the cleanest signal this data has.

This also produces a ceiling worth remembering: **the draft strength difference
explains about 1.8% of outcome variance** (R²=0.018, n=35,503, t=25.7, highly
significant). That is a property of the problem, not a defect in the model.

### 4. A bigger model (embedding dimension)

Sweeping the embedding dimension on the full 53,212-match pool against the fixed
newest 10% test set, three seeds per configuration:

| Dim | Epochs | Test AUC | Same-state pairwise | Train AUC | Interaction/bias |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 12 | 0.5671 | 0.5361 | 0.5517 | 0.010 |
| 16 | 12 | 0.5671 | 0.5359 | 0.5518 | 0.022 |
| 32 | 12 | 0.5671 | 0.5369 | 0.5518 | 0.043 |
| 64 | 12 | 0.5672 | 0.5376 | 0.5519 | 0.090 |
| 128 | 30 | 0.5665 | 0.5372 | 0.5520 | 0.129 |

**Sixteen times the capacity moves test AUC by 0.0007**, against a seed standard
deviation of 0.0003 to 0.0015. Thirty epochs is slightly worse than twelve.

The decisive column is train AUC, which also does not move (0.5517 to 0.5520). This
is not overfitting: the model cannot fit even the *training* data better with more
parameters, because there is nothing further in it to fit. The model is not
capacity-limited.

The interaction/bias ratio climbs from 0.010 to 0.129 with dimension while every
held-out metric stays flat. **The embeddings absorb whatever capacity they are
given, as noise**, so a rising ratio alone is not evidence that they have learned
synergy or counters.

### Data volume: AUC still climbing, ranking already converged

The same split, swept by training size, three seeds each:

| Training matches | Round-3 AUC | Same-state pairwise |
| ---: | ---: | ---: |
| 2,000 | 0.5277 | 0.5083 |
| 5,000 | 0.5346 | 0.5128 |
| 10,000 | 0.5513 | 0.5180 |
| 20,000 | 0.5603 | 0.5287 |
| 35,000 | 0.5641 | **0.5373** |
| 53,212 | 0.5671 | **0.5359** |

**AUC has not converged**: from 10,000 onward it gains a steady ~0.005 per doubling
with no sign of a plateau. **Ranking has**: 52% more data from 35,000 to 53,212
moves it by -0.0014, which is noise.

They diverge for the structural reason that runs through this whole document. More
data sharpens `state_strength`, and that term adds the same amount to every
candidate, so it never touches the ordering. The ranking is driven by
`candidate_bias`, 127 parameters that converged long ago.

**Budget implication**: buying more 7.41 data makes "your draft is at 44%" more
accurate and **does not change which heroes get recommended**. At the observed rate,
lifting AUC from 0.567 to 0.60 needs about 6.6 doublings, roughly 6.4 million
matches. The spend worth making is **the next patch** — hero strengths drift, the
ranking is essentially a strength list, and no amount of extra 7.41 data fixes that.

Reproduce with `python -m d2draft.negative_results --check capacity` (a few minutes).

### Logistic regression matches the neural network

A plain logistic regression trained on the same split — one ally coefficient, one
enemy coefficient, and one candidate coefficient per hero, 384 parameters — against
the current network at roughly 8,385 parameters:

| Model | Params | Round-3 AUC | LogLoss | Calibration | Same-state pairwise | Hit gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Neural network (current) | 8,385 | 0.5694 | 0.6860 | 0.0071 | 0.5386 | 2.74 pts |
| Logistic regression | 384 | 0.5679 | 0.6864 | 0.0079 | 0.5356 | 3.19 pts |
| Static hero strength | 127 | 0.5259 | 5.0356 | 0.4766 | 0.5370 | 2.78 pts |

Twenty-two times fewer parameters, every metric tied within noise, 14 seconds to
train. Note the third row: the static tier list **ranks just as well but its
probabilities are uncalibrated** (log loss 5.04, calibration error 0.477), so it
cannot simply replace the model — the lineup evaluation depends on calibrated
probabilities. Logistic regression is a calibrated tier list.

Caveat: this regression was fitted in a scratch script with a single L2 setting, no
tuning, and only on the all-ranks bracket.

**Decision: keep the neural network, do not switch to logistic regression.** Logistic
regression structurally cannot represent interactions, so no amount of data ever helps
it, whereas the network's synergy and counter terms already exist and are merely held
down by regularisation at this data scale — retuning L2 releases them once data grows.
The two tie today, so holding that option costs nothing, while migrating would touch
the bundle format, the recommender, and the retraining pipeline for a measured gain of
zero.

For calibration: RecSys 2018 still found only 0.03 AUC between logistic regression and
a neural network at 3 million human matches, and JueWuDraft reports 0.642 to 0.694 at
30 million. Meaningful gains on human data therefore appear somewhere between 3M and
30M matches, against the 66,515 available — a factor of 45 to 450.

**How to monitor it**: after each retrain, run
`python -m d2draft.negative_results --check interactions` and watch the ratio of the
interaction spread to the fixed per-hero bias, currently **0.024**. **A rising ratio is
not evidence on its own.** Raising the embedding dimension from 8 to 128 lifts that
ratio from 0.010 to 0.129 while test AUC, same-state pairwise, and the hit gap all stay
flat: the embeddings absorb whatever capacity they are given, as noise. Only a rising
ratio *together with* improving held-out metrics means they have learned anything.

The same sweep shows the model is **not capacity-limited**. Sixteen times the embedding
dimension moves even the *training* AUC from 0.5517 to 0.5520. A bigger model is not the
answer; the limit is in the data.

**How it interacts with retraining**: if retrains become frequent and each uses a
shorter recent window, an 8,385-parameter model is most exposed to overfitting exactly
when per-retrain data shrinks. Retraining should therefore keep using the fine-tuning
path in `d2draft/train_outcome.py`, warm-starting from the previous bundle, rather
than training from scratch on a short window.

### A product direction that is explicitly closed

Player proficiency and account history are the only extra signal the literature
validates — DraftRec lifts League of Legends accuracy from 0.5255 to 0.5535, and
nearly all of that comes from player history rather than hero interactions. **The
user has decided against it**: it would require binding a Steam account, which is
too much product complexity, and players already skip heroes they cannot play. Do
not raise it again.
