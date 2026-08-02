# Dota 2 BP Helper 离线 Benchmark

以下数字来自当前随 App 发布的 Dota 7.41 模型和按时间留出的测试集。

## 历史胜率关联（不能解释为因果提升）

“推荐前 N 内”表示真实最后一手落在模型推荐前 N 名。不同组不是随机分配，玩家、
位置和阵容难度都可能造成差异，所以表格只描述历史关联，不能声称模型让胜率提高了
相同数值。

| 模型人群 | 范围 | 范围内胜率 | 范围外胜率 | 观察差值 | 范围内样本 | 粗略 95% 区间 |
|---|---:|---:|---:|---:|---:|---:|
| 传奇及以上 | 前5 | 50.80% | 49.74% | +1.06 点 | 1,185 | -2.21～+4.33 |
| 传奇及以上 | 前10 | 51.29% | 49.12% | +2.17 点 | 1,979 | -0.69～+5.02 |
| 统帅及以下 | 前5 | 51.13% | 49.61% | +1.52 点 | 1,551 | -1.37～+4.40 |
| 统帅及以下 | 前10 | 51.23% | 49.15% | +2.08 点 | 2,477 | -0.48～+4.64 |
| 全段位 | 前5 | 50.60% | 49.80% | +0.80 点 | 2,840 | -1.33～+2.92 |
| 全段位 | 前10 | 50.50% | 49.64% | +0.85 点 | 4,717 | -1.01～+2.72 |

三个区间都跨过零，因此当前不能把观察差值宣传成已证实的胜率提升。

## 阵容条件推荐能力

这里比较“读取双方已选英雄的阵容模型”和“完全不看阵容、只按该分段常见选择排序”。
前 10 覆盖率表示真实玩家下一手出现在推荐前十中的比例。

| 模型人群 | 轮次 | 阵容模型前10 | 不看阵容榜前10 | 每100次多覆盖 |
|---|---:|---:|---:|---:|
| 传奇及以上 | 第一轮 | 34.09% | 34.34% | -0.25 |
| 传奇及以上 | 第二轮 | 24.69% | 23.52% | +1.17 |
| 传奇及以上 | 第三轮 | 40.57% | 29.09% | +11.48 |
| 统帅及以下 | 第一轮 | 34.11% | 34.11% | +0.00 |
| 统帅及以下 | 第二轮 | 24.23% | 22.96% | +1.28 |
| 统帅及以下 | 第三轮 | 40.83% | 33.76% | +7.07 |
| 全段位 | 第一轮 | 34.16% | 34.16% | +0.00 |
| 全段位 | 第二轮 | 24.72% | 22.96% | +1.75 |
| 全段位 | 第三轮 | 41.65% | 31.59% | +10.06 |

第一轮双方都没有阵容信息，使用常见选择榜是预期行为。第三轮已有公开 4v4 阵容，
阵容模型的优势最明显。

## 怎样验证真实胜率提升

公开比赛只能观测实际选择和结果，无法知道同一局改选另一个英雄会怎样。若要声称
实际胜率从 50% 提升至 53%，需要预先定义的随机对照实验；理想 1:1 分组、95%
显著性和 80% 检验功效下约需 8,700 场完成对局，推荐未被采纳时还需要更多样本。

---

# Dota 2 BP Helper Offline Benchmark (English)

These figures come from the Dota 7.41 models bundled with the app and a
chronologically held-out test set.

## Historical win-rate association (not a causal estimate)

“Inside top N” means the actual final pick appeared among the model's top N
recommendations. Players, roles, and lineup difficulty were not randomly assigned,
so these figures describe historical association and must not be presented as an
equal causal win-rate improvement.

| Model population | Range | Win rate inside | Win rate outside | Observed difference | Inside samples | Approx. 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legend and above | Top 5 | 50.80% | 49.74% | +1.06 points | 1,185 | -2.21 to +4.33 |
| Legend and above | Top 10 | 51.29% | 49.12% | +2.17 points | 1,979 | -0.69 to +5.02 |
| Archon and below | Top 5 | 51.13% | 49.61% | +1.52 points | 1,551 | -1.37 to +4.40 |
| Archon and below | Top 10 | 51.23% | 49.15% | +2.08 points | 2,477 | -0.48 to +4.64 |
| All ranks | Top 5 | 50.60% | 49.80% | +0.80 points | 2,840 | -1.33 to +2.92 |
| All ranks | Top 10 | 50.50% | 49.64% | +0.85 points | 4,717 | -1.01 to +2.72 |

All three intervals cross zero. The current evidence therefore does not establish a
causal win-rate improvement.

## Lineup-conditioned recommendation ability

This compares the lineup-aware model against a list that ignores the lineup and
ranks only the most frequent picks in the same bracket. Hit@10 is the share of actual
next picks found in the first ten recommendations.

| Model population | Round | Lineup model Hit@10 | No-lineup list Hit@10 | Extra hits per 100 |
| --- | ---: | ---: | ---: | ---: |
| Legend and above | 1 | 34.09% | 34.34% | -0.25 |
| Legend and above | 2 | 24.69% | 23.52% | +1.17 |
| Legend and above | 3 | 40.57% | 29.09% | +11.48 |
| Archon and below | 1 | 34.11% | 34.11% | +0.00 |
| Archon and below | 2 | 24.23% | 22.96% | +1.28 |
| Archon and below | 3 | 40.83% | 33.76% | +7.07 |
| All ranks | 1 | 34.16% | 34.16% | +0.00 |
| All ranks | 2 | 24.72% | 22.96% | +1.75 |
| All ranks | 3 | 41.65% | 31.59% | +10.06 |

There is no lineup information in round 1, so using the pick-frequency list is
expected. The lineup-aware model has its clearest advantage in round 3, after a public
4v4 lineup is available.

## How to verify a real win-rate improvement

Public match data observes only the chosen hero and outcome; it cannot reveal what
the same match would have looked like with another pick. A claim that the app raises
win rate from 50% to 53% requires a pre-registered randomized controlled experiment.
With ideal 1:1 assignment, 95% significance and 80% power, roughly 8,700 completed
matches are needed, and more are required when users do not follow recommendations.
