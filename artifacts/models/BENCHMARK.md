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
