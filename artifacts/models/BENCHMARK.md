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

### 离线策略评估当前不可用

`off_policy_value` 使用 Policy 作为倾向性模型做自归一化 IPS。当前三套模型上它
都被标记为 `usable: false`：有效样本量只有 19%～21%，且把目标策略换成均匀分布
后估计值只变化 0.006～0.007，说明数字由 `1/μ` 分母主导，与被评估的模型无关。
在 Policy 的 Hit@1 提高之前，不要引用这个估计值。

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

### Off-policy evaluation is currently unusable

`off_policy_value` runs self-normalised IPS with Policy as the propensity model.
It is currently marked `usable: false` for all three models: effective sample size
is only 19–21%, and swapping the target for a uniform policy moves the estimate by
0.006–0.007, showing the number is driven by the `1/mu` denominator rather than by
the model under evaluation. Do not quote it until Policy's Hit@1 improves.
