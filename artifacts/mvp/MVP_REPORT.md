# Dota 2 BP Helper — Free-data MVP report

> Historical report for the original 1,966-match ordered-BP checkpoint. The current
> production recommender now uses a candidate-conditioned Outcome objective trained
> on 66,515 ordered matches. See `artifacts/models/BENCHMARK.md`; the Policy/Value
> blend below is retained only as development history.

Generated: 2026-08-01

## Outcome

The first offline MVP is feasible, but the current neural network is not uniformly better than simple baselines. The evidence supports a modular hybrid:

- phase 1: global pick-frequency prior;
- phase 2 (2v2 visible): frequency baseline for now;
- phase 3 (4v4 visible): neural policy model is promising;
- final 5v5 value: smoothed individual-hero strength baseline for now.

No paid OpenDota usage was needed for this experiment.

## Data

- 26,193 ranked All Pick matches with complete final 5v5 lineups and outcomes;
- 1,994 downloaded match-detail records;
- 1,966/1,994 (98.6%) strictly reconstruct the 2+2 / 2+2 / 1+1 phases;
- 28 malformed/incomplete phase records excluded;
- 2,000 keyed network attempts, below the 3,000-call daily free allowance;
- raw responses retained in compressed JSONL; normalized records retained in SQLite.

The candidate matches were sampled across seven daily windows. All evaluation splits are chronological: oldest 80% for training, newest 20% for testing.

## Value model

Test set: 5,239 matches. Radiant won 53.56%, so the majority-class accuracy is 53.56%.

| Model | AUC | Accuracy | Log loss | Brier |
|---|---:|---:|---:|---:|
| Smoothed hero-strength baseline | **0.5708** | **55.39%** | **0.6833** | **0.2451** |
| One-hidden-layer neural network | 0.5683 | 54.95% | 0.6874 | 0.2471 |

Across five neural initializations, Value AUC ranged from 0.5666 to 0.5700, with a mean of 0.5680. It did not beat the baseline in any run. The available data contains a real but modest lineup signal; it does not justify using the neural Value model yet.

## Policy model

Training: 1,572 matches / 9,432 conditioned pick targets.  
Testing: 394 matches / 2,364 conditioned pick targets.

The baseline ranks legal heroes by phase-specific training popularity. The neural model consumes ally heroes, enemy heroes and phase, while masking already-visible heroes.

### Phase 2 — 2v2 visible

| Model | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|
| Popularity baseline | **13.13%** | **22.08%** | **0.1009** |
| Neural mean over 5 seeds | 10.63% | 18.83% | — |

The neural model is not justified for phase 2 with this sample size.

### Phase 3 — 4v4 visible

| Model | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|
| Popularity baseline | 14.21% | 26.90% | 0.1043 |
| Neural mean over 5 seeds | **17.06%** | **28.81%** | **0.1210** |

Every tested neural initialization improved phase-3 Hit@5 over the baseline. This is the first evidence that the visible 4v4 composition contains learnable interaction information beyond global pick popularity. The absolute gain is still small and must be validated with more matches.

## Interpretation

“Predict the hero a player actually picked” is not the same as “recommend the uniquely optimal hero”: multiple picks can be reasonable, and the dataset records only one. Hit@K is therefore a policy-signal diagnostic, not product correctness.

Likewise, 55.39% winner accuracy is only 1.83 percentage points above the Radiant-majority baseline. The useful metrics are the chronological AUC/log loss and whether recommendation rankings improve prospective outcomes—not a claim that the system is already an expert drafter.

## Next experiment before spending money

1. Implement the hybrid recommender using the winning component for each phase.
2. Backtest recommendation rankings without additional API calls.
3. Add learning curves using subsets of the existing data.
4. Spend on more ordered BP only if phase-3 validation improves consistently as sample size grows.

Current evidence does not justify consuming the project's $10 budget.

## Runnable hybrid recommender

The unified artifact is `hybrid_model.npz`. The CLI accepts hero IDs or English names and
automatically excludes visible heroes. The components used at runtime are:

| Phase | Policy | Validation-selected Value blend |
|---|---|---:|
| 1 (0v0 visible) | pick popularity | 0.0 |
| 2 (2v2 visible) | phase-2 popularity | 0.1 |
| 3 (4v4 visible) | neural Policy | 0.1 |

The blend weights were selected only on the chronological 80–90% slice. The newest 10%
(197 matches) was retained for a final backtest:

| Phase | Targets | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| 1 | 788 | 19.67% | 32.74% | 0.1539 |
| 2 | 788 | 13.96% | 23.48% | 0.1044 |
| 3 | 394 | 17.51% | 30.46% | 0.1139 |

For phase 3, the pure neural Policy scored 16.75% Hit@5 and 27.41% Hit@10 on this final
slice. The validation-selected 0.1 Value blend increased those to 17.51% and 30.46%,
respectively. This is encouraging but noisy because the final slice contains only 197
matches.

Example:

```powershell
python -m d2draft.recommend_cli `
  --phase 3 `
  --allies "Axe,Crystal Maiden,Juggernaut,Pudge" `
  --enemies "Anti-Mage,Lion,Invoker,Sniper" `
  --top-k 10
```
