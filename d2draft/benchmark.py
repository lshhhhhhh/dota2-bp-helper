from __future__ import annotations

from typing import Any


def policy_benchmark(backtest: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    selected = backtest.get("final_test_selected", {})
    baseline = backtest.get("final_test_policy_baseline", {})
    result: dict[str, dict[str, float | int]] = {}
    for phase in (1, 2, 3):
        name = f"phase_{phase}"
        model = selected.get(name, {})
        base = baseline.get(name, {})
        hit_5 = float(model.get("hit_at_5", 0.0))
        hit_10 = float(model.get("hit_at_10", 0.0))
        baseline_hit_5 = float(base.get("hit_at_5", 0.0))
        baseline_hit_10 = float(base.get("hit_at_10", 0.0))
        result[name] = {
            "examples": int(model.get("examples", 0)),
            "hit_at_5": hit_5,
            "hit_at_10": hit_10,
            "baseline_hit_at_5": baseline_hit_5,
            "baseline_hit_at_10": baseline_hit_10,
            "hit_at_5_lift_points": (hit_5 - baseline_hit_5) * 100.0,
            "hit_at_10_lift_points": (hit_10 - baseline_hit_10) * 100.0,
            "median_rank": float(model.get("median_rank", 0.0)),
        }
    return result


def approximate_ab_test_matches(
    *,
    baseline_win_rate: float = 0.50,
    absolute_lift: float = 0.03,
    z_alpha: float = 1.96,
    z_power: float = 0.84,
) -> int:
    """Approximate total sample for an equal-size two-arm win-rate experiment."""

    treatment = baseline_win_rate + absolute_lift
    pooled = (baseline_win_rate + treatment) / 2.0
    numerator = (
        z_alpha * (2.0 * pooled * (1.0 - pooled)) ** 0.5
        + z_power
        * (
            baseline_win_rate * (1.0 - baseline_win_rate)
            + treatment * (1.0 - treatment)
        )
        ** 0.5
    ) ** 2
    per_group = numerator / (absolute_lift**2)
    return int(round(per_group * 2.0))
