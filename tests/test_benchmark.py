from __future__ import annotations

import json
import unittest
from pathlib import Path

from d2draft.benchmark import approximate_ab_test_matches, policy_benchmark
from d2draft.model_bundle import ModelBundle


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkTest(unittest.TestCase):
    def test_legend_plus_phase_three_lift(self) -> None:
        report = json.loads(
            (
                ROOT / "artifacts" / "models" / "legend_plus" / "backtest.json"
            ).read_text(encoding="utf-8")
        )
        phase_three = policy_benchmark(report)["phase_3"]
        self.assertGreater(phase_three["examples"], 1_000)
        self.assertGreater(float(phase_three["hit_at_10_lift_points"]), 5.0)

    def test_three_point_win_rate_ab_test_needs_thousands(self) -> None:
        matches = approximate_ab_test_matches()
        self.assertGreater(matches, 8_000)
        self.assertLess(matches, 10_000)

    def test_legend_plus_outcome_benchmark(self) -> None:
        bundle = ModelBundle.load(ROOT / "artifacts" / "models" / "legend_plus")
        benchmark = bundle.outcome_benchmark
        metrics = benchmark["outcome_prediction_metrics"]["phase_3"]
        baseline = benchmark["global_hero_winrate_prediction_baseline"]["phase_3"]
        top_five = benchmark["historical_winrate_association"]["phase_3"][
            "outcome_recommender"
        ]["top_5"]
        self.assertGreater(float(metrics["auc"]), float(baseline["auc"]) + 0.02)
        self.assertGreater(top_five["followed_decisions"], 400)
        self.assertEqual(len(top_five["approximate_95_ci_points"]), 2)
        self.assertGreater(float(top_five["approximate_95_ci_points"][0]), 0.0)
        self.assertIn("not causal", benchmark["interpretation"])


if __name__ == "__main__":
    unittest.main()
