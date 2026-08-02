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

    def test_legend_plus_historical_advantage_report(self) -> None:
        bundle = ModelBundle.load(ROOT / "artifacts" / "models" / "legend_plus")
        benchmark = bundle.advantage_benchmark
        top_five = benchmark["groups"]["top_5"]
        self.assertGreater(top_five["followed_decisions"], 100)
        self.assertEqual(len(top_five["approximate_95_ci_points"]), 2)
        self.assertIn("not a causal", benchmark["interpretation"])


if __name__ == "__main__":
    unittest.main()
