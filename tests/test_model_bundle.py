from __future__ import annotations

import unittest
from pathlib import Path

from d2draft.model_bundle import ModelBundle
from d2draft.recommender import HeroCatalog


ROOT = Path(__file__).resolve().parents[1]


class ModelBundleTest(unittest.TestCase):
    def test_production_bundle_is_compatible_with_catalog(self) -> None:
        catalog = HeroCatalog(ROOT / "data" / "heroes.json")
        bundle = ModelBundle.load(
            ROOT / "artifacts" / "mvp",
            expected_hero_ids=catalog.by_id,
        )
        self.assertEqual(bundle.patch_label, "7.41")
        self.assertEqual(bundle.manifest["hero_count"], 127)
        self.assertEqual(len(bundle.short_hash), 12)
        self.assertTrue(bundle.report)
        self.assertTrue(bundle.backtest)
        self.assertTrue(bundle.outcome_benchmark)
        self.assertEqual(
            bundle.manifest["recommendation_objective"],
            "maximize predicted match win probability",
        )

    def test_rank_specific_bundles_declare_their_training_population(self) -> None:
        catalog = HeroCatalog(ROOT / "data" / "heroes.json")
        expected = {
            "legend_plus": "legend_plus",
            "archon_below": "archon_below",
        }
        for directory, bracket_id in expected.items():
            bundle = ModelBundle.load(
                ROOT / "artifacts" / "models" / directory,
                expected_hero_ids=catalog.by_id,
            )
            policy = bundle.report["policy"]
            outcome = bundle.report["outcome"]
            self.assertEqual(bundle.rank_bracket_id, bracket_id)
            self.assertGreater(int(policy["train_matches"]), 10_000)
            self.assertGreater(int(policy["test_matches"]), 1_000)
            self.assertGreater(int(outcome["train_examples"]), 200_000)


if __name__ == "__main__":
    unittest.main()
