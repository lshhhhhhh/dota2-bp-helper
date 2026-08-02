from __future__ import annotations

import unittest
from pathlib import Path

from d2draft.recommender import HeroCatalog, HybridRecommender
from d2draft.state import DraftState


ROOT = Path(__file__).resolve().parents[1]


class RecommenderIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = HeroCatalog(ROOT / "data" / "heroes.json")
        cls.model = HybridRecommender(
            ROOT / "artifacts" / "mvp" / "hybrid_model.npz", cls.catalog
        )

    def test_name_resolution(self) -> None:
        self.assertEqual(self.catalog.resolve("Anti-Mage"), self.catalog.resolve("antimage"))
        self.assertEqual(self.catalog.resolve("npc_dota_hero_axe"), self.catalog.resolve("Axe"))

    def test_chinese_names_nicknames_and_abbreviations(self) -> None:
        juggernaut = self.catalog.resolve("Juggernaut")
        for alias in ("主宰", "剑圣", "jugg", "jug", "zhuzai"):
            self.assertEqual(self.catalog.resolve(alias), juggernaut)
        self.assertEqual(self.catalog.info(juggernaut).chinese_name, "主宰")
        self.assertEqual(self.catalog.search("剑")[0].hero_id, juggernaut)

    def test_alias_search(self) -> None:
        results = self.catalog.search("剑圣")
        self.assertTrue(results)
        self.assertEqual(results[0].hero_id, self.catalog.resolve("Juggernaut"))

    def test_visible_heroes_are_never_recommended(self) -> None:
        visible = tuple(
            self.catalog.resolve(name)
            for name in ("Axe", "Crystal Maiden", "Juggernaut", "Pudge")
        )
        enemies = tuple(
            self.catalog.resolve(name)
            for name in ("Anti-Mage", "Lion", "Invoker", "Sniper")
        )
        recommendations, kind = self.model.recommend(
            DraftState(phase=3, allies=visible, enemies=enemies), top_k=20, value_blend=0.1
        )
        self.assertEqual(kind, "neural")
        recommended = {item.hero_id for item in recommendations}
        self.assertFalse(recommended & set(visible + enemies))
        self.assertEqual(len(recommendations), 20)

    def test_phase_two_uses_neural_policy(self) -> None:
        allies = (self.catalog.resolve("Axe"), self.catalog.resolve("Crystal Maiden"))
        enemies = (self.catalog.resolve("Anti-Mage"), self.catalog.resolve("Lion"))
        recommendations, kind = self.model.recommend(
            DraftState(phase=2, allies=allies, enemies=enemies), top_k=5, value_blend=0.1
        )
        self.assertEqual(kind, "neural")
        self.assertEqual(len(recommendations), 5)


if __name__ == "__main__":
    unittest.main()
