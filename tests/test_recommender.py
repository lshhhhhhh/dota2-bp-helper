from __future__ import annotations

import unittest
from pathlib import Path

from d2draft.recommender import (
    DEFAULT_POLICY_SURPRISE,
    HeroCatalog,
    HybridRecommender,
)
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

    def test_production_recommender_optimizes_outcome(self) -> None:
        self.assertEqual(self.model.objective, "outcome")
        recommendations, _ = self.model.recommend(
            DraftState(phase=1, allies=(), enemies=()), top_k=5, candidate_pool=None
        )
        probabilities = [
            item.predicted_win_probability for item in recommendations
        ]
        self.assertTrue(all(value is not None for value in probabilities))
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))

    def _opposite_round_two_drafts(self) -> tuple[DraftState, DraftState]:
        supports = (self.catalog.resolve("Crystal Maiden"), self.catalog.resolve("Witch Doctor"))
        cores = (self.catalog.resolve("Spectre"), self.catalog.resolve("Anti-Mage"))
        return (
            DraftState(phase=2, allies=supports, enemies=cores),
            DraftState(phase=2, allies=cores, enemies=supports),
        )

    def _top_ids(
        self, state: DraftState, pool: int | None, *, surprise: float = 0.0
    ) -> set[int]:
        recommendations, _ = self.model.recommend(
            state, top_k=5, candidate_pool=pool, policy_surprise=surprise
        )
        return {item.hero_id for item in recommendations}

    def test_without_the_pool_opposite_drafts_get_the_same_list(self) -> None:
        # Ranking every legal hero by win probability is close to a fixed tier list,
        # so two drafts that need opposite things receive identical advice.
        first, second = self._opposite_round_two_drafts()
        self.assertEqual(self._top_ids(first, None), self._top_ids(second, None))

    def test_the_pool_makes_the_list_respond_to_the_draft(self) -> None:
        first, second = self._opposite_round_two_drafts()
        overlap = self._top_ids(first, 20) & self._top_ids(second, 20)
        self.assertLessEqual(len(overlap), 3)

    def test_the_surprise_term_makes_the_list_respond_without_any_pool(self) -> None:
        first, second = self._opposite_round_two_drafts()
        overlap = self._top_ids(first, None, surprise=DEFAULT_POLICY_SURPRISE) & self._top_ids(
            second, None, surprise=DEFAULT_POLICY_SURPRISE
        )
        self.assertLessEqual(len(overlap), 3)

    def test_the_surprise_term_is_inert_in_round_one(self) -> None:
        # With nothing revealed the policy is the marginal, so they cancel exactly.
        empty = DraftState(phase=1, allies=(), enemies=())
        self.assertEqual(
            self._top_ids(empty, None, surprise=0.0),
            self._top_ids(empty, None, surprise=DEFAULT_POLICY_SURPRISE),
        )

    def test_the_surprise_term_ranks_a_clashing_carry_below_the_pool(self) -> None:
        # A second farm-hungry carry behind Spectre is the kind of suggestion that
        # costs a user's trust. Neither mechanism removes it -- the policy only
        # says it is 7x less likely, not impossible -- but the surprise term keeps
        # it further down the list than a candidate pool does.
        enemies = tuple(
            self.catalog.resolve(n)
            for n in ("Phantom Lancer", "Hoodwink", "Necrophos", "Bane")
        )
        allies = tuple(
            self.catalog.resolve(n)
            for n in ("Legion Commander", "Skywrath Mage", "Tusk", "Spectre")
        )
        carry = self.catalog.resolve("Sven")
        state = DraftState(phase=3, allies=allies, enemies=enemies)

        def rank_of(**options) -> int:
            recommendations, _ = self.model.recommend(state, top_k=127, **options)
            return next(i.rank for i in recommendations if i.hero_id == carry)

        pooled = rank_of(candidate_pool=20, policy_surprise=0.0)
        surprised = rank_of(candidate_pool=None, policy_surprise=DEFAULT_POLICY_SURPRISE)
        self.assertGreater(surprised, pooled)

    def test_a_smaller_pool_separates_the_two_drafts_further(self) -> None:
        first, second = self._opposite_round_two_drafts()
        wide = len(self._top_ids(first, 60) & self._top_ids(second, 60))
        narrow = len(self._top_ids(first, 10) & self._top_ids(second, 10))
        self.assertLess(narrow, wide)

    def test_a_pool_wider_than_the_roster_changes_nothing(self) -> None:
        state, _ = self._opposite_round_two_drafts()
        self.assertEqual(self._top_ids(state, 500), self._top_ids(state, None))

    def test_the_list_still_fills_past_the_pool(self) -> None:
        state, _ = self._opposite_round_two_drafts()
        recommendations, _ = self.model.recommend(state, top_k=50, candidate_pool=10)
        self.assertEqual(len(recommendations), 50)
        self.assertEqual(len({item.hero_id for item in recommendations}), 50)

    def test_the_pool_reorders_without_changing_reported_probabilities(self) -> None:
        state, _ = self._opposite_round_two_drafts()
        unfiltered, _ = self.model.recommend(state, top_k=127, candidate_pool=None)
        pooled, _ = self.model.recommend(state, top_k=127, candidate_pool=10)
        self.assertEqual(
            {item.hero_id: item.predicted_win_probability for item in unfiltered},
            {item.hero_id: item.predicted_win_probability for item in pooled},
        )

    def test_recommends_when_one_side_is_a_pick_ahead(self) -> None:
        allies = tuple(
            self.catalog.resolve(name) for name in ("Axe", "Crystal Maiden", "Zeus")
        )
        enemies = tuple(self.catalog.resolve(name) for name in ("Spectre", "Anti-Mage"))
        recommendations, _ = self.model.recommend(
            DraftState.for_next_pick(allies, enemies), top_k=5
        )
        self.assertEqual(len(recommendations), 5)
        self.assertFalse({item.hero_id for item in recommendations} & set(allies + enemies))
        self.assertTrue(
            all(0.0 < item.predicted_win_probability < 1.0 for item in recommendations)
        )

    def test_recommends_against_a_side_that_has_finished_drafting(self) -> None:
        allies = tuple(
            self.catalog.resolve(name)
            for name in ("Axe", "Crystal Maiden", "Zeus", "Tidehunter")
        )
        enemies = tuple(
            self.catalog.resolve(name)
            for name in ("Spectre", "Anti-Mage", "Lich", "Riki", "Lion")
        )
        recommendations, _ = self.model.recommend(
            DraftState.for_next_pick(allies, enemies), top_k=5
        )
        self.assertEqual(len(recommendations), 5)
        self.assertFalse({item.hero_id for item in recommendations} & set(allies + enemies))

    def test_uneven_boards_do_not_inflate_the_win_probability(self) -> None:
        # state_strength is centred, so revealing one more hero per side shifts the
        # estimate by that hero's own strength rather than by a count artefact.
        allies = tuple(self.catalog.resolve(name) for name in ("Crystal Maiden", "Witch Doctor"))
        enemies = tuple(self.catalog.resolve(name) for name in ("Spectre", "Anti-Mage"))
        balanced, _ = self.model.recommend(
            DraftState(phase=2, allies=allies, enemies=enemies), top_k=1
        )
        ahead, _ = self.model.recommend(
            DraftState(phase=2, allies=allies, enemies=enemies[:1]), top_k=1
        )
        self.assertLess(
            abs(
                balanced[0].predicted_win_probability
                - ahead[0].predicted_win_probability
            ),
            0.10,
        )

    def test_the_pool_never_suggests_a_visible_hero(self) -> None:
        state, _ = self._opposite_round_two_drafts()
        recommendations, _ = self.model.recommend(state, top_k=20, candidate_pool=10)
        visible = set(state.allies) | set(state.enemies)
        self.assertFalse({item.hero_id for item in recommendations} & visible)


if __name__ == "__main__":
    unittest.main()
