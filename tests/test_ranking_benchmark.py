from __future__ import annotations

import unittest

import numpy as np

from d2draft.ranking_benchmark import (
    evaluate_method,
    legal_mask,
    off_policy_value,
    ranking_examples,
    same_state_pairwise,
    state_responsiveness,
    stratified_association,
)


HEROES = 12


def match_row(
    match_id: int,
    *,
    radiant_win: int,
    radiant: tuple[list[int], list[int], list[int]],
    dire: tuple[list[int], list[int], list[int]],
) -> dict:
    import json

    return {
        "match_id": match_id,
        "radiant_win": radiant_win,
        "phase_1_radiant": json.dumps(radiant[0]),
        "phase_2_radiant": json.dumps(radiant[1]),
        "phase_3_radiant": json.dumps(radiant[2]),
        "phase_1_dire": json.dumps(dire[0]),
        "phase_2_dire": json.dumps(dire[1]),
        "phase_3_dire": json.dumps(dire[2]),
    }


def two_matches() -> list[dict]:
    return [
        match_row(
            1,
            radiant_win=1,
            radiant=([0, 1], [2, 3], [4]),
            dire=([5, 6], [7, 8], [9]),
        ),
        match_row(
            2,
            radiant_win=0,
            radiant=([0, 5], [2, 7], [10]),
            dire=([1, 6], [3, 8], [11]),
        ),
    ]


IDENTITY = {hero: hero for hero in range(HEROES)}


class ExampleConstructionTest(unittest.TestCase):
    def test_examples_keep_match_and_side(self) -> None:
        rex = ranking_examples(two_matches(), IDENTITY)
        self.assertEqual(len(rex), 20)
        self.assertEqual(sorted(set(rex.match_index.tolist())), [0, 1])
        self.assertEqual(sorted(set(rex.side.tolist())), [0, 1])

        phase_3 = rex.examples.phase == 3
        self.assertEqual(int(phase_3.sum()), 4)
        winners = rex.examples.candidate[phase_3 & (rex.examples.outcome == 1)]
        self.assertEqual(sorted(winners.tolist()), [4, 11])

    def test_state_holds_only_earlier_rounds(self) -> None:
        rex = ranking_examples(two_matches(), IDENTITY)
        row = np.flatnonzero(
            (rex.examples.phase == 3)
            & (rex.match_index == 0)
            & (rex.side == 0)
        )[0]
        self.assertEqual(sorted(rex.examples.allies[row].tolist()), [0, 1, 2, 3])
        self.assertEqual(sorted(rex.examples.enemies[row].tolist()), [5, 6, 7, 8])
        self.assertEqual(int(rex.examples.candidate[row]), 4)

    def test_legal_mask_excludes_revealed_heroes(self) -> None:
        rex = ranking_examples(two_matches(), IDENTITY)
        legal = legal_mask(rex.examples, HEROES)
        row = np.flatnonzero(
            (rex.examples.phase == 3) & (rex.match_index == 0) & (rex.side == 0)
        )[0]
        for hero in (0, 1, 2, 3, 5, 6, 7, 8):
            self.assertFalse(legal[row, hero], hero)
        for hero in (4, 9, 10, 11):
            self.assertTrue(legal[row, hero], hero)


class StateResponsivenessTest(unittest.TestCase):
    def test_a_fixed_tier_list_scores_zero(self) -> None:
        fixed = np.linspace(1.0, 0.0, HEROES)
        scores = np.tile(fixed, (500, 1))
        report = state_responsiveness(scores, top_k=5, sample=None)
        self.assertEqual(report["distinct_top_1"], 1)
        self.assertEqual(report["distinct_top_5"], 5)
        self.assertEqual(report["top_5_change_rate"], 0.0)
        self.assertAlmostEqual(report["mean_spearman_vs_static"], 1.0, places=9)
        self.assertEqual(report["max_rank_swing"], 0.0)

    def test_a_state_dependent_ranking_scores_high(self) -> None:
        rng = np.random.default_rng(0)
        scores = rng.normal(size=(500, HEROES))
        report = state_responsiveness(scores, top_k=5, sample=None)
        self.assertGreater(report["distinct_top_1"], 5)
        self.assertGreater(report["top_5_change_rate"], 0.9)
        self.assertLess(abs(report["mean_spearman_vs_static"]), 0.2)
        self.assertGreater(report["max_rank_swing"], 5.0)

    def test_small_perturbation_of_a_tier_list_is_still_nearly_static(self) -> None:
        rng = np.random.default_rng(1)
        fixed = np.linspace(1.0, 0.0, HEROES)
        scores = fixed + 0.001 * rng.normal(size=(500, HEROES))
        report = state_responsiveness(scores, top_k=5, sample=None)
        self.assertGreater(report["mean_spearman_vs_static"], 0.99)
        self.assertEqual(report["distinct_top_5"], 5)


class SameStatePairwiseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rex = ranking_examples(two_matches(), IDENTITY)
        self.legal = legal_mask(self.rex.examples, HEROES)

    def scores_favouring(self, heroes: list[int], amount: float = 5.0) -> np.ndarray:
        scores = np.zeros((len(self.rex), HEROES))
        for hero in heroes:
            scores[:, hero] = amount
        return scores

    def test_perfect_ranker_scores_one(self) -> None:
        # heroes 4 and 11 are the winning side's round-3 picks
        report = same_state_pairwise(self.rex, self.scores_favouring([4, 11]), self.legal)
        self.assertEqual(report["matches"], 2)
        self.assertEqual(report["comparisons"], 4)
        self.assertEqual(report["accuracy"], 1.0)

    def test_inverted_ranker_scores_zero(self) -> None:
        report = same_state_pairwise(self.rex, self.scores_favouring([9, 10]), self.legal)
        self.assertEqual(report["accuracy"], 0.0)

    def test_constant_ranker_scores_chance(self) -> None:
        scores = np.zeros((len(self.rex), HEROES))
        report = same_state_pairwise(self.rex, scores, self.legal)
        self.assertEqual(report["accuracy"], 0.5)

    def test_a_fixed_tier_list_is_evaluated_not_skipped(self) -> None:
        # A static list still gets a verdict; it just cannot beat chance reliably.
        fixed = np.linspace(1.0, 0.0, HEROES)
        report = same_state_pairwise(
            self.rex, np.tile(fixed, (len(self.rex), 1)), self.legal
        )
        self.assertEqual(report["comparisons"], 4)
        self.assertIn(report["accuracy"], (0.0, 0.25, 0.5, 0.75, 1.0))


def stratum(*, win_rate: float, followed: int, size: int = 100) -> np.ndarray:
    """Outcomes where the top-K group and the rest share the stratum win rate."""

    outcomes = np.zeros(size)
    outcomes[: round(win_rate * followed)] = 1.0
    rest = size - followed
    outcomes[followed : followed + round(win_rate * rest)] = 1.0
    return outcomes


class StratifiedAssociationTest(unittest.TestCase):
    def test_confounded_association_collapses_after_stratifying(self) -> None:
        # Inside each stratum the top-5 group wins exactly as often as the rest,
        # so the only association is the strong-state/strong-pick path.
        plan = ((0.0, 0.4, 10), (1.0, 0.6, 90))
        ranks = np.concatenate(
            [np.where(np.arange(100) < followed, 1, 50) for _, _, followed in plan]
        )
        outcomes = np.concatenate(
            [stratum(win_rate=rate, followed=followed) for _, rate, followed in plan]
        )
        state = np.concatenate([np.full(100, value) for value, _, _ in plan])

        report = stratified_association(
            ranks, outcomes, state, top_k=5, strata=2
        )
        self.assertGreater(report["unstratified"]["observed_difference_points"], 10.0)
        self.assertAlmostEqual(report["stratified_difference_points"], 0.0, places=9)


class OffPolicyValueTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.n, self.heroes = 400, 8
        self.behavior = rng.normal(size=(self.n, self.heroes))
        self.legal = np.ones((self.n, self.heroes), dtype=bool)
        probabilities = np.exp(self.behavior) / np.exp(self.behavior).sum(1, keepdims=True)
        self.chosen = np.array(
            [rng.choice(self.heroes, p=row) for row in probabilities]
        )
        self.outcomes = (rng.random(self.n) < 0.5).astype(float)

    def test_target_equal_to_behavior_recovers_observed_win_rate(self) -> None:
        report = off_policy_value(
            self.behavior, self.behavior, self.chosen, self.outcomes, self.legal,
            temperature=1.0, behavior_temperature=1.0,
        )
        self.assertAlmostEqual(report["estimate"], float(self.outcomes.mean()), places=9)
        self.assertAlmostEqual(report["effective_fraction"], 1.0, places=9)
        self.assertEqual(report["clipped_fraction"], 0.0)

    def test_a_policy_favouring_winning_actions_scores_higher(self) -> None:
        target = self.behavior.copy()
        target[np.arange(self.n), self.chosen] += np.where(self.outcomes > 0, 6.0, -6.0)
        report = off_policy_value(
            target, self.behavior, self.chosen, self.outcomes, self.legal,
            temperature=1.0, behavior_temperature=1.0,
        )
        self.assertGreater(report["estimate"], float(self.outcomes.mean()) + 0.2)

    def test_a_well_specified_propensity_model_is_marked_usable(self) -> None:
        report = off_policy_value(
            self.behavior, self.behavior, self.chosen, self.outcomes, self.legal,
            temperature=1.0, behavior_temperature=1.0,
        )
        self.assertTrue(report["usable"])
        self.assertEqual(report["effective_fraction"], 1.0)

    def test_a_propensity_model_that_misses_observed_picks_is_marked_unusable(self) -> None:
        # A behaviour model anti-correlated with what was actually chosen drives
        # most weights past the clip, which biases the estimate even though
        # flattening them onto the clip value keeps effective sample size high.
        broken = -self.behavior * 8.0
        report = off_policy_value(
            self.behavior, broken, self.chosen, self.outcomes, self.legal,
            temperature=1.0, behavior_temperature=1.0,
        )
        self.assertGreater(report["clipped_fraction"], 0.05)
        self.assertFalse(report["usable"])

    def test_reference_estimate_exposes_a_target_that_does_not_matter(self) -> None:
        # Outcomes independent of the action: every policy has the same value, so
        # the target moves the estimate by nothing.
        report = off_policy_value(
            self.behavior, self.behavior, self.chosen, self.outcomes, self.legal,
            temperature=1.0, behavior_temperature=1.0,
        )
        self.assertIsNotNone(report["uniform_target_estimate"])
        self.assertLess(report["target_sensitivity"], 0.1)

    def test_effective_sample_size_falls_as_the_policy_sharpens(self) -> None:
        target = np.zeros_like(self.behavior)
        target[:, 0] = 1.0
        loose = off_policy_value(
            target, self.behavior, self.chosen, self.outcomes, self.legal,
            temperature=1.0,
        )
        sharp = off_policy_value(
            target, self.behavior, self.chosen, self.outcomes, self.legal,
            temperature=0.01,
        )
        self.assertLess(sharp["effective_fraction"], loose["effective_fraction"])
        self.assertLessEqual(sharp["effective_fraction"], 1.0)


class EvaluateMethodTest(unittest.TestCase):
    def test_report_covers_every_family(self) -> None:
        rex = ranking_examples(two_matches(), IDENTITY)
        legal = legal_mask(rex.examples, HEROES)
        rng = np.random.default_rng(11)
        scores = rng.normal(size=(len(rex), HEROES))
        behavior = rng.normal(size=(len(rex), HEROES))
        state_value = rng.normal(size=len(rex))

        report = evaluate_method(
            rex, scores, legal,
            behavior_scores=behavior, state_value=state_value, phase=3,
        )
        self.assertEqual(report["decisions"], 4)
        for key in (
            "state_responsiveness",
            "same_state_pairwise",
            "stratified_association",
            "off_policy_value",
        ):
            self.assertIn(key, report)


if __name__ == "__main__":
    unittest.main()
