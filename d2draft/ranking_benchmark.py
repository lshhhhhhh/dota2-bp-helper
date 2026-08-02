"""Measure recommendation ranking quality, which outcome AUC does not capture.

The outcome model scores ``P(win | state, candidate)``. Its state term is constant
across candidates, so it moves AUC without moving the ranking. A model whose
ranking is a fixed tier list can therefore look good on every metric the project
had before this module. Everything here is designed to separate those two things.

Four families, in increasing order of how much they assume:

``state_responsiveness``
    Does the ranking react to the draft at all? Pure diagnostics, no assumptions.
    A state-independent scorer gets zero here by construction.

``same_state_pairwise``
    Both round-3 picks of one match, scored at the *same* state. The state term
    cancels exactly, so this isolates candidate ranking. The hero that won and
    the hero that lost come from the same match, so match-level confounders
    cancel too. Chance is 0.5.

``stratified_association``
    The historical top-K win-rate association, conditioned on state value and
    computed for every method over an identical decision set. The unstratified
    version compares different subsets per method, which is not a comparison.

``off_policy_value``
    Self-normalised inverse propensity scoring with the policy model as the
    behaviour model. This is the closest offline answer to "what would happen if
    users followed the recommendation", and it is still not causal: it assumes
    no unobserved confounding, which player skill violates.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import UTC, datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .outcome import OutcomeEmbeddingModel, OutcomeExamples


RADIANT, DIRE = 0, 1

# Conventional rule of thumb for inverse propensity estimates: below this share of
# effective samples the weights are concentrated on too few decisions to trust.
MINIMUM_EFFECTIVE_FRACTION = 0.3
# Clipping trades variance for bias, and it also inflates effective sample size by
# flattening the largest weights onto one value. Both checks are needed.
MAXIMUM_CLIPPED_FRACTION = 0.05


@dataclass(frozen=True)
class RankingExamples:
    """Outcome examples that remember which match and side each decision is from."""

    examples: OutcomeExamples
    match_index: np.ndarray
    side: np.ndarray

    def __len__(self) -> int:
        return len(self.examples)


def ranking_examples(
    rows: list[sqlite3.Row], hero_to_index: dict[int, int]
) -> RankingExamples:
    records: list[tuple[list[int], list[int], int, int, float, int, int]] = []
    for match_index, row in enumerate(rows):
        picks = {
            f"phase_{phase}_{side}": [
                hero_to_index[int(hero)]
                for hero in json.loads(row[f"phase_{phase}_{side}"])
            ]
            for phase in (1, 2, 3)
            for side in ("radiant", "dire")
        }
        radiant_win = float(row["radiant_win"])
        for side_id, (side, other, won) in enumerate(
            (
                ("radiant", "dire", radiant_win),
                ("dire", "radiant", 1.0 - radiant_win),
            )
        ):
            p1_allies = picks[f"phase_1_{side}"]
            p1_enemies = picks[f"phase_1_{other}"]
            p2_allies = picks[f"phase_2_{side}"]
            p2_enemies = picks[f"phase_2_{other}"]
            for candidate in p1_allies:
                records.append(([], [], candidate, 1, won, match_index, side_id))
            for candidate in p2_allies:
                records.append(
                    (p1_allies, p1_enemies, candidate, 2, won, match_index, side_id)
                )
            for candidate in picks[f"phase_3_{side}"]:
                records.append(
                    (
                        p1_allies + p2_allies,
                        p1_enemies + p2_enemies,
                        candidate,
                        3,
                        won,
                        match_index,
                        side_id,
                    )
                )

    count = len(records)
    allies = np.full((count, 4), -1, dtype=np.int16)
    enemies = np.full((count, 4), -1, dtype=np.int16)
    candidate = np.empty(count, dtype=np.int16)
    phase = np.empty(count, dtype=np.int8)
    outcome = np.empty(count, dtype=np.float32)
    match_index = np.empty(count, dtype=np.int32)
    side = np.empty(count, dtype=np.int8)
    for index, record in enumerate(records):
        own, opposing, action, round_number, won, match, side_id = record
        allies[index, : len(own)] = own
        enemies[index, : len(opposing)] = opposing
        candidate[index] = action
        phase[index] = round_number
        outcome[index] = won
        match_index[index] = match
        side[index] = side_id
    return RankingExamples(
        examples=OutcomeExamples(allies, enemies, candidate, phase, outcome),
        match_index=match_index,
        side=side,
    )


def legal_mask(examples: OutcomeExamples, heroes: int) -> np.ndarray:
    legal = np.ones((len(examples), heroes), dtype=bool)
    rows = np.arange(len(examples))
    for slot in range(4):
        for table in (examples.allies, examples.enemies):
            values = table[:, slot]
            valid = values >= 0
            legal[rows[valid], values[valid]] = False
    return legal


def _rank_transform(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=-1, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    positions = np.arange(values.shape[-1], dtype=np.float64)
    np.put_along_axis(ranks, order, np.broadcast_to(positions, order.shape), axis=-1)
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ra, rb = _rank_transform(a), _rank_transform(b)
    ra = ra - ra.mean(axis=-1, keepdims=True)
    rb = rb - rb.mean(axis=-1, keepdims=True)
    numerator = (ra * rb).sum(axis=-1)
    denominator = np.sqrt((ra**2).sum(axis=-1) * (rb**2).sum(axis=-1))
    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    )


def state_responsiveness(
    scores: np.ndarray, *, top_k: int = 5, sample: int | None = 4000, seed: int = 0
) -> dict[str, Any]:
    """How much the ranking moves with the draft state, ignoring legality.

    Legality alone forces some variation, so it is deliberately excluded here: a
    scorer that only reacts to heroes being unavailable is still a fixed list.
    """

    if len(scores) == 0:
        return {"decisions": 0}
    rows = np.arange(len(scores))
    if sample is not None and len(rows) > sample:
        rows = np.random.default_rng(seed).choice(len(scores), sample, replace=False)
    subset = np.asarray(scores[rows], dtype=np.float64)

    static = subset.mean(axis=0)
    static_order = np.argsort(-static)
    static_top = set(static_order[:top_k].tolist())

    order = np.argsort(-subset, axis=1)
    top = order[:, :top_k]
    changed = sum(1 for row in top if set(row.tolist()) != static_top)

    per_state_rank = _rank_transform(-subset)
    swing = per_state_rank.max(axis=0) - per_state_rank.min(axis=0)

    return {
        "decisions": int(len(subset)),
        "top_k": top_k,
        "distinct_top_1": int(len(np.unique(order[:, 0]))),
        f"distinct_top_{top_k}": int(len(np.unique(top))),
        "heroes": int(subset.shape[1]),
        f"top_{top_k}_change_rate": float(changed / len(subset)),
        "mean_spearman_vs_static": float(
            _spearman(subset, np.broadcast_to(static, subset.shape)).mean()
        ),
        "median_rank_swing": float(np.median(swing)),
        "max_rank_swing": float(swing.max()),
    }


def _phase_3_pairs(rex: RankingExamples) -> np.ndarray:
    """Row index pairs (winning-side decision, losing-side decision) per match."""

    mask = rex.examples.phase == 3
    rows = np.flatnonzero(mask)
    if len(rows) == 0:
        return np.empty((0, 2), dtype=np.int64)
    order = np.lexsort((rex.side[rows], rex.match_index[rows]))
    rows = rows[order]
    matches = rex.match_index[rows]
    outcomes = rex.examples.outcome[rows]

    pairs: list[tuple[int, int]] = []
    index = 0
    while index + 1 < len(rows):
        if matches[index] != matches[index + 1]:
            index += 1
            continue
        first, second = rows[index], rows[index + 1]
        if outcomes[index] > outcomes[index + 1]:
            pairs.append((int(first), int(second)))
        elif outcomes[index + 1] > outcomes[index]:
            pairs.append((int(second), int(first)))
        index += 2
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def same_state_pairwise(
    rex: RankingExamples, scores: np.ndarray, legal: np.ndarray
) -> dict[str, Any]:
    """Prefer the winning hero over the losing hero, both scored at one state.

    Both round-3 picks of a match were made simultaneously from the same public
    4v4, so each is a legal candidate at the other side's state. Scoring the pair
    at a single state cancels the state term exactly, leaving only the candidate
    ranking. Chance is 0.5.
    """

    pairs = _phase_3_pairs(rex)
    if len(pairs) == 0:
        return {"comparisons": 0, "accuracy": None}

    candidates = rex.examples.candidate
    wins: list[float] = []
    for evaluated in (0, 1):
        state_rows = pairs[:, evaluated]
        winning_hero = candidates[pairs[:, 0]].astype(np.int64)
        losing_hero = candidates[pairs[:, 1]].astype(np.int64)
        usable = (
            (winning_hero != losing_hero)
            & legal[state_rows, winning_hero]
            & legal[state_rows, losing_hero]
        )
        if not np.any(usable):
            continue
        rows = state_rows[usable]
        winner_score = scores[rows, winning_hero[usable]]
        loser_score = scores[rows, losing_hero[usable]]
        wins.append(
            np.where(winner_score > loser_score, 1.0, np.where(winner_score < loser_score, 0.0, 0.5))
        )

    if not wins:
        return {"comparisons": 0, "accuracy": None}
    flat = np.concatenate(wins)
    accuracy = float(flat.mean())
    error = math.sqrt(max(accuracy * (1.0 - accuracy), 1e-12) / len(flat))
    return {
        "comparisons": int(len(flat)),
        "matches": int(len(pairs)),
        "accuracy": accuracy,
        "approximate_95_ci": [accuracy - 1.96 * error, accuracy + 1.96 * error],
        "points_above_chance": (accuracy - 0.5) * 100.0,
    }


def _difference(followed: np.ndarray, other: np.ndarray) -> dict[str, Any]:
    if len(followed) == 0 or len(other) == 0:
        return {
            "followed_decisions": int(len(followed)),
            "other_decisions": int(len(other)),
            "observed_difference_points": None,
        }
    a, b = float(followed.mean()), float(other.mean())
    error = math.sqrt(
        a * (1.0 - a) / len(followed) + b * (1.0 - b) / len(other)
    )
    return {
        "followed_decisions": int(len(followed)),
        "followed_win_rate": a,
        "other_decisions": int(len(other)),
        "other_win_rate": b,
        "observed_difference_points": (a - b) * 100.0,
        "approximate_95_ci_points": [
            (a - b - 1.96 * error) * 100.0,
            (a - b + 1.96 * error) * 100.0,
        ],
    }


def stratified_association(
    ranks: np.ndarray,
    outcomes: np.ndarray,
    state_value: np.ndarray,
    *,
    top_k: int = 5,
    strata: int = 10,
) -> dict[str, Any]:
    """Top-K win-rate association, pooled across state-value strata.

    Conditioning on state value removes the "strong lineup wins and also had a
    strong pick available" path. It does not remove player self-selection.
    """

    order = np.argsort(state_value, kind="mergesort")
    bounds = np.array_split(order, min(strata, max(len(order), 1)))
    weighted_difference, total_weight = 0.0, 0.0
    per_stratum: list[dict[str, Any]] = []
    for stratum in bounds:
        if len(stratum) == 0:
            continue
        followed = outcomes[stratum][ranks[stratum] <= top_k]
        other = outcomes[stratum][ranks[stratum] > top_k]
        report = _difference(followed, other)
        per_stratum.append(report)
        if report.get("observed_difference_points") is None:
            continue
        weight = float(len(followed))
        weighted_difference += weight * report["observed_difference_points"]
        total_weight += weight

    overall = _difference(outcomes[ranks <= top_k], outcomes[ranks > top_k])
    return {
        "top_k": top_k,
        "strata": len(per_stratum),
        "unstratified": overall,
        "stratified_difference_points": (
            weighted_difference / total_weight if total_weight > 0 else None
        ),
        "followed_decisions": int(np.sum(ranks <= top_k)),
    }


def _softmax(scores: np.ndarray, legal: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.where(legal, scores.astype(np.float64) / max(temperature, 1e-6), -np.inf)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def off_policy_value(
    target_scores: np.ndarray,
    behavior_scores: np.ndarray,
    chosen: np.ndarray,
    outcomes: np.ndarray,
    legal: np.ndarray,
    *,
    temperature: float = 0.1,
    behavior_temperature: float = 1.0,
    clip: float = 20.0,
    reference: bool = True,
) -> dict[str, Any]:
    """Self-normalised IPS estimate of the win rate under a target ranking policy.

    Assumes no unobserved confounding and that the behaviour model is a usable
    propensity model. Player skill breaks the first.

    ``usable`` gates on effective sample size, the conventional rule of thumb for
    the second. Also reported, because they diagnose *why* an estimate is weak:
    the behaviour model's probability on the observed picks, and the estimate a
    uniform target policy would receive. If the real target and a uniform target
    land in the same place, the ``1/mu`` denominator is driving the number and it
    says nothing about the recommender.
    """

    rows = np.arange(len(chosen))
    target = _softmax(target_scores, legal, temperature)
    behavior = _softmax(behavior_scores, legal, behavior_temperature)
    numerator = target[rows, chosen]
    denominator = behavior[rows, chosen]
    ratio = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    )
    clipped = np.minimum(ratio, clip)
    total = clipped.sum()
    if total <= 0:
        return {"estimate": None, "decisions": int(len(chosen))}
    estimate = float((clipped * outcomes).sum() / total)
    effective = float(total**2 / max((clipped**2).sum(), 1e-12))
    spread = float(np.sqrt(np.sum((clipped * (outcomes - estimate)) ** 2)) / total)

    uniform = 1.0 / np.maximum(legal.sum(axis=1), 1)
    effective_fraction = effective / len(chosen)
    clipped_fraction = float(np.mean(ratio > clip))
    report = {
        "estimate": estimate,
        "approximate_95_ci": [estimate - 1.96 * spread, estimate + 1.96 * spread],
        "decisions": int(len(chosen)),
        "effective_sample_size": effective,
        "effective_fraction": effective_fraction,
        "clipped_fraction": clipped_fraction,
        "behavior_probability_median": float(np.median(denominator)),
        "behavior_below_uniform_fraction": float(np.mean(denominator < uniform)),
        "usable": bool(
            effective_fraction >= MINIMUM_EFFECTIVE_FRACTION
            and clipped_fraction <= MAXIMUM_CLIPPED_FRACTION
        ),
        "temperature": temperature,
        "clip": clip,
    }
    if reference:
        flat = off_policy_value(
            np.zeros_like(target_scores),
            behavior_scores,
            chosen,
            outcomes,
            legal,
            temperature=1.0,
            behavior_temperature=behavior_temperature,
            clip=clip,
            reference=False,
        )
        report["uniform_target_estimate"] = flat["estimate"]
        report["target_sensitivity"] = (
            None
            if flat["estimate"] is None
            else abs(estimate - flat["estimate"])
        )
    return report


def evaluate_method(
    rex: RankingExamples,
    scores: np.ndarray,
    legal: np.ndarray,
    *,
    behavior_scores: np.ndarray | None = None,
    state_value: np.ndarray | None = None,
    phase: int = 3,
    top_k: int = 5,
    temperature: float = 0.1,
) -> dict[str, Any]:
    mask = rex.examples.phase == phase
    rows = np.flatnonzero(mask)
    phase_scores = scores[rows]
    phase_legal = legal[rows]
    chosen = rex.examples.candidate[rows].astype(np.int64)
    outcomes = rex.examples.outcome[rows].astype(np.float64)

    masked = np.where(phase_legal, phase_scores, -np.inf)
    picked = masked[np.arange(len(rows)), chosen]
    ranks = 1 + np.sum(masked > picked[:, None], axis=1)

    report: dict[str, Any] = {
        "phase": phase,
        "decisions": int(len(rows)),
        "state_responsiveness": state_responsiveness(phase_scores, top_k=top_k),
        "same_state_pairwise": same_state_pairwise(rex, scores, legal),
    }
    if state_value is not None:
        report["stratified_association"] = stratified_association(
            ranks, outcomes, state_value[rows], top_k=top_k
        )
    if behavior_scores is not None:
        report["off_policy_value"] = off_policy_value(
            phase_scores,
            behavior_scores[rows],
            chosen,
            outcomes,
            phase_legal,
            temperature=temperature,
        )
    return report


def _batched_scores(
    model: OutcomeEmbeddingModel, examples: OutcomeExamples, heroes: int
) -> np.ndarray:
    scores = np.empty((len(examples), heroes), dtype=np.float32)
    for start in range(0, len(examples), 4096):
        stop = min(start + 4096, len(examples))
        indices = np.arange(start, stop)
        scores[start:stop] = model.score_all_candidates(examples.subset(indices))
    return scores


def run(args: argparse.Namespace) -> dict[str, Any]:
    from .experiment import load_policy_rows
    from .model_bundle import ModelBundle
    from .outcome_benchmark import _policy_scores
    from .recommender import HeroCatalog

    catalog = HeroCatalog(args.heroes)
    bundle = ModelBundle.load(args.model_dir, expected_hero_ids=catalog.by_id)
    with np.load(bundle.artifact_path, allow_pickle=False) as artifact:
        hero_ids = artifact["hero_ids"].astype(np.int64)
        hero_to_index = {int(hero): index for index, hero in enumerate(hero_ids)}
        connection = sqlite3.connect(args.database)
        connection.row_factory = sqlite3.Row
        rows = load_policy_rows(
            connection,
            (args.patch,),
            minimum_rank_tier=args.min_rank_tier,
            maximum_rank_tier_exclusive=args.max_rank_tier_exclusive,
        )
        connection.close()

        test_rows = rows[int(len(rows) * 0.9) :]
        rex = ranking_examples(test_rows, hero_to_index)
        heroes = len(hero_ids)
        model = OutcomeEmbeddingModel.from_artifact(artifact)

        outcome_scores = _batched_scores(model, rex.examples, heroes)
        behavior = _policy_scores(rex.examples, artifact)
        static = np.broadcast_to(
            (artifact["value_weight"][0] * artifact["hero_strength"]).astype(np.float32),
            outcome_scores.shape,
        )

    legal = legal_mask(rex.examples, heroes)
    shuffled = np.random.default_rng(args.seed).normal(
        size=outcome_scores.shape
    ).astype(np.float32)

    # A state value that any scorer can share: how good the public draft looks to
    # the outcome model, averaged over the candidates that are still available.
    masked = np.where(legal, outcome_scores, np.nan)
    state_value = np.nanmean(masked, axis=1)

    methods = {
        "outcome_recommender": outcome_scores,
        "static_hero_strength": static,
        "pick_prediction_policy": behavior,
        "random": shuffled,
    }
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_id": bundle.model_id,
        "rank_bracket": bundle.rank_bracket_label,
        "patch": args.patch,
        "split": "oldest 80% training, next 10% untouched, newest 10% benchmark",
        "test_matches": len(test_rows),
        "test_decisions": len(rex),
        "phase": args.phase,
        "methods": {
            name: evaluate_method(
                rex,
                scores,
                legal,
                behavior_scores=behavior,
                state_value=state_value,
                phase=args.phase,
                top_k=args.top_k,
                temperature=args.temperature,
            )
            for name, scores in methods.items()
        },
        "interpretation": (
            "state_responsiveness and same_state_pairwise are assumption-free "
            "diagnostics of the ranking itself. stratified_association removes the "
            "state-strength path but not player self-selection. off_policy_value "
            "assumes no unobserved confounding, which player skill violates, so "
            "compare policies to each other rather than reading it as a forecast."
        ),
    }

    output = (
        Path(args.output)
        if args.output
        else Path(args.model_dir) / "ranking_benchmark.json"
    )
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Held-out benchmark for recommendation ranking quality"
    )
    parser.add_argument("--database", default="data/collection/draft_matches.sqlite3")
    parser.add_argument("--heroes", default="data/heroes.json")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--patch", default="7.41")
    parser.add_argument("--min-rank-tier", type=int, default=None)
    parser.add_argument("--max-rank-tier-exclusive", type=int, default=None)
    parser.add_argument("--phase", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
