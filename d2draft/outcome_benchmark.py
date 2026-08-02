from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .experiment import load_policy_rows
from .metrics import binary_metrics
from .model_bundle import ModelBundle
from .outcome import OutcomeEmbeddingModel, OutcomeExamples, outcome_examples
from .recommender import HeroCatalog


def _association(ranks: np.ndarray, outcomes: np.ndarray, top_k: int) -> dict[str, Any]:
    followed = outcomes[ranks <= top_k]
    other = outcomes[ranks > top_k]
    if len(followed) == 0 or len(other) == 0:
        return {
            "top_k": top_k,
            "followed_decisions": int(len(followed)),
            "other_decisions": int(len(other)),
            "observed_difference_points": None,
            "approximate_95_ci_points": None,
        }
    followed_rate = float(followed.mean())
    other_rate = float(other.mean())
    difference = followed_rate - other_rate
    standard_error = math.sqrt(
        followed_rate * (1.0 - followed_rate) / len(followed)
        + other_rate * (1.0 - other_rate) / len(other)
    )
    return {
        "top_k": top_k,
        "followed_decisions": int(len(followed)),
        "followed_win_rate": followed_rate,
        "other_decisions": int(len(other)),
        "other_win_rate": other_rate,
        "observed_difference_points": difference * 100.0,
        "approximate_95_ci_points": [
            (difference - 1.96 * standard_error) * 100.0,
            (difference + 1.96 * standard_error) * 100.0,
        ],
    }


def _legal_mask(examples: OutcomeExamples, heroes: int) -> np.ndarray:
    legal = np.ones((len(examples), heroes), dtype=bool)
    rows = np.arange(len(examples))
    for slot in range(4):
        ally = examples.allies[:, slot]
        valid = ally >= 0
        legal[rows[valid], ally[valid]] = False
        enemy = examples.enemies[:, slot]
        valid = enemy >= 0
        legal[rows[valid], enemy[valid]] = False
    return legal


def _ranks(scores: np.ndarray, examples: OutcomeExamples, legal: np.ndarray) -> np.ndarray:
    masked = np.where(legal, scores, -np.inf)
    chosen = masked[np.arange(len(examples)), examples.candidate]
    return 1 + np.sum(masked > chosen[:, None], axis=1)


def _policy_scores(examples: OutcomeExamples, artifact: Any) -> np.ndarray:
    hero_ids = artifact["hero_ids"]
    heroes = len(hero_ids)
    scores = np.empty((len(examples), heroes), dtype=np.float32)
    phase_one = examples.phase == 1
    scores[phase_one] = np.log(artifact["phase_frequency"][0])
    contextual = ~phase_one
    if np.any(contextual):
        selected = np.flatnonzero(contextual)
        x = np.zeros((len(selected), heroes * 2 + 2), dtype=np.float32)
        own = examples.allies[selected]
        opposing = examples.enemies[selected]
        rows = np.arange(len(selected))
        for slot in range(4):
            values = own[:, slot]
            valid = values >= 0
            x[rows[valid], values[valid]] = 1.0
            values = opposing[:, slot]
            valid = values >= 0
            x[rows[valid], heroes + values[valid]] = 1.0
        phase_columns = heroes * 2 + examples.phase[selected].astype(np.int64) - 2
        x[rows, phase_columns] = 1.0
        hidden = np.maximum(0.0, x @ artifact["policy_w1"] + artifact["policy_b1"])
        scores[selected] = hidden @ artifact["policy_w2"] + artifact["policy_b2"]
    return scores


def _method_report(
    ranks: np.ndarray, outcomes: np.ndarray
) -> dict[str, dict[str, Any]]:
    return {
        f"top_{top_k}": _association(ranks, outcomes, top_k)
        for top_k in (1, 5, 10)
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    catalog = HeroCatalog(args.heroes)
    bundle = ModelBundle.load(args.model_dir, expected_hero_ids=catalog.by_id)
    with np.load(bundle.artifact_path, allow_pickle=False) as artifact:
        if "outcome_candidate_bias" not in artifact.files:
            raise ValueError("model bundle does not contain an outcome recommender")
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
        examples = outcome_examples(test_rows, hero_to_index)
        model = OutcomeEmbeddingModel.from_artifact(artifact)

        outcome_scores: list[np.ndarray] = []
        policy_scores: list[np.ndarray] = []
        for start in range(0, len(examples), 4096):
            indices = np.arange(start, min(start + 4096, len(examples)))
            batch = examples.subset(indices)
            outcome_scores.append(model.score_all_candidates(batch))
            policy_scores.append(_policy_scores(batch, artifact))
        q_scores = np.concatenate(outcome_scores, axis=0)
        behavior_scores = np.concatenate(policy_scores, axis=0)
        hero_strength_scores = np.broadcast_to(
            artifact["value_weight"][0] * artifact["hero_strength"],
            q_scores.shape,
        )
        baseline_bias = float(artifact["value_bias"][0])
        baseline_weight = float(artifact["value_weight"][0])
        baseline_strength = artifact["hero_strength"].astype(np.float64)

    legal = _legal_mask(examples, len(hero_ids))
    q_ranks = _ranks(q_scores, examples, legal)
    policy_ranks = _ranks(behavior_scores, examples, legal)
    strength_ranks = _ranks(hero_strength_scores, examples, legal)
    selected_probability = q_scores[
        np.arange(len(examples)), examples.candidate
    ]
    baseline_selected_probability = 1.0 / (
        1.0
        + np.exp(
            -(
                baseline_bias
                + baseline_weight * baseline_strength[examples.candidate]
            )
        )
    )
    best_probability = np.where(legal, q_scores, -np.inf).max(axis=1)

    metrics = {
        "overall": binary_metrics(examples.outcome, selected_probability),
        **{
            f"phase_{phase}": binary_metrics(
                examples.outcome[examples.phase == phase],
                selected_probability[examples.phase == phase],
            )
            for phase in (1, 2, 3)
        },
    }
    baseline_metrics = {
        "overall": binary_metrics(
            examples.outcome, baseline_selected_probability
        ),
        **{
            f"phase_{phase}": binary_metrics(
                examples.outcome[examples.phase == phase],
                baseline_selected_probability[examples.phase == phase],
            )
            for phase in (1, 2, 3)
        },
    }
    association = {}
    for phase in (1, 2, 3):
        mask = examples.phase == phase
        association[f"phase_{phase}"] = {
            "outcome_recommender": _method_report(
                q_ranks[mask], examples.outcome[mask]
            ),
            "pick_prediction_baseline": _method_report(
                policy_ranks[mask], examples.outcome[mask]
            ),
            "global_hero_winrate_baseline": _method_report(
                strength_ranks[mask], examples.outcome[mask]
            ),
        }

    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_id": bundle.model_id,
        "rank_bracket": bundle.rank_bracket_label,
        "patch": args.patch,
        "split": "oldest 80% training, next 10% untouched, newest 10% benchmark",
        "test_matches": len(test_rows),
        "test_decisions": len(examples),
        "objective": "maximize predicted win probability for the candidate pick",
        "outcome_prediction_metrics": metrics,
        "global_hero_winrate_prediction_baseline": baseline_metrics,
        "lineup_information_auc_gain": {
            f"phase_{phase}": float(
                metrics[f"phase_{phase}"]["auc"]
                - baseline_metrics[f"phase_{phase}"]["auc"]
            )
            for phase in (1, 2, 3)
        },
        "historical_winrate_association": association,
        "model_estimated_advantage_points": {
            f"phase_{phase}": float(
                100.0
                * np.mean(
                    best_probability[examples.phase == phase]
                    - selected_probability[examples.phase == phase]
                )
            )
            for phase in (1, 2, 3)
        },
        "interpretation": (
            "Outcome AUC/log loss/calibration are genuine held-out predictive metrics. "
            "Top-K win-rate differences are observational associations, not causal "
            "effects. Model-estimated advantage is a counterfactual projection from the "
            "same Q model and must not be presented as measured win-rate lift."
        ),
    }
    output = (
        Path(args.output)
        if args.output
        else Path(args.model_dir) / "outcome_benchmark.json"
    )
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Held-out benchmark for the outcome-driven recommender"
    )
    parser.add_argument("--database", default="data/collection/draft_matches.sqlite3")
    parser.add_argument("--heroes", default="data/heroes.json")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--patch", default="7.41")
    parser.add_argument("--min-rank-tier", type=int, default=None)
    parser.add_argument("--max-rank-tier-exclusive", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
