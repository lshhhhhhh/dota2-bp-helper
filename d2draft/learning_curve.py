from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .experiment import (
    PolicyMLP,
    all_hero_ids,
    load_value_rows,
    policy_examples,
    ranking_metrics,
    softmax,
)


PHASE_FIELDS = (
    "phase_1_radiant",
    "phase_1_dire",
    "phase_2_radiant",
    "phase_2_dire",
    "phase_3_radiant",
    "phase_3_dire",
)


def load_rows(
    connection: sqlite3.Connection,
    *,
    patch: str,
    minimum_rank_tier: int | None,
    maximum_rank_tier_exclusive: int | None,
) -> list[sqlite3.Row]:
    conditions = ["reconstructable = 1", "canonical_patch = ?"]
    parameters: list[object] = [patch]
    if minimum_rank_tier is not None:
        conditions.append("avg_rank_tier >= ?")
        parameters.append(minimum_rank_tier)
    if maximum_rank_tier_exclusive is not None:
        conditions.append("avg_rank_tier < ?")
        parameters.append(maximum_rank_tier_exclusive)
    return connection.execute(
        f"""
        SELECT match_id, start_time, radiant_win, {", ".join(PHASE_FIELDS)}
        FROM matches
        WHERE {" AND ".join(conditions)}
        ORDER BY start_time, match_id
        """,
        tuple(parameters),
    ).fetchall()


def automatic_sizes(pool_size: int) -> list[int]:
    candidates = [
        1000,
        2000,
        4000,
        6000,
        8000,
        10000,
        15000,
        20000,
        25000,
        30000,
        35000,
        40000,
    ]
    sizes = [size for size in candidates if size < pool_size]
    sizes.append(pool_size)
    return sorted(set(sizes))


def negative_log_likelihood(logits: np.ndarray, target: np.ndarray) -> float:
    probability = softmax(logits)
    selected = probability[np.arange(len(target)), target]
    return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())


def win_association(
    logits: np.ndarray,
    target: np.ndarray,
    phase: np.ndarray,
    outcomes: np.ndarray,
    *,
    top_k: int,
) -> dict[str, Any]:
    phase_three = phase == 3
    phase_logits = logits[phase_three]
    phase_target = target[phase_three]
    order = np.argsort(-phase_logits, axis=1)
    positions = np.argmax(order == phase_target[:, None], axis=1) + 1
    followed = outcomes[positions <= top_k]
    other = outcomes[positions > top_k]
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


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def load_or_create_split_state(
    path: Path,
    *,
    patch: str,
    brackets: list[tuple[str, list[sqlite3.Row]]],
    holdout_fraction: float,
    subset_seed: int,
) -> dict[str, Any]:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("patch") != patch:
            raise ValueError(
                f"split state patch {state.get('patch')} does not match requested {patch}"
            )
    else:
        state = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "patch": patch,
            "holdout_fraction": holdout_fraction,
            "subset_seed": subset_seed,
            "brackets": {},
        }

    changed = not path.exists()
    for bracket_index, (label, rows) in enumerate(brackets):
        current_ids = [int(row["match_id"]) for row in rows]
        current_set = set(current_ids)
        saved = state["brackets"].get(label)
        if saved is None:
            cut = int(len(rows) * (1.0 - holdout_fraction))
            old_train_ids = current_ids[:cut]
            rng = np.random.default_rng(subset_seed + bracket_index)
            training_order = [
                old_train_ids[int(index)] for index in rng.permutation(len(old_train_ids))
            ]
            saved = {
                "holdout_match_ids": current_ids[cut:],
                "training_order_match_ids": training_order,
            }
            state["brackets"][label] = saved
            changed = True

        holdout_ids = {int(value) for value in saved["holdout_match_ids"]}
        missing_holdout = holdout_ids - current_set
        if missing_holdout:
            raise ValueError(
                f"fixed {label} holdout lost {len(missing_holdout)} match IDs"
            )
        training_order = [int(value) for value in saved["training_order_match_ids"]]
        known = holdout_ids | set(training_order)
        additions = [match_id for match_id in current_ids if match_id not in known]
        if additions:
            training_order.extend(additions)
            saved["training_order_match_ids"] = training_order
            saved["last_appended_at_utc"] = datetime.now(UTC).isoformat()
            saved["last_appended_matches"] = len(additions)
            changed = True

    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def convergence_assessment(points: list[dict[str, Any]]) -> dict[str, Any]:
    increments: list[dict[str, Any]] = []
    for previous, current in zip(points, points[1:]):
        previous_runs = {
            run["seed"]: run["phase_3_hit_at_10"] for run in previous["runs"]
        }
        paired = [
            (run["phase_3_hit_at_10"] - previous_runs[run["seed"]]) * 100.0
            for run in current["runs"]
        ]
        delta_matches = current["train_matches"] - previous["train_matches"]
        mean = float(np.mean(paired))
        standard_deviation = float(np.std(paired, ddof=1)) if len(paired) > 1 else 0.0
        standard_error = standard_deviation / math.sqrt(len(paired)) if paired else 0.0
        normalized = mean * 2000.0 / delta_matches
        increments.append(
            {
                "from_matches": previous["train_matches"],
                "to_matches": current["train_matches"],
                "hit_at_10_change_points": mean,
                "change_per_2000_matches_points": normalized,
                "paired_seed_approximate_95_ci_points": [
                    mean - 1.96 * standard_error,
                    mean + 1.96 * standard_error,
                ],
            }
        )

    # Tiny remainder steps (for example 8,000 -> 8,054) are useful to retain in
    # the report, but are too small for a stable plateau decision.
    meaningful = [
        item
        for item in increments
        if item["to_matches"] - item["from_matches"] >= 1000
    ]
    recent = meaningful[-2:]
    plateau = len(recent) == 2 and all(
        abs(item["change_per_2000_matches_points"]) < 0.5
        and item["paired_seed_approximate_95_ci_points"][0] <= 0.0
        <= item["paired_seed_approximate_95_ci_points"][1]
        for item in recent
    )
    return {
        "status": "practical_plateau" if plateau else "not_yet_demonstrated",
        "rule": (
            "Practical plateau requires the last two additions to change phase-3 "
            "Hit@10 by less than 0.5 percentage point per 2,000 matches, with each "
            "paired-seed approximate 95% interval including zero."
        ),
        "increments": increments,
        "minimum_increment_matches_for_assessment": 1000,
    }


def run_bracket(
    rows: list[sqlite3.Row],
    *,
    label: str,
    hero_to_index: dict[int, int],
    training_order_ids: list[int],
    holdout_ids: list[int],
    model_seeds: list[int],
    requested_sizes: list[int] | None,
    policy_hidden: int,
    policy_epochs: int,
    cached_points: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    by_id = {int(row["match_id"]): row for row in rows}
    train_pool = [by_id[int(match_id)] for match_id in training_order_ids]
    holdout_set = {int(match_id) for match_id in holdout_ids}
    test_rows = [row for row in rows if int(row["match_id"]) in holdout_set]
    sizes = automatic_sizes(len(train_pool)) if requested_sizes is None else [
        size for size in requested_sizes if 0 < size <= len(train_pool)
    ]
    if len(train_pool) not in sizes:
        sizes.append(len(train_pool))
    sizes = sorted(set(sizes))

    test_x, test_used, test_target, test_phase = policy_examples(test_rows, hero_to_index)
    # policy_examples emits phase-three Radiant then Dire for every match.
    outcomes = np.asarray(
        [
            outcome
            for row in test_rows
            for outcome in (float(row["radiant_win"]), 1.0 - float(row["radiant_win"]))
        ],
        dtype=np.float32,
    )

    points: list[dict[str, Any]] = []
    for size in sizes:
        if cached_points and size in cached_points:
            points.append(cached_points[size])
            continue
        subset = train_pool[:size]
        train_x, train_used, train_target, _ = policy_examples(subset, hero_to_index)
        seed_runs: list[dict[str, Any]] = []
        for seed in model_seeds:
            rng = np.random.default_rng(seed)
            model = PolicyMLP.create(
                train_x.shape[1], policy_hidden, len(hero_to_index), rng
            )
            model.fit(
                train_x,
                train_used,
                train_target,
                epochs=policy_epochs,
                batch_size=256,
                learning_rate=1e-3,
                l2=1e-4,
                rng=rng,
            )
            logits = model.logits(test_x, test_used)
            metrics = ranking_metrics(logits, test_target, test_phase)
            association = win_association(
                logits, test_target, test_phase, outcomes, top_k=5
            )
            seed_runs.append(
                {
                    "seed": seed,
                    "phase_2_hit_at_10": metrics["phase_2"]["hit_at_10"],
                    "phase_3_hit_at_10": metrics["phase_3"]["hit_at_10"],
                    "phase_3_mrr": metrics["phase_3"]["mrr"],
                    "negative_log_likelihood": negative_log_likelihood(
                        logits, test_target
                    ),
                    "top_5_observed_win_difference_points": association[
                        "observed_difference_points"
                    ],
                    "top_5_win_association": association,
                }
            )
        points.append(
            {
                "train_matches": size,
                "train_examples": len(train_target),
                "runs": seed_runs,
                "phase_2_hit_at_10": summarize(
                    [run["phase_2_hit_at_10"] for run in seed_runs]
                ),
                "phase_3_hit_at_10": summarize(
                    [run["phase_3_hit_at_10"] for run in seed_runs]
                ),
                "phase_3_mrr": summarize([run["phase_3_mrr"] for run in seed_runs]),
                "negative_log_likelihood": summarize(
                    [run["negative_log_likelihood"] for run in seed_runs]
                ),
                "top_5_observed_win_difference_points": summarize(
                    [
                        run["top_5_observed_win_difference_points"]
                        for run in seed_runs
                        if run["top_5_observed_win_difference_points"] is not None
                    ]
                ),
            }
        )

    return {
        "label": label,
        "available_matches": len(rows),
        "train_pool_matches": len(train_pool),
        "fixed_newest_holdout_matches": len(test_rows),
        "fixed_newest_holdout_start_time": int(test_rows[0]["start_time"]),
        "fixed_newest_holdout_end_time": int(test_rows[-1]["start_time"]),
        "points": points,
        "convergence": convergence_assessment(points),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    connection = sqlite3.connect(args.database)
    connection.row_factory = sqlite3.Row
    heroes = all_hero_ids(load_value_rows(connection))
    hero_to_index = {hero: index for index, hero in enumerate(heroes)}
    brackets = [
        (
            "legend_plus",
            load_rows(
                connection,
                patch=args.patch,
                minimum_rank_tier=50,
                maximum_rank_tier_exclusive=None,
            ),
        ),
        (
            "archon_below",
            load_rows(
                connection,
                patch=args.patch,
                minimum_rank_tier=None,
                maximum_rank_tier_exclusive=50,
            ),
        ),
    ]
    connection.close()
    requested_sizes = (
        [int(value) for value in args.sizes.split(",") if value.strip()]
        if args.sizes
        else None
    )
    model_seeds = [args.seed + offset for offset in range(args.seeds)]
    output = Path(args.output)
    previous_report: dict[str, Any] | None = None
    if output.exists() and not args.no_cache:
        candidate = json.loads(output.read_text(encoding="utf-8"))
        method = candidate.get("method", {})
        if (
            candidate.get("patch") == args.patch
            and method.get("model_seeds") == model_seeds
            and method.get("policy_epochs") == args.policy_epochs
        ):
            previous_report = candidate
    split_state_path = Path(args.split_state)
    split_state = load_or_create_split_state(
        split_state_path,
        patch=args.patch,
        brackets=brackets,
        holdout_fraction=args.holdout_fraction,
        subset_seed=args.subset_seed,
    )
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "database": str(Path(args.database).resolve()),
        "patch": args.patch,
        "hero_count": len(heroes),
        "method": {
            "holdout": (
                f"newest {args.holdout_fraction:.0%} per rank bracket; fixed for every point"
            ),
            "training_subsets": (
                "persistent nested training order; newly collected matches append without "
                "changing earlier checkpoints"
            ),
            "split_state": str(split_state_path.resolve()),
            "model_seeds": model_seeds,
            "policy_epochs": args.policy_epochs,
            "primary_metric": "phase_3_hit_at_10",
            "important_caveat": (
                "Historical win-rate association is observational and is not a causal "
                "estimate of the advantage produced by following recommendations."
            ),
        },
        "brackets": {},
    }
    for label, rows in brackets:
        if len(rows) < 100:
            raise ValueError(f"insufficient {label} rows: {len(rows)}")
        saved_split = split_state["brackets"][label]
        cached_points = None
        if previous_report is not None:
            old_bracket = previous_report.get("brackets", {}).get(label, {})
            cached_points = {
                int(point["train_matches"]): point
                for point in old_bracket.get("points", [])
            }
        report["brackets"][label] = run_bracket(
            rows,
            label=label,
            hero_to_index=hero_to_index,
            training_order_ids=saved_split["training_order_match_ids"],
            holdout_ids=saved_split["holdout_match_ids"],
            model_seeds=model_seeds,
            requested_sizes=requested_sizes,
            policy_hidden=args.policy_hidden,
            policy_epochs=args.policy_epochs,
            cached_points=cached_points,
        )
    report["elapsed_seconds"] = time.monotonic() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure fixed-holdout policy-model learning curves by rank bracket"
    )
    parser.add_argument("--database", default="data/collection/draft_matches.sqlite3")
    parser.add_argument(
        "--output", default="artifacts/learning_curve/learning_curve.json"
    )
    parser.add_argument("--patch", default="7.41")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--subset-seed", type=int, default=4171)
    parser.add_argument(
        "--split-state",
        default="artifacts/learning_curve/fixed_split.json",
        help="persistent holdout IDs and nested training order",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--sizes", default=None, help="comma-separated training sizes")
    parser.add_argument("--policy-hidden", type=int, default=96)
    parser.add_argument("--policy-epochs", type=int, default=35)
    parser.add_argument(
        "--no-cache", action="store_true", help="retrain unchanged curve checkpoints"
    )
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
