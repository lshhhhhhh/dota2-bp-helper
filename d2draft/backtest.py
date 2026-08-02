from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .experiment import load_policy_rows, policy_examples
from .patches import install_patch_schema


def row_zscore(values: np.ndarray, legal: np.ndarray) -> np.ndarray:
    output = np.full_like(values, -1e9, dtype=np.float64)
    for row in range(len(values)):
        selected = values[row, legal[row]]
        output[row, legal[row]] = (selected - selected.mean()) / max(float(selected.std()), 1e-6)
    return output


def phase_one_targets(rows: list[sqlite3.Row], hero_to_index: dict[int, int]) -> np.ndarray:
    result: list[int] = []
    for row in rows:
        for field in ("phase_1_radiant", "phase_1_dire"):
            result.extend(hero_to_index[int(hero)] for hero in json.loads(row[field]))
    return np.asarray(result, dtype=np.int64)


def rank_summary(logits: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    order = np.argsort(-logits, axis=1)
    rank = np.argmax(order == target[:, None], axis=1) + 1
    return {
        "examples": int(len(rank)),
        "hit_at_5": float(np.mean(rank <= 5)),
        "hit_at_10": float(np.mean(rank <= 10)),
        "mrr": float(np.mean(1.0 / rank)),
        "median_rank": float(np.median(rank)),
    }


def score_split(
    rows: list[sqlite3.Row],
    *,
    hero_to_index: dict[int, int],
    artifact: Any,
    blends: list[float],
) -> dict[str, dict[str, Any]]:
    heroes = len(hero_to_index)
    frequency = artifact["phase_frequency"].astype(np.float64)
    value_delta = float(artifact["value_weight"][0]) * artifact["hero_strength"].astype(
        np.float64
    )

    result: dict[str, dict[str, Any]] = {"phase_1": {}, "phase_2": {}, "phase_3": {}}
    p1_target = phase_one_targets(rows, hero_to_index)
    p1_policy = np.broadcast_to(np.log(frequency[0]), (len(p1_target), heroes)).copy()
    p1_legal = np.ones_like(p1_policy, dtype=bool)
    p1_policy_z = row_zscore(p1_policy, p1_legal)
    p1_value_z = row_zscore(np.broadcast_to(value_delta, p1_policy.shape), p1_legal)
    for blend in blends:
        result["phase_1"][str(blend)] = rank_summary(
            p1_policy_z + blend * p1_value_z, p1_target
        )

    x, used, target, phase = policy_examples(rows, hero_to_index)
    hidden = np.maximum(
        0.0,
        x @ artifact["policy_w1"] + artifact["policy_b1"],
    )
    policy_logits = hidden @ artifact["policy_w2"] + artifact["policy_b2"]
    policy_logits = np.where(used, -1e9, policy_logits)
    policy_z = row_zscore(policy_logits, ~used)
    value_z = row_zscore(np.broadcast_to(value_delta, policy_logits.shape), ~used)
    for phase_number in (2, 3):
        mask = phase == phase_number
        for blend in blends:
            combined = np.where(used, -1e9, policy_z + blend * value_z)
            result[f"phase_{phase_number}"][str(blend)] = rank_summary(
                combined[mask], target[mask]
            )
    return result


def policy_baseline_split(
    rows: list[sqlite3.Row], *, hero_to_index: dict[int, int], artifact: Any
) -> dict[str, Any]:
    """Score the phase-frequency baseline on the exact same held-out rows."""

    heroes = len(hero_to_index)
    frequency = artifact["phase_frequency"].astype(np.float64)
    p1_target = phase_one_targets(rows, hero_to_index)
    p1_logits = np.broadcast_to(np.log(frequency[0]), (len(p1_target), heroes)).copy()
    result: dict[str, Any] = {"phase_1": rank_summary(p1_logits, p1_target)}
    _, used, target, phase = policy_examples(rows, hero_to_index)
    logits = np.stack([np.log(frequency[int(number) - 1]) for number in phase])
    logits = np.where(used, -1e9, logits)
    for phase_number in (2, 3):
        mask = phase == phase_number
        result[f"phase_{phase_number}"] = rank_summary(logits[mask], target[mask])
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    connection = sqlite3.connect(args.database)
    connection.row_factory = sqlite3.Row
    install_patch_schema(connection)
    patches = tuple(args.patch or ())
    policy_rows = load_policy_rows(
        connection,
        patches,
        minimum_rank_tier=args.min_rank_tier,
        maximum_rank_tier_exclusive=args.max_rank_tier_exclusive,
    )
    if len(policy_rows) < 20:
        raise ValueError(f"insufficient policy rows for patch filter: {patches or 'all'}")
    artifact = np.load(args.model)
    hero_ids = artifact["hero_ids"].astype(np.int64)
    hero_to_index = {int(hero): index for index, hero in enumerate(hero_ids)}
    validation_rows = policy_rows[int(len(policy_rows) * 0.8) : int(len(policy_rows) * 0.9)]
    test_rows = policy_rows[int(len(policy_rows) * 0.9) :]
    validation = score_split(
        validation_rows,
        hero_to_index=hero_to_index,
        artifact=artifact,
        blends=args.blends,
    )
    final_test = score_split(
        test_rows,
        hero_to_index=hero_to_index,
        artifact=artifact,
        blends=args.blends,
    )
    validation_baseline = policy_baseline_split(
        validation_rows, hero_to_index=hero_to_index, artifact=artifact
    )
    final_test_baseline = policy_baseline_split(
        test_rows, hero_to_index=hero_to_index, artifact=artifact
    )
    selected: dict[str, float] = {}
    selected_test: dict[str, Any] = {}
    for phase_name in ("phase_1", "phase_2", "phase_3"):
        # Select only on validation MRR; ties prefer the smaller Value contribution.
        best = max(
            args.blends,
            key=lambda blend: (validation[phase_name][str(blend)]["mrr"], -blend),
        )
        selected[phase_name] = float(best)
        selected_test[phase_name] = final_test[phase_name][str(best)]

    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "split": "oldest 80% model training, next 10% blend selection, newest 10% final test",
        "validation_matches": len(validation_rows),
        "test_matches": len(test_rows),
        "patch_filter": list(patches) if patches else "all",
        "rank_filter": {
            "minimum_avg_rank_tier": args.min_rank_tier,
            "maximum_avg_rank_tier_exclusive": args.max_rank_tier_exclusive,
        },
        "interpretation": (
            "Hit@K measures agreement with an observed pick, not whether the recommendation "
            "would have produced a better counterfactual outcome."
        ),
        "validation": validation,
        "validation_policy_baseline": validation_baseline,
        "selected_value_blend": selected,
        "final_test_selected": selected_test,
        "final_test_policy_baseline": final_test_baseline,
        "final_test_all_blends": final_test,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the offline hybrid recommender")
    parser.add_argument("--database", default="data/collection/draft_matches.sqlite3")
    parser.add_argument("--model", default="artifacts/mvp/hybrid_model.npz")
    parser.add_argument("--output", default="artifacts/mvp/backtest.json")
    parser.add_argument(
        "--blends", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 1.0]
    )
    parser.add_argument(
        "--patch",
        action="append",
        help="canonical patch to include, e.g. --patch 7.41; repeat to include several",
    )
    parser.add_argument("--min-rank-tier", type=int, default=None)
    parser.add_argument("--max-rank-tier-exclusive", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
