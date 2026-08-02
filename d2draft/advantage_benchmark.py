from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .model_bundle import ModelBundle
from .recommender import HeroCatalog, HybridRecommender
from .state import DraftState


PHASE_FIELDS = (
    "phase_1_radiant",
    "phase_1_dire",
    "phase_2_radiant",
    "phase_2_dire",
    "phase_3_radiant",
    "phase_3_dire",
)


def _group_summary(records: list[tuple[int, float]], top_k: int) -> dict[str, Any]:
    followed = [win for rank, win in records if rank <= top_k]
    other = [win for rank, win in records if rank > top_k]
    followed_rate = sum(followed) / len(followed)
    other_rate = sum(other) / len(other)
    difference = followed_rate - other_rate
    standard_error = math.sqrt(
        followed_rate * (1.0 - followed_rate) / len(followed)
        + other_rate * (1.0 - other_rate) / len(other)
    )
    return {
        "top_k": top_k,
        "followed_decisions": len(followed),
        "followed_win_rate": followed_rate,
        "other_decisions": len(other),
        "other_win_rate": other_rate,
        "observed_difference_points": difference * 100.0,
        "approximate_95_ci_points": [
            (difference - 1.96 * standard_error) * 100.0,
            (difference + 1.96 * standard_error) * 100.0,
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = Path(args.model_dir)
    catalog = HeroCatalog(Path(args.heroes))
    bundle = ModelBundle.load(model_dir, expected_hero_ids=catalog.by_id)
    recommender = HybridRecommender(bundle.artifact_path, catalog)
    blend = float(bundle.backtest.get("selected_value_blend", {}).get("phase_3", 0.0))

    conditions = ["reconstructable = 1", "canonical_patch = ?"]
    parameters: list[object] = [args.patch]
    if args.min_rank_tier is not None:
        conditions.append("avg_rank_tier >= ?")
        parameters.append(args.min_rank_tier)
    if args.max_rank_tier_exclusive is not None:
        conditions.append("avg_rank_tier < ?")
        parameters.append(args.max_rank_tier_exclusive)
    connection = sqlite3.connect(args.database)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        f"""
        SELECT match_id, start_time, radiant_win, {", ".join(PHASE_FIELDS)}
        FROM matches
        WHERE {" AND ".join(conditions)}
        ORDER BY start_time, match_id
        """,
        tuple(parameters),
    ).fetchall()
    connection.close()
    test_rows = rows[int(len(rows) * 0.9) :]

    records: list[tuple[int, float]] = []
    for row in test_rows:
        picks = {
            field: [int(hero) for hero in json.loads(row[field])]
            for field in PHASE_FIELDS
        }
        for side, other in (("radiant", "dire"), ("dire", "radiant")):
            allies = picks[f"phase_1_{side}"] + picks[f"phase_2_{side}"]
            enemies = picks[f"phase_1_{other}"] + picks[f"phase_2_{other}"]
            target = picks[f"phase_3_{side}"][0]
            recommendations, _ = recommender.recommend(
                DraftState(3, tuple(allies), tuple(enemies)),
                top_k=10,
                value_blend=blend,
            )
            rank = next(
                (
                    index
                    for index, recommendation in enumerate(recommendations, 1)
                    if recommendation.hero_id == target
                ),
                99,
            )
            win = (
                float(row["radiant_win"])
                if side == "radiant"
                else 1.0 - float(row["radiant_win"])
            )
            records.append((rank, win))

    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_id": bundle.model_id,
        "rank_bracket": bundle.rank_bracket_label,
        "patch": args.patch,
        "phase": 3,
        "test_matches": len(test_rows),
        "test_decisions": len(records),
        "value_blend": blend,
        "groups": {
            f"top_{top_k}": _group_summary(records, top_k) for top_k in (1, 5, 10)
        },
        "interpretation": (
            "Historical association only. The groups are not randomized, so the "
            "observed win-rate difference is not a causal treatment effect."
        ),
    }
    output = Path(args.output) if args.output else model_dir / "advantage_benchmark.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Historical win-rate association benchmark")
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
