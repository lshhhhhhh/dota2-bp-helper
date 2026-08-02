from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .experiment import load_policy_rows
from .metrics import binary_metrics
from .model_bundle import ModelBundle, write_model_manifest
from .outcome import OutcomeEmbeddingModel, outcome_examples
from .recommender import HeroCatalog


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.base_model_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    catalog = HeroCatalog(args.heroes)
    bundle = ModelBundle.load(source, expected_hero_ids=catalog.by_id)
    with np.load(bundle.artifact_path, allow_pickle=False) as artifact:
        arrays = {name: artifact[name] for name in artifact.files}
        hero_ids = artifact["hero_ids"].astype(np.int64)
    hero_to_index = {int(hero): index for index, hero in enumerate(hero_ids)}

    connection = sqlite3.connect(args.database)
    connection.row_factory = sqlite3.Row
    rows = load_policy_rows(
        connection,
        tuple(args.patch or ()),
        minimum_rank_tier=args.min_rank_tier,
        maximum_rank_tier_exclusive=args.max_rank_tier_exclusive,
    )
    connection.close()
    cut = int(len(rows) * 0.8)
    train = outcome_examples(rows[:cut], hero_to_index)
    test = outcome_examples(rows[cut:], hero_to_index)

    rng = np.random.default_rng(args.seed)
    if args.outcome_init_model_dir:
        init_bundle = ModelBundle.load(
            args.outcome_init_model_dir, expected_hero_ids=catalog.by_id
        )
        with np.load(init_bundle.artifact_path, allow_pickle=False) as init_artifact:
            model = OutcomeEmbeddingModel.from_artifact(init_artifact)
    else:
        model = OutcomeEmbeddingModel.create(
            len(hero_ids), args.dimensions, rng
        )
    model.fit(
        train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        l2=args.l2,
        rng=rng,
    )
    probability = model.predict(test)
    metrics = {
        "overall": binary_metrics(test.outcome, probability),
        **{
            f"phase_{phase}": binary_metrics(
                test.outcome[test.phase == phase],
                probability[test.phase == phase],
            )
            for phase in (1, 2, 3)
        },
    }
    arrays.update(model.artifact_parameters())
    np.savez_compressed(output / "hybrid_model.npz", **arrays)

    for filename in ("policy_model.npz", "value_model.npz"):
        source_file = source / filename
        if source_file.exists():
            shutil.copy2(source_file, output / filename)

    generated_at = datetime.now(UTC).isoformat()
    report = dict(bundle.report)
    report["generated_at_utc"] = generated_at
    report["outcome"] = {
        "objective": "predict candidate win probability from the public draft state",
        "architecture": "candidate-conditioned embedding interaction network",
        "split": "oldest 80% train, newest 20% test",
        "train_matches": cut,
        "test_matches": len(rows) - cut,
        "train_examples": len(train),
        "test_examples": len(test),
        "embedding_dimensions": args.dimensions,
        "epochs": args.epochs,
        "metrics": metrics,
        "losing_pick_feedback": (
            "Every chosen hero on the losing side is labeled 0; every chosen hero "
            "on the winning side is labeled 1."
        ),
    }
    limitations = list(report.get("limitations", []))
    counterfactual_note = (
        "Outcomes for unchosen candidates are counterfactual estimates, not observed labels."
    )
    if counterfactual_note not in limitations:
        limitations.append(counterfactual_note)
    report["limitations"] = limitations
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_model_manifest(
        output,
        generated_at_utc=generated_at,
        patches=tuple(args.patch or ()),
        hero_count=len(hero_ids),
        minimum_rank_tier=args.min_rank_tier,
        maximum_rank_tier_exclusive=args.max_rank_tier_exclusive,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upgrade an existing policy bundle with an outcome recommender"
    )
    parser.add_argument("--database", default="data/collection/draft_matches.sqlite3")
    parser.add_argument("--heroes", default="data/heroes.json")
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--outcome-init-model-dir",
        default=None,
        help="optional outcome bundle used to initialize transfer learning",
    )
    parser.add_argument("--patch", action="append", default=[])
    parser.add_argument("--min-rank-tier", type=int, default=None)
    parser.add_argument("--max-rank-tier-exclusive", type=int, default=None)
    parser.add_argument("--dimensions", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--l2", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
