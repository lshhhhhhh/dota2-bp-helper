from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .model_bundle import write_model_manifest
from .patches import install_patch_schema

from .metrics import binary_metrics


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class Adam:
    def __init__(self, parameters: dict[str, np.ndarray]) -> None:
        self.m = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.v = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.step = 0

    def update(
        self,
        parameters: dict[str, np.ndarray],
        gradients: dict[str, np.ndarray],
        learning_rate: float,
    ) -> None:
        self.step += 1
        for name, value in parameters.items():
            gradient = gradients[name]
            self.m[name] = 0.9 * self.m[name] + 0.1 * gradient
            self.v[name] = 0.999 * self.v[name] + 0.001 * gradient * gradient
            m_hat = self.m[name] / (1.0 - 0.9**self.step)
            v_hat = self.v[name] / (1.0 - 0.999**self.step)
            value -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)


def batches(size: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = rng.permutation(size)
    return [order[start : start + batch_size] for start in range(0, size, batch_size)]


@dataclass
class BinaryMLP:
    parameters: dict[str, np.ndarray]

    @classmethod
    def create(cls, inputs: int, hidden: int, rng: np.random.Generator) -> "BinaryMLP":
        return cls(
            {
                "w1": (rng.normal(size=(inputs, hidden)) * math.sqrt(2.0 / inputs)).astype(np.float32),
                "b1": np.zeros(hidden, dtype=np.float32),
                "w2": (rng.normal(size=(hidden, 1)) / math.sqrt(hidden)).astype(np.float32),
                "b2": np.zeros(1, dtype=np.float32),
            }
        )

    def predict(self, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        outputs = []
        p = self.parameters
        for start in range(0, len(x), batch_size):
            xb = x[start : start + batch_size]
            hidden = np.maximum(0.0, xb @ p["w1"] + p["b1"])
            outputs.append(sigmoid((hidden @ p["w2"] + p["b2"]).ravel()))
        return np.concatenate(outputs)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        l2: float,
        rng: np.random.Generator,
    ) -> None:
        p = self.parameters
        optimizer = Adam(p)
        for _ in range(epochs):
            for index in batches(len(x), batch_size, rng):
                xb = x[index]
                yb = y[index]
                pre = xb @ p["w1"] + p["b1"]
                hidden = np.maximum(0.0, pre)
                probability = sigmoid((hidden @ p["w2"] + p["b2"]).ravel())
                dlogit = ((probability - yb) / len(index))[:, None]
                gradients = {
                    "w2": hidden.T @ dlogit + l2 * p["w2"],
                    "b2": dlogit.sum(axis=0),
                }
                dhidden = dlogit @ p["w2"].T
                dpre = dhidden * (pre > 0)
                gradients["w1"] = xb.T @ dpre + l2 * p["w1"]
                gradients["b1"] = dpre.sum(axis=0)
                optimizer.update(p, gradients, learning_rate)


@dataclass
class PolicyMLP:
    parameters: dict[str, np.ndarray]

    @classmethod
    def create(
        cls, inputs: int, hidden: int, outputs: int, rng: np.random.Generator
    ) -> "PolicyMLP":
        return cls(
            {
                "w1": (rng.normal(size=(inputs, hidden)) * math.sqrt(2.0 / inputs)).astype(np.float32),
                "b1": np.zeros(hidden, dtype=np.float32),
                "w2": (rng.normal(size=(hidden, outputs)) / math.sqrt(hidden)).astype(np.float32),
                "b2": np.zeros(outputs, dtype=np.float32),
            }
        )

    def logits(self, x: np.ndarray, used: np.ndarray) -> np.ndarray:
        p = self.parameters
        hidden = np.maximum(0.0, x @ p["w1"] + p["b1"])
        logits = hidden @ p["w2"] + p["b2"]
        return np.where(used, -1e9, logits)

    def fit(
        self,
        x: np.ndarray,
        used: np.ndarray,
        target: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        l2: float,
        rng: np.random.Generator,
    ) -> None:
        p = self.parameters
        optimizer = Adam(p)
        for _ in range(epochs):
            for index in batches(len(x), batch_size, rng):
                xb = x[index]
                ub = used[index]
                yb = target[index]
                pre = xb @ p["w1"] + p["b1"]
                hidden = np.maximum(0.0, pre)
                logits = hidden @ p["w2"] + p["b2"]
                logits = np.where(ub, -1e9, logits)
                probability = softmax(logits)
                probability[np.arange(len(index)), yb] -= 1.0
                dlogit = probability / len(index)
                gradients = {
                    "w2": hidden.T @ dlogit + l2 * p["w2"],
                    "b2": dlogit.sum(axis=0),
                }
                dhidden = dlogit @ p["w2"].T
                dpre = dhidden * (pre > 0)
                gradients["w1"] = xb.T @ dpre + l2 * p["w1"]
                gradients["b1"] = dpre.sum(axis=0)
                optimizer.update(p, gradients, learning_rate)


def _filter_clause(
    patches: tuple[str, ...] | None,
    *,
    minimum_rank_tier: int | None = None,
    maximum_rank_tier_exclusive: int | None = None,
    base_conditions: tuple[str, ...] = (),
) -> tuple[str, tuple[object, ...]]:
    conditions = list(base_conditions)
    parameters: list[object] = []
    if patches:
        placeholders = ",".join("?" for _ in patches)
        conditions.append(f"canonical_patch IN ({placeholders})")
        parameters.extend(patches)
    if minimum_rank_tier is not None:
        conditions.append("avg_rank_tier >= ?")
        parameters.append(minimum_rank_tier)
    if maximum_rank_tier_exclusive is not None:
        conditions.append("avg_rank_tier < ?")
        parameters.append(maximum_rank_tier_exclusive)
    clause = " WHERE " + " AND ".join(conditions) if conditions else ""
    return clause, tuple(parameters)


def load_value_rows(
    connection: sqlite3.Connection,
    patches: tuple[str, ...] | None = None,
    *,
    minimum_rank_tier: int | None = None,
    maximum_rank_tier_exclusive: int | None = None,
) -> list[sqlite3.Row]:
    clause, parameters = _filter_clause(
        patches,
        minimum_rank_tier=minimum_rank_tier,
        maximum_rank_tier_exclusive=maximum_rank_tier_exclusive,
    )
    return connection.execute(
        f"""
        SELECT match_id, start_time, radiant_win, radiant_team, dire_team
        FROM candidates
        {clause}
        ORDER BY start_time, match_id
        """,
        parameters,
    ).fetchall()


def load_policy_rows(
    connection: sqlite3.Connection,
    patches: tuple[str, ...] | None = None,
    *,
    minimum_rank_tier: int | None = None,
    maximum_rank_tier_exclusive: int | None = None,
) -> list[sqlite3.Row]:
    clause, parameters = _filter_clause(
        patches,
        minimum_rank_tier=minimum_rank_tier,
        maximum_rank_tier_exclusive=maximum_rank_tier_exclusive,
        base_conditions=("reconstructable = 1",),
    )
    return connection.execute(
        f"""
        SELECT match_id, start_time,
               phase_1_radiant, phase_1_dire,
               phase_2_radiant, phase_2_dire,
               phase_3_radiant, phase_3_dire
        FROM matches
        {clause}
        ORDER BY start_time, match_id
        """,
        parameters,
    ).fetchall()


def all_hero_ids(value_rows: list[sqlite3.Row]) -> list[int]:
    heroes: set[int] = set()
    for row in value_rows:
        heroes.update(json.loads(row["radiant_team"]))
        heroes.update(json.loads(row["dire_team"]))
    return sorted(heroes)


def value_arrays(
    rows: list[sqlite3.Row], hero_to_index: dict[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((len(rows), len(hero_to_index)), dtype=np.float32)
    y = np.zeros(len(rows), dtype=np.float32)
    for i, row in enumerate(rows):
        for hero in json.loads(row["radiant_team"]):
            x[i, hero_to_index[int(hero)]] = 1.0
        for hero in json.loads(row["dire_team"]):
            x[i, hero_to_index[int(hero)]] = -1.0
        y[i] = float(row["radiant_win"])
    return x, y


def policy_examples(
    rows: list[sqlite3.Row], hero_to_index: dict[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    examples: list[tuple[list[int], list[int], int, int]] = []
    for row in rows:
        p1r = [int(x) for x in json.loads(row["phase_1_radiant"])]
        p1d = [int(x) for x in json.loads(row["phase_1_dire"])]
        p2r = [int(x) for x in json.loads(row["phase_2_radiant"])]
        p2d = [int(x) for x in json.loads(row["phase_2_dire"])]
        p3r = [int(x) for x in json.loads(row["phase_3_radiant"])]
        p3d = [int(x) for x in json.loads(row["phase_3_dire"])]
        for target in p2r:
            examples.append((p1r, p1d, 2, target))
        for target in p2d:
            examples.append((p1d, p1r, 2, target))
        for target in p3r:
            examples.append((p1r + p2r, p1d + p2d, 3, target))
        for target in p3d:
            examples.append((p1d + p2d, p1r + p2r, 3, target))

    h = len(hero_to_index)
    x = np.zeros((len(examples), h * 2 + 2), dtype=np.float32)
    used = np.zeros((len(examples), h), dtype=bool)
    target = np.zeros(len(examples), dtype=np.int64)
    phase = np.zeros(len(examples), dtype=np.int8)
    for i, (allies, enemies, phase_number, hero) in enumerate(examples):
        for ally in allies:
            index = hero_to_index[ally]
            x[i, index] = 1.0
            used[i, index] = True
        for enemy in enemies:
            index = hero_to_index[enemy]
            x[i, h + index] = 1.0
            used[i, index] = True
        x[i, h + h + phase_number - 2] = 1.0
        target[i] = hero_to_index[hero]
        phase[i] = phase_number
    return x, used, target, phase


def phase_frequencies(
    rows: list[sqlite3.Row],
    train_target: np.ndarray,
    train_phase: np.ndarray,
    hero_to_index: dict[int, int],
) -> np.ndarray:
    """Laplace-smoothed target counts for phases 1, 2 and 3."""

    frequency = np.ones((3, len(hero_to_index)), dtype=np.float64)
    for row in rows:
        for field in ("phase_1_radiant", "phase_1_dire"):
            for hero in json.loads(row[field]):
                frequency[0, hero_to_index[int(hero)]] += 1.0
    for target, phase_number in zip(train_target, train_phase, strict=True):
        frequency[int(phase_number) - 1, target] += 1.0
    return frequency


def fit_strength_baseline(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    h = x.shape[1]
    games = np.zeros(h, dtype=np.float64)
    wins = np.zeros(h, dtype=np.float64)
    for row, label in zip(x, y, strict=True):
        radiant = row > 0
        dire = row < 0
        games += radiant | dire
        wins += radiant * label + dire * (1.0 - label)
    prior = float(y.mean())
    alpha = 100.0
    rate = (wins + alpha * 0.5) / (games + alpha)
    strength = np.log(np.clip(rate, 1e-5, 1 - 1e-5) / np.clip(1 - rate, 1e-5, 1))
    score = x @ strength
    weight = 0.2
    bias = math.log(prior / (1.0 - prior))
    for _ in range(800):
        probability = sigmoid(weight * score + bias)
        error = probability - y
        weight -= 0.05 * float(np.mean(error * score))
        bias -= 0.05 * float(error.mean())
    return strength.astype(np.float32), float(weight), float(bias)


def ranking_metrics(logits: np.ndarray, target: np.ndarray, phase: np.ndarray) -> dict[str, Any]:
    order = np.argsort(-logits, axis=1)
    positions = np.argmax(order == target[:, None], axis=1) + 1

    def summarize(mask: np.ndarray) -> dict[str, float]:
        rank = positions[mask]
        return {
            "examples": int(len(rank)),
            "hit_at_5": float(np.mean(rank <= 5)),
            "hit_at_10": float(np.mean(rank <= 10)),
            "mrr": float(np.mean(1.0 / rank)),
            "median_rank": float(np.median(rank)),
        }

    return {
        "overall": summarize(np.ones(len(target), dtype=bool)),
        "phase_2": summarize(phase == 2),
        "phase_3": summarize(phase == 3),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    connection = sqlite3.connect(args.database)
    connection.row_factory = sqlite3.Row
    install_patch_schema(connection)
    patches = tuple(args.patch or ())
    all_value_rows = load_value_rows(connection)
    value_rows = load_value_rows(
        connection,
        patches,
        minimum_rank_tier=args.min_rank_tier,
        maximum_rank_tier_exclusive=args.max_rank_tier_exclusive,
    )
    policy_rows = load_policy_rows(
        connection,
        patches,
        minimum_rank_tier=args.min_rank_tier,
        maximum_rank_tier_exclusive=args.max_rank_tier_exclusive,
    )
    if len(value_rows) < 20 or len(policy_rows) < 20:
        raise ValueError(f"insufficient rows for patch filter: {patches or 'all'}")
    heroes = all_hero_ids(all_value_rows)
    hero_to_index = {hero: index for index, hero in enumerate(heroes)}

    value_cut = int(len(value_rows) * 0.8)
    value_train_rows, value_test_rows = value_rows[:value_cut], value_rows[value_cut:]
    value_train_x, value_train_y = value_arrays(value_train_rows, hero_to_index)
    value_test_x, value_test_y = value_arrays(value_test_rows, hero_to_index)

    strength, baseline_weight, baseline_bias = fit_strength_baseline(value_train_x, value_train_y)
    baseline_probability = sigmoid(value_test_x @ strength * baseline_weight + baseline_bias)
    value_baseline = binary_metrics(value_test_y, baseline_probability)

    # Side-swap augmentation teaches the network that swapping teams should invert the label.
    augmented_x = np.concatenate([value_train_x, -value_train_x], axis=0)
    augmented_y = np.concatenate([value_train_y, 1.0 - value_train_y], axis=0)
    value_model = BinaryMLP.create(len(heroes), args.value_hidden, rng)
    value_model.fit(
        augmented_x,
        augmented_y,
        epochs=args.value_epochs,
        batch_size=256,
        learning_rate=8e-4,
        l2=2e-4,
        rng=rng,
    )
    value_neural = binary_metrics(value_test_y, value_model.predict(value_test_x))

    policy_cut = int(len(policy_rows) * 0.8)
    policy_train_rows, policy_test_rows = policy_rows[:policy_cut], policy_rows[policy_cut:]
    train_x, train_used, train_target, train_phase = policy_examples(
        policy_train_rows, hero_to_index
    )
    test_x, test_used, test_target, test_phase = policy_examples(policy_test_rows, hero_to_index)

    frequency = phase_frequencies(
        policy_train_rows, train_target, train_phase, hero_to_index
    )
    baseline_logits = np.log(frequency[test_phase - 1])
    baseline_logits = np.where(test_used, -1e9, baseline_logits)
    policy_baseline = ranking_metrics(baseline_logits, test_target, test_phase)

    policy_model = PolicyMLP.create(train_x.shape[1], args.policy_hidden, len(heroes), rng)
    policy_model.fit(
        train_x,
        train_used,
        train_target,
        epochs=args.policy_epochs,
        batch_size=256,
        learning_rate=1e-3,
        l2=1e-4,
        rng=rng,
    )
    policy_neural = ranking_metrics(
        policy_model.logits(test_x, test_used), test_target, test_phase
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "value_model.npz",
        hero_ids=np.asarray(heroes),
        **value_model.parameters,
    )
    np.savez_compressed(
        output_dir / "policy_model.npz",
        hero_ids=np.asarray(heroes),
        **policy_model.parameters,
    )
    np.savez_compressed(
        output_dir / "hybrid_model.npz",
        hero_ids=np.asarray(heroes, dtype=np.int64),
        hero_strength=strength,
        value_weight=np.asarray([baseline_weight], dtype=np.float32),
        value_bias=np.asarray([baseline_bias], dtype=np.float32),
        phase_frequency=frequency.astype(np.float32),
        policy_w1=policy_model.parameters["w1"],
        policy_b1=policy_model.parameters["b1"],
        policy_w2=policy_model.parameters["w2"],
        policy_b2=policy_model.parameters["b2"],
    )
    generated_at_utc = datetime.now(UTC).isoformat()
    report = {
        "generated_at_utc": generated_at_utc,
        "seed": args.seed,
        "split": "oldest 80% train, newest 20% test",
        "heroes": len(heroes),
        "patch_filter": list(patches) if patches else "all",
        "rank_filter": {
            "minimum_avg_rank_tier": args.min_rank_tier,
            "maximum_avg_rank_tier_exclusive": args.max_rank_tier_exclusive,
        },
        "value": {
            "train_matches": len(value_train_rows),
            "test_matches": len(value_test_rows),
            "baseline": value_baseline,
            "neural": value_neural,
        },
        "policy": {
            "train_matches": len(policy_train_rows),
            "test_matches": len(policy_test_rows),
            "train_examples": len(train_target),
            "test_examples": len(test_target),
            "baseline": policy_baseline,
            "neural": policy_neural,
        },
        "limitations": [
            "The policy sample is intentionally small and is only a signal check.",
            "Observed player picks are not a unique ground-truth optimal recommendation.",
            "The public match sample is visibility-biased.",
            "No role, lane, player proficiency, or expert tags are used.",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_model_manifest(
        output_dir,
        generated_at_utc=generated_at_utc,
        patches=patches,
        hero_count=len(heroes),
        minimum_rank_tier=args.min_rank_tier,
        maximum_rank_tier_exclusive=args.max_rank_tier_exclusive,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare Dota draft MVP models")
    parser.add_argument("--database", default="data/collection/draft_matches.sqlite3")
    parser.add_argument("--output-dir", default="artifacts/mvp")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--value-hidden", type=int, default=64)
    parser.add_argument("--policy-hidden", type=int, default=96)
    parser.add_argument("--value-epochs", type=int, default=25)
    parser.add_argument("--policy-epochs", type=int, default=35)
    parser.add_argument(
        "--min-rank-tier",
        type=int,
        default=None,
        help="minimum OpenDota avg_rank_tier to include, e.g. 50 for Legend+",
    )
    parser.add_argument(
        "--max-rank-tier-exclusive",
        type=int,
        default=None,
        help="exclusive maximum avg_rank_tier, e.g. 50 for Archon and below",
    )
    parser.add_argument(
        "--patch",
        action="append",
        help="canonical patch to include, e.g. --patch 7.41; repeat to include several",
    )
    args = parser.parse_args()
    started = time.monotonic()
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Experiment completed in {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
