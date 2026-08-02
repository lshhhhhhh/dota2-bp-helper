from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class OutcomeExamples:
    allies: np.ndarray
    enemies: np.ndarray
    candidate: np.ndarray
    phase: np.ndarray
    outcome: np.ndarray

    def __len__(self) -> int:
        return len(self.candidate)

    def subset(self, indices: np.ndarray) -> "OutcomeExamples":
        return OutcomeExamples(
            allies=self.allies[indices],
            enemies=self.enemies[indices],
            candidate=self.candidate[indices],
            phase=self.phase[indices],
            outcome=self.outcome[indices],
        )


def outcome_examples(
    rows: list[sqlite3.Row], hero_to_index: dict[int, int]
) -> OutcomeExamples:
    """Build contextual action/outcome examples from reconstructable ranked drafts.

    Every final hero on the winning side receives label 1 and every final hero on
    the losing side receives label 0. Only information public before the current
    simultaneous pick round is included in the state.
    """

    records: list[tuple[list[int], list[int], int, int, float]] = []
    for row in rows:
        picks = {
            f"phase_{phase}_{side}": [
                hero_to_index[int(hero)]
                for hero in json.loads(row[f"phase_{phase}_{side}"])
            ]
            for phase in (1, 2, 3)
            for side in ("radiant", "dire")
        }
        radiant_win = float(row["radiant_win"])
        for side, other, won in (
            ("radiant", "dire", radiant_win),
            ("dire", "radiant", 1.0 - radiant_win),
        ):
            p1_allies = picks[f"phase_1_{side}"]
            p1_enemies = picks[f"phase_1_{other}"]
            p2_allies = picks[f"phase_2_{side}"]
            p2_enemies = picks[f"phase_2_{other}"]
            for candidate in p1_allies:
                records.append(([], [], candidate, 1, won))
            for candidate in p2_allies:
                records.append((p1_allies, p1_enemies, candidate, 2, won))
            for candidate in picks[f"phase_3_{side}"]:
                records.append(
                    (
                        p1_allies + p2_allies,
                        p1_enemies + p2_enemies,
                        candidate,
                        3,
                        won,
                    )
                )

    count = len(records)
    allies = np.full((count, 4), -1, dtype=np.int16)
    enemies = np.full((count, 4), -1, dtype=np.int16)
    candidate = np.empty(count, dtype=np.int16)
    phase = np.empty(count, dtype=np.int8)
    outcome = np.empty(count, dtype=np.float32)
    for index, (own, opposing, action, round_number, won) in enumerate(records):
        allies[index, : len(own)] = own
        enemies[index, : len(opposing)] = opposing
        candidate[index] = action
        phase[index] = round_number
        outcome[index] = won
    return OutcomeExamples(allies, enemies, candidate, phase, outcome)


class _Adam:
    def __init__(self, parameters: dict[str, np.ndarray]) -> None:
        self.m = {key: np.zeros_like(value) for key, value in parameters.items()}
        self.v = {key: np.zeros_like(value) for key, value in parameters.items()}
        self.step = 0

    def update(
        self,
        parameters: dict[str, np.ndarray],
        gradients: dict[str, np.ndarray],
        learning_rate: float,
    ) -> None:
        self.step += 1
        for key, value in parameters.items():
            gradient = gradients[key]
            self.m[key] = 0.9 * self.m[key] + 0.1 * gradient
            self.v[key] = 0.999 * self.v[key] + 0.001 * gradient * gradient
            m_hat = self.m[key] / (1.0 - 0.9**self.step)
            v_hat = self.v[key] / (1.0 - 0.999**self.step)
            value -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)


@dataclass
class OutcomeEmbeddingModel:
    """Small candidate-conditioned outcome network with learned hero embeddings."""

    parameters: dict[str, np.ndarray]

    @classmethod
    def create(
        cls, heroes: int, dimensions: int, rng: np.random.Generator
    ) -> "OutcomeEmbeddingModel":
        scale = 0.04
        return cls(
            {
                "candidate_bias": np.zeros(heroes, dtype=np.float32),
                "state_strength": np.zeros(heroes, dtype=np.float32),
                "synergy_candidate": rng.normal(
                    0.0, scale, size=(heroes, dimensions)
                ).astype(np.float32),
                "synergy_ally": rng.normal(
                    0.0, scale, size=(heroes, dimensions)
                ).astype(np.float32),
                "counter_candidate": rng.normal(
                    0.0, scale, size=(heroes, dimensions)
                ).astype(np.float32),
                "counter_enemy": rng.normal(
                    0.0, scale, size=(heroes, dimensions)
                ).astype(np.float32),
                "phase_bias": np.zeros(3, dtype=np.float32),
            }
        )

    @classmethod
    def from_artifact(cls, artifact: Any) -> "OutcomeEmbeddingModel":
        names = (
            "candidate_bias",
            "state_strength",
            "synergy_candidate",
            "synergy_ally",
            "counter_candidate",
            "counter_enemy",
            "phase_bias",
        )
        return cls(
            {
                name: artifact[f"outcome_{name}"].astype(np.float32)
                for name in names
            }
        )

    @property
    def interaction_scale(self) -> float:
        # Embeddings are initialized at a small scale; an additional 1/sqrt(d)
        # factor made the global hero bias dominate and nearly erased lineup-specific
        # changes in the recommendation ranking.
        return 1.0

    def _components(
        self, examples: OutcomeExamples
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p = self.parameters
        ally_mask = examples.allies >= 0
        enemy_mask = examples.enemies >= 0
        ally_index = np.where(ally_mask, examples.allies, 0)
        enemy_index = np.where(enemy_mask, examples.enemies, 0)
        ally_embeddings = p["synergy_ally"][ally_index] * ally_mask[..., None]
        enemy_embeddings = p["counter_enemy"][enemy_index] * enemy_mask[..., None]
        ally_sum = ally_embeddings.sum(axis=1)
        enemy_sum = enemy_embeddings.sum(axis=1)
        state = (
            (p["state_strength"][ally_index] * ally_mask).sum(axis=1)
            - (p["state_strength"][enemy_index] * enemy_mask).sum(axis=1)
        )
        candidate_synergy = p["synergy_candidate"][examples.candidate]
        candidate_counter = p["counter_candidate"][examples.candidate]
        return ally_mask, enemy_mask, ally_sum, enemy_sum, (
            p["phase_bias"][examples.phase - 1]
            + p["candidate_bias"][examples.candidate]
            + state
            + self.interaction_scale
            * (candidate_synergy * ally_sum).sum(axis=1)
            + self.interaction_scale
            * (candidate_counter * enemy_sum).sum(axis=1)
        )

    def logits(self, examples: OutcomeExamples) -> np.ndarray:
        return self._components(examples)[-1]

    def predict(self, examples: OutcomeExamples) -> np.ndarray:
        return _sigmoid(self.logits(examples))

    def score_state(
        self,
        allies: list[int],
        enemies: list[int],
        phase: int,
    ) -> np.ndarray:
        heroes = len(self.parameters["candidate_bias"])
        # Every term sums over the slot axis, so the width only has to fit the
        # heroes actually revealed. A side may be one pick ahead of the other.
        slots = max(4, len(allies), len(enemies))
        own = np.full((heroes, slots), -1, dtype=np.int16)
        opposing = np.full((heroes, slots), -1, dtype=np.int16)
        own[:, : len(allies)] = np.asarray(allies, dtype=np.int16)
        opposing[:, : len(enemies)] = np.asarray(enemies, dtype=np.int16)
        examples = OutcomeExamples(
            allies=own,
            enemies=opposing,
            candidate=np.arange(heroes, dtype=np.int16),
            phase=np.full(heroes, phase, dtype=np.int8),
            outcome=np.zeros(heroes, dtype=np.float32),
        )
        return self.predict(examples)

    def score_all_candidates(self, examples: OutcomeExamples) -> np.ndarray:
        """Return candidate win probabilities for every state and every hero."""

        p = self.parameters
        ally_mask = examples.allies >= 0
        enemy_mask = examples.enemies >= 0
        ally_index = np.where(ally_mask, examples.allies, 0)
        enemy_index = np.where(enemy_mask, examples.enemies, 0)
        ally_sum = (
            p["synergy_ally"][ally_index] * ally_mask[..., None]
        ).sum(axis=1)
        enemy_sum = (
            p["counter_enemy"][enemy_index] * enemy_mask[..., None]
        ).sum(axis=1)
        state = (
            (p["state_strength"][ally_index] * ally_mask).sum(axis=1)
            - (p["state_strength"][enemy_index] * enemy_mask).sum(axis=1)
        )
        logits = (
            p["phase_bias"][examples.phase - 1, None]
            + p["candidate_bias"][None, :]
            + state[:, None]
            + self.interaction_scale
            * (ally_sum @ p["synergy_candidate"].T)
            + self.interaction_scale
            * (enemy_sum @ p["counter_candidate"].T)
        )
        return _sigmoid(logits)

    def fit(
        self,
        examples: OutcomeExamples,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        l2: float,
        rng: np.random.Generator,
    ) -> None:
        p = self.parameters
        optimizer = _Adam(p)
        scale = self.interaction_scale
        size = len(examples)
        for _ in range(epochs):
            order = rng.permutation(size)
            for start in range(0, size, batch_size):
                indices = order[start : start + batch_size]
                batch = examples.subset(indices)
                ally_mask, enemy_mask, ally_sum, enemy_sum, logits = self._components(
                    batch
                )
                error = (_sigmoid(logits) - batch.outcome) / len(batch)
                gradients = {
                    key: np.zeros_like(value) for key, value in p.items()
                }
                np.add.at(gradients["phase_bias"], batch.phase - 1, error)
                np.add.at(gradients["candidate_bias"], batch.candidate, error)

                candidate_synergy = p["synergy_candidate"][batch.candidate]
                candidate_counter = p["counter_candidate"][batch.candidate]
                np.add.at(
                    gradients["synergy_candidate"],
                    batch.candidate,
                    error[:, None] * ally_sum * scale,
                )
                np.add.at(
                    gradients["counter_candidate"],
                    batch.candidate,
                    error[:, None] * enemy_sum * scale,
                )

                for slot in range(4):
                    ally_valid = ally_mask[:, slot]
                    if np.any(ally_valid):
                        ally_index = batch.allies[ally_valid, slot]
                        ally_error = error[ally_valid]
                        np.add.at(
                            gradients["state_strength"], ally_index, ally_error
                        )
                        np.add.at(
                            gradients["synergy_ally"],
                            ally_index,
                            ally_error[:, None]
                            * candidate_synergy[ally_valid]
                            * scale,
                        )
                    enemy_valid = enemy_mask[:, slot]
                    if np.any(enemy_valid):
                        enemy_index = batch.enemies[enemy_valid, slot]
                        enemy_error = error[enemy_valid]
                        np.add.at(
                            gradients["state_strength"], enemy_index, -enemy_error
                        )
                        np.add.at(
                            gradients["counter_enemy"],
                            enemy_index,
                            enemy_error[:, None]
                            * candidate_counter[enemy_valid]
                            * scale,
                        )

                for key in (
                    "candidate_bias",
                    "state_strength",
                    "synergy_candidate",
                    "synergy_ally",
                    "counter_candidate",
                    "counter_enemy",
                ):
                    gradients[key] += l2 * p[key]
                optimizer.update(p, gradients, learning_rate)

    def artifact_parameters(self) -> dict[str, np.ndarray]:
        return {f"outcome_{key}": value for key, value in self.parameters.items()}
