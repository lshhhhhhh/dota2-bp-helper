from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .outcome import OutcomeEmbeddingModel
from .state import DraftState


# How much weight the ranking gives to heroes this particular draft calls for,
# measured as log P(hero | draft) - log P(hero | round): how far the board moved
# human preference for a hero relative to that hero's own popularity.
#
# This is not a conflict detector. Players do avoid clashes -- every hard carry's
# pick probability collapses 7-20x once Spectre is on the team -- but the term
# only reads preference relative to a hero's baseline, and a hero can stay above
# its baseline while still being a poor fit. It promotes heroes the board asks
# for, which in practice pushes supports up when a team needs them.
#
# The raw policy cannot be used for this: it is dominated by popularity and ranks
# at chance. Subtracting the marginal leaves the part driven by the draft. At 0.15
# it beats a plain candidate pool on the winner/loser hit gap in all three
# brackets and on pairwise accuracy in two, with two to three times more heroes
# reaching the top five.
#
# Phase 1 is inert by construction: with nothing revealed the conditional and the
# marginal are the same distribution over the same legal heroes, so the term is
# exactly zero and the ranking falls back to predicted win probability.
DEFAULT_POLICY_SURPRISE = 0.15

# Superseded by the surprise term above, which measures the same thing
# continuously and scores better. Kept because it is a useful lever on its own:
# pass an integer to rank that many likely picks ahead of the rest. Note the
# effect is not monotonic -- a pool of 40 scores 0.5113 on legend-and-above, whose
# interval [0.4984, 0.5243] includes chance. Stacking a pool on top of the
# surprise term scores worse than the surprise term alone in every bracket.
DEFAULT_CANDIDATE_POOL = None
_POOL_DEMOTION = 100.0


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _zscore_legal(values: np.ndarray, legal: np.ndarray) -> np.ndarray:
    output = np.full_like(values, -1e9, dtype=np.float64)
    selected = values[legal].astype(np.float64)
    scale = float(selected.std())
    output[legal] = (selected - float(selected.mean())) / max(scale, 1e-6)
    return output


def normalize_hero_name(value: str) -> str:
    # ``\w`` is Unicode-aware in Python, so Chinese names survive normalization.
    return re.sub(r"[\W_]+", "", value.casefold(), flags=re.UNICODE)


@dataclass(frozen=True)
class HeroInfo:
    hero_id: int
    name: str
    roles: tuple[str, ...]
    internal_name: str = ""
    chinese_name: str = ""
    aliases: tuple[str, ...] = ()

    @property
    def display_name(self) -> str:
        return f"{self.chinese_name} · {self.name}" if self.chinese_name else self.name


class HeroCatalog:
    def __init__(self, path: str | Path, aliases_path: str | Path | None = None) -> None:
        source = Path(path)
        raw = json.loads(source.read_text(encoding="utf-8-sig"))
        alias_source = Path(aliases_path) if aliases_path else source.with_name("hero_aliases_zh.json")
        alias_data: dict[str, dict[str, object]] = {}
        if alias_source.exists():
            alias_data = json.loads(alias_source.read_text(encoding="utf-8-sig"))
        self.by_id: dict[int, HeroInfo] = {}
        self.alias_to_ids: dict[str, set[int]] = {}
        for value in raw.values():
            localized = alias_data.get(str(value["id"]), {})
            hero = HeroInfo(
                hero_id=int(value["id"]),
                name=str(value["localized_name"]),
                roles=tuple(str(role) for role in value.get("roles", [])),
                internal_name=str(value.get("name", "")),
                chinese_name=str(localized.get("chinese_name", "")),
                aliases=tuple(str(alias) for alias in localized.get("aliases", [])),
            )
            self.by_id[hero.hero_id] = hero
            aliases = {
                str(hero.hero_id),
                hero.name,
                hero.chinese_name,
                str(value.get("name", "")),
                str(value.get("name", "")).removeprefix("npc_dota_hero_"),
                *hero.aliases,
            }
            for alias in aliases:
                normalized = normalize_hero_name(alias)
                if normalized:
                    self.alias_to_ids.setdefault(normalized, set()).add(hero.hero_id)
        # Compatibility for callers that only need unambiguous exact aliases.
        self.by_name = {
            alias: next(iter(hero_ids))
            for alias, hero_ids in self.alias_to_ids.items()
            if len(hero_ids) == 1
        }

    def resolve(self, value: str | int) -> int:
        if isinstance(value, int) or str(value).strip().isdigit():
            hero_id = int(value)
            if hero_id not in self.by_id:
                raise ValueError(f"unknown hero id: {hero_id}")
            return hero_id
        normalized = normalize_hero_name(str(value))
        hero_ids = self.alias_to_ids.get(normalized)
        if not hero_ids:
            raise ValueError(f"unknown hero name: {value}")
        if len(hero_ids) > 1:
            choices = "、".join(self.info(hero_id).display_name for hero_id in sorted(hero_ids))
            raise ValueError(f"英雄简称“{value}”有歧义：{choices}")
        return next(iter(hero_ids))

    def search(self, query: str, *, limit: int = 12) -> list[HeroInfo]:
        needle = normalize_hero_name(query)
        if not needle:
            return sorted(self.by_id.values(), key=lambda hero: hero.hero_id)[:limit]
        ranked: list[tuple[int, int, str, HeroInfo]] = []
        for hero in self.by_id.values():
            fields = {
                hero.name,
                hero.chinese_name,
                hero.internal_name.removeprefix("npc_dota_hero_"),
                *hero.aliases,
            }
            normalized_fields = [normalize_hero_name(field) for field in fields if field]
            matches = [field for field in normalized_fields if needle in field]
            if not matches:
                continue
            if needle in normalized_fields:
                quality = 0
            elif any(field.startswith(needle) for field in matches):
                quality = 1
            else:
                quality = 2
            ranked.append((quality, min(len(field) for field in matches), hero.name, hero))
        ranked.sort(key=lambda item: item[:3])
        return [item[3] for item in ranked[:limit]]

    def info(self, hero_id: int) -> HeroInfo:
        return self.by_id.get(hero_id, HeroInfo(hero_id, f"Hero {hero_id}", ()))


@dataclass(frozen=True)
class Recommendation:
    rank: int
    hero_id: int
    name: str
    roles: tuple[str, ...]
    combined_score: float
    policy_probability: float
    value_log_odds_delta: float
    predicted_win_probability: float | None = None


class HybridRecommender:
    """Phase-aware offline MVP with independently replaceable Policy and Value parts."""

    def __init__(self, model_path: str | Path, catalog: HeroCatalog) -> None:
        artifact = np.load(model_path)
        self.hero_ids = artifact["hero_ids"].astype(np.int64)
        self.hero_to_index = {int(hero): i for i, hero in enumerate(self.hero_ids)}
        self.hero_strength = artifact["hero_strength"].astype(np.float64)
        self.value_weight = float(artifact["value_weight"][0])
        self.value_bias = float(artifact["value_bias"][0])
        self.phase_frequency = artifact["phase_frequency"].astype(np.float64)
        self.policy_w1 = artifact["policy_w1"].astype(np.float64)
        self.policy_b1 = artifact["policy_b1"].astype(np.float64)
        self.policy_w2 = artifact["policy_w2"].astype(np.float64)
        self.policy_b2 = artifact["policy_b2"].astype(np.float64)
        self.catalog = catalog
        outcome_keys = {
            "outcome_candidate_bias",
            "outcome_state_strength",
            "outcome_synergy_candidate",
            "outcome_synergy_ally",
            "outcome_counter_candidate",
            "outcome_counter_enemy",
            "outcome_phase_bias",
        }
        self.outcome_model = (
            OutcomeEmbeddingModel.from_artifact(artifact)
            if outcome_keys.issubset(set(artifact.files))
            else None
        )

    @property
    def objective(self) -> str:
        return "outcome" if self.outcome_model is not None else "legacy_hybrid"

    def _policy_logits(self, state: DraftState) -> tuple[np.ndarray, str]:
        if state.phase == 1:
            return np.log(self.phase_frequency[state.phase - 1]), "popularity"
        h = len(self.hero_ids)
        x = np.zeros(h * 2 + 2, dtype=np.float64)
        for hero in state.allies:
            if hero in self.hero_to_index:
                x[self.hero_to_index[hero]] = 1.0
        for hero in state.enemies:
            if hero in self.hero_to_index:
                x[h + self.hero_to_index[hero]] = 1.0
        x[h * 2 + state.phase - 2] = 1.0
        hidden = np.maximum(0.0, x @ self.policy_w1 + self.policy_b1)
        return hidden @ self.policy_w2 + self.policy_b2, "neural"

    def _pool_penalty(
        self, policy: np.ndarray, legal: np.ndarray, candidate_pool: int | None
    ) -> np.ndarray:
        """Push heroes players almost never pick below the ones they do.

        Ranking every legal hero by predicted win probability produces a list that
        barely moves between drafts, because the outcome model's ordering is close
        to a fixed tier list and its top entries are picked about 11% of the time.
        Ordering the heroes players actually consider first makes the list respond
        to the draft. Nothing is dropped: heroes outside the pool keep their order
        and follow the pool, so a long list still fills.
        """

        penalty = np.zeros(len(self.hero_ids), dtype=np.float64)
        if candidate_pool is None or candidate_pool <= 0:
            return penalty
        available = int(legal.sum())
        if candidate_pool >= available:
            return penalty
        ranked = np.argsort(-np.where(legal, policy, -np.inf))
        penalty[ranked[candidate_pool:]] = _POOL_DEMOTION
        return penalty

    def _policy_surprise(
        self, policy: np.ndarray, legal: np.ndarray, phase: int
    ) -> np.ndarray:
        """How far this draft moved human preference, net of a hero's popularity.

        ``log P(hero | this draft) - log P(hero | this round)``. A hero that is
        merely rare scores zero, so this promotes what the board asks for rather
        than what is popular.

        It does not detect heroes clashing. A hero can lose most of its appeal to
        the board and still sit above its own baseline: a second farm-hungry carry
        behind Spectre drops 7x in pick probability yet still scores positive here,
        because it remains likelier than that hero's average round. Such a pick
        falls down the list only because better-fitting heroes rise past it.
        """

        def normalize(values: np.ndarray) -> np.ndarray:
            logits = np.where(legal, values, -np.inf)
            logits = logits - logits.max()
            return logits - np.log(np.exp(logits).sum())

        conditional = normalize(policy)
        # Normalised the same way and over the same legal heroes, so at phase 1 --
        # where the policy *is* the marginal -- the two cancel exactly and the
        # ranking falls back to predicted win probability alone.
        marginal = normalize(np.log(np.maximum(self.phase_frequency[phase - 1], 1.0)))
        difference = np.subtract(
            conditional, marginal, out=np.zeros_like(conditional), where=legal
        )
        return np.where(legal, difference, 0.0)

    def score_arrays(
        self,
        state: DraftState,
        *,
        value_blend: float = 0.25,
        candidate_pool: int | None = DEFAULT_CANDIDATE_POOL,
        policy_surprise: float = DEFAULT_POLICY_SURPRISE,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        policy, policy_kind = self._policy_logits(state)
        legal = np.ones(len(self.hero_ids), dtype=bool)
        for hero in state.used:
            index = self.hero_to_index.get(hero)
            if index is not None:
                legal[index] = False
        if self.outcome_model is not None:
            allies = [self.hero_to_index[hero] for hero in state.allies if hero in self.hero_to_index]
            enemies = [self.hero_to_index[hero] for hero in state.enemies if hero in self.hero_to_index]
            value_delta = self.outcome_model.score_state(allies, enemies, state.phase)
            combined = np.log(
                np.clip(value_delta, 1e-7, 1.0 - 1e-7)
                / np.clip(1.0 - value_delta, 1e-7, 1.0)
            )
        else:
            value_delta = self.value_weight * self.hero_strength
            combined = _zscore_legal(policy, legal) + value_blend * _zscore_legal(value_delta, legal)
        if policy_surprise:
            combined = combined + policy_surprise * self._policy_surprise(
                policy, legal, state.phase
            )
        combined = combined - self._pool_penalty(policy, legal, candidate_pool)
        combined[~legal] = -1e9
        legal_policy = policy.copy()
        legal_policy[~legal] = -1e9
        probabilities = np.zeros(len(self.hero_ids), dtype=np.float64)
        probabilities[legal] = _softmax(legal_policy[legal])
        return combined, probabilities, value_delta, legal, policy_kind

    def recommend(
        self,
        state: DraftState,
        *,
        top_k: int = 10,
        value_blend: float = 0.25,
        candidate_pool: int | None = DEFAULT_CANDIDATE_POOL,
        policy_surprise: float = DEFAULT_POLICY_SURPRISE,
    ) -> tuple[list[Recommendation], str]:
        combined, probability, value_delta, legal, policy_kind = self.score_arrays(
            state,
            value_blend=value_blend,
            candidate_pool=candidate_pool,
            policy_surprise=policy_surprise,
        )
        order = np.argsort(-combined)
        result: list[Recommendation] = []
        for index in order:
            if not legal[index]:
                continue
            hero_id = int(self.hero_ids[index])
            info = self.catalog.info(hero_id)
            result.append(
                Recommendation(
                    rank=len(result) + 1,
                    hero_id=hero_id,
                    name=info.name,
                    roles=info.roles,
                    combined_score=float(combined[index]),
                    policy_probability=float(probability[index]),
                    value_log_odds_delta=(
                        float(combined[index])
                        if self.outcome_model is not None
                        else float(value_delta[index])
                    ),
                    predicted_win_probability=(
                        float(value_delta[index])
                        if self.outcome_model is not None
                        else None
                    ),
                )
            )
            if len(result) >= top_k:
                break
        return result, policy_kind


def resolve_many(catalog: HeroCatalog, values: Iterable[str]) -> tuple[int, ...]:
    return tuple(catalog.resolve(value) for value in values if str(value).strip())
