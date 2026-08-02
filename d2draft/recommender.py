from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .state import DraftState


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

    def score_arrays(
        self, state: DraftState, *, value_blend: float = 0.25
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        policy, policy_kind = self._policy_logits(state)
        legal = np.ones(len(self.hero_ids), dtype=bool)
        for hero in state.used:
            index = self.hero_to_index.get(hero)
            if index is not None:
                legal[index] = False
        value_delta = self.value_weight * self.hero_strength
        combined = _zscore_legal(policy, legal) + value_blend * _zscore_legal(value_delta, legal)
        combined[~legal] = -1e9
        legal_policy = policy.copy()
        legal_policy[~legal] = -1e9
        probabilities = np.zeros(len(self.hero_ids), dtype=np.float64)
        probabilities[legal] = _softmax(legal_policy[legal])
        return combined, probabilities, value_delta, legal, policy_kind

    def recommend(
        self, state: DraftState, *, top_k: int = 10, value_blend: float = 0.25
    ) -> tuple[list[Recommendation], str]:
        combined, probability, value_delta, legal, policy_kind = self.score_arrays(
            state, value_blend=value_blend
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
                    value_log_odds_delta=float(value_delta[index]),
                )
            )
            if len(result) >= top_k:
                break
        return result, policy_kind


def resolve_many(catalog: HeroCatalog, values: Iterable[str]) -> tuple[int, ...]:
    return tuple(catalog.resolve(value) for value in values if str(value).strip())
