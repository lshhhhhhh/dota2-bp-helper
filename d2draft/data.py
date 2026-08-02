from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class Match:
    match_id: int
    start_time: int
    radiant_win: bool
    radiant: tuple[int, ...]
    dire: tuple[int, ...]
    avg_rank_tier: int | None = None


def _parse_team(value: object) -> tuple[int, ...]:
    if isinstance(value, list):
        result = tuple(int(x) for x in value)
    elif isinstance(value, str):
        raw = value.strip()
        if raw.startswith("{") and raw.endswith("}"):
            raw = "[" + raw[1:-1] + "]"
        parsed = json.loads(raw)
        result = tuple(int(x) for x in parsed)
    else:
        raise ValueError(f"unsupported team value: {value!r}")
    if len(result) != 5 or len(set(result)) != 5 or any(x <= 0 for x in result):
        raise ValueError(f"invalid five-hero team: {result!r}")
    return result


def match_from_row(row: dict[str, object]) -> Match:
    radiant = _parse_team(row["radiant_team"])
    dire = _parse_team(row["dire_team"])
    if set(radiant) & set(dire):
        raise ValueError("duplicate hero across teams")
    return Match(
        match_id=int(row["match_id"]),
        start_time=int(row["start_time"]),
        radiant_win=bool(row["radiant_win"]),
        radiant=radiant,
        dire=dire,
        avg_rank_tier=int(row["avg_rank_tier"]) if row.get("avg_rank_tier") is not None else None,
    )


def load_matches(path: str | Path) -> list[Match]:
    matches: list[Match] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                matches.append(match_from_row(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid match at line {line_number}: {exc}") from exc
    matches.sort(key=lambda m: (m.start_time, m.match_id))
    return matches


def save_matches(path: str | Path, rows: Iterable[dict[str, object]]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            try:
                match = match_from_row(row)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            handle.write(
                json.dumps(
                    {
                        "match_id": match.match_id,
                        "start_time": match.start_time,
                        "radiant_win": match.radiant_win,
                        "avg_rank_tier": match.avg_rank_tier,
                        "radiant_team": list(match.radiant),
                        "dire_team": list(match.dire),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count


def time_split(
    matches: Sequence[Match], test_fraction: float = 0.2
) -> tuple[list[Match], list[Match]]:
    if not 0.05 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.05 and 0.5")
    if len(matches) < 20:
        raise ValueError("at least 20 matches are required")
    ordered = sorted(matches, key=lambda m: (m.start_time, m.match_id))
    cut = max(1, min(len(ordered) - 1, int(len(ordered) * (1.0 - test_fraction))))
    return ordered[:cut], ordered[cut:]


def hero_ids(matches: Sequence[Match]) -> list[int]:
    return sorted({hero for match in matches for hero in match.radiant + match.dire})


def build_index(ids: Sequence[int]) -> tuple[dict[int, int], np.ndarray]:
    unique = np.asarray(sorted(set(int(x) for x in ids)), dtype=np.int64)
    return {int(hero_id): i for i, hero_id in enumerate(unique)}, unique


def multihot_matches(
    matches: Sequence[Match], hero_to_index: dict[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(matches)
    h = len(hero_to_index)
    radiant = np.zeros((n, h), dtype=np.float32)
    dire = np.zeros((n, h), dtype=np.float32)
    labels = np.zeros(n, dtype=np.float32)
    for i, match in enumerate(matches):
        for hero_id in match.radiant:
            if hero_id in hero_to_index:
                radiant[i, hero_to_index[hero_id]] = 1.0
        for hero_id in match.dire:
            if hero_id in hero_to_index:
                dire[i, hero_to_index[hero_id]] = 1.0
        labels[i] = float(match.radiant_win)
    return radiant, dire, labels


def masked_policy_examples(
    matches: Sequence[Match],
    hero_to_index: dict[int, int],
    rng: np.random.Generator,
    repeats: int = 1,
) -> Iterator[tuple[np.ndarray, np.ndarray, int, int]]:
    """Yield ally/enemy multihot, phase index, and an observed completion target."""

    h = len(hero_to_index)
    known_by_phase = (0, 2, 4)
    for match in matches:
        for _ in range(repeats):
            for phase_index, known_count in enumerate(known_by_phase):
                perspective_radiant = bool(rng.integers(0, 2))
                own = match.radiant if perspective_radiant else match.dire
                enemy = match.dire if perspective_radiant else match.radiant
                target = int(rng.choice(own))
                remaining_own = [x for x in own if x != target]
                known_own = (
                    rng.choice(remaining_own, size=known_count, replace=False).tolist()
                    if known_count
                    else []
                )
                known_enemy = (
                    rng.choice(enemy, size=known_count, replace=False).tolist()
                    if known_count
                    else []
                )
                ally_vec = np.zeros(h, dtype=np.float32)
                enemy_vec = np.zeros(h, dtype=np.float32)
                for hero_id in known_own:
                    ally_vec[hero_to_index[int(hero_id)]] = 1.0
                for hero_id in known_enemy:
                    enemy_vec[hero_to_index[int(hero_id)]] = 1.0
                yield ally_vec, enemy_vec, phase_index, hero_to_index[target]

