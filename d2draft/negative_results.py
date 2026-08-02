"""Re-run the checks that ruled out three directions for improving the model.

Each of these looked promising and each was abandoned after measurement. They are
kept runnable so a future maintainer can confirm the numbers rather than retrying
the work. All three need the private collection database, which is never shipped.

    python -m d2draft.negative_results --check interactions
    python -m d2draft.negative_results --check composition
    python -m d2draft.negative_results --check margin-targets
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ARTIFACT = "artifacts/mvp/hybrid_model.npz"
DEFAULT_RAW = "data/collection/raw_details.jsonl.gz"


def _teams(match: dict) -> tuple[list[dict], list[dict]] | None:
    players = match.get("players")
    if not players or len(players) != 10 or match.get("radiant_win") is None:
        return None
    radiant = [p for p in players if p.get("player_slot", 255) < 128]
    dire = [p for p in players if p.get("player_slot", 255) >= 128]
    if len(radiant) != 5 or len(dire) != 5:
        return None
    return radiant, dire


def _stream(path: Path, limit: int):
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            try:
                match = json.loads(line)
            except ValueError:
                continue
            sides = _teams(match)
            if sides is None:
                continue
            yield match, sides
            count += 1
            if count >= limit:
                return


def interactions(artifact_path: str, **_: Any) -> None:
    """The synergy and counter embeddings are too small to reorder the top of the list."""

    with np.load(artifact_path, allow_pickle=False) as artifact:
        bias = artifact["outcome_candidate_bias"].astype(np.float64)
        synergy_candidate = artifact["outcome_synergy_candidate"].astype(np.float64)
        synergy_ally = artifact["outcome_synergy_ally"].astype(np.float64)
        counter_candidate = artifact["outcome_counter_candidate"].astype(np.float64)
        counter_enemy = artifact["outcome_counter_enemy"].astype(np.float64)

    heroes = len(bias)
    rng = np.random.default_rng(0)
    states = 5000
    interaction = np.empty((states, heroes))
    top_five = set()
    best = np.full(heroes, heroes)
    worst = np.zeros(heroes, dtype=int)
    for index in range(states):
        drafted = rng.choice(heroes, size=8, replace=False)
        value = synergy_ally[drafted[:4]].sum(0) @ synergy_candidate.T
        value = value + counter_enemy[drafted[4:]].sum(0) @ counter_candidate.T
        interaction[index] = value
        order = np.argsort(-(bias + value))
        top_five.update(order[:5].tolist())
        rank = np.argsort(np.argsort(-(bias + value)))
        best = np.minimum(best, rank)
        worst = np.maximum(worst, rank)

    print(f"heroes                                  {heroes}")
    print(f"candidate_bias spread (std)             {bias.std():.4f}")
    print(f"interaction spread (mean std)           {interaction.std(axis=1).mean():.4f}")
    print(f"ratio interaction / bias                {interaction.std(axis=1).mean() / bias.std():.3f}")
    print(f"distinct heroes ever in top 5           {len(top_five)} of {heroes}")
    print(f"median rank swing across states         {np.median(worst - best):.0f}")
    print(f"best rank reachable from outside top 20 {best[np.argsort(np.argsort(-bias)) >= 20].min() + 1}")

    pairs = heroes * (heroes - 1) // 2
    matches = 66_515
    per_pair = matches * 2 * 10 / pairs
    power = 1.96 + 0.84
    print(f"\nhero pairs                              {pairs:,}")
    print(f"ally-pair observations per pair         {per_pair:.0f}")
    print(f"minimum detectable win-rate gap         {power * math.sqrt(2 * 0.25 / per_pair) * 100:.1f} points")
    needed = 2 * power**2 * 0.25 / 0.02**2
    print(f"observations needed for a 2-point gap   {needed:,.0f}")
    print(f"matches that would require              {matches * needed / per_pair / 1e6:.1f} million")


def composition(raw_path: str, limit: int = 40_000, **_: Any) -> None:
    """Human drafts never become role-unbalanced, so there is nothing to detect."""

    positions: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(5))
    drafts: list[tuple[list[int], list[int], int]] = []
    for match, (radiant, dire) in _stream(Path(raw_path), limit):
        record = {}
        usable = True
        for side, players in (("radiant", radiant), ("dire", dire)):
            entries = []
            for player in players:
                hero, worth = player.get("hero_id"), player.get("net_worth")
                if not hero or worth is None:
                    usable = False
                    break
                entries.append((worth, hero))
            if not usable:
                break
            entries.sort(key=lambda item: -item[0])
            for rank, (_, hero) in enumerate(entries):
                positions[hero][rank] += 1
            record[side] = [hero for _, hero in entries]
        if usable:
            drafts.append((record["radiant"], record["dire"], int(match["radiant_win"])))

    print(f"matches parsed                          {len(drafts):,}")
    distribution = {
        hero: counts / counts.sum()
        for hero, counts in positions.items()
        if counts.sum() >= 200
    }
    entropy = []
    for values in distribution.values():
        share = values[values > 0]
        entropy.append(-(share * np.log(share)).sum() / np.log(5))
    print(f"heroes with >=200 games                 {len(distribution)}")
    print(f"position entropy (0 fixed, 1 uniform)   median {np.median(entropy):.3f}, min {min(entropy):.3f}")

    carry = {hero: values[0] for hero, values in distribution.items()}
    counts = []
    labels = []
    for radiant, dire, radiant_win in drafts:
        for team, won in ((radiant, radiant_win), (dire, 1 - radiant_win)):
            counts.append(sum(carry.get(hero, 0.2) for hero in team))
            labels.append(won)
    counts, labels = np.asarray(counts), np.asarray(labels, dtype=float)
    print(f"expected natural position-1 per team    mean {counts.mean():.2f}, std {counts.std():.2f}")
    print(f"  5th to 95th percentile                [{np.percentile(counts, 5):.2f}, {np.percentile(counts, 95):.2f}]")
    print("\nwin rate by composition bin:")
    edges = np.percentile(counts, [0, 10, 25, 50, 75, 90, 100])
    for index in range(len(edges) - 1):
        upper = counts <= edges[index + 1] if index == len(edges) - 2 else counts < edges[index + 1]
        selected = (counts >= edges[index]) & upper
        if selected.sum() > 50:
            print(
                f"  [{edges[index]:.2f}, {edges[index + 1]:.2f})  n={selected.sum():>6}"
                f"  win rate {labels[selected].mean():.4f}"
            )
    print("\nCaveat: position is proxied by end-of-game net worth rank, which is noisy,")
    print("and the position-1 distribution is estimated on the same matches.")


def margin_targets(artifact_path: str, raw_path: str, limit: int = 40_000, **_: Any) -> None:
    """Binary win/loss is a more efficient training target than any margin."""

    with np.load(artifact_path, allow_pickle=False) as artifact:
        hero_ids = artifact["hero_ids"].astype(int)
        strength = artifact["hero_strength"].astype(float)
    lookup = {int(hero): value for hero, value in zip(hero_ids, strength)}

    def standing(status: object) -> int:
        return bin(int(status or 0)).count("1")

    records = []
    for match, (radiant, dire) in _stream(Path(raw_path), limit):
        if not match.get("duration"):
            continue
        try:
            draft = sum(lookup[p["hero_id"]] for p in radiant) - sum(
                lookup[p["hero_id"]] for p in dire
            )
            worth = sum(p["net_worth"] for p in radiant) - sum(p["net_worth"] for p in dire)
        except (KeyError, TypeError):
            continue
        records.append(
            (
                draft,
                int(match["radiant_win"]),
                worth,
                match["duration"],
                standing(match.get("tower_status_radiant")) - standing(match.get("tower_status_dire")),
                standing(match.get("barracks_status_radiant")) - standing(match.get("barracks_status_dire")),
                match.get("radiant_score", 0) - match.get("dire_score", 0),
            )
        )

    table = np.asarray(records, dtype=float)
    print(f"matches parsed                          {len(table):,}")
    predictor = table[:, 0]
    predictor = (predictor - predictor.mean()) / predictor.std()
    targets = {
        "binary win/loss (in use)": table[:, 1],
        "kill difference": table[:, 6],
        "barracks difference": table[:, 5],
        "tower difference": table[:, 4],
        "net worth difference": table[:, 2],
        "net worth per minute": table[:, 2] / (table[:, 3] / 60.0),
    }
    print(f"\n{'target':<28}{'t':>10}{'R^2':>10}{'relative':>10}")
    baseline = None
    for name, values in targets.items():
        values = (values - values.mean()) / values.std()
        beta = float((predictor * values).mean())
        error = math.sqrt((1 - beta * beta) / (len(predictor) - 2))
        t = beta / error
        baseline = baseline or t
        print(f"{name:<28}{t:>10.2f}{beta * beta:>10.5f}{t / baseline:>9.2f}x")
    print("\nThe draft explains ~1.8% of outcome variance at best; margins add")
    print("skill-gap and duration variance, which is noise for this purpose.")


CHECKS = {
    "interactions": interactions,
    "composition": composition,
    "margin-targets": margin_targets,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", choices=sorted(CHECKS), required=True)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    parser.add_argument("--raw", default=DEFAULT_RAW)
    parser.add_argument("--limit", type=int, default=40_000)
    arguments = parser.parse_args()
    CHECKS[arguments.check](
        artifact_path=arguments.artifact, raw_path=arguments.raw, limit=arguments.limit
    )


if __name__ == "__main__":
    main()
