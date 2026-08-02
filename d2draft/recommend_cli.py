from __future__ import annotations

import argparse
import json
from pathlib import Path

from .recommender import (
    DEFAULT_CANDIDATE_POOL,
    DEFAULT_POLICY_SURPRISE,
    HeroCatalog,
    HybridRecommender,
    resolve_many,
)
from .state import DraftState


def comma_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Dota 2 ranked draft recommender")
    parser.add_argument("--model", default="artifacts/mvp/hybrid_model.npz")
    parser.add_argument("--heroes", default="data/heroes.json")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--allies", default="", help="comma-separated hero names or IDs")
    parser.add_argument("--enemies", default="", help="comma-separated hero names or IDs")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--value-blend",
        type=float,
        default=None,
        help=(
            "override the phase-specific validation-selected weight; "
            "0 uses only observed-pick policy"
        ),
    )
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=DEFAULT_CANDIDATE_POOL,
        help=(
            "rank this many of the heroes players are most likely to pick ahead of "
            "the rest; 0 disables the pool and ranks every legal hero by win "
            "probability, which produces a nearly fixed list"
        ),
    )
    parser.add_argument(
        "--policy-surprise",
        type=float,
        default=DEFAULT_POLICY_SURPRISE,
        help=(
            "weight on how far this draft moved human preference for a hero, "
            "relative to that hero's popularity for the round; 0 ranks purely by "
            "predicted win probability and cannot see heroes clashing"
        ),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    catalog = HeroCatalog(Path(args.heroes))
    state = DraftState(
        phase=args.phase,
        allies=resolve_many(catalog, comma_values(args.allies)),
        enemies=resolve_many(catalog, comma_values(args.enemies)),
    )
    recommender = HybridRecommender(Path(args.model), catalog)
    value_blend = args.value_blend
    if value_blend is None:
        defaults = {1: 0.0, 2: 0.1, 3: 0.1}
        backtest_path = Path(args.model).with_name("backtest.json")
        if backtest_path.exists():
            backtest = json.loads(backtest_path.read_text(encoding="utf-8"))
            selected = backtest.get("selected_value_blend", {})
            value_blend = float(selected.get(f"phase_{args.phase}", defaults[args.phase]))
        else:
            value_blend = defaults[args.phase]
    recommendations, policy_kind = recommender.recommend(
        state,
        top_k=args.top_k,
        value_blend=value_blend,
        candidate_pool=args.candidate_pool or None,
        policy_surprise=args.policy_surprise,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "phase": state.phase,
                    "allies": state.allies,
                    "enemies": state.enemies,
                    "policy_model": policy_kind,
                    "recommendation_objective": recommender.objective,
                    "value_blend": value_blend,
                    "recommendations": [recommendation.__dict__ for recommendation in recommendations],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print(f"Phase {state.phase} | Objective={recommender.objective}")
    print("rank  hero                         pick_p     win_p     roles")
    for item in recommendations:
        roles = ", ".join(item.roles[:3])
        if item.predicted_win_probability is not None:
            score_text = f"{item.predicted_win_probability:>7.2%}"
        else:
            score_text = f"{item.value_log_odds_delta:>+7.3f}"
        print(
            f"{item.rank:>4}  {item.name:<27} "
            f"{item.policy_probability:>7.2%}  {score_text}  {roles}"
        )


if __name__ == "__main__":
    main()
