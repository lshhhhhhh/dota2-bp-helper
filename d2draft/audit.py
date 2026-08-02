from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


API_BASE = "https://api.opendota.com/api"


def load_dotenv(path: Path) -> None:
    """Load only this tool's settings without logging any value."""

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in {"OPENDOTA_API_KEY", "OPENDOTA_MIN_INTERVAL"} and key not in os.environ:
            os.environ[key] = value


class OpenDotaClient:
    def __init__(self, api_key: str | None = None, min_interval: float = 1.05) -> None:
        self.api_key = api_key
        self.min_interval = min_interval
        self.last_request = 0.0
        self.remaining_minute: int | None = None
        self.remaining_day: int | None = None

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = dict(params or {})
        if self.api_key:
            query["api_key"] = self.api_key
        url = f"{API_BASE}{path}"
        if query:
            url += "?" + urllib.parse.urlencode(query)

        for attempt in range(5):
            wait = self.min_interval - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait)
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "d2draft-data-audit/0.1"},
            )
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    self.last_request = time.monotonic()
                    minute = response.headers.get("X-Rate-Limit-Remaining-Minute")
                    day = response.headers.get("X-Rate-Limit-Remaining-Day")
                    self.remaining_minute = int(minute) if minute is not None else None
                    self.remaining_day = int(day) if day is not None else None
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                self.last_request = time.monotonic()
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 4:
                    raise
                retry_after = float(exc.headers.get("Retry-After", 2**attempt))
                time.sleep(max(retry_after, 1.0))
            except (TimeoutError, urllib.error.URLError):
                self.last_request = time.monotonic()
                if attempt == 4:
                    raise
                time.sleep(float(2**attempt))
        raise RuntimeError("unreachable")

    def explorer(self, sql: str) -> list[dict[str, Any]]:
        payload = self.get_json("/explorer", {"sql": sql})
        return list(payload.get("rows", []))


def parse_pg_team(value: Any) -> tuple[int, ...]:
    if isinstance(value, list):
        return tuple(int(x) for x in value)
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("{") and raw.endswith("}"):
            raw = "[" + raw[1:-1] + "]"
        return tuple(int(x) for x in json.loads(raw))
    return ()


@dataclass
class PhaseAudit:
    match_id: int
    avg_rank_tier: int | None
    raw_pick_records: int
    final_pick_records: int
    discarded_collision_records: int
    has_picks_bans: bool
    reconstructable: bool
    phase_teams: list[list[int]]
    known_player_ranks: int
    immortal_players: int
    divine_or_higher_players: int
    minimum_known_rank_tier: int | None
    leaderboard_players: int
    error: str | None = None


def audit_phase_order(row: dict[str, Any], detail: dict[str, Any]) -> PhaseAudit:
    match_id = int(row["match_id"])
    radiant = set(parse_pg_team(row.get("radiant_team")))
    dire = set(parse_pg_team(row.get("dire_team")))
    final_ids = radiant | dire
    events = sorted(
        (event for event in (detail.get("picks_bans") or []) if bool(event.get("is_pick"))),
        key=lambda event: int(event.get("order", event.get("ord", 999))),
    )
    final_events = [event for event in events if int(event.get("hero_id", 0)) in final_ids]
    phase_teams: list[list[int]] = []
    valid = len(final_ids) == 10 and len(final_events) == 10
    expected = ((2, 2), (2, 2), (1, 1))
    cursor = 0
    if valid:
        for radiant_count, dire_count in expected:
            width = radiant_count + dire_count
            phase = final_events[cursor : cursor + width]
            teams = [int(event.get("team", -1)) for event in phase]
            phase_teams.append(teams)
            valid = valid and teams.count(0) == radiant_count and teams.count(1) == dire_count
            cursor += width
    players = detail.get("players") or []
    ranks = [int(player["rank_tier"]) for player in players if player.get("rank_tier")]
    leaderboard_players = sum(player.get("leaderboard_rank") is not None for player in players)
    return PhaseAudit(
        match_id=match_id,
        avg_rank_tier=int(row["avg_rank_tier"]) if row.get("avg_rank_tier") is not None else None,
        raw_pick_records=len(events),
        final_pick_records=len(final_events),
        discarded_collision_records=len(events) - len(final_events),
        has_picks_bans=bool(detail.get("picks_bans")),
        reconstructable=bool(valid),
        phase_teams=phase_teams,
        known_player_ranks=len(ranks),
        immortal_players=sum(rank >= 80 for rank in ranks),
        divine_or_higher_players=sum(rank >= 70 for rank in ranks),
        minimum_known_rank_tier=min(ranks) if ranks else None,
        leaderboard_players=leaderboard_players,
    )


def latest_valid_rows(
    client: OpenDotaClient, limit: int, min_avg_rank_tier: int = 70
) -> list[dict[str, Any]]:
    sql = f"""
SELECT match_id, start_time, radiant_win, avg_rank_tier, radiant_team, dire_team
FROM public_matches
WHERE avg_rank_tier >= {int(min_avg_rank_tier)}
  AND lobby_type = 7
  AND game_mode = 22
  AND duration >= 600
ORDER BY match_id DESC
LIMIT {int(limit)}
""".strip()
    return client.explorer(sql)


def daily_counts(
    client: OpenDotaClient, newest: int, days: int, min_avg_rank_tier: int
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for offset in range(days):
        end = newest - offset * 86400
        start = end - 86400
        sql = f"""
SELECT avg_rank_tier, COUNT(*) AS matches
FROM public_matches
WHERE start_time >= {start}
  AND start_time < {end}
  AND avg_rank_tier >= {int(min_avg_rank_tier)}
  AND lobby_type = 7
  AND game_mode = 22
  AND duration >= 600
GROUP BY avg_rank_tier
ORDER BY avg_rank_tier
""".strip()
        rows = client.explorer(sql)
        counts = {str(int(row["avg_rank_tier"])): int(row["matches"]) for row in rows}
        results.append(
            {
                "start_time": start,
                "end_time": end,
                "start_utc": datetime.fromtimestamp(start, UTC).isoformat(),
                "end_utc": datetime.fromtimestamp(end, UTC).isoformat(),
                "counts_by_avg_rank_tier": counts,
                "total": sum(counts.values()),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit OpenDota data before choosing a model")
    parser.add_argument("--days", type=int, default=3, help="24-hour windows to count")
    parser.add_argument("--detail-sample", type=int, default=30, help="match details to inspect")
    parser.add_argument(
        "--min-avg-rank-tier",
        type=int,
        default=75,
        help="minimum public_matches average tier for detail sampling",
    )
    parser.add_argument("--output", default="data/audit", help="output directory")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--use-env-key",
        action="store_true",
        help="use OPENDOTA_API_KEY from the env file (off by default for safe auditing)",
    )
    args = parser.parse_args()

    load_dotenv(Path(args.env_file))
    detected_api_key = os.environ.get("OPENDOTA_API_KEY")
    api_key = detected_api_key if args.use_env_key else None
    # Keep the public limit even when a key exists unless the caller explicitly overrides it.
    interval = float(os.environ.get("OPENDOTA_MIN_INTERVAL", "1.05"))
    client = OpenDotaClient(api_key=api_key, min_interval=interval)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows = latest_valid_rows(
        client, max(args.detail_sample, 100), min_avg_rank_tier=args.min_avg_rank_tier
    )
    if not rows:
        raise RuntimeError("OpenDota returned no matching public matches")
    newest = max(int(row["start_time"]) for row in rows)
    counts = daily_counts(client, newest, args.days, args.min_avg_rank_tier)

    audits: list[PhaseAudit] = []
    for index, row in enumerate(rows[: args.detail_sample], start=1):
        try:
            detail = client.get_json(f"/matches/{int(row['match_id'])}")
            audits.append(audit_phase_order(row, detail))
        except Exception as exc:  # keep an audit trail instead of hiding failed records
            audits.append(
                PhaseAudit(
                    match_id=int(row["match_id"]),
                    avg_rank_tier=int(row["avg_rank_tier"]) if row.get("avg_rank_tier") else None,
                    raw_pick_records=0,
                    final_pick_records=0,
                    discarded_collision_records=0,
                    has_picks_bans=False,
                    reconstructable=False,
                    phase_teams=[],
                    known_player_ranks=0,
                    immortal_players=0,
                    divine_or_higher_players=0,
                    minimum_known_rank_tier=None,
                    leaderboard_players=0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        if index % 10 == 0:
            print(f"Audited {index}/{args.detail_sample} match details", flush=True)

    with (output / "phase_sample.jsonl").open("w", encoding="utf-8") as handle:
        for audit in audits:
            handle.write(json.dumps(asdict(audit), ensure_ascii=False) + "\n")

    valid_arrays = 0
    for row in rows:
        radiant = parse_pg_team(row.get("radiant_team"))
        dire = parse_pg_team(row.get("dire_team"))
        if len(radiant) == len(dire) == 5 and len(set(radiant + dire)) == 10:
            valid_arrays += 1

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "filters": {
            "avg_rank_tier_min": args.min_avg_rank_tier,
            "lobby_type": 7,
            "game_mode": 22,
            "duration_min_seconds": 600,
        },
        "api_key_detected": bool(detected_api_key),
        "api_key_used": bool(api_key),
        "rate_limit_remaining_minute": client.remaining_minute,
        "rate_limit_remaining_day": client.remaining_day,
        "latest_sample_size": len(rows),
        "latest_sample_complete_final_lineups": valid_arrays,
        "daily_counts": counts,
        "detail_sample_size": len(audits),
        "detail_min_avg_rank_tier": args.min_avg_rank_tier,
        "picks_bans_available": sum(a.has_picks_bans for a in audits),
        "phase_reconstructable": sum(a.reconstructable for a in audits),
        "detail_errors": sum(a.error is not None for a in audits),
        "raw_pick_record_histogram": dict(Counter(a.raw_pick_records for a in audits)),
        "collision_record_histogram": dict(Counter(a.discarded_collision_records for a in audits)),
        "known_player_rank_histogram": dict(Counter(a.known_player_ranks for a in audits)),
        "immortal_player_count_histogram": dict(Counter(a.immortal_players for a in audits)),
        "matches_with_10_known_ranks": sum(a.known_player_ranks == 10 for a in audits),
        "matches_with_10_immortals": sum(a.immortal_players == 10 for a in audits),
        "matches_with_10_divine_or_higher": sum(
            a.divine_or_higher_players == 10 for a in audits
        ),
        "limitations": [
            "public_matches is a visibility-biased sample, not every Dota match",
            "avg_rank_tier 75 must not be relabeled Immortal without further evidence",
            "phase reconstruction filters attempted picks absent from the final lineup",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
