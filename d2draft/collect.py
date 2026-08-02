from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from .patches import canonical_patch_for_time, install_patch_schema


API_BASE = "https://api.opendota.com/api"
PRICE_PER_CALL_USD = 0.0001


def load_dotenv(path: Path) -> None:
    """Load supported settings without ever printing their values."""

    if not path.exists():
        return
    aliases = {
        "open_dota_api": "OPENDOTA_API_KEY",
        "opendota_api_key": "OPENDOTA_API_KEY",
        "OPENDOTA_API_KEY": "OPENDOTA_API_KEY",
        "stratz_api": "STRATZ_API_KEY",
        "stratz_api_key": "STRATZ_API_KEY",
        "STRATZ_API_KEY": "STRATZ_API_KEY",
    }
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        target = aliases.get(name.strip())
        if target and target not in os.environ:
            os.environ[target] = value.strip().strip('"').strip("'")


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self.interval = 1.0 / requests_per_second
        self.lock = threading.Lock()
        self.next_start = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_start - now)
            self.next_start = max(now, self.next_start) + self.interval
        if delay:
            time.sleep(delay)


class AttemptBudget:
    """A conservative request budget. Every network attempt consumes one unit."""

    def __init__(self, used: int, maximum: int) -> None:
        self.used = used
        self.maximum = maximum
        self.lock = threading.Lock()

    def claim(self) -> bool:
        with self.lock:
            if self.used >= self.maximum:
                return False
            self.used += 1
            return True


@dataclass
class FetchResult:
    match_id: int
    status: str
    detail: dict[str, Any] | None
    error: str | None
    source: str = "opendota"


def request_json(
    path: str,
    *,
    api_key: str | None = None,
    limiter: RateLimiter | None = None,
    budget: AttemptBudget | None = None,
    timeout: float = 30.0,
) -> Any:
    if budget is not None and not budget.claim():
        raise RuntimeError("request budget exhausted")
    if limiter is not None:
        limiter.wait()
    params = {"api_key": api_key} if api_key else {}
    url = f"{API_BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "d2draft-collector/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def explorer(sql: str, limiter: RateLimiter) -> list[dict[str, Any]]:
    limiter.wait()
    url = f"{API_BASE}/explorer?" + urllib.parse.urlencode({"sql": sql})
    request = urllib.request.Request(url, headers={"User-Agent": "d2draft-collector/0.1"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return list(json.loads(response.read().decode("utf-8")).get("rows", []))


def parse_team(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(hero) for hero in value]
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("{") and raw.endswith("}"):
            raw = "[" + raw[1:-1] + "]"
        return [int(hero) for hero in json.loads(raw)]
    return []


def init_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS candidates (
            match_id INTEGER PRIMARY KEY,
            start_time INTEGER NOT NULL,
            avg_rank_tier INTEGER,
            radiant_team TEXT NOT NULL,
            dire_team TEXT NOT NULL,
            radiant_win INTEGER NOT NULL,
            sample_day INTEGER NOT NULL,
            priority TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            error TEXT,
            canonical_patch TEXT
        );
        CREATE INDEX IF NOT EXISTS candidates_status_priority
            ON candidates(status, priority);

        CREATE TABLE IF NOT EXISTS matches (
            match_id INTEGER PRIMARY KEY,
            match_seq_num INTEGER,
            start_time INTEGER NOT NULL,
            duration INTEGER,
            patch INTEGER,
            region INTEGER,
            cluster INTEGER,
            avg_rank_tier INTEGER,
            radiant_win INTEGER NOT NULL,
            radiant_team TEXT NOT NULL,
            dire_team TEXT NOT NULL,
            raw_picks TEXT NOT NULL,
            final_picks TEXT NOT NULL,
            phase_1_radiant TEXT,
            phase_1_dire TEXT,
            phase_2_radiant TEXT,
            phase_2_dire TEXT,
            phase_3_radiant TEXT,
            phase_3_dire TEXT,
            player_ranks TEXT NOT NULL,
            reconstructable INTEGER NOT NULL,
            retrieved_at INTEGER NOT NULL,
            data_source TEXT,
            source_patch_id INTEGER,
            canonical_patch TEXT
        );

        CREATE TABLE IF NOT EXISTS state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    install_patch_schema(connection)
    connection.commit()
    return connection


def get_state_int(connection: sqlite3.Connection, key: str, default: int = 0) -> int:
    row = connection.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
    return int(row[0]) if row else default


def set_state(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        "INSERT INTO state(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )


def discover_candidates(
    connection: sqlite3.Connection,
    *,
    days: int,
    modulo: int,
    per_day_limit: int,
) -> int:
    existing = connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    if existing:
        print(
            f"Candidate database already contains {existing:,} rows; "
            "rescanning recent windows for unseen matches.",
            flush=True,
        )

    limiter = RateLimiter(0.8)
    latest_rows = explorer("SELECT MAX(start_time)::bigint AS latest FROM public_matches", limiter)
    newest = int(latest_rows[0]["latest"])
    inserted = 0
    for day in range(days):
        end = newest - day * 86400
        start = end - 86400
        remainder = (day * 37 + 11) % modulo
        sql = f"""
SELECT match_id, start_time, radiant_win, avg_rank_tier, radiant_team, dire_team
FROM public_matches
WHERE start_time >= {start}
  AND start_time < {end}
  AND lobby_type = 7
  AND game_mode = 22
  AND duration >= 600
  AND MOD(match_id, {modulo}) = {remainder}
ORDER BY match_id DESC
LIMIT {per_day_limit}
""".strip()
        rows = explorer(sql, limiter)
        before = connection.total_changes
        for row in rows:
            match_id = int(row["match_id"])
            radiant = parse_team(row.get("radiant_team"))
            dire = parse_team(row.get("dire_team"))
            if len(radiant) != 5 or len(dire) != 5:
                continue
            priority = hashlib.sha256(str(match_id).encode("ascii")).hexdigest()[:16]
            start_time = int(row["start_time"])
            canonical_patch = canonical_patch_for_time(connection, start_time)
            connection.execute(
                """
                INSERT OR IGNORE INTO candidates(
                    match_id, start_time, avg_rank_tier, radiant_team, dire_team,
                    radiant_win, sample_day, priority
                    , canonical_patch
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match_id,
                    start_time,
                    int(row["avg_rank_tier"]) if row.get("avg_rank_tier") is not None else None,
                    json.dumps(radiant, separators=(",", ":")),
                    json.dumps(dire, separators=(",", ":")),
                    int(bool(row["radiant_win"])),
                    day,
                    priority,
                    canonical_patch,
                ),
            )
        connection.commit()
        added = connection.total_changes - before
        inserted += added
        print(
            f"Candidate day {day + 1:02d}/{days}: {added:,} added "
            f"({inserted:,} total)",
            flush=True,
        )
    total = int(connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
    print(f"Candidate discovery complete: {inserted:,} new; {total:,} total.", flush=True)
    return total


def normalize_match(
    candidate: sqlite3.Row, detail: dict[str, Any], *, source: str = "opendota"
) -> dict[str, Any]:
    radiant = set(json.loads(candidate["radiant_team"]))
    dire = set(json.loads(candidate["dire_team"]))
    final_heroes = radiant | dire
    raw_picks = sorted(
        [event for event in (detail.get("picks_bans") or []) if bool(event.get("is_pick"))],
        key=lambda event: int(event.get("order", event.get("ord", 999))),
    )
    compact_raw = [
        {
            "hero_id": int(event.get("hero_id", 0)),
            "team": int(event.get("team", -1)),
            "order": int(event.get("order", event.get("ord", -1))),
        }
        for event in raw_picks
    ]
    final_picks = [event for event in compact_raw if event["hero_id"] in final_heroes]
    reconstructable = len(final_heroes) == 10 and len(final_picks) == 10
    phases: list[tuple[list[int], list[int]]] = []
    cursor = 0
    for width, expected_each in ((4, 2), (4, 2), (2, 1)):
        events = final_picks[cursor : cursor + width]
        r = [event["hero_id"] for event in events if event["team"] == 0]
        d = [event["hero_id"] for event in events if event["team"] == 1]
        reconstructable = reconstructable and len(r) == expected_each and len(d) == expected_each
        phases.append((r, d))
        cursor += width
    ranks = [
        int(player["rank_tier"]) if player.get("rank_tier") is not None else None
        for player in (detail.get("players") or [])
    ]
    return {
        "match_id": int(candidate["match_id"]),
        "match_seq_num": detail.get("match_seq_num"),
        "start_time": int(candidate["start_time"]),
        "duration": detail.get("duration"),
        "patch": detail.get("patch"),
        "region": detail.get("region"),
        "cluster": detail.get("cluster"),
        "avg_rank_tier": candidate["avg_rank_tier"],
        "radiant_win": int(candidate["radiant_win"]),
        "radiant_team": candidate["radiant_team"],
        "dire_team": candidate["dire_team"],
        "raw_picks": json.dumps(compact_raw, separators=(",", ":")),
        "final_picks": json.dumps(final_picks, separators=(",", ":")),
        "phases": phases,
        "player_ranks": json.dumps(ranks, separators=(",", ":")),
        "reconstructable": int(bool(reconstructable)),
        "retrieved_at": int(time.time()),
        "data_source": source,
        "source_patch_id": detail.get("patch"),
        "canonical_patch": candidate["canonical_patch"] if "canonical_patch" in candidate.keys() else None,
    }


def fetch_detail(
    match_id: int,
    *,
    api_key: str,
    limiter: RateLimiter,
    budget: AttemptBudget,
    retries: int,
) -> FetchResult:
    for attempt in range(retries + 1):
        try:
            detail = request_json(
                f"/matches/{match_id}",
                api_key=api_key,
                limiter=limiter,
                budget=budget,
            )
            return FetchResult(match_id, "ok", detail, None)
        except RuntimeError as exc:
            if "budget exhausted" in str(exc):
                return FetchResult(match_id, "budget", None, "request budget exhausted")
            return FetchResult(match_id, "error", None, type(exc).__name__)
        except urllib.error.HTTPError as exc:
            # Never stringify HTTPError: it contains the URL and therefore the API key.
            code = int(exc.code)
            if code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(min(8.0, 1.5 * 2**attempt))
                continue
            return FetchResult(match_id, "error", None, f"HTTP {code}")
        except (TimeoutError, urllib.error.URLError):
            if attempt < retries:
                time.sleep(min(8.0, 1.5 * 2**attempt))
                continue
            return FetchResult(match_id, "error", None, "network error")
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            return FetchResult(match_id, "error", None, type(exc).__name__)
    return FetchResult(match_id, "error", None, "unreachable")


def write_raw_batch(path: Path, details: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Each batch is a separate gzip member, so a stopped run only risks its current small batch.
    with path.open("ab") as raw_file:
        with gzip.GzipFile(fileobj=raw_file, mode="wb", compresslevel=6) as compressed:
            for detail in details:
                compressed.write(json.dumps(detail, separators=(",", ":")).encode("utf-8"))
                compressed.write(b"\n")


def store_batch(
    connection: sqlite3.Connection,
    batch: list[tuple[sqlite3.Row, FetchResult]],
    raw_path: Path,
    attempts_used: int,
) -> None:
    successful_raw = [result.detail for _, result in batch if result.detail is not None]
    if successful_raw:
        write_raw_batch(raw_path, successful_raw)
    for candidate, result in batch:
        if result.detail is None:
            status = "pending" if result.status == "budget" else "error"
            connection.execute(
                "UPDATE candidates SET status = ?, error = ? WHERE match_id = ?",
                (status, result.error, result.match_id),
            )
            continue
        normalized = normalize_match(candidate, result.detail, source=result.source)
        phases = normalized.pop("phases")
        connection.execute(
            """
            INSERT OR REPLACE INTO matches(
                match_id, match_seq_num, start_time, duration, patch, region, cluster,
                avg_rank_tier, radiant_win, radiant_team, dire_team, raw_picks, final_picks,
                phase_1_radiant, phase_1_dire, phase_2_radiant, phase_2_dire,
                phase_3_radiant, phase_3_dire, player_ranks, reconstructable, retrieved_at
                , data_source, source_patch_id, canonical_patch
            ) VALUES (
                :match_id, :match_seq_num, :start_time, :duration, :patch, :region, :cluster,
                :avg_rank_tier, :radiant_win, :radiant_team, :dire_team, :raw_picks, :final_picks,
                :phase_1_radiant, :phase_1_dire, :phase_2_radiant, :phase_2_dire,
                :phase_3_radiant, :phase_3_dire, :player_ranks, :reconstructable, :retrieved_at
                , :data_source, :source_patch_id, :canonical_patch
            )
            """,
            {
                **normalized,
                "phase_1_radiant": json.dumps(phases[0][0], separators=(",", ":")),
                "phase_1_dire": json.dumps(phases[0][1], separators=(",", ":")),
                "phase_2_radiant": json.dumps(phases[1][0], separators=(",", ":")),
                "phase_2_dire": json.dumps(phases[1][1], separators=(",", ":")),
                "phase_3_radiant": json.dumps(phases[2][0], separators=(",", ":")),
                "phase_3_dire": json.dumps(phases[2][1], separators=(",", ":")),
            },
        )
        connection.execute(
            "UPDATE candidates SET status = ?, error = NULL WHERE match_id = ?",
            ("usable" if normalized["reconstructable"] else "invalid", result.match_id),
        )
    set_state(connection, "paid_attempts", attempts_used)
    connection.commit()


def write_manifest(connection: sqlite3.Connection, path: Path, maximum_attempts: int) -> None:
    counts = dict(connection.execute("SELECT status, COUNT(*) FROM candidates GROUP BY status"))
    attempts = get_state_int(connection, "paid_attempts")
    payload = {
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "candidate_status": counts,
        "stored_matches": int(connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0]),
        "reconstructable_matches": int(
            connection.execute("SELECT COUNT(*) FROM matches WHERE reconstructable = 1").fetchone()[0]
        ),
        "paid_key_network_attempts": attempts,
        "conservative_cost_usd": round(attempts * PRICE_PER_CALL_USD, 4),
        "maximum_attempts": maximum_attempts,
        "maximum_conservative_cost_usd": round(maximum_attempts * PRICE_PER_CALL_USD, 2),
        "free_stratz_queries": get_state_int(connection, "stratz_queries"),
        "free_stratz_requested_matches": get_state_int(connection, "stratz_requested_matches"),
        "free_stratz_stored_matches": get_state_int(connection, "stratz_stored_matches"),
        "note": "Cost estimate counts every keyed network attempt as billable and ignores the free daily allowance.",
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def collect(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    connection = init_database(data_dir / "draft_matches.sqlite3")
    connection.row_factory = sqlite3.Row
    candidate_count = discover_candidates(
        connection,
        days=args.days,
        modulo=args.sample_modulo,
        per_day_limit=args.per_day_limit,
    )
    if candidate_count < args.target_details:
        raise RuntimeError(
            f"Only {candidate_count:,} candidates were found for target {args.target_details:,}. "
            "Increase --days or reduce --sample-modulo."
        )

    already_stored = int(connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0])
    if already_stored >= args.target_details:
        print(f"Target already satisfied: {already_stored:,} matches stored.", flush=True)
        write_manifest(connection, data_dir / "manifest.json", args.max_attempts)
        return

    used_attempts = get_state_int(connection, "paid_attempts")
    budget = AttemptBudget(used_attempts, args.max_attempts)
    limiter = RateLimiter(args.requests_per_second)
    rows = connection.execute(
        "SELECT * FROM candidates "
        "WHERE status IN ('pending', 'error', 'stratz_missing', 'stratz_invalid') "
        "ORDER BY CASE status "
        "WHEN 'stratz_missing' THEN 0 WHEN 'stratz_invalid' THEN 1 "
        "WHEN 'error' THEN 2 ELSE 3 END, priority"
    ).fetchall()
    remaining_target = args.target_details - already_stored
    rows = rows[: min(len(rows), remaining_target + args.workers * 4)]
    print(
        f"Starting details: stored={already_stored:,}, target={args.target_details:,}, "
        f"attempt_budget={used_attempts:,}/{args.max_attempts:,}, rps={args.requests_per_second:g}",
        flush=True,
    )

    started = time.monotonic()
    completed_this_run = 0
    batch: list[tuple[sqlite3.Row, FetchResult]] = []
    row_iter = iter(rows)
    inflight: dict[Any, sqlite3.Row] = {}

    def submit_one(pool: ThreadPoolExecutor) -> bool:
        try:
            row = next(row_iter)
        except StopIteration:
            return False
        future = pool.submit(
            fetch_detail,
            int(row["match_id"]),
            api_key=args.api_key,
            limiter=limiter,
            budget=budget,
            retries=args.retries,
        )
        inflight[future] = row
        return True

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for _ in range(min(args.workers * 2, len(rows))):
                if not submit_one(pool):
                    break
            while inflight:
                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for future in done:
                    row = inflight.pop(future)
                    result = future.result()
                    batch.append((row, result))
                    completed_this_run += int(result.detail is not None)
                    current_total = already_stored + completed_this_run
                    if len(batch) >= args.commit_every:
                        store_batch(connection, batch, data_dir / "raw_details.jsonl.gz", budget.used)
                        batch.clear()
                        write_manifest(connection, data_dir / "manifest.json", args.max_attempts)
                    if current_total < args.target_details and budget.used < budget.maximum:
                        submit_one(pool)
                if completed_this_run and completed_this_run % args.progress_every < len(done):
                    elapsed = max(0.001, time.monotonic() - started)
                    speed = completed_this_run / elapsed
                    usable = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM matches WHERE reconstructable = 1"
                        ).fetchone()[0]
                    )
                    print(
                        f"Progress stored~{already_stored + completed_this_run:,}/"
                        f"{args.target_details:,}; usable_committed={usable:,}; "
                        f"attempts={budget.used:,}/{budget.maximum:,}; "
                        f"speed={speed:.1f}/s; conservative_cost=${budget.used * PRICE_PER_CALL_USD:.2f}",
                        flush=True,
                    )
                if already_stored + completed_this_run >= args.target_details:
                    break
    except KeyboardInterrupt:
        print("Interrupted; committing completed responses before exit.", flush=True)
    finally:
        if batch:
            store_batch(connection, batch, data_dir / "raw_details.jsonl.gz", budget.used)
        else:
            set_state(connection, "paid_attempts", budget.used)
            connection.commit()
        write_manifest(connection, data_dir / "manifest.json", args.max_attempts)

    stored = int(connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0])
    usable = int(connection.execute("SELECT COUNT(*) FROM matches WHERE reconstructable = 1").fetchone()[0])
    print(
        f"Finished: stored={stored:,}; reconstructable={usable:,}; "
        f"attempts={budget.used:,}; conservative_cost=${budget.used * PRICE_PER_CALL_USD:.2f}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Budget-capped OpenDota ranked draft collector")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--data-dir", default="data/collection")
    # MVP-safe defaults stay below the observed remaining free daily allowance.
    # Spending requires an explicit larger --max-attempts supplied by the operator.
    parser.add_argument("--target-details", type=int, default=2_000)
    parser.add_argument("--max-attempts", type=int, default=2_000)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--sample-modulo", type=int, default=127)
    parser.add_argument("--per-day-limit", type=int, default=4_000)
    parser.add_argument("--requests-per-second", type=float, default=45.0)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--commit-every", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=1_000)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    load_dotenv(Path(args.env_file))
    args.api_key = os.environ.get("OPENDOTA_API_KEY")
    if not args.api_key:
        parser.error(
            "OpenDota key missing. Set OPENDOTA_API_KEY or open_dota_api in the selected .env file."
        )
    if args.max_attempts > 100_000:
        parser.error("Refusing max-attempts above 100,000 ($10 conservative ceiling).")
    # target-details is a cumulative database size, whereas max-attempts is a
    # cumulative network-attempt ceiling. After the first incremental run the
    # target can legitimately be larger than the request ceiling.
    if args.requests_per_second > 48:
        parser.error("requests-per-second above 48 leaves too little margin below the live limit")
    collect(args)


if __name__ == "__main__":
    main()
