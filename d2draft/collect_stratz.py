from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .collect import (
    FetchResult,
    get_state_int,
    init_database,
    load_dotenv,
    normalize_match,
    set_state,
    store_batch,
    write_manifest,
)


STRATZ_URL = "https://api.stratz.com/graphql"


def _query(rows: list[sqlite3.Row], token: str, timeout: float) -> dict[str, Any]:
    fields = "id gameVersionId pickBans { isPick heroId order isRadiant }"
    selections = " ".join(
        f'm{index}: match(id: {int(row["match_id"])}) {{ {fields} }}'
        for index, row in enumerate(rows)
    )
    body = json.dumps({"query": f"query DraftBatch {{ {selections} }}"}).encode("utf-8")
    request = urllib.request.Request(
        STRATZ_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "d2draft-collector/0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if payload.get("errors") and not payload.get("data"):
        messages = "; ".join(str(error.get("message", "GraphQL error")) for error in payload["errors"])
        raise RuntimeError(messages[:500])
    return dict(payload.get("data") or {})


def _opendota_shape(value: dict[str, Any]) -> dict[str, Any]:
    events = []
    for event in value.get("pickBans") or []:
        order = event.get("order")
        hero_id = event.get("heroId")
        if order is None or hero_id is None:
            continue
        events.append(
            {
                "is_pick": bool(event.get("isPick")),
                "hero_id": int(hero_id),
                "order": int(order),
                "team": 0 if bool(event.get("isRadiant")) else 1,
            }
        )
    return {
        "match_seq_num": None,
        "patch": value.get("gameVersionId"),
        "picks_bans": events,
        "players": [],
    }


def collect(args: argparse.Namespace) -> None:
    database = Path(args.database)
    connection = init_database(database)
    connection.row_factory = sqlite3.Row
    token = os.environ.get("STRATZ_API_KEY")
    if not token:
        raise RuntimeError("STRATZ_API_KEY is missing")

    initial_usable = int(
        connection.execute("SELECT COUNT(*) FROM matches WHERE reconstructable = 1").fetchone()[0]
    )
    if initial_usable >= args.target_usable:
        print(f"STRATZ target already satisfied: {initial_usable:,} usable matches.", flush=True)
        return
    rows = connection.execute(
        "SELECT * FROM candidates WHERE status = 'pending' ORDER BY priority"
    ).fetchall()
    total_queries = get_state_int(connection, "stratz_queries")
    total_requested = get_state_int(connection, "stratz_requested_matches")
    total_stored = get_state_int(connection, "stratz_stored_matches")
    run_queries = 0
    run_requested = 0
    run_stored = 0
    started = time.monotonic()
    interval = 1.0 / args.requests_per_second

    for offset in range(0, len(rows), args.batch_size):
        usable = int(
            connection.execute("SELECT COUNT(*) FROM matches WHERE reconstructable = 1").fetchone()[0]
        )
        if usable >= args.target_usable or run_queries >= args.max_queries:
            break
        batch = rows[offset : offset + args.batch_size]
        payload: dict[str, Any] | None = None
        for attempt in range(args.retries + 1):
            try:
                tick = time.monotonic()
                payload = _query(batch, token, args.timeout)
                elapsed = time.monotonic() - tick
                if elapsed < interval:
                    time.sleep(interval - elapsed)
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt >= args.retries:
                    raise RuntimeError(f"STRATZ HTTP {exc.code}") from exc
                time.sleep(min(15.0, 2.0**attempt))
            except (TimeoutError, urllib.error.URLError):
                if attempt >= args.retries:
                    raise
                time.sleep(min(15.0, 2.0**attempt))
        if payload is None:
            break

        run_queries += 1
        run_requested += len(batch)
        successful: list[tuple[sqlite3.Row, FetchResult]] = []
        for index, row in enumerate(batch):
            value = payload.get(f"m{index}")
            if not value or not value.get("pickBans"):
                connection.execute(
                    "UPDATE candidates SET status = 'stratz_missing', error = 'STRATZ pickBans unavailable' "
                    "WHERE match_id = ?",
                    (int(row["match_id"]),),
                )
                continue
            detail = _opendota_shape(value)
            normalized = normalize_match(row, detail)
            if not normalized["reconstructable"]:
                connection.execute(
                    "UPDATE candidates SET status = 'stratz_invalid', error = 'STRATZ BP not reconstructable' "
                    "WHERE match_id = ?",
                    (int(row["match_id"]),),
                )
                continue
            successful.append(
                (row, FetchResult(int(row["match_id"]), "ok", detail, None, "stratz"))
            )

        if successful:
            paid_attempts = get_state_int(connection, "paid_attempts")
            store_batch(
                connection,
                successful,
                database.with_name("raw_details_stratz.jsonl.gz"),
                paid_attempts,
            )
            run_stored += len(successful)
        total_queries += 1
        total_requested += len(batch)
        total_stored += len(successful)
        set_state(connection, "stratz_queries", total_queries)
        set_state(connection, "stratz_requested_matches", total_requested)
        set_state(connection, "stratz_stored_matches", total_stored)
        connection.commit()

        if run_queries % args.progress_every == 0:
            current_usable = int(
                connection.execute("SELECT COUNT(*) FROM matches WHERE reconstructable = 1").fetchone()[0]
            )
            speed = run_requested / max(time.monotonic() - started, 0.001)
            print(
                f"STRATZ progress queries={run_queries:,}/{args.max_queries:,}; "
                f"requested={run_requested:,}; stored={run_stored:,}; "
                f"usable={current_usable:,}/{args.target_usable:,}; {speed:.1f} matches/s",
                flush=True,
            )
            write_manifest(connection, database.with_name("manifest.json"), get_state_int(connection, "paid_attempts"))

    current_usable = int(
        connection.execute("SELECT COUNT(*) FROM matches WHERE reconstructable = 1").fetchone()[0]
    )
    write_manifest(connection, database.with_name("manifest.json"), get_state_int(connection, "paid_attempts"))
    print(
        f"STRATZ finished queries={run_queries:,}; requested={run_requested:,}; "
        f"stored={run_stored:,}; usable={current_usable:,}; "
        f"OpenDota paid attempts added=0",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Free batched STRATZ ordered-draft collector")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--database", default="data/collection/draft_matches.sqlite3")
    parser.add_argument("--target-usable", type=int, default=11_966)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-queries", type=int, default=600)
    parser.add_argument("--requests-per-second", type=float, default=2.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 50:
        parser.error("batch-size must be between 1 and 50")
    if not 0.1 <= args.requests_per_second <= 2.0:
        parser.error("requests-per-second must be between 0.1 and 2.0")
    if not 1 <= args.max_queries <= 1_000:
        parser.error("max-queries must be between 1 and 1,000")
    load_dotenv(Path(args.env_file))
    collect(args)


if __name__ == "__main__":
    main()
