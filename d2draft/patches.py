from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


OPENDOTA_PATCHES_URL = (
    "https://raw.githubusercontent.com/odota/dotaconstants/master/build/patch.json"
)
STRATZ_URL = "https://api.stratz.com/graphql"

# Offline bootstrap. Network sync extends this table; it is intentionally safe
# to start the application without network access.
BOOTSTRAP_PATCHES = (
    ("7.38", "2025-02-19T13:48:29.412Z", "opendota", 57),
    ("7.39", "2025-05-22T23:36:01.602Z", "opendota", 58),
    ("7.40", "2025-12-16T00:50:40.281Z", "opendota", 59),
    ("7.40b", "2025-12-24T00:00:00Z", "stratz", 182),
    ("7.41", "2026-03-24T00:50:59.580Z", "opendota", 60),
)


def _timestamp(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def _ensure_column(
    connection: sqlite3.Connection, table: str, column: str, declaration: str
) -> None:
    columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def install_patch_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS patch_catalog (
            patch_name TEXT PRIMARY KEY,
            start_time INTEGER NOT NULL,
            source TEXT NOT NULL,
            synced_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS patch_catalog_start_time
            ON patch_catalog(start_time);
        CREATE TABLE IF NOT EXISTS patch_source_ids (
            source TEXT NOT NULL,
            source_patch_id INTEGER NOT NULL,
            patch_name TEXT NOT NULL,
            PRIMARY KEY(source, source_patch_id)
        );
        """
    )
    _ensure_column(connection, "candidates", "canonical_patch", "TEXT")
    _ensure_column(connection, "matches", "data_source", "TEXT")
    _ensure_column(connection, "matches", "source_patch_id", "INTEGER")
    _ensure_column(connection, "matches", "canonical_patch", "TEXT")
    now = int(time.time())
    for name, date, source, source_id in BOOTSTRAP_PATCHES:
        connection.execute(
            "INSERT INTO patch_catalog(patch_name,start_time,source,synced_at) VALUES(?,?,?,?) "
            "ON CONFLICT(patch_name) DO UPDATE SET "
            "start_time=excluded.start_time, source=excluded.source, synced_at=excluded.synced_at",
            (name, _timestamp(date), source, now),
        )
        connection.execute(
            "INSERT OR REPLACE INTO patch_source_ids(source,source_patch_id,patch_name) "
            "VALUES(?,?,?)",
            (source, source_id, name),
        )
    connection.commit()


def canonical_patch_for_time(connection: sqlite3.Connection, start_time: int) -> str | None:
    row = connection.execute(
        "SELECT patch_name FROM patch_catalog WHERE start_time <= ? "
        "ORDER BY start_time DESC LIMIT 1",
        (int(start_time),),
    ).fetchone()
    return str(row[0]) if row else None


def backfill_patch_metadata(connection: sqlite3.Connection) -> None:
    install_patch_schema(connection)
    connection.execute(
        "UPDATE candidates SET canonical_patch = ("
        "SELECT patch_name FROM patch_catalog WHERE patch_catalog.start_time <= candidates.start_time "
        "ORDER BY patch_catalog.start_time DESC LIMIT 1)"
    )
    connection.execute(
        "UPDATE matches SET source_patch_id = COALESCE(source_patch_id, patch), "
        "data_source = COALESCE(data_source, CASE WHEN match_seq_num IS NULL THEN 'stratz' ELSE 'opendota' END), "
        "canonical_patch = (SELECT patch_name FROM patch_catalog "
        "WHERE patch_catalog.start_time <= matches.start_time "
        "ORDER BY patch_catalog.start_time DESC LIMIT 1)"
    )
    connection.commit()


def _upsert_patch(
    connection: sqlite3.Connection,
    *,
    name: str,
    start_time: int,
    source: str,
    source_id: int,
) -> None:
    now = int(time.time())
    connection.execute(
        "INSERT INTO patch_catalog(patch_name,start_time,source,synced_at) VALUES(?,?,?,?) "
        "ON CONFLICT(patch_name) DO UPDATE SET start_time=excluded.start_time, synced_at=excluded.synced_at",
        (name, int(start_time), source, now),
    )
    connection.execute(
        "INSERT OR REPLACE INTO patch_source_ids(source,source_patch_id,patch_name) VALUES(?,?,?)",
        (source, int(source_id), name),
    )


def sync_patch_catalog(
    connection: sqlite3.Connection, *, stratz_token: str | None = None
) -> dict[str, int]:
    install_patch_schema(connection)
    with urllib.request.urlopen(OPENDOTA_PATCHES_URL, timeout=30) as response:
        opendota = json.load(response)
    for value in opendota:
        _upsert_patch(
            connection,
            name=str(value["name"]),
            start_time=_timestamp(str(value["date"])),
            source="opendota",
            source_id=int(value["id"]),
        )
    stratz_count = 0
    if stratz_token:
        query = "query { constants { gameVersions { id name asOfDateTime } } }"
        request = urllib.request.Request(
            STRATZ_URL,
            data=json.dumps({"query": query}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {stratz_token}",
                "Content-Type": "application/json",
                "User-Agent": "d2draft-collector/0.1",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
        versions = payload.get("data", {}).get("constants", {}).get("gameVersions", [])
        for value in versions:
            if value.get("id") is None or not value.get("name") or value.get("asOfDateTime") is None:
                continue
            _upsert_patch(
                connection,
                name=str(value["name"]),
                start_time=int(value["asOfDateTime"]),
                source="stratz",
                source_id=int(value["id"]),
            )
            stratz_count += 1
    connection.commit()
    backfill_patch_metadata(connection)
    return {"opendota": len(opendota), "stratz": stratz_count}


def patch_summary(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT canonical_patch,
               COUNT(*) AS stored_matches,
               SUM(reconstructable) AS reconstructable_matches,
               SUM(CASE WHEN data_source='stratz' THEN 1 ELSE 0 END) AS stratz_matches,
               SUM(CASE WHEN data_source='opendota' THEN 1 ELSE 0 END) AS opendota_matches,
               MIN(start_time) AS first_match_time,
               MAX(start_time) AS last_match_time
        FROM matches
        GROUP BY canonical_patch
        ORDER BY first_match_time
        """
    ).fetchall()
    return [
        {
            "canonical_patch": row[0],
            "stored_matches": int(row[1]),
            "reconstructable_matches": int(row[2] or 0),
            "stratz_matches": int(row[3] or 0),
            "opendota_matches": int(row[4] or 0),
            "first_match_time": int(row[5]),
            "last_match_time": int(row[6]),
        }
        for row in rows
    ]


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        if "=" not in raw or raw.lstrip().startswith("#"):
            continue
        name, value = raw.split("=", 1)
        if name.strip().casefold() in {"stratz_api", "stratz_api_key"}:
            os.environ.setdefault("STRATZ_API_KEY", value.strip().strip('"').strip("'"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dota patch catalog and database migration")
    parser.add_argument("--database", default="data/collection/draft_matches.sqlite3")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--sync", action="store_true")
    args = parser.parse_args()
    connection = sqlite3.connect(args.database)
    install_patch_schema(connection)
    if args.sync:
        _load_env(Path(args.env_file))
        print(json.dumps(sync_patch_catalog(connection, stratz_token=os.environ.get("STRATZ_API_KEY"))))
    else:
        backfill_patch_metadata(connection)
    print(json.dumps(patch_summary(connection), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
