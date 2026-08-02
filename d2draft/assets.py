from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


CDN_BASE = "https://cdn.cloudflare.steamstatic.com"


def download_one(hero: dict[str, Any], output: Path) -> tuple[int, str | None]:
    hero_id = int(hero["id"])
    target = output / f"{hero_id}.png"
    if target.exists() and target.stat().st_size > 1_000:
        return hero_id, None
    url = CDN_BASE + str(hero["img"]).rstrip("?")
    request = urllib.request.Request(url, headers={"User-Agent": "d2draft-assets/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
        if len(payload) < 1_000:
            return hero_id, "response too small"
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(target)
        return hero_id, None
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        return hero_id, type(exc).__name__


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official Dota hero portrait templates")
    parser.add_argument("--heroes", default="data/heroes.json")
    parser.add_argument("--output", default="data/hero_portraits")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    raw = json.loads(Path(args.heroes).read_text(encoding="utf-8-sig"))
    heroes = list(raw.values())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, hero, output) for hero in heroes]
        for future in as_completed(futures):
            hero_id, error = future.result()
            if error:
                failures.append((hero_id, error))
    print(f"Portraits ready: {len(heroes) - len(failures)}/{len(heroes)}")
    if failures:
        print("Failures:", failures)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
