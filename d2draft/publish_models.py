"""Package model bundles and build the model-index.json used by in-app updates.

This only writes local files. Uploading the result is a manual step so that a
model release never replaces the app's "latest release" by accident: publish the
assets under a pre-release tag, which GitHub never reports as latest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .model_bundle import ModelBundle
from .model_updates import (
    DEFAULT_CHANNEL,
    MAX_BUNDLE_BYTES,
    SUPPORTED_BUNDLE_FORMAT,
    SUPPORTED_INDEX_FORMAT,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = (
    ROOT / "artifacts" / "models" / "legend_plus",
    ROOT / "artifacts" / "models" / "archon_below",
    ROOT / "artifacts" / "mvp",
)
BUNDLE_FILES = (
    "model_manifest.json",
    "hybrid_model.npz",
    "report.json",
    "backtest.json",
    "advantage_benchmark.json",
    "outcome_benchmark.json",
)
DEFAULT_TAG = "models-latest"
DEFAULT_REPOSITORY = "lshhhhhhh/dota2-bp-helper"


def _asset_name(bracket_id: str) -> str:
    return bracket_id.replace("_", "-") + ".zip"


def _phase_3_metrics(bundle: ModelBundle) -> dict[str, float]:
    try:
        metrics = bundle.outcome_benchmark["outcome_prediction_metrics"]["phase_3"]
    except (KeyError, TypeError):
        return {}
    summary: dict[str, float] = {}
    for source, target in (
        ("auc", "phase_3_auc"),
        ("log_loss", "phase_3_log_loss"),
        ("brier", "phase_3_brier"),
        ("matches", "phase_3_test_matches"),
    ):
        try:
            summary[target] = float(metrics[source])
        except (KeyError, TypeError, ValueError):
            continue
    return summary


def _pack(bundle: ModelBundle, destination: Path) -> Path:
    archive = destination / _asset_name(bundle.rank_bracket_id)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for name in BUNDLE_FILES:
            path = bundle.directory / name
            if path.is_file():
                handle.write(path, name)
    if archive.stat().st_size > MAX_BUNDLE_BYTES:
        raise SystemExit(f"{archive.name} is larger than the app's download limit")
    return archive


def build(
    sources: tuple[Path, ...],
    destination: Path,
    *,
    base_url: str,
    minimum_app_version: str,
    channel: str,
    notes_zh: str,
    notes_en: str,
) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for source in sources:
        bundle = ModelBundle.load(source)
        archive = _pack(bundle, destination)
        payload = archive.read_bytes()
        entries.append(
            {
                "rank_bracket_id": bundle.rank_bracket_id,
                "model_id": bundle.model_id,
                "display_name": bundle.display_name,
                "url": f"{base_url.rstrip('/')}/{archive.name}",
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bundle_format": SUPPORTED_BUNDLE_FORMAT,
                "hero_count": len(bundle.hero_ids),
                "game_patches": list(bundle.manifest.get("game_patches", [])),
                "minimum_app_version": minimum_app_version,
                "created_at_utc": str(bundle.manifest.get("created_at_utc", "")),
                "release_notes": {"zh": notes_zh, "en": notes_en},
                "benchmark_summary": _phase_3_metrics(bundle),
            }
        )

    index = {
        "index_format": SUPPORTED_INDEX_FORMAT,
        "channel": channel,
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": entries,
    }
    (destination / "model-index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "models")
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--base-url")
    parser.add_argument("--channel", default=DEFAULT_CHANNEL)
    parser.add_argument("--minimum-app-version", default=__version__)
    parser.add_argument("--notes-zh", default="常规模型更新。")
    parser.add_argument("--notes-en", default="Routine model update.")
    arguments = parser.parse_args()

    sources = tuple(arguments.source) if arguments.source else DEFAULT_SOURCES
    base_url = arguments.base_url or (
        f"https://github.com/{arguments.repository}/releases/download/{arguments.tag}"
    )
    index = build(
        sources,
        arguments.out,
        base_url=base_url,
        minimum_app_version=arguments.minimum_app_version,
        channel=arguments.channel,
        notes_zh=arguments.notes_zh,
        notes_en=arguments.notes_en,
    )

    print(f"wrote {arguments.out / 'model-index.json'}")
    for entry in index["models"]:
        print(
            f"  {entry['rank_bracket_id']:<13} {entry['model_id']}  "
            f"{entry['size'] / 1024:.0f} KB  {entry['sha256'][:12]}…"
        )
    assets = " ".join(
        f'"{(arguments.out / _asset_name(entry["rank_bracket_id"])).as_posix()}"'
        for entry in index["models"]
    )
    print(
        "\nPublish manually as a pre-release so the app's latest release is untouched:\n"
        f"  gh release create {arguments.tag} --repo {arguments.repository} "
        f"--prerelease --title \"Model bundles\" --notes \"{arguments.notes_en}\" "
        f'{assets} "{(arguments.out / "model-index.json").as_posix()}"\n'
        f"  gh release upload {arguments.tag} --repo {arguments.repository} --clobber "
        f'{assets} "{(arguments.out / "model-index.json").as_posix()}"'
    )


if __name__ == "__main__":
    main()
