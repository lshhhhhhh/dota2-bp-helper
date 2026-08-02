from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class ModelBundleError(ValueError):
    pass


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise ModelBundleError(f"missing model metadata: {path.name}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelBundleError(f"invalid model metadata: {path.name}") from exc
    if not isinstance(value, dict):
        raise ModelBundleError(f"model metadata must be an object: {path.name}")
    return value


@dataclass(frozen=True)
class ModelBundle:
    directory: Path
    manifest: dict[str, Any]
    report: dict[str, Any]
    backtest: dict[str, Any]
    advantage_benchmark: dict[str, Any]
    artifact_path: Path
    hero_ids: tuple[int, ...]

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        expected_hero_ids: Iterable[int] | None = None,
    ) -> "ModelBundle":
        root = Path(directory)
        manifest = _read_json(root / "model_manifest.json")
        if int(manifest.get("format_version", 0)) != 1:
            raise ModelBundleError("unsupported model bundle format")

        artifact_name = str(manifest.get("artifact", ""))
        if not artifact_name or Path(artifact_name).name != artifact_name:
            raise ModelBundleError("model artifact must be a filename inside its bundle")
        artifact_path = root / artifact_name
        if not artifact_path.exists():
            raise ModelBundleError(f"missing model artifact: {artifact_name}")

        expected_hash = str(manifest.get("artifact_sha256", "")).lower()
        actual_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if expected_hash and actual_hash != expected_hash:
            raise ModelBundleError("model artifact hash does not match its manifest")

        required_arrays = {
            "hero_ids",
            "hero_strength",
            "value_weight",
            "value_bias",
            "phase_frequency",
            "policy_w1",
            "policy_b1",
            "policy_w2",
            "policy_b2",
        }
        try:
            with np.load(artifact_path, allow_pickle=False) as artifact:
                missing = required_arrays.difference(artifact.files)
                if missing:
                    raise ModelBundleError(
                        "model artifact is missing arrays: " + ", ".join(sorted(missing))
                    )
                hero_ids = tuple(int(value) for value in artifact["hero_ids"])
        except (OSError, ValueError) as exc:
            if isinstance(exc, ModelBundleError):
                raise
            raise ModelBundleError("model artifact cannot be opened") from exc

        if len(hero_ids) != int(manifest.get("hero_count", -1)):
            raise ModelBundleError("model hero count does not match its manifest")
        if expected_hero_ids is not None and set(hero_ids) != set(expected_hero_ids):
            raise ModelBundleError("model hero table is incompatible with this app")

        return cls(
            directory=root,
            manifest=manifest,
            report=_read_json(root / "report.json", required=False),
            backtest=_read_json(root / "backtest.json", required=False),
            advantage_benchmark=_read_json(
                root / "advantage_benchmark.json", required=False
            ),
            artifact_path=artifact_path,
            hero_ids=hero_ids,
        )

    @property
    def model_id(self) -> str:
        return str(self.manifest.get("model_id", "unknown"))

    @property
    def display_name(self) -> str:
        return str(self.manifest.get("display_name", self.model_id))

    @property
    def patch_label(self) -> str:
        patches = self.manifest.get("game_patches", [])
        if isinstance(patches, list) and len(patches) == 1:
            return str(patches[0])
        return "混合版本" if patches else "未知"

    @property
    def short_hash(self) -> str:
        return str(self.manifest.get("artifact_sha256", ""))[:12] or "未知"

    @property
    def rank_bracket_id(self) -> str:
        bracket = self.manifest.get("rank_bracket", {})
        return str(bracket.get("id", "all")) if isinstance(bracket, dict) else "all"

    @property
    def rank_bracket_label(self) -> str:
        bracket = self.manifest.get("rank_bracket", {})
        return str(bracket.get("label", "全段位")) if isinstance(bracket, dict) else "全段位"


def write_model_manifest(
    directory: str | Path,
    *,
    generated_at_utc: str,
    patches: tuple[str, ...],
    hero_count: int,
    minimum_rank_tier: int | None = None,
    maximum_rank_tier_exclusive: int | None = None,
) -> dict[str, Any]:
    root = Path(directory)
    artifact = root / "hybrid_model.npz"
    patch_label = "-".join(patches) if patches else "mixed"
    compact_time = generated_at_utc.replace("-", "").replace(":", "").split(".")[0]
    if minimum_rank_tier == 50 and maximum_rank_tier_exclusive is None:
        rank_id, rank_label = "legend_plus", "传奇及以上"
    elif minimum_rank_tier is None and maximum_rank_tier_exclusive == 50:
        rank_id, rank_label = "archon_below", "统帅及以下"
    elif minimum_rank_tier is None and maximum_rank_tier_exclusive is None:
        rank_id, rank_label = "all", "全段位"
    else:
        lower = str(minimum_rank_tier) if minimum_rank_tier is not None else "any"
        upper = (
            str(maximum_rank_tier_exclusive)
            if maximum_rank_tier_exclusive is not None
            else "any"
        )
        rank_id, rank_label = f"rank_{lower}_{upper}", f"段位 {lower}–{upper}"
    manifest = {
        "format_version": 1,
        "model_id": f"mvp-{patch_label}-{rank_id}-{compact_time}",
        "display_name": f"Dota 2 BP Hybrid · {rank_label}",
        "artifact": artifact.name,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "created_at_utc": generated_at_utc,
        "game_patches": list(patches),
        "hero_count": hero_count,
        "rank_bracket": {
            "id": rank_id,
            "label": rank_label,
            "minimum_avg_rank_tier": minimum_rank_tier,
            "maximum_avg_rank_tier_exclusive": maximum_rank_tier_exclusive,
        },
        "model_family": "phase-aware hybrid recommender",
        "input_contract": "BP phase + allied hero IDs + enemy hero IDs",
        "output_contract": "ranked legal hero IDs with policy and value scores",
    }
    (root / "model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
