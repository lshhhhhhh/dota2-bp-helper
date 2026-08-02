from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path

import numpy as np

from d2draft.model_bundle import ModelBundle
from d2draft.model_updates import (
    MAX_BUNDLE_BYTES,
    ModelUpdateError,
    ModelUpdater,
    UpdateState,
    merge_bundles,
    parse_index,
    safe_extract,
)


HERO_IDS = (1, 2, 3, 4, 5)
INDEX_URL = "https://models.example.test/model-index.json"
BUNDLE_URL = "https://models.example.test/legend-plus.zip"
ARRAY_NAMES = (
    "hero_strength",
    "value_weight",
    "value_bias",
    "phase_frequency",
    "policy_w1",
    "policy_b1",
    "policy_w2",
    "policy_b2",
    "outcome_candidate_bias",
    "outcome_state_strength",
    "outcome_synergy_candidate",
    "outcome_synergy_ally",
    "outcome_counter_candidate",
    "outcome_counter_enemy",
    "outcome_phase_bias",
)


def write_bundle(
    directory: Path,
    *,
    model_id: str,
    created_at: str,
    hero_ids: tuple[int, ...] = HERO_IDS,
    bracket: str = "legend_plus",
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / "hybrid_model.npz"
    arrays = {name: np.zeros((2, 2), dtype=np.float32) for name in ARRAY_NAMES}
    arrays["hero_ids"] = np.array(hero_ids, dtype=np.int64)
    np.savez(artifact, **arrays)
    manifest = {
        "format_version": 1,
        "model_id": model_id,
        "display_name": f"Dota 2 BP Outcome · {bracket}",
        "artifact": artifact.name,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "created_at_utc": created_at,
        "game_patches": ["7.41"],
        "hero_count": len(hero_ids),
        "rank_bracket": {"id": bracket, "label": bracket},
        "recommendation_objective": "maximize predicted match win probability",
    }
    (directory / "model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def zip_bundle(directory: Path, archive: Path, *, prefix: str = "") -> bytes:
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                handle.write(path, prefix + path.relative_to(directory).as_posix())
    return archive.read_bytes()


def index_payload(entries: list[dict], *, channel: str = "stable", fmt: int = 1) -> bytes:
    document = {
        "index_format": fmt,
        "channel": channel,
        "published_at_utc": "2026-08-05T00:00:00+00:00",
        "models": entries,
    }
    return json.dumps(document).encode("utf-8")


def entry_for(payload: bytes, *, model_id: str, created_at: str, **overrides) -> dict:
    entry = {
        "rank_bracket_id": "legend_plus",
        "model_id": model_id,
        "display_name": "Dota 2 BP Outcome",
        "url": BUNDLE_URL,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bundle_format": 1,
        "hero_count": len(HERO_IDS),
        "game_patches": ["7.41"],
        "minimum_app_version": "0.4.0",
        "created_at_utc": created_at,
        "release_notes": {"zh": "新模型", "en": "New model"},
        "benchmark_summary": {"phase_3_auc": 0.571},
    }
    entry.update(overrides)
    return entry


class FakeResponse:
    def __init__(self, payload: bytes, headers: dict[str, str]) -> None:
        self._payload = payload
        self._offset = 0
        self.headers = headers

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._payload) - self._offset
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


class FakeOpener:
    def __init__(
        self,
        routes: dict[str, bytes],
        *,
        error: Exception | None = None,
        declare_length: bool = True,
    ) -> None:
        self.routes = routes
        self.error = error
        self.declare_length = declare_length
        self.requested: list[str] = []

    def open(self, request, timeout=None):  # noqa: ANN001 - urllib signature
        self.requested.append(request.full_url)
        if self.error is not None:
            raise self.error
        payload = self.routes.get(request.full_url)
        if payload is None:
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)
        headers = {"Content-Length": str(len(payload))} if self.declare_length else {}
        return FakeResponse(payload, headers)


class ModelUpdatesTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.tmp = Path(self._temporary.name)
        self.install_root = self.tmp / "install"
        self.source = self.tmp / "source"
        self.manifest = write_bundle(
            self.source, model_id="model-new", created_at="2026-08-05T00:00:00+00:00"
        )
        self.payload = zip_bundle(self.source, self.tmp / "bundle.zip")

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def updater(self, opener: FakeOpener, **kwargs) -> ModelUpdater:
        options = {
            "hero_ids": HERO_IDS,
            "install_root": self.install_root,
            "index_url": INDEX_URL,
            "app_version": "0.4.0",
            "opener": opener,
        }
        options.update(kwargs)
        return ModelUpdater(**options)

    def opener_for(self, entries: list[dict], **kwargs) -> FakeOpener:
        return FakeOpener(
            {INDEX_URL: index_payload(entries, **kwargs), BUNDLE_URL: self.payload}
        )

    def default_entry(self, **overrides) -> dict:
        entry = entry_for(
            self.payload,
            model_id="model-new",
            created_at="2026-08-05T00:00:00+00:00",
        )
        entry.update(overrides)
        return entry

    def test_install_then_rollback(self) -> None:
        updater = self.updater(self.opener_for([self.default_entry()]))
        updates = updater.check(updater.installed_bundles())
        self.assertEqual([remote.model_id for remote in updates], ["model-new"])

        installed = updater.download_and_install(updates[0])
        self.assertEqual(installed.model_id, "model-new")
        self.assertEqual(
            [bundle.model_id for bundle in updater.installed_bundles()], ["model-new"]
        )
        self.assertFalse(updater.has_previous("legend_plus"))

        newer_source = self.tmp / "newer"
        write_bundle(
            newer_source, model_id="model-newer", created_at="2026-08-09T00:00:00+00:00"
        )
        newer_payload = zip_bundle(newer_source, self.tmp / "newer.zip")
        opener = FakeOpener(
            {
                INDEX_URL: index_payload(
                    [
                        entry_for(
                            newer_payload,
                            model_id="model-newer",
                            created_at="2026-08-09T00:00:00+00:00",
                        )
                    ]
                ),
                BUNDLE_URL: newer_payload,
            }
        )
        updater = self.updater(opener)
        updates = updater.check(updater.installed_bundles())
        self.assertEqual([remote.model_id for remote in updates], ["model-newer"])
        updater.download_and_install(updates[0])
        self.assertTrue(updater.has_previous("legend_plus"))

        restored = updater.rollback("legend_plus")
        self.assertEqual(restored.model_id, "model-new")
        self.assertEqual(
            ModelBundle.load(
                self.install_root / "models" / "legend_plus" / "previous",
                expected_hero_ids=HERO_IDS,
            ).model_id,
            "model-newer",
        )

    def test_already_current_model_is_not_offered(self) -> None:
        updater = self.updater(self.opener_for([self.default_entry()]))
        updater.download_and_install(updater.check(updater.installed_bundles())[0])
        self.assertEqual(updater.check(updater.installed_bundles()), [])

    def test_corrupted_hash_keeps_current_model(self) -> None:
        updater = self.updater(self.opener_for([self.default_entry()]))
        updater.download_and_install(updater.check(updater.installed_bundles())[0])

        tampered = bytearray(self.payload)
        tampered[-1] ^= 0xFF
        opener = FakeOpener(
            {
                INDEX_URL: index_payload(
                    [
                        entry_for(
                            bytes(tampered),
                            model_id="model-tampered",
                            created_at="2026-08-09T00:00:00+00:00",
                            sha256=hashlib.sha256(self.payload).hexdigest(),
                        )
                    ]
                ),
                BUNDLE_URL: bytes(tampered),
            }
        )
        updater = self.updater(opener)
        remote = updater.check(updater.installed_bundles())[0]
        with self.assertRaises(ModelUpdateError):
            updater.download_and_install(remote)
        self.assertEqual(
            [bundle.model_id for bundle in updater.installed_bundles()], ["model-new"]
        )
        self.assertFalse((self.install_root / "models" / "legend_plus" / ".staging").exists())

    def test_oversized_declared_size_is_rejected(self) -> None:
        entry = self.default_entry(size=MAX_BUNDLE_BYTES + 1)
        updater = self.updater(self.opener_for([entry]))
        with self.assertRaises(ModelUpdateError):
            updater.fetch_index()

    def test_body_larger_than_declared_size_is_rejected(self) -> None:
        entry = self.default_entry(size=len(self.payload) - 16)
        opener = FakeOpener(
            {INDEX_URL: index_payload([entry]), BUNDLE_URL: self.payload},
            declare_length=False,
        )
        updater = self.updater(opener)
        remote = updater.fetch_index().models[0]
        with self.assertRaises(ModelUpdateError):
            updater.download_and_install(remote)

    def test_offline_check_reports_error_and_installs_nothing(self) -> None:
        opener = FakeOpener({}, error=urllib.error.URLError("offline"))
        updater = self.updater(opener)
        with self.assertRaises(ModelUpdateError):
            updater.check(updater.installed_bundles())
        state = updater.load_state()
        self.assertTrue(state.last_error)
        self.assertGreater(state.last_checked_at, 0.0)
        self.assertEqual(updater.installed_bundles(), [])

    def test_http_index_url_is_rejected(self) -> None:
        updater = self.updater(
            FakeOpener({}), index_url="http://models.example.test/model-index.json"
        )
        with self.assertRaises(ModelUpdateError):
            updater.fetch_index()

    def test_download_host_must_match_index_host(self) -> None:
        entry = self.default_entry(url="https://elsewhere.example.test/legend-plus.zip")
        updater = self.updater(self.opener_for([entry]))
        with self.assertRaises(ModelUpdateError):
            updater.fetch_index()

    def test_unexpected_channel_is_rejected(self) -> None:
        updater = self.updater(self.opener_for([self.default_entry()], channel="nightly"))
        with self.assertRaises(ModelUpdateError):
            updater.fetch_index()

    def test_unsupported_index_format_is_rejected(self) -> None:
        updater = self.updater(self.opener_for([self.default_entry()], fmt=99))
        with self.assertRaises(ModelUpdateError):
            updater.fetch_index()

    def test_incompatible_models_are_skipped(self) -> None:
        entries = [
            self.default_entry(model_id="a", rank_bracket_id="all", hero_count=126),
            self.default_entry(model_id="b", rank_bracket_id="archon_below", bundle_format=2),
            self.default_entry(model_id="c", minimum_app_version="9.0.0"),
        ]
        updater = self.updater(self.opener_for(entries))
        index = updater.fetch_index()
        self.assertEqual(updater.available_updates(index, []), [])
        self.assertEqual(
            [updater.incompatibility_reason(model) for model in index.models],
            ["hero_count", "bundle_format", "app_version"],
        )

    def test_hero_table_mismatch_is_rejected_during_install(self) -> None:
        foreign = self.tmp / "foreign"
        write_bundle(
            foreign,
            model_id="model-foreign",
            created_at="2026-08-09T00:00:00+00:00",
            hero_ids=(1, 2, 3, 4, 9),
        )
        payload = zip_bundle(foreign, self.tmp / "foreign.zip")
        opener = FakeOpener(
            {
                INDEX_URL: index_payload(
                    [
                        entry_for(
                            payload,
                            model_id="model-foreign",
                            created_at="2026-08-09T00:00:00+00:00",
                        )
                    ]
                ),
                BUNDLE_URL: payload,
            }
        )
        updater = self.updater(opener)
        remote = updater.fetch_index().models[0]
        with self.assertRaises(ModelUpdateError):
            updater.download_and_install(remote)
        self.assertEqual(updater.installed_bundles(), [])

    def test_model_id_must_match_the_published_entry(self) -> None:
        entry = self.default_entry(model_id="model-claimed")
        updater = self.updater(self.opener_for([entry]))
        with self.assertRaises(ModelUpdateError):
            updater.download_and_install(updater.fetch_index().models[0])

    def test_bundle_inside_a_single_top_level_directory(self) -> None:
        payload = zip_bundle(
            self.source, self.tmp / "nested.zip", prefix="legend-plus/"
        )
        opener = FakeOpener(
            {
                INDEX_URL: index_payload(
                    [
                        entry_for(
                            payload,
                            model_id="model-new",
                            created_at="2026-08-05T00:00:00+00:00",
                        )
                    ]
                ),
                BUNDLE_URL: payload,
            }
        )
        updater = self.updater(opener)
        installed = updater.download_and_install(updater.fetch_index().models[0])
        self.assertEqual(installed.model_id, "model-new")

    def test_rollback_without_previous_model_fails(self) -> None:
        updater = self.updater(FakeOpener({}))
        with self.assertRaises(ModelUpdateError):
            updater.rollback("legend_plus")

    def test_check_throttling_and_auto_update_toggle(self) -> None:
        now = [1_000_000.0]
        updater = self.updater(
            self.opener_for([self.default_entry()]), clock=lambda: now[0]
        )
        self.assertTrue(updater.should_check())
        updater.check(updater.installed_bundles())
        self.assertFalse(updater.should_check())

        now[0] += 24 * 60 * 60
        self.assertTrue(updater.should_check())

        state = updater.load_state()
        state.auto_update = False
        updater.save_state(state)
        self.assertFalse(updater.should_check())

    def test_update_state_survives_a_corrupt_file(self) -> None:
        self.install_root.mkdir(parents=True, exist_ok=True)
        (self.install_root / "update_state.json").write_text("{ broken", encoding="utf-8")
        state = UpdateState.load(self.install_root / "update_state.json")
        self.assertTrue(state.auto_update)
        self.assertEqual(state.last_checked_at, 0.0)

    def test_configured_index_url_must_be_https(self) -> None:
        self.install_root.mkdir(parents=True, exist_ok=True)
        (self.install_root / "update_config.json").write_text(
            json.dumps({"index_url": "http://insecure.example.test/model-index.json"}),
            encoding="utf-8",
        )
        updater = ModelUpdater(hero_ids=HERO_IDS, install_root=self.install_root)
        self.assertTrue(updater.index_url.startswith("https://"))

        (self.install_root / "update_config.json").write_text(
            json.dumps({"index_url": "https://mirror.example.test/model-index.json"}),
            encoding="utf-8",
        )
        updater = ModelUpdater(hero_ids=HERO_IDS, install_root=self.install_root)
        self.assertEqual(
            updater.index_url, "https://mirror.example.test/model-index.json"
        )

    def test_merge_prefers_the_newest_bundle_per_bracket(self) -> None:
        old_dir = self.tmp / "builtin"
        new_dir = self.tmp / "installed"
        write_bundle(old_dir, model_id="old", created_at="2026-07-01T00:00:00+00:00")
        write_bundle(new_dir, model_id="new", created_at="2026-08-05T00:00:00+00:00")
        other = self.tmp / "other"
        write_bundle(
            other,
            model_id="all-ranks",
            created_at="2026-07-01T00:00:00+00:00",
            bracket="all",
        )
        builtin = [
            ModelBundle.load(old_dir, expected_hero_ids=HERO_IDS),
            ModelBundle.load(other, expected_hero_ids=HERO_IDS),
        ]
        installed = [ModelBundle.load(new_dir, expected_hero_ids=HERO_IDS)]
        merged = merge_bundles(builtin, installed)
        self.assertEqual([bundle.model_id for bundle in merged], ["new", "all-ranks"])


class SafeExtractTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.tmp = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def archive_with(self, name: str, *, symlink: bool = False) -> Path:
        archive = self.tmp / "evil.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            info = zipfile.ZipInfo(name)
            if symlink:
                info.external_attr = (0o120777 << 16) | 0o200000
                handle.writestr(info, "../../secret.txt")
            else:
                handle.writestr(info, "owned")
        return archive

    def test_parent_directory_traversal_is_rejected(self) -> None:
        with self.assertRaises(ModelUpdateError):
            safe_extract(self.archive_with("../escaped.txt"), self.tmp / "out")
        self.assertFalse((self.tmp / "escaped.txt").exists())

    def test_nested_parent_traversal_is_rejected(self) -> None:
        with self.assertRaises(ModelUpdateError):
            safe_extract(self.archive_with("a/../../escaped.txt"), self.tmp / "out")

    def test_backslash_traversal_is_rejected(self) -> None:
        with self.assertRaises(ModelUpdateError):
            safe_extract(self.archive_with("..\\escaped.txt"), self.tmp / "out")

    def test_absolute_path_is_rejected(self) -> None:
        with self.assertRaises(ModelUpdateError):
            safe_extract(self.archive_with("/etc/hosts"), self.tmp / "out")

    def test_drive_qualified_path_is_rejected(self) -> None:
        with self.assertRaises(ModelUpdateError):
            safe_extract(self.archive_with("C:/windows/system32/evil.dll"), self.tmp / "out")

    def test_symlink_entry_is_rejected(self) -> None:
        with self.assertRaises(ModelUpdateError):
            safe_extract(self.archive_with("link.txt", symlink=True), self.tmp / "out")

    def test_too_many_entries_are_rejected(self) -> None:
        archive = self.tmp / "many.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            for number in range(128):
                handle.writestr(f"file-{number}.txt", "x")
        with self.assertRaises(ModelUpdateError):
            safe_extract(archive, self.tmp / "out")

    def test_zip_bomb_is_rejected(self) -> None:
        archive = self.tmp / "bomb.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
            handle.writestr("big.bin", b"\0" * (48 << 20))
        with self.assertRaises(ModelUpdateError):
            safe_extract(archive, self.tmp / "out")

    def test_unreadable_archive_is_rejected(self) -> None:
        archive = self.tmp / "broken.zip"
        archive.write_bytes(b"not a zip file")
        with self.assertRaises(ModelUpdateError):
            safe_extract(archive, self.tmp / "out")

    def test_valid_archive_extracts(self) -> None:
        archive = self.tmp / "good.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("model_manifest.json", "{}")
            handle.writestr("nested/report.json", "{}")
        destination = self.tmp / "out"
        safe_extract(archive, destination)
        self.assertTrue((destination / "model_manifest.json").is_file())
        self.assertTrue((destination / "nested" / "report.json").is_file())


class PublishRoundTripTest(unittest.TestCase):
    """What the publisher writes must be installable by the in-app updater."""

    SOURCE = Path(__file__).resolve().parents[1] / "artifacts" / "models" / "legend_plus"

    def setUp(self) -> None:
        if not (self.SOURCE / "model_manifest.json").is_file():
            self.skipTest("built-in model bundles are not present")
        self._temporary = tempfile.TemporaryDirectory()
        self.tmp = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_published_bundle_installs_and_loads(self) -> None:
        from d2draft.publish_models import build

        published = self.tmp / "published"
        index = build(
            (self.SOURCE,),
            published,
            base_url="https://models.example.test",
            minimum_app_version="0.4.0",
            channel="stable",
            notes_zh="测试",
            notes_en="test",
        )
        payload = (published / "model-index.json").read_bytes()
        parsed = parse_index(payload, index_url=INDEX_URL)
        self.assertEqual(len(parsed.models), 1)
        remote = parsed.models[0]

        archive = published / "legend-plus.zip"
        self.assertEqual(remote.size, archive.stat().st_size)
        self.assertEqual(
            remote.sha256, hashlib.sha256(archive.read_bytes()).hexdigest()
        )
        self.assertEqual(remote.rank_bracket_id, "legend_plus")
        self.assertIn("phase_3_auc", remote.benchmark_summary)
        self.assertEqual(index["models"][0]["model_id"], remote.model_id)

        hero_ids = ModelBundle.load(self.SOURCE).hero_ids
        updater = ModelUpdater(
            hero_ids=hero_ids,
            install_root=self.tmp / "install",
            index_url=INDEX_URL,
            app_version="0.4.0",
            opener=FakeOpener({INDEX_URL: payload, remote.url: archive.read_bytes()}),
        )
        installed = updater.download_and_install(updater.fetch_index().models[0])
        self.assertEqual(installed.model_id, remote.model_id)
        self.assertEqual(installed.hero_ids, hero_ids)
        self.assertTrue(installed.outcome_benchmark)

    def test_published_bundle_drives_the_recommender(self) -> None:
        from d2draft.publish_models import build
        from d2draft.recommender import HeroCatalog, HybridRecommender

        repository = Path(__file__).resolve().parents[1]
        published = self.tmp / "published"
        build(
            (self.SOURCE,),
            published,
            base_url="https://models.example.test",
            minimum_app_version="0.4.0",
            channel="stable",
            notes_zh="测试",
            notes_en="test",
        )
        extracted = self.tmp / "extracted"
        safe_extract(published / "legend-plus.zip", extracted)
        catalog = HeroCatalog(repository / "data" / "heroes.json")
        bundle = ModelBundle.load(extracted, expected_hero_ids=catalog.by_id)
        recommender = HybridRecommender(bundle.artifact_path, catalog)
        self.assertIsNotNone(recommender)


class IndexParsingTest(unittest.TestCase):
    def test_invalid_json_is_rejected(self) -> None:
        with self.assertRaises(ModelUpdateError):
            parse_index(b"{ not json", index_url=INDEX_URL)

    def test_invalid_sha256_is_rejected(self) -> None:
        payload = index_payload(
            [entry_for(b"x", model_id="m", created_at="", sha256="zz")]
        )
        with self.assertRaises(ModelUpdateError):
            parse_index(payload, index_url=INDEX_URL)

    def test_missing_identity_is_rejected(self) -> None:
        payload = index_payload(
            [entry_for(b"x", model_id="", created_at="")]
        )
        with self.assertRaises(ModelUpdateError):
            parse_index(payload, index_url=INDEX_URL)


if __name__ == "__main__":
    unittest.main()
