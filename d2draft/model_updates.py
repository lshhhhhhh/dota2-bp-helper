from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from . import __version__
from .model_bundle import ModelBundle, ModelBundleError


class ModelUpdateError(RuntimeError):
    """An update failed. ``kind`` selects the message the desktop app shows."""

    def __init__(self, message: str, *, kind: str = "verify") -> None:
        super().__init__(message)
        self.kind = kind


SUPPORTED_INDEX_FORMAT = 1
SUPPORTED_BUNDLE_FORMAT = 1
DEFAULT_CHANNEL = "stable"
DEFAULT_INDEX_URL = (
    "https://github.com/lshhhhhhh/dota2-bp-helper/releases/download/"
    "models-latest/model-index.json"
)
DEFAULT_TIMEOUT = 20.0
CHECK_INTERVAL_SECONDS = 24 * 60 * 60
MAX_INDEX_BYTES = 1 << 20
MAX_BUNDLE_BYTES = 16 << 20
MAX_EXTRACTED_BYTES = 32 << 20
MAX_ZIP_ENTRIES = 64
_CHUNK = 64 * 1024
USER_AGENT = (
    f"Dota2BPHelper/{__version__} "
    "(+https://github.com/lshhhhhhh/dota2-bp-helper)"
)


def default_install_root() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
    if base:
        return Path(base) / "Dota2BPHelper"
    return Path.home() / ".dota2_bp_helper"


def _version_tuple(value: object) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in str(value or "0").split("."):
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        parts.append(int(digits) if digits else 0)
    return tuple(parts) or (0,)


def _parse_timestamp(value: object) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return 0.0
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


class _HttpsOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise ModelUpdateError(
                "the update server redirected to a non-HTTPS address", kind="network"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _require_https(url: str, *, label: str) -> urllib.parse.SplitResult:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ModelUpdateError(f"{label} must be an HTTPS address", kind="network")
    return parsed


@dataclass(frozen=True)
class RemoteModel:
    rank_bracket_id: str
    model_id: str
    display_name: str
    url: str
    size: int
    sha256: str
    bundle_format: int
    hero_count: int
    game_patches: tuple[str, ...]
    minimum_app_version: str
    created_at_utc: str
    release_notes: dict[str, str] = field(default_factory=dict)
    benchmark_summary: dict[str, float] = field(default_factory=dict)

    @property
    def created_at(self) -> float:
        return _parse_timestamp(self.created_at_utc)

    @property
    def patch_label(self) -> str:
        if len(self.game_patches) == 1:
            return self.game_patches[0]
        return "mixed" if self.game_patches else "unknown"

    def notes(self, language: str) -> str:
        return (
            self.release_notes.get(language)
            or self.release_notes.get("en")
            or self.release_notes.get("zh")
            or ""
        )


@dataclass(frozen=True)
class ModelIndex:
    channel: str
    published_at_utc: str
    models: tuple[RemoteModel, ...]


def _as_object(payload: object, *, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ModelUpdateError(f"{label} is not a JSON object")
    return payload


def _parse_remote_model(payload: object, *, index_host: str) -> RemoteModel:
    entry = _as_object(payload, label="model index entry")

    parsed = _require_https(str(entry.get("url", "")), label="model download URL")
    if parsed.hostname.lower() != index_host:
        raise ModelUpdateError("model download URL must use the model index host")

    digest = str(entry.get("sha256", "")).strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ModelUpdateError("model index entry has an invalid SHA-256")

    try:
        size = int(entry.get("size", 0))
        bundle_format = int(entry.get("bundle_format", 0))
        hero_count = int(entry.get("hero_count", 0))
    except (TypeError, ValueError) as exc:
        raise ModelUpdateError("model index entry has a non-numeric field") from exc
    if not 0 < size <= MAX_BUNDLE_BYTES:
        raise ModelUpdateError("model index entry declares an unsupported size")

    bracket = str(entry.get("rank_bracket_id", "")).strip()
    model_id = str(entry.get("model_id", "")).strip()
    if not bracket or not model_id:
        raise ModelUpdateError("model index entry is missing its identity")

    patches = entry.get("game_patches", [])
    if not isinstance(patches, list):
        raise ModelUpdateError("model index entry has an invalid patch list")

    notes = entry.get("release_notes", {})
    summary = entry.get("benchmark_summary", {})
    return RemoteModel(
        rank_bracket_id=bracket,
        model_id=model_id,
        display_name=str(entry.get("display_name", model_id)),
        url=parsed.geturl(),
        size=size,
        sha256=digest,
        bundle_format=bundle_format,
        hero_count=hero_count,
        game_patches=tuple(str(patch) for patch in patches),
        minimum_app_version=str(entry.get("minimum_app_version", "0")),
        created_at_utc=str(entry.get("created_at_utc", "")),
        release_notes={
            str(key): str(value)
            for key, value in (notes.items() if isinstance(notes, dict) else ())
        },
        benchmark_summary={
            str(key): float(value)
            for key, value in (summary.items() if isinstance(summary, dict) else ())
            if isinstance(value, (int, float))
        },
    )


def parse_index(payload: bytes, *, index_url: str) -> ModelIndex:
    host = _require_https(index_url, label="model index URL").hostname.lower()
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ModelUpdateError("the model index is not valid JSON") from exc

    document = _as_object(document, label="the model index")
    if int(document.get("index_format", 0) or 0) != SUPPORTED_INDEX_FORMAT:
        raise ModelUpdateError("this app does not support that model index format")

    entries = document.get("models", [])
    if not isinstance(entries, list):
        raise ModelUpdateError("the model index has an invalid model list")

    return ModelIndex(
        channel=str(document.get("channel", DEFAULT_CHANNEL)),
        published_at_utc=str(document.get("published_at_utc", "")),
        models=tuple(
            _parse_remote_model(entry, index_host=host) for entry in entries
        ),
    )


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    try:
        handle = zipfile.ZipFile(archive)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ModelUpdateError("the downloaded model is not a readable archive") from exc

    with handle as bundle:
        entries = bundle.infolist()
        if len(entries) > MAX_ZIP_ENTRIES:
            raise ModelUpdateError("the downloaded model contains too many files")
        if sum(int(entry.file_size) for entry in entries) > MAX_EXTRACTED_BYTES:
            raise ModelUpdateError("the downloaded model expands beyond the size limit")

        for entry in entries:
            mode = (entry.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ModelUpdateError("the downloaded model contains a symbolic link")
            if mode not in (0, 0o100000, 0o040000):
                raise ModelUpdateError("the downloaded model contains a special file")

            normalized = entry.filename.replace("\\", "/")
            if normalized.startswith("/"):
                raise ModelUpdateError("the downloaded model contains an absolute path")
            parts = [part for part in normalized.split("/") if part not in ("", ".")]
            if any(part == ".." for part in parts):
                raise ModelUpdateError(
                    "the downloaded model contains a parent directory reference"
                )
            if any(":" in part for part in parts):
                raise ModelUpdateError("the downloaded model contains an invalid path")
            if not parts:
                continue

            target = (root / Path(*parts)).resolve()
            if target != root and root not in target.parents:
                raise ModelUpdateError(
                    "the downloaded model escapes its destination directory"
                )
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with bundle.open(entry) as source, open(target, "wb") as sink:
                while True:
                    chunk = source.read(_CHUNK)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_EXTRACTED_BYTES:
                        raise ModelUpdateError(
                            "the downloaded model expands beyond the size limit"
                        )
                    sink.write(chunk)


def _locate_bundle_directory(root: Path) -> Path:
    if (root / "model_manifest.json").is_file():
        return root
    children = [child for child in sorted(root.iterdir()) if child.is_dir()]
    if len(children) == 1 and (children[0] / "model_manifest.json").is_file():
        return children[0]
    raise ModelUpdateError("the downloaded model has no model_manifest.json")


@dataclass
class UpdateState:
    auto_update: bool = True
    last_checked_at: float = 0.0
    last_error: str = ""
    acknowledged_model_ids: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "UpdateState":
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if not isinstance(document, dict):
            return cls()
        acknowledged = document.get("acknowledged_model_ids", [])
        return cls(
            auto_update=bool(document.get("auto_update", True)),
            last_checked_at=float(document.get("last_checked_at", 0.0) or 0.0),
            last_error=str(document.get("last_error", "")),
            acknowledged_model_ids=[
                str(value)
                for value in (acknowledged if isinstance(acknowledged, list) else ())
            ][-64:],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "auto_update": self.auto_update,
            "last_checked_at": self.last_checked_at,
            "last_error": self.last_error,
            "acknowledged_model_ids": self.acknowledged_model_ids[-64:],
        }
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)


class ModelUpdater:
    def __init__(
        self,
        *,
        hero_ids: Iterable[int],
        install_root: str | Path | None = None,
        index_url: str | None = None,
        channel: str = DEFAULT_CHANNEL,
        app_version: str = __version__,
        opener: Any | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.hero_ids = tuple(sorted({int(value) for value in hero_ids}))
        self.root = Path(install_root) if install_root else default_install_root()
        self.models_root = self.root / "models"
        self.state_path = self.root / "update_state.json"
        self.config_path = self.root / "update_config.json"
        self.channel = channel
        self.app_version = str(app_version)
        self.timeout = float(timeout)
        self._clock = clock
        self._opener = opener
        self.index_url = index_url or self._configured_index_url()

    def _configured_index_url(self) -> str:
        try:
            document = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return DEFAULT_INDEX_URL
        candidate = str(document.get("index_url", "")) if isinstance(document, dict) else ""
        try:
            _require_https(candidate, label="model index URL")
        except ModelUpdateError:
            return DEFAULT_INDEX_URL
        return candidate

    def _bracket_directory(self, bracket_id: str) -> Path:
        safe = "".join(
            char for char in bracket_id.lower() if char.isalnum() or char in "_-"
        )
        if not safe:
            raise ModelUpdateError("the model has an invalid rank bracket id")
        return self.models_root / safe

    def load_state(self) -> UpdateState:
        return UpdateState.load(self.state_path)

    def save_state(self, state: UpdateState) -> None:
        state.save(self.state_path)

    def should_check(self, state: UpdateState | None = None) -> bool:
        state = state or self.load_state()
        if not state.auto_update:
            return False
        return (self._clock() - state.last_checked_at) >= CHECK_INTERVAL_SECONDS

    def installed_bundles(self) -> list[ModelBundle]:
        bundles: list[ModelBundle] = []
        if not self.models_root.is_dir():
            return bundles
        for bracket in sorted(self.models_root.iterdir()):
            current = bracket / "current"
            if not (current / "model_manifest.json").is_file():
                continue
            try:
                bundles.append(
                    ModelBundle.load(current, expected_hero_ids=self.hero_ids)
                )
            except ModelBundleError:
                continue
        return bundles

    def has_previous(self, bracket_id: str) -> bool:
        previous = self._bracket_directory(bracket_id) / "previous"
        return (previous / "model_manifest.json").is_file()

    def incompatibility_reason(self, remote: RemoteModel) -> str | None:
        if remote.bundle_format != SUPPORTED_BUNDLE_FORMAT:
            return "bundle_format"
        if _version_tuple(self.app_version) < _version_tuple(remote.minimum_app_version):
            return "app_version"
        if remote.hero_count != len(self.hero_ids):
            return "hero_count"
        return None

    def fetch_index(self) -> ModelIndex:
        payload = self._read(self.index_url, limit=MAX_INDEX_BYTES)
        try:
            index = parse_index(payload, index_url=self.index_url)
        except ModelUpdateError as exc:
            raise ModelUpdateError(str(exc), kind="index") from exc
        if index.channel != self.channel:
            raise ModelUpdateError(
                "the model index is not on the expected channel", kind="index"
            )
        return index

    def available_updates(
        self, index: ModelIndex, installed: Iterable[ModelBundle]
    ) -> list[RemoteModel]:
        local = {bundle.rank_bracket_id: bundle for bundle in installed}
        updates: list[RemoteModel] = []
        for remote in index.models:
            if self.incompatibility_reason(remote) is not None:
                continue
            current = local.get(remote.rank_bracket_id)
            if current is not None and not _is_newer(remote, current):
                continue
            updates.append(remote)
        return updates

    def check(self, installed: Iterable[ModelBundle]) -> list[RemoteModel]:
        state = self.load_state()
        try:
            updates = self.available_updates(self.fetch_index(), installed)
        except ModelUpdateError as exc:
            state.last_checked_at = self._clock()
            state.last_error = str(exc)
            self.save_state(state)
            raise
        state.last_checked_at = self._clock()
        state.last_error = ""
        self.save_state(state)
        return updates

    def download_and_install(
        self,
        remote: RemoteModel,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> ModelBundle:
        reason = self.incompatibility_reason(remote)
        if reason is not None:
            raise ModelUpdateError(f"the model is not compatible with this app: {reason}")

        bracket = self._bracket_directory(remote.rank_bracket_id)
        bracket.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="d2bp-", dir=str(bracket)) as workspace:
            work = Path(workspace)
            payload = self._read(remote.url, limit=remote.size, progress=progress)
            if len(payload) != remote.size:
                raise ModelUpdateError("the downloaded model has an unexpected size")
            if hashlib.sha256(payload).hexdigest() != remote.sha256:
                raise ModelUpdateError(
                    "the downloaded model does not match its published SHA-256"
                )

            archive = work / "bundle.zip"
            archive.write_bytes(payload)
            extracted = work / "extracted"
            safe_extract(archive, extracted)
            source = _locate_bundle_directory(extracted)
            try:
                candidate = ModelBundle.load(source, expected_hero_ids=self.hero_ids)
            except ModelBundleError as exc:
                raise ModelUpdateError(str(exc)) from exc
            if candidate.model_id != remote.model_id:
                raise ModelUpdateError(
                    "the downloaded model does not match its published model id"
                )
            if candidate.rank_bracket_id != remote.rank_bracket_id:
                raise ModelUpdateError(
                    "the downloaded model does not match its published rank bracket"
                )

            staging = bracket / ".staging"
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            shutil.move(str(source), str(staging))
            self._install_staged(bracket, staging)

        installed = ModelBundle.load(bracket / "current", expected_hero_ids=self.hero_ids)
        state = self.load_state()
        if remote.model_id not in state.acknowledged_model_ids:
            state.acknowledged_model_ids.append(remote.model_id)
        self.save_state(state)
        return installed

    def _install_staged(self, bracket: Path, staging: Path) -> None:
        current = bracket / "current"
        previous = bracket / "previous"
        if previous.exists():
            shutil.rmtree(previous, ignore_errors=True)
        rotated = False
        if current.exists():
            os.replace(current, previous)
            rotated = True
        try:
            os.replace(staging, current)
        except OSError as exc:
            if rotated:
                os.replace(previous, current)
            raise ModelUpdateError(
                "the downloaded model could not be installed", kind="install"
            ) from exc

    def rollback(self, bracket_id: str) -> ModelBundle:
        bracket = self._bracket_directory(bracket_id)
        current = bracket / "current"
        previous = bracket / "previous"
        if not (previous / "model_manifest.json").is_file():
            raise ModelUpdateError("there is no previous model to restore", kind="install")

        spare = bracket / ".rollback"
        if spare.exists():
            shutil.rmtree(spare, ignore_errors=True)
        rotated = False
        if current.exists():
            os.replace(current, spare)
            rotated = True
        try:
            os.replace(previous, current)
        except OSError as exc:
            if rotated:
                os.replace(spare, current)
            raise ModelUpdateError(
                "the previous model could not be restored", kind="install"
            ) from exc
        if rotated:
            os.replace(spare, previous)
        try:
            return ModelBundle.load(current, expected_hero_ids=self.hero_ids)
        except ModelBundleError as exc:
            raise ModelUpdateError(str(exc)) from exc

    def _read(
        self,
        url: str,
        *,
        limit: int,
        progress: Callable[[int, int], None] | None = None,
    ) -> bytes:
        _require_https(url, label="the update address")
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"}
        )
        opener = self._opener or urllib.request.build_opener(_HttpsOnlyRedirectHandler)
        try:
            with opener.open(request, timeout=self.timeout) as response:
                declared = response.headers.get("Content-Length")
                if declared is not None and str(declared).isdigit():
                    if int(declared) > limit:
                        raise ModelUpdateError(
                            "the update server sent too much data", kind="network"
                        )
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > limit:
                        raise ModelUpdateError(
                            "the update server sent too much data", kind="network"
                        )
                    chunks.append(chunk)
                    if progress is not None:
                        progress(total, limit)
        except ModelUpdateError:
            raise
        except urllib.error.HTTPError as exc:
            raise ModelUpdateError(
                f"the update server returned HTTP {exc.code}", kind="network"
            ) from exc
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
            raise ModelUpdateError(
                "the update server could not be reached", kind="network"
            ) from exc
        return b"".join(chunks)


def _is_newer(remote: RemoteModel, local: ModelBundle) -> bool:
    if remote.model_id == local.model_id:
        return False
    return remote.created_at > bundle_created_at(local)


def bundle_created_at(bundle: ModelBundle) -> float:
    return _parse_timestamp(bundle.manifest.get("created_at_utc"))


BRACKET_PRIORITY = {"legend_plus": 0, "archon_below": 1, "all": 2}


def merge_bundles(
    builtin: Iterable[ModelBundle], installed: Iterable[ModelBundle]
) -> list[ModelBundle]:
    chosen: dict[str, ModelBundle] = {}
    for bundle in [*builtin, *installed]:
        current = chosen.get(bundle.rank_bracket_id)
        if current is None or bundle_created_at(bundle) > bundle_created_at(current):
            chosen[bundle.rank_bracket_id] = bundle
    return sorted(
        chosen.values(),
        key=lambda bundle: (
            BRACKET_PRIORITY.get(bundle.rank_bracket_id, 9),
            bundle.patch_label,
            bundle.model_id,
        ),
    )
