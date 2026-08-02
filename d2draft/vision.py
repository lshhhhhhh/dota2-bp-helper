from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps

from .recommender import HeroCatalog


Rect = tuple[int, int, int, int]


@dataclass(frozen=True)
class ViewportCandidate:
    rect: Rect
    score: float
    source: str = "auto"


@dataclass(frozen=True)
class CaptureConfig:
    screen_width: int
    screen_height: int
    allies_box: Rect
    enemies_box: Rect
    orientation: str = "horizontal"

    @classmethod
    def default_for_screen(cls, width: int, height: int) -> "CaptureConfig":
        """Scale the measured 2560x1440 Dota 7.41 top-card geometry."""

        sx, sy = width / 2560.0, height / 1440.0

        def scaled(rect: Rect) -> Rect:
            return tuple(
                int(round(value * (sx if index % 2 == 0 else sy)))
                for index, value in enumerate(rect)
            )  # type: ignore[return-value]

        return cls(
            screen_width=width,
            screen_height=height,
            allies_box=scaled((280, 0, 1100, 105)),
            enemies_box=scaled((1460, 0, 2280, 105)),
        )

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        temporary.replace(output)

    @classmethod
    def load(cls, path: str | Path) -> "CaptureConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            screen_width=int(raw["screen_width"]),
            screen_height=int(raw["screen_height"]),
            allies_box=tuple(int(x) for x in raw["allies_box"]),
            enemies_box=tuple(int(x) for x in raw["enemies_box"]),
            orientation=str(raw.get("orientation", "horizontal")),
        )


@dataclass(frozen=True)
class VisionMatch:
    slot: int
    hero_id: int | None
    name: str
    similarity: float
    margin: float
    accepted: bool


def split_slots(box: Rect, count: int = 5, orientation: str = "horizontal") -> list[Rect]:
    left, top, right, bottom = box
    if right <= left or bottom <= top:
        raise ValueError(f"invalid capture box: {box}")
    result: list[Rect] = []
    if orientation == "horizontal":
        edges = np.linspace(left, right, count + 1).round().astype(int)
        for index in range(count):
            result.append((int(edges[index]), top, int(edges[index + 1]), bottom))
    elif orientation == "vertical":
        edges = np.linspace(top, bottom, count + 1).round().astype(int)
        for index in range(count):
            result.append((left, int(edges[index]), right, int(edges[index + 1])))
    else:
        raise ValueError("orientation must be horizontal or vertical")
    return result


def locate_windowed_viewports(
    screenshot: Image.Image,
    *,
    max_candidates: int = 6,
    minimum_width_fraction: float = 0.25,
) -> list[ViewportCandidate]:
    """Find likely 16:9 video/game viewports inside a desktop screenshot.

    Windowed livestream players commonly letterbox 16:9 content.  Long transitions
    between a dark bar and textured content provide candidate top/bottom edges.  We
    deliberately return several candidates; Dota portrait recognition decides which
    one is the actual game viewport.
    """

    gray = np.asarray(screenshot.convert("L"), dtype=np.int16)
    height, width = gray.shape
    if height < 240 or width < 400:
        return []
    row_difference = np.abs(np.diff(gray, axis=0))
    minimum_width = max(400, int(round(width * minimum_width_fraction)))
    gap_tolerance = max(10, int(round(width * 0.007)))
    raw: list[ViewportCandidate] = []

    def add_candidate(left: int, top: int, right: int, bottom: int, score: float) -> None:
        if left < 0 or top < 0 or right > width or bottom > height:
            return
        if right - left < minimum_width or bottom - top < 225:
            return
        raw.append(ViewportCandidate((left, top, right, bottom), score))

    for row_index, difference in enumerate(row_difference):
        changed = np.flatnonzero(difference > 10)
        if len(changed) == 0 or int(changed[-1] - changed[0] + 1) < minimum_width:
            continue
        split_at = np.flatnonzero(np.diff(changed) > gap_tolerance) + 1
        for run in np.split(changed, split_at):
            if len(run) == 0:
                continue
            left, right = int(run[0]), int(run[-1]) + 1
            candidate_width = right - left
            if candidate_width < minimum_width:
                continue
            candidate_height = int(round(candidate_width * 9.0 / 16.0))
            boundary_y = row_index + 1
            transition = float(difference[left:right].mean())

            # Dark band above, textured content below: candidate top edge.
            above = gray[max(0, boundary_y - 12) : boundary_y, left:right]
            inside_below = gray[
                boundary_y : min(height, boundary_y + 40), left:right
            ]
            if above.size and inside_below.size:
                dark_fraction = float((above < 8).mean())
                texture = float(inside_below.std())
                score = (
                    dark_fraction * 50.0
                    + min(transition, 50.0)
                    + min(texture, 50.0)
                    + candidate_width / width * 10.0
                )
                if dark_fraction >= 0.45:
                    add_candidate(
                        left,
                        boundary_y,
                        right,
                        boundary_y + candidate_height,
                        score,
                    )

            # Textured content above, dark band below: candidate bottom edge.
            below = gray[
                boundary_y : min(height, boundary_y + 12), left:right
            ]
            inside_above = gray[
                max(0, boundary_y - 40) : boundary_y, left:right
            ]
            if below.size and inside_above.size:
                dark_fraction = float((below < 8).mean())
                texture = float(inside_above.std())
                score = (
                    dark_fraction * 50.0
                    + min(transition, 50.0)
                    + min(texture, 50.0)
                    + candidate_width / width * 10.0
                )
                if dark_fraction >= 0.45:
                    add_candidate(
                        left,
                        boundary_y - candidate_height,
                        right,
                        boundary_y,
                        score,
                    )

    # Collapse multiple adjacent edge rows into one candidate.
    selected: list[ViewportCandidate] = []
    for candidate in sorted(raw, key=lambda item: item.score, reverse=True):
        left, top, right, bottom = candidate.rect
        duplicate = any(
            abs(left - other.rect[0]) < 30
            and abs(top - other.rect[1]) < 20
            and abs(right - other.rect[2]) < 30
            and abs(bottom - other.rect[3]) < 20
            for other in selected
        )
        if not duplicate:
            selected.append(candidate)
        if len(selected) >= max_candidates:
            break
    return selected


def _feature(image: Image.Image) -> np.ndarray:
    image = ImageOps.fit(image.convert("RGB"), (64, 36), method=Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    # Standardized color retains portrait layout while reducing brightness/contrast sensitivity.
    color = (array - array.mean(axis=(0, 1), keepdims=True)) / (
        array.std(axis=(0, 1), keepdims=True) + 0.08
    )
    gray = array @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    dx = np.diff(gray, axis=1, prepend=gray[:, :1])
    dy = np.diff(gray, axis=0, prepend=gray[:1, :])
    vector = np.concatenate([color.ravel() * 0.55, dx.ravel(), dy.ravel()]).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-8)


def _inset_image(image: Image.Image, x_fraction: float = 0.06, y_fraction: float = 0.08) -> Image.Image:
    width, height = image.size
    inset_x = max(1, int(width * x_fraction))
    inset_y = max(1, int(height * y_fraction))
    return image.crop((inset_x, inset_y, width - inset_x, height - inset_y))


class PortraitMatcher:
    def __init__(
        self,
        portraits_dir: str | Path | Iterable[str | Path],
        catalog: HeroCatalog,
        *,
        minimum_similarity: float = 0.48,
        minimum_margin: float = 0.015,
        minimum_active_contrast: float = 30.0,
    ) -> None:
        self.catalog = catalog
        self.minimum_similarity = minimum_similarity
        self.minimum_margin = minimum_margin
        self.minimum_active_contrast = minimum_active_contrast
        template_hero_ids: list[int] = []
        features: list[np.ndarray] = []
        directories = (
            [Path(portraits_dir)]
            if isinstance(portraits_dir, (str, Path))
            else [Path(path) for path in portraits_dir]
        )
        internal_names = sorted(
            (
                (hero.internal_name, hero.hero_id)
                for hero in catalog.by_id.values()
                if hero.internal_name
            ),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        seen: set[tuple[int, str]] = set()
        for directory in directories:
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.png")):
                hero_id: int | None = None
                if path.stem.isdigit():
                    candidate = int(path.stem)
                    hero_id = candidate if candidate in catalog.by_id else None
                else:
                    resource_name = path.stem.removesuffix("_png")
                    for internal_name, candidate in internal_names:
                        if resource_name == internal_name or resource_name.startswith(
                            internal_name + "_"
                        ):
                            hero_id = candidate
                            break
                if hero_id is None or (hero_id, path.name) in seen:
                    continue
                with Image.open(path) as image:
                    features.append(_feature(_inset_image(image)))
                template_hero_ids.append(hero_id)
                seen.add((hero_id, path.name))
        if not features:
            raise ValueError(f"no portrait templates found in {directories}")
        self.template_hero_ids = np.asarray(template_hero_ids, dtype=np.int64)
        self.hero_ids = np.asarray(sorted(set(template_hero_ids)), dtype=np.int64)
        self.features = np.stack(features)

    def classify(self, image: Image.Image, slot: int = 0) -> VisionMatch:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        if rgb.size == 0 or float(rgb.std()) < 8.0:
            return VisionMatch(slot, None, "Empty", 0.0, 0.0, False)
        # Dota darkens a hero portrait when it is merely being proposed/hovered.
        # Measure the portrait area only: the lower card decorations can otherwise
        # make an inactive card look deceptively high-contrast.  This is a state
        # check, independent of which hero happens to be shown.
        portrait_height = max(1, int(round(rgb.shape[0] * 0.8)))
        portrait = rgb[:portrait_height]
        gray = portrait @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
        active_contrast = float(np.percentile(gray, 90) - np.percentile(gray, 10))
        if active_contrast < self.minimum_active_contrast:
            return VisionMatch(slot, None, "Empty", 0.0, 0.0, False)
        feature = _feature(image)
        template_similarity = self.features @ feature
        similarity = np.asarray(
            [
                float(template_similarity[self.template_hero_ids == hero_id].max())
                for hero_id in self.hero_ids
            ],
            dtype=np.float32,
        )
        order = np.argsort(-similarity)
        best, second = int(order[0]), int(order[1])
        score = float(similarity[best])
        margin = score - float(similarity[second])
        accepted = score >= self.minimum_similarity and margin >= self.minimum_margin
        hero_id = int(self.hero_ids[best]) if accepted else None
        name = self.catalog.info(int(self.hero_ids[best])).name
        if not accepted:
            name = f"? {name}"
        return VisionMatch(slot, hero_id, name, score, margin, accepted)

    def recognize_box(
        self,
        screenshot: Image.Image,
        box: Rect,
        *,
        count: int = 5,
        orientation: str = "horizontal",
    ) -> list[VisionMatch]:
        matches: list[VisionMatch] = []
        for slot, rectangle in enumerate(split_slots(box, count, orientation)):
            left, top, right, bottom = rectangle
            width, height = right - left, bottom - top
            # Ignore borders, player-name labels and pick-state frames around the portrait.
            crop = _inset_image(screenshot.crop((left, top, right, bottom)))
            matches.append(self.classify(crop, slot))
        return matches


def accepted_heroes(matches: Iterable[VisionMatch]) -> tuple[int, ...]:
    return tuple(match.hero_id for match in matches if match.hero_id is not None)
