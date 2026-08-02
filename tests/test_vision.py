from __future__ import annotations

import unittest
from pathlib import Path

from PIL import Image

from d2draft.recommender import HeroCatalog
from d2draft.vision import (
    MINIMUM_MARGIN,
    CaptureConfig,
    PortraitMatcher,
    accepted_heroes,
    locate_windowed_viewports,
    split_slots,
)


ROOT = Path(__file__).resolve().parents[1]


class VisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = HeroCatalog(ROOT / "data" / "heroes.json")
        cls.matcher = PortraitMatcher(ROOT / "data" / "hero_portraits", cls.catalog)

    def test_split_slots(self) -> None:
        self.assertEqual(
            split_slots((0, 0, 500, 100)),
            [(0, 0, 100, 100), (100, 0, 200, 100), (200, 0, 300, 100),
             (300, 0, 400, 100), (400, 0, 500, 100)],
        )

    def test_synthetic_portrait_strip(self) -> None:
        hero_ids = [1, 2, 3, 4, 5]
        canvas = Image.new("RGB", (5 * 256, 144), "black")
        for slot, hero_id in enumerate(hero_ids):
            with Image.open(ROOT / "data" / "hero_portraits" / f"{hero_id}.png") as image:
                canvas.paste(image.convert("RGB").resize((256, 144)), (slot * 256, 0))
        matches = self.matcher.recognize_box(canvas, (0, 0, 5 * 256, 144))
        self.assertEqual(accepted_heroes(matches), tuple(hero_ids))
        self.assertTrue(all(match.similarity > 0.55 for match in matches))

    def test_config_round_trip_shape(self) -> None:
        config = CaptureConfig(1920, 1080, (10, 20, 510, 120), (1000, 20, 1500, 120))
        self.assertEqual(config.orientation, "horizontal")

    def test_real_2560_screenshot(self) -> None:
        screenshot_path = ROOT / "screenshot" / "8f8f7c2d-b301-45fc-802f-1c29252095f5.png"
        if not screenshot_path.exists():
            self.skipTest("user-provided screenshot is not available")
        matcher = PortraitMatcher(
            [
                ROOT / "data" / "hero_portraits",
                ROOT / "data" / "game_hero_images" / "panorama" / "images" / "heroes",
            ],
            self.catalog,
        )
        with Image.open(screenshot_path) as screenshot:
            config = CaptureConfig.default_for_screen(*screenshot.size)
            allies = matcher.recognize_box(screenshot, config.allies_box)
            enemies = matcher.recognize_box(screenshot, config.enemies_box)
        self.assertEqual(accepted_heroes(allies), (1, 101, 37, 36, 7))
        self.assertEqual(accepted_heroes(enemies), (81, 8, 64, 11, 30))

    def test_zero_selected_screenshot_has_no_false_positives(self) -> None:
        screenshot_path = ROOT / "screenshot" / "0_selected.png"
        if not screenshot_path.exists():
            self.skipTest("user-provided zero-pick screenshot is not available")
        matcher = PortraitMatcher(
            [
                ROOT / "data" / "hero_portraits",
                ROOT / "data" / "game_hero_images" / "panorama" / "images" / "heroes",
            ],
            self.catalog,
        )
        with Image.open(screenshot_path) as screenshot:
            config = CaptureConfig.default_for_screen(*screenshot.size)
            radiant = matcher.recognize_box(screenshot, config.allies_box)
            dire = matcher.recognize_box(screenshot, config.enemies_box)
        self.assertEqual(accepted_heroes(radiant), ())
        self.assertEqual(accepted_heroes(dire), ())
        # What keeps an empty card out is that it stays ambiguous, not that it
        # scores low. Masking the rank banner lifts every similarity, so bounding
        # the raw score here only pinned the old feature's scale.
        self.assertLess(max(match.margin for match in radiant + dire), MINIMUM_MARGIN)

    def test_livestream_suggested_spectre_is_not_a_locked_pick(self) -> None:
        screenshot_path = ROOT / "screenshot" / "live_proposed_spectre.png"
        if not screenshot_path.exists():
            self.skipTest("user-provided livestream screenshot is not available")
        matcher = PortraitMatcher(
            [
                ROOT / "data" / "hero_portraits",
                ROOT / "data" / "game_hero_images" / "panorama" / "images" / "heroes",
            ],
            self.catalog,
        )
        with Image.open(screenshot_path) as screenshot:
            config = CaptureConfig.default_for_screen(*screenshot.size)
            radiant = matcher.recognize_box(screenshot, config.allies_box)
            dire = matcher.recognize_box(screenshot, config.enemies_box)

        self.assertEqual(accepted_heroes(radiant), (123, 3))
        self.assertEqual(accepted_heroes(dire), (101, 100))
        self.assertIsNone(radiant[3].hero_id)
        self.assertFalse(radiant[3].accepted)

    def test_locates_dota_inside_windowed_livestream(self) -> None:
        screenshot_path = ROOT / "screenshot" / "windowed_livestream_desktop.png"
        if not screenshot_path.exists():
            self.skipTest("user-provided windowed livestream screenshot is not available")
        matcher = PortraitMatcher(
            [
                ROOT / "data" / "hero_portraits",
                ROOT / "data" / "game_hero_images" / "panorama" / "images" / "heroes",
            ],
            self.catalog,
        )
        with Image.open(screenshot_path) as screenshot:
            candidates = locate_windowed_viewports(screenshot)
            self.assertTrue(candidates)
            viewport = screenshot.crop(candidates[0].rect)
            config = CaptureConfig.default_for_screen(*viewport.size)
            radiant = matcher.recognize_box(viewport, config.allies_box)
            dire = matcher.recognize_box(viewport, config.enemies_box)

        expected = (114, 353, 1425, 1090)
        self.assertTrue(
            all(abs(actual - target) <= 2 for actual, target in zip(candidates[0].rect, expected))
        )
        self.assertEqual(accepted_heroes(radiant), (123, 3))
        self.assertEqual(accepted_heroes(dire), (101, 100))


if __name__ == "__main__":
    unittest.main()
