from __future__ import annotations

import unittest

from d2draft.i18n import LANGUAGE_LABELS, rank_label, tr, validate_translations


class I18nTest(unittest.TestCase):
    def test_translation_catalogs_have_the_same_keys(self) -> None:
        validate_translations()

    def test_main_controls_are_bilingual(self) -> None:
        self.assertEqual(tr("zh", "recognize_screen"), "识别屏幕")
        self.assertEqual(tr("en", "recognize_screen"), "Scan Screen")
        self.assertEqual(tuple(LANGUAGE_LABELS.values()), ("中文", "English"))

    def test_rank_labels_are_localized_from_stable_ids(self) -> None:
        self.assertEqual(rank_label("zh", "legend_plus"), "传奇及以上")
        self.assertEqual(rank_label("en", "legend_plus"), "Legend and above")
        self.assertEqual(rank_label("en", "archon_below"), "Archon and below")

    def test_formatted_status_messages_work_in_both_languages(self) -> None:
        chinese = tr(
            "zh",
            "recognized_phase",
            source="屏幕 1",
            radiant=2,
            dire=2,
            confidence="",
            phase=2,
        )
        english = tr(
            "en",
            "recognized_phase",
            source="Display 1",
            radiant=2,
            dire=2,
            confidence="",
            phase=2,
        )
        self.assertIn("第 2 轮", chinese)
        self.assertIn("round 2", english)


if __name__ == "__main__":
    unittest.main()
