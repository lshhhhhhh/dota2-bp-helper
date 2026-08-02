from __future__ import annotations

import unittest

from d2draft.screen_capture import MonitorInfo, rectangles_intersect


class ScreenCaptureTest(unittest.TestCase):
    def test_monitor_dimensions_and_intersection(self) -> None:
        monitor = MonitorInfo("display", 2560, 0, 5120, 1440, False)
        self.assertEqual((monitor.width, monitor.height), (2560, 1440))
        self.assertTrue(rectangles_intersect((2600, 0, 3000, 100), monitor.rect))
        self.assertFalse(rectangles_intersect((0, 0, 2500, 100), monitor.rect))


if __name__ == "__main__":
    unittest.main()
