from __future__ import annotations

import unittest
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:  # pragma: no cover - Tk missing entirely
    tk = None

from d2draft.screen_capture import MonitorInfo


ROOT = Path(__file__).resolve().parents[1]


def tk_available() -> bool:
    if tk is None:
        return False
    try:
        probe = tk.Tk()
    except tk.TclError:
        return False
    probe.destroy()
    return True


@unittest.skipUnless(tk_available(), "no display available for Tk")
class DesktopUiTest(unittest.TestCase):
    def setUp(self) -> None:
        from d2draft.desktop import DraftDesktopApp

        self.root = tk.Tk()
        self.app = DraftDesktopApp(self.root)
        self.app.updater = None  # never touch the network from a test
        self.root.update()

    def tearDown(self) -> None:
        self.root.destroy()

    def _combo(self, side: str) -> ttk.Combobox:
        def walk(widget):
            yield widget
            for child in widget.winfo_children():
                yield from walk(child)

        return next(
            widget
            for widget in walk(self.root)
            if isinstance(widget, ttk.Combobox)
            and str(widget.cget("textvariable")) == str(self.app.input_vars[side])
        )

    def test_typing_is_never_interrupted_by_the_suggestion_list(self) -> None:
        # Posting the combobox's own dropdown grabs the keyboard, which swallowed
        # every character after the first.
        combo = self._combo("radiant")
        typed = ""
        for character in "juggern":
            typed += character
            self.app.input_vars["radiant"].set(typed)
            self.app._update_suggestions("radiant", combo, character)
            self.root.update()
            self.assertIsNone(self.root.grab_current(), f"grab set after {typed!r}")
            self.assertEqual(self.app.input_vars["radiant"].get(), typed)
        self.assertTrue(self.app.suggestion_ids["radiant"])

    def test_arrow_keys_and_return_pick_a_suggestion(self) -> None:
        combo = self._combo("radiant")
        self.app.input_vars["radiant"].set("juggern")
        self.app._update_suggestions("radiant", combo, "n")
        self.root.update()
        self.assertEqual(self.app._move_suggestion("radiant", 1), "break")
        self.app._accept_suggestion("radiant", combo)
        self.root.update()
        picked = [hero for hero in self.app.team_ids["radiant"] if hero]
        self.assertEqual(picked, [self.app.catalog.resolve("Juggernaut")])

    def test_an_empty_query_closes_the_suggestion_list(self) -> None:
        combo = self._combo("radiant")
        self.app.input_vars["radiant"].set("jug")
        self.app._update_suggestions("radiant", combo, "g")
        self.root.update()
        window = self.app._suggestion_windows["radiant"]
        self.assertTrue(window.winfo_viewable())
        self.app.input_vars["radiant"].set("")
        self.app._update_suggestions("radiant", combo, "BackSpace")
        self.root.update()
        self.assertFalse(window.winfo_viewable())

    def test_the_window_hides_whenever_it_shares_a_screen_with_the_capture(self) -> None:
        # Comparing against the default portrait boxes missed windowed clients,
        # leaving an always-on-top window covering what recognition reads.
        here = MonitorInfo(device="A", left=0, top=0, right=1920, bottom=1080, primary=True)
        elsewhere = MonitorInfo(device="B", left=1920, top=0, right=3840, bottom=1080)
        self.root.geometry("1240x880+100+100")
        self.root.update_idletasks()
        self.assertTrue(self.app._window_overlaps_capture_area([here]))
        self.assertFalse(self.app._window_overlaps_capture_area([elsewhere]))

    def test_the_suggestion_popup_is_hidden_before_a_capture(self) -> None:
        # It is override-redirect, so withdrawing the main window leaves it on
        # screen and it would land in the screenshot.
        combo = self._combo("radiant")
        self.app.input_vars["radiant"].set("jug")
        self.app._update_suggestions("radiant", combo, "g")
        self.root.update()
        self.assertTrue(self.app._suggestion_windows["radiant"].winfo_viewable())
        for side in self.app.team_ids:
            self.app._hide_suggestions(side)
        self.root.update()
        self.assertFalse(self.app._suggestion_windows["radiant"].winfo_viewable())


if __name__ == "__main__":
    unittest.main()
