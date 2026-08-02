from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class MonitorInfo:
    device: str
    left: int
    top: int
    right: int
    bottom: int
    primary: bool = False

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def enumerate_monitors(fallback_width: int, fallback_height: int) -> list[MonitorInfo]:
    """Return Windows monitors with the primary display first."""

    if os.name != "nt":
        return [MonitorInfo("primary", 0, 0, fallback_width, fallback_height, True)]

    from ctypes import wintypes

    class MonitorInfoExW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", wintypes.RECT),
            ("rcWork", wintypes.RECT),
            ("dwFlags", wintypes.DWORD),
            ("szDevice", wintypes.WCHAR * 32),
        ]

    callback_type = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(wintypes.RECT),
        wintypes.LPARAM,
    )
    result: list[MonitorInfo] = []

    def callback(monitor: int, _dc: int, _rect: object, _data: int) -> int:
        details = MonitorInfoExW()
        details.cbSize = ctypes.sizeof(MonitorInfoExW)
        if ctypes.windll.user32.GetMonitorInfoW(monitor, ctypes.byref(details)):
            rect = details.rcMonitor
            result.append(
                MonitorInfo(
                    details.szDevice,
                    int(rect.left),
                    int(rect.top),
                    int(rect.right),
                    int(rect.bottom),
                    bool(details.dwFlags & 1),
                )
            )
        return 1

    callback_function = callback_type(callback)
    ctypes.windll.user32.EnumDisplayMonitors(None, None, callback_function, 0)
    if not result:
        return [MonitorInfo("primary", 0, 0, fallback_width, fallback_height, True)]
    result.sort(key=lambda monitor: (not monitor.primary, monitor.left, monitor.top))
    return result


def rectangles_intersect(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> bool:
    return not (
        first[2] <= second[0]
        or first[0] >= second[2]
        or first[3] <= second[1]
        or first[1] >= second[3]
    )
