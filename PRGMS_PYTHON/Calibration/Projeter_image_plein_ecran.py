import argparse
import ctypes
import ctypes.wintypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pygame


SWP_FRAMECHANGED = 0x0020
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040
HWND_TOPMOST = -1
GWL_STYLE = -16
WS_POPUP = 0x80000000
WS_VISIBLE = 0x10000000


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class MONITORINFOEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_ulong),
        ("rcMonitor", RECT),
        ("rcWork", RECT),
        ("dwFlags", ctypes.c_ulong),
        ("szDevice", ctypes.c_wchar * 32),
    ]


@dataclass
class MonitorBounds:
    x: int
    y: int
    width: int
    height: int


def get_monitor_bounds(index: int) -> MonitorBounds:
    user32 = ctypes.windll.user32
    enum_monitors = []

    monitor_enum_proc = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        ctypes.wintypes.HMONITOR,
        ctypes.wintypes.HDC,
        ctypes.POINTER(RECT),
        ctypes.wintypes.LPARAM,
    )

    def callback(hmonitor, _hdc, _rect, _lparam):
        enum_monitors.append(hmonitor)
        return 1

    user32.EnumDisplayMonitors(0, 0, monitor_enum_proc(callback), 0)
    if index < 0 or index >= len(enum_monitors):
        raise RuntimeError(f"Ecran {index + 1} introuvable. Ecrans detectes: {len(enum_monitors)}.")

    info = MONITORINFOEXW()
    info.cbSize = ctypes.sizeof(MONITORINFOEXW)
    if not user32.GetMonitorInfoW(enum_monitors[index], ctypes.byref(info)):
        raise RuntimeError("Impossible de lire la geometrie de l'ecran cible.")

    rect = info.rcMonitor
    return MonitorBounds(rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)


def force_borderless_fullscreen(window_handle: int, bounds: MonitorBounds) -> None:
    user32 = ctypes.windll.user32
    user32.SetWindowLongW(window_handle, GWL_STYLE, WS_POPUP | WS_VISIBLE)
    user32.SetWindowPos(
        window_handle,
        HWND_TOPMOST,
        bounds.x,
        bounds.y,
        bounds.width,
        bounds.height,
        SWP_FRAMECHANGED | SWP_SHOWWINDOW | SWP_NOACTIVATE,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Projette une image en plein ecran sur le 2e moniteur.")
    parser.add_argument("image_path", help="Chemin de l'image a projeter.")
    parser.add_argument("--monitor", type=int, default=1, help="Index de l'ecran cible, 0-based. Defaut: 1.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image_path)
    if not image_path.exists():
        raise SystemExit(f"Image introuvable: {image_path}")
    if sys.platform != "win32":
        raise SystemExit("Ce script cible Windows.")

    ctypes.windll.user32.SetProcessDPIAware()
    bounds = get_monitor_bounds(args.monitor)

    os.environ["SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS"] = "0"
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{bounds.x},{bounds.y}"

    pygame.init()
    pygame.display.init()
    pygame.mouse.set_visible(False)

    screen = pygame.display.set_mode((bounds.width, bounds.height), pygame.NOFRAME)
    pygame.display.set_caption("Projection calibration emetteur")
    window_handle = pygame.display.get_wm_info()["window"]
    force_borderless_fullscreen(window_handle, bounds)

    image = pygame.image.load(str(image_path))
    image = pygame.transform.smoothscale(image, (bounds.width, bounds.height))

    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 0
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                return 0
            if hasattr(pygame, "WINDOWFOCUSLOST") and event.type == pygame.WINDOWFOCUSLOST:
                force_borderless_fullscreen(window_handle, bounds)

        screen.blit(image, (0, 0))
        pygame.display.flip()
        force_borderless_fullscreen(window_handle, bounds)
        clock.tick(60)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        pygame.quit()
