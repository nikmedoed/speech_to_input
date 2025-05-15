from __future__ import annotations

import time
from typing import Callable

from pynput import keyboard


class HotKeyListener:
    """
    Tracks Ctrl+Alt+R  and  Ctrl+` / Ctrl+ё combinations with debouncing.
    """

    def __init__(self, on_toggle: Callable[[], None], min_interval: float = 0.2) -> None:
        self.on_toggle = on_toggle
        self.min_interval = min_interval
        self._last_call = 0.0

        # State flags
        self._ctrl = False
        self._alt = False

        # Start background listener
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()

    # ---------- internal helpers ---------- #
    def _fire(self) -> None:
        now = time.time()
        if now - self._last_call >= self.min_interval:
            self._last_call = now
            self.on_toggle()

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self._ctrl = True
            return
        if key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
            self._alt = True
            return

        # --- main combos --- #
        vk = getattr(key, "vk", None)
        char = getattr(key, "char", None)

        # 1. Ctrl + Alt + R
        if self._ctrl and self._alt and vk == 0x52:  # 0x52 == ord('R')
            self._fire()
            return

        # 2. Ctrl + `  (рус. Ctrl + ё тоже работает)
        if self._ctrl and (vk == 0xC0 or char in ("`", "ё")):
            self._fire()

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self._ctrl = False
        elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
            self._alt = False
