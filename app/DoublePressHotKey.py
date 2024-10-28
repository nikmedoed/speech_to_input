import keyboard
import time


class DoubleKeyPress:
    def __init__(self, key, callback, max_interval=0.25):
        self.key = key
        self.callback = callback
        self.max_interval = max_interval
        self.last_time_pressed = 0
        keyboard.on_press_key(self.key, self._handle_key)

    def _handle_key(self, event):
        current_time = time.time()
        if current_time - self.last_time_pressed < self.max_interval:
            self.callback()
        self.last_time_pressed = current_time
