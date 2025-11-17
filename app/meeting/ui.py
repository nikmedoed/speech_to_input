import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional


class MeetingTranscriberUI:
    """Simple Tkinter window that shows live transcription with timestamps."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Транскрибатор собеседника")
        self.root.geometry("640x480")
        self.root.minsize(480, 320)

        self._toggle_handler: Optional[Callable[[], None]] = None
        self._running = False

        self.status_var = tk.StringVar(value="Остановлено")

        self._build_layout()

    # ----------------------- layout ----------------------- #
    def _build_layout(self):
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill="both", expand=True)

        button_frame = ttk.Frame(container)
        button_frame.pack(fill="x", pady=(0, 8))

        self.toggle_btn = ttk.Button(button_frame, text="Начать прослушивание", command=self._handle_toggle)
        self.toggle_btn.pack(side="left")

        ttk.Label(button_frame, textvariable=self.status_var).pack(side="left", padx=(12, 0))

        text_frame = ttk.Frame(container)
        text_frame.pack(fill="both", expand=True)

        self.text = tk.Text(
            text_frame,
            wrap="word",
            font=("Segoe UI", 11),
            undo=False,
        )
        self.text.bind("<Key>", lambda _: "break")
        self.text.bind("<<Paste>>", lambda _: "break")
        scrollbar = ttk.Scrollbar(text_frame, command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)
        self.text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    # ----------------------- callbacks ----------------------- #
    def bind_toggle(self, callback: Callable[[], None]):
        self._toggle_handler = callback

    def _handle_toggle(self):
        if self._toggle_handler:
            self._toggle_handler()

    # ----------------------- UI helpers ----------------------- #
    def set_running(self, is_running: bool):
        def inner():
            self._running = is_running
            text = "Остановить" if is_running else "Начать прослушивание"
            status = "Слушаем системный звук..." if is_running else "Остановлено"
            self.toggle_btn.config(text=text)
            self.status_var.set(status)

        self.root.after(0, inner)

    def append_chunk(self, timestamp: str, text: str):
        display = f"[{timestamp}] {text.strip()}\n"
        self._append_text(display)

    def add_message(self, text: str):
        self._append_text(f"{text.strip()}\n")

    def clear(self):
        def inner():
            self.text.delete("1.0", tk.END)

        self.root.after(0, inner)

    def _append_text(self, payload: str):
        def inner():
            self.text.insert(tk.END, payload)
            self.text.see(tk.END)

        self.root.after(0, inner)
