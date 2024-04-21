import tkinter as tk
from datetime import datetime


class RecordingIndicator:
    size = 56
    default_color = 'red'
    stop_color = 'blue'

    def __init__(self):
        self.root = tk.Tk()
        self.root_size = self.size + 2
        self.root.overrideredirect(True)
        self.root.attributes('-alpha', 0.7)
        self.root.attributes('-topmost', True)
        self.root.attributes('-transparentcolor', 'grey15')

        self.canvas = tk.Canvas(
            self.root, width=self.root_size, height=self.root_size,
            bg='grey15', highlightthickness=0)
        self.canvas.pack()
        self.circle = self.canvas.create_oval(0, 0, self.size, self.size, fill=self.default_color, outline='')

        self.label_asr = self.canvas.create_text(
            self.size / 2, self.size * 0.25, text="ASR", fill='white',
            font=("Helvetica", 8), anchor='center')
        self.timer_label = self.canvas.create_text(
            self.size / 2, self.size * 0.56, text="00:00", fill='white',
            font=("Helvetica", 12), anchor='center')
        self.arrow = self.canvas.create_text(
            self.size / 2, self.size * 0.8, text="↓↓↓", fill='white',
            font=("Helvetica", 8), anchor='center')

        self.root.geometry(f"{self.root_size}x{self.root_size}")
        self.root.withdraw()
        self.start_time = None

        # Event handlers for moving the window
        self.canvas.bind("<Button-1>", self.start_move)
        self.canvas.bind("<B1-Motion>", self.do_move)
        self._drag_data = {"x": 0, "y": 0}  # Data for drag and drop motion

    def start_move(self, event):
        """ Record the initial position for dragging the window. """
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def do_move(self, event):
        """ Handle dragging the window. """
        dx = event.x - self._drag_data["x"]
        dy = event.y - self._drag_data["y"]
        x = self.root.winfo_x() + dx
        y = self.root.winfo_y() + dy
        self.root.geometry(f"+{x}+{y}")

    def update_time(self):
        if self.start_time:
            elapsed_time = datetime.now() - self.start_time
            mins, secs = divmod(elapsed_time.seconds, 60)
            self.canvas.itemconfig(self.timer_label, text=f"{mins:02}:{secs:02}")
        else:
            self.start_time = datetime.now()
        self.root.after(1000, self.update_time)

    def show(self, x, y):
        self.start_time = datetime.now()
        half = self.size // 2
        adjusted_x = x - half
        adjusted_y = y - half - 50
        # ToDo Определение границ экрана
        # get_monitors() - не подходит, потому что ломает скелинг
        if adjusted_y < 0:
            adjusted_y = y + half + 50

        self.root.geometry(f'+{adjusted_x}+{adjusted_y}')
        self.root.deiconify()
        self.update_time()

    def hide(self):
        self.root.withdraw()
        self.start_time = None
        self.canvas.itemconfig(self.timer_label, text="00:00")
        self.canvas.itemconfig(self.circle, fill=self.stop_color)

    def stop_recording(self):
        self.canvas.itemconfig(self.circle, fill=self.stop_color)
