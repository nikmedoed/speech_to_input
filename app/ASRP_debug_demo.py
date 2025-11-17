import random
import time

import numpy as np

from app.ASRProcessor import ASRChunk


class ASRProcessorDemo:
    def __init__(self, asr, sampling_rate):
        self.sampling_rate = sampling_rate
        self.audio_buffer = np.array([], dtype=np.float32)
        self.pause_time = 0
        self.out = ""

    def insert_audio_chunk(self, new_chunk):
        lc = len(self.audio_buffer) / self.sampling_rate
        self.audio_buffer = np.append(self.audio_buffer, new_chunk)
        nl = len(self.audio_buffer) / self.sampling_rate
        self.out = f"buff {nl:.2f} new {nl - lc:.2f} pause {self.pause_time}"
        self.pause_time = random.randint(1, 5)
        time.sleep(self.pause_time)

    def gel_all_text(self):
        return "The full text will be here"

    def finish(self):
        tl = len(self.audio_buffer) / self.sampling_rate
        self.audio_buffer = np.array([], dtype=np.float32)
        return f"finished total: {tl :.2f}\n"

    def process_iter(self):
        return self.process_iter_chunk().text

    def process_iter_chunk(self):
        out = self.out
        self.out = "N/A"
        return ASRChunk(text=f"{out}\n", words=[], start=None, end=None)
