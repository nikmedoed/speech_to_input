import pyaudio
import numpy as np
import queue
from settings import Settings


class AudioStreamManager:
    stream: pyaudio.Stream = None

    def __init__(self, settings: Settings):
        self.p = pyaudio.PyAudio()
        self.settings = settings
        self.queue_audio_buffer = queue.Queue()
        self.open_stream()

    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768
        self.queue_audio_buffer.put(audio_data)
        return in_data, pyaudio.paContinue

    def close(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def open_stream(self):
        if self.stream is not None:
            self.close()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.settings.sample_rate,
            input=True,
            input_device_index=self.settings.active_microphone_device,
            frames_per_buffer=1024,
            stream_callback=self.audio_callback
        )

    def start_stream(self):
        try:
            self.stream.start_stream()
        except:
            self.open_stream()

    def stop_stream(self):
        self.stop_stream()

    def get_audio_data(self):
        data_list = []
        while not self.queue_audio_buffer.empty():
            data_list.append(self.queue_audio_buffer.get())
        return data_list

    def __del__(self):
        self.close()
        self.p.terminate()
