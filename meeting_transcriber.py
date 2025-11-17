import threading
import time
from datetime import datetime

import numpy as np
import pyperclip

from app.ASRProcessor import ASRProcessor, ASRChunk
from app.meeting.audio import DesktopAudioStreamManager
from app.meeting.ui import MeetingTranscriberUI
from app.meeting.vad import WebRTCVADFilter
from app.models.FasterWhisper import FasterWhisperASR
from app.select_device import select_output_loopback_device, select_soundcard_loopback_device
from settings import Settings


class MeetingTranscriberController:
    def __init__(
        self,
        settings: Settings,
        processor: ASRProcessor,
        stream: DesktopAudioStreamManager,
        ui: MeetingTranscriberUI,
        vad_filter: WebRTCVADFilter | None = None,
    ):
        self.settings = settings
        self.processor = processor
        self.stream = stream
        self.ui = ui
        self.vad_filter = vad_filter
        self._output_tail = ""
        self._output_tail_limit = 400

        self._recording = threading.Event()
        self._shutdown = threading.Event()
        self._session_start_wall_time: float | None = None

        self.ui.bind_toggle(self.toggle_recording)
        self.ui.set_running(False)

        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    # ----------------------- lifecycle ----------------------- #
    def shutdown(self):
        self._shutdown.set()
        self._recording.clear()
        self.stream.stop_stream()
        self.stream.close()

    # ----------------------- actions ----------------------- #
    def toggle_recording(self):
        if self._recording.is_set():
            self._stop_session()
        else:
            self._start_session()

    def _start_session(self):
        self.processor.reset()
        self.stream.start_stream()
        self._session_start_wall_time = time.time()
        self._recording.set()
        self.ui.set_running(True)
        self.ui.clear()
        self._output_tail = ""

    def _stop_session(self):
        self._recording.clear()
        self.stream.stop_stream()

    # ----------------------- worker ----------------------- #
    def _loop(self):
        record_event = self._recording
        settings = self.settings
        vad_filter = self.vad_filter

        while not self._shutdown.is_set():
            progressive_work = record_event.is_set() or not settings.stop_immediately

            if self.stream.empty():
                if record_event.is_set():
                    time.sleep(0.3)
                else:
                    record_event.wait(0.1)
                continue

            all_text = ""
            data_list = self.stream.get_audio_data()
            if not data_list:
                continue
            raw_chunk = np.concatenate(data_list)
            if vad_filter:
                raw_chunk = vad_filter.process(raw_chunk)
                if raw_chunk.size == 0:
                    continue

            chunk = None
            if progressive_work:
                self.processor.insert_audio_chunk(raw_chunk)
                chunk = self.processor.process_iter_chunk()

            if self.stream.empty() and not record_event.is_set():
                all_text = self.processor.gel_all_text().lstrip()
                tail = self.processor.finish()
                if tail:
                    all_text = (all_text + " " + tail).strip()
                self._emit_text(all_text, time.time())
                continue

            if chunk and chunk.text.strip():
                cleaned = settings.typewrite and chunk.text or chunk.text
                cleaned = self.processor.remove_stop_phrases(cleaned).strip()
                cleaned = self._extract_new_text(cleaned)
                if cleaned:
                    ts = self._format_chunk(chunk)
                    self.ui.append_chunk(ts, cleaned)

            if all_text and settings.copy_to_buffer:
                cleaned = self.processor.remove_stop_phrases(all_text)
                if cleaned:
                    pyperclip.copy(cleaned)

    def _emit_text(self, text: str, wall_time: float):
        cleaned = self.processor.remove_stop_phrases(text).strip()
        cleaned = self._extract_new_text(cleaned)
        if cleaned:
            timestamp = self._format_timestamp(wall_time)
            self.ui.append_chunk(timestamp, cleaned)
            if self.settings.copy_to_buffer:
                pyperclip.copy(cleaned)

    def _format_chunk(self, chunk: ASRChunk) -> str:
        if chunk.end is not None and self._session_start_wall_time:
            wall = self._session_start_wall_time + chunk.end
            return self._format_timestamp(wall)
        return self._format_duration(chunk.end or 0.0)

    @staticmethod
    def _format_timestamp(value: float) -> str:
        return datetime.fromtimestamp(value).strftime("%H:%M:%S")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        mins, secs = divmod(int(seconds), 60)
        hours, mins = divmod(mins, 60)
        return f"{hours:02}:{mins:02}:{secs:02}"

    def _extract_new_text(self, text: str) -> str:
        """Strip any prefix that was already emitted earlier."""
        if not text:
            return ""
        overlap = 0
        tail = self._output_tail
        max_overlap = min(len(tail), len(text))
        for size in range(max_overlap, 0, -1):
            if tail[-size:] == text[:size]:
                overlap = size
                break
        new_part = text[overlap:]
        if not new_part:
            return ""
        tail = (tail + new_part)[-self._output_tail_limit:]
        self._output_tail = tail
        return new_part


def main():
    settings = Settings(profile="meeting")
    settings.copy_to_buffer = False
    settings.typewrite = False

    output_device = select_output_loopback_device()
    settings.active_output_device = output_device

    soundcard_loopback = settings.soundcard_loopback_id or select_soundcard_loopback_device()
    settings.soundcard_loopback_id = soundcard_loopback

    asr = FasterWhisperASR(modelsize=settings.model_size,
                           lan=settings.model_language,
                           vad=settings.model_vad)
    stream = DesktopAudioStreamManager(settings.sample_rate,
                                       output_device,
                                       soundcard_loopback)
    processor = ASRProcessor(asr, settings.sample_rate)
    vad_filter = None
    if settings.model_vad and settings.model_vad_backend == "webrtc":
        vad_filter = WebRTCVADFilter(sample_rate=settings.sample_rate,
                                     aggressiveness=2,
                                     hangover_ms=300)

    ui = MeetingTranscriberUI()
    controller = MeetingTranscriberController(settings, processor, stream, ui, vad_filter)

    def on_close():
        controller.shutdown()
        ui.root.destroy()

    ui.root.protocol("WM_DELETE_WINDOW", on_close)
    ui.root.mainloop()


if __name__ == "__main__":
    main()
