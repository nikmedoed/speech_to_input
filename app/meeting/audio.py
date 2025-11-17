import queue
import threading
import sys
import site
from typing import Optional

import librosa
import numpy as np
import pyaudio


def _import_soundcard():
    try:
        import soundcard as _sc  # type: ignore
        return _sc
    except ImportError:
        usersite = getattr(site, "getusersitepackages", lambda: None)()
        if usersite and usersite not in sys.path:
            sys.path.append(usersite)
        try:
            import soundcard as _sc  # type: ignore
            return _sc
        except ImportError:
            return None


sc = _import_soundcard()


class DesktopAudioStreamManager:
    """Stream manager that captures outgoing audio (system mix or chosen output)."""

    def __init__(
        self,
        target_sample_rate: int,
        output_device_index: Optional[int] = None,
        soundcard_device_id: Optional[str] = None,
        frames_per_buffer: int = 4096,
    ):
        self._target_sample_rate = target_sample_rate
        self._frames_per_buffer = frames_per_buffer
        self._queue = queue.Queue()

        self._backend = None
        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._pa_stream: Optional[pyaudio.Stream] = None
        self._pa_device_info = None
        self._pa_source_rate: Optional[int] = None
        self._pa_channels: int = 0

        self._soundcard_device_id = soundcard_device_id
        self._sc_speaker = None
        self._sc_microphone = None
        self._sc_rate: Optional[int] = None
        self._sc_channels: int = 0
        self._sc_thread: Optional[threading.Thread] = None
        self._sc_stop = threading.Event()

        self._init_backend(output_device_index)
        if self._backend == "pyaudio":
            self.open_stream()
            self.stop_stream()

    # ----------------------- backend selection ----------------------- #
    def _init_backend(self, output_device_index: Optional[int]):
        pa_supported = hasattr(pyaudio, "paWASAPI")
        if pa_supported:
            try:
                self._init_pyaudio_backend(output_device_index)
                self._backend = "pyaudio"
                return
            except Exception as exc:
                print(f"PyAudio loopback недоступен ({exc}). Пробуем soundcard.")
        self._init_soundcard_backend(self._soundcard_device_id)
        self._backend = "soundcard"

    def _init_pyaudio_backend(self, output_device_index: Optional[int]):
        self._pyaudio = pyaudio.PyAudio()
        self._pa_device_info = self._resolve_device(output_device_index)
        self._pa_source_rate = int(self._pa_device_info["defaultSampleRate"])
        self._pa_channels = int(max(1, self._pa_device_info.get("maxInputChannels", 1)))

    def _init_soundcard_backend(self, device_id: Optional[str]):
        if sc is None:
            raise RuntimeError(
                "Для захвата системного вывода требуется soundcard (pip install soundcard)."
            )
        if device_id:
            self._sc_microphone = sc.get_microphone(id=device_id, include_loopback=True)
            self._sc_speaker = None
        else:
            self._sc_speaker = sc.default_speaker()
            if self._sc_speaker is None:
                raise RuntimeError("Не удалось получить устройство вывода через soundcard.")
            try:
                self._sc_microphone = sc.get_microphone(
                    id=self._sc_speaker.id, include_loopback=True
                )
            except Exception as exc:
                raise RuntimeError(f"Не удалось создать loopback-микрофон через soundcard: {exc}")
        self._sc_rate = int(self._detect_soundcard_samplerate())
        self._sc_channels = self._detect_soundcard_channels()

    # ----------------------- public API ----------------------- #
    def open_stream(self) -> None:
        if self._backend == "pyaudio":
            if self._pa_stream is not None:
                self.close()
            kwargs = {
                "format": pyaudio.paInt16,
                "channels": self._pa_channels,
                "rate": self._pa_source_rate,
                "input": True,
                "frames_per_buffer": self._frames_per_buffer,
                "input_device_index": self._pa_device_info["index"],
                "stream_callback": self._pyaudio_callback,
            }
            if self._pa_device_info.get("isLoopbackDevice"):
                kwargs["as_loopback"] = True
            self._pa_stream = self._pyaudio.open(**kwargs)
        elif self._backend == "soundcard":
            pass
        else:
            raise RuntimeError("Не выбран backend аудио.")

    def start_stream(self) -> None:
        if self._backend == "pyaudio":
            if self._pa_stream is None:
                self.open_stream()
            if not self._pa_stream.is_active():
                self._pa_stream.start_stream()
        else:
            if self._sc_thread and self._sc_thread.is_alive():
                return
            self._sc_stop.clear()
            self._sc_thread = threading.Thread(target=self._soundcard_loop, daemon=True)
            self._sc_thread.start()

    def stop_stream(self) -> None:
        if self._backend == "pyaudio":
            if self._pa_stream is not None and self._pa_stream.is_active():
                self._pa_stream.stop_stream()
        else:
            self._sc_stop.set()
            if self._sc_thread and self._sc_thread.is_alive():
                self._sc_thread.join(timeout=1.0)
                self._sc_thread = None

    def close(self) -> None:
        if self._backend == "pyaudio":
            if self._pa_stream is not None:
                if self._pa_stream.is_active():
                    self._pa_stream.stop_stream()
                self._pa_stream.close()
                self._pa_stream = None
        elif self._backend == "soundcard":
            self.stop_stream()

    def get_audio_data(self) -> list[np.ndarray]:
        data = []
        while not self._queue.empty():
            data.append(self._queue.get())
        return data

    def empty(self) -> bool:
        return self._queue.empty()

    def __del__(self):
        self.close()
        if self._pyaudio is not None:
            self._pyaudio.terminate()

    # ----------------------- callbacks ----------------------- #
    def _pyaudio_callback(self, in_data, frame_count, time_info, status_flags):
        audio = np.frombuffer(in_data, dtype=np.int16)
        if self._pa_channels > 1:
            audio = audio.reshape(-1, self._pa_channels).mean(axis=1)
        audio = audio.astype(np.float32) / 32768.0
        if self._pa_source_rate != self._target_sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=self._pa_source_rate,
                target_sr=self._target_sample_rate,
            )
        self._queue.put(audio)
        return None, pyaudio.paContinue

    def _soundcard_loop(self):
        mic = self._sc_microphone
        if mic is None:
            raise RuntimeError("Loopback-микрофон soundcard не инициализирован.")
        source_rate = self._sc_rate or self._target_sample_rate
        frames = self._frames_per_buffer
        channel_indices = list(range(max(1, self._sc_channels)))
        with mic.recorder(
            samplerate=source_rate,
            channels=channel_indices,
            blocksize=self._frames_per_buffer,
        ) as recorder:
            while not self._sc_stop.is_set():
                chunk = recorder.record(frames)
                if chunk.size == 0:
                    continue
                audio = chunk.astype(np.float32)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if source_rate != self._target_sample_rate:
                    audio = librosa.resample(
                        audio,
                        orig_sr=source_rate,
                        target_sr=self._target_sample_rate,
                    )
                self._queue.put(audio)

    # ----------------------- helpers ----------------------- #
    def _resolve_device(self, user_index: Optional[int]):
        if user_index is not None:
            info = self._pyaudio.get_device_info_by_index(user_index)
            if not info.get("isLoopbackDevice"):
                raise ValueError(
                    "Устройство вывода должно быть выбрано из списка loopback-устройств WASAPI."
                )
            return info
        wasapi_info = self._pyaudio.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_output_index = wasapi_info.get("defaultOutputDevice")
        if default_output_index is None:
            raise RuntimeError("Не удалось найти устройство вывода WASAPI.")
        info = self._pyaudio.get_device_info_by_index(default_output_index)
        if info.get("isLoopbackDevice"):
            return info
        base_name = info.get("name")
        for idx in range(self._pyaudio.get_device_count()):
            candidate = self._pyaudio.get_device_info_by_index(idx)
            if candidate.get("isLoopbackDevice") and candidate.get("name", "").startswith(base_name):
                return candidate
        raise RuntimeError("Loopback-устройство по умолчанию не найдено.")

    # ----------------------- soundcard helpers ----------------------- #
    def _detect_soundcard_samplerate(self) -> int:
        rate_candidates = [
            getattr(self._sc_microphone, "samplerate", None),
            getattr(self._sc_speaker, "nominal_samplerate", None),
            getattr(self._sc_speaker, "samplerate", None),
        ]
        for rate in rate_candidates:
            if isinstance(rate, (int, float)) and rate > 0:
                return int(rate)
        return self._target_sample_rate

    def _detect_soundcard_channels(self) -> int:
        channels = getattr(self._sc_microphone, "channels", None)
        if isinstance(channels, int) and channels > 0:
            return channels
        if isinstance(channels, (list, tuple)) and channels:
            return len(channels)
        speaker_channels = getattr(self._sc_speaker, "channels", None)
        if isinstance(speaker_channels, int) and speaker_channels > 0:
            return speaker_channels
        return 2
