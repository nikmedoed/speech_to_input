from __future__ import annotations

import numpy as np


class WebRTCVADFilter:
    """Filters out silence using lightweight WebRTC VAD."""

    def __init__(
        self,
        sample_rate: int,
        frame_ms: int = 30,
        aggressiveness: int = 2,
        hangover_ms: int = 300,
    ):
        import webrtcvad  # Imported lazily to avoid dependency for не-VAD режимов

        if frame_ms not in {10, 20, 30}:
            raise ValueError("WebRTC VAD поддерживает размер окна 10, 20 или 30 мс.")
        self.sample_rate = sample_rate
        self._frame_samples = int(sample_rate * frame_ms / 1000)
        self._hangover_frames = max(1, hangover_ms // frame_ms)
        self._speech_countdown = 0
        self._residual = np.empty(0, dtype=np.float32)

        self._vad = webrtcvad.Vad(aggressiveness)

    def process(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return np.empty(0, dtype=np.float32)

        if audio.ndim > 1:
            audio = audio.reshape(-1)

        audio = audio.astype(np.float32, copy=False)
        data = np.concatenate((self._residual, audio))

        frame_len = self._frame_samples
        if data.size < frame_len:
            self._residual = data
            return np.empty(0, dtype=np.float32)

        usable = data.size - (data.size % frame_len)
        frames = data[:usable].reshape(-1, frame_len)
        self._residual = data[usable:]

        kept_frames: list[np.ndarray] = []
        for frame in frames:
            speech = self._is_speech(frame)
            if speech:
                self._speech_countdown = self._hangover_frames
                kept_frames.append(frame)
            elif self._speech_countdown > 0:
                kept_frames.append(frame)
                self._speech_countdown -= 1

        if not kept_frames:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(kept_frames)

    def reset(self):
        self._residual = np.empty(0, dtype=np.float32)
        self._speech_countdown = 0

    def _is_speech(self, frame: np.ndarray) -> bool:
        pcm16 = np.clip(frame, -1.0, 1.0)
        pcm16 = (pcm16 * 32768.0).astype(np.int16).tobytes()
        return self._vad.is_speech(pcm16, self.sample_rate)
