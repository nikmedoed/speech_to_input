from dataclasses import asdict
import os

import numpy as np
from .types import Word

# huggingface_hub's experimental Xet downloader fails on Windows/Dropbox setups
# with "cannot create file, file exists (os error 183)". Force-disable it by
# removing hf_xet support (hf_hub falls back to regular HTTP downloads).
os.environ.setdefault("HF_HUB_ENABLE_XET", "0")
try:
    from huggingface_hub.utils import _runtime as _hf_runtime

    _hf_runtime._package_versions["hf_xet"] = "N/A"
except Exception:
    pass


class FasterWhisperASR:
    sep = ""  # join transcribe words with this character "" for faster-whisper because it emits the spaces when neeeded)

    STOP_PHRASES = {
        'Субтитры сделал DimaTorzok',
        'Субтитры создавал DimaTorzok',
        'Продолжение следует...',
        'Редактор субтитров А.Семкин Корректор А.Егорова',
        'Спасибо за внимание.',
        'Продолжение',
        'Продолжение серии.',
        'следует...',
        'Спасибо за внимание!',
        'Субтитры подогнал «Симон»!',
        'Корректор А.Кулакова.',
        'До новых встреч.',
        'Субтитры подогнал «Симон»',
        'ПОДПИШИСЬ НА КАНАЛ, ЧТОБЫ НЕ ПРОПУСТИТЬ НОЛИКИ.',
        'Смотритев следующей части.',
        'Смотритев следующей серии.',
        ' сделал DimaTorzok',
        "И, конечно же, я надеюсь, что вам понравилось это видео. Если вам понравилось это видео, пожалуйста, ставьте лайки и подписывайтесь на мой канал. До новых встреч!",
        "Добро пожаловать на наш канал!"
    }

    def __init__(self, lan=None, modelsize='large-v3', vad=True):
        from faster_whisper import WhisperModel
        self.transcribe_kargs = {"vad_filter": vad}
        self.original_language = lan
        self.model = WhisperModel(modelsize, device="cuda", compute_type="float16")
        # warm up the ASR, because the very first transcribe takes much more time than the other
        self.transcribe(np.zeros(16000, dtype=np.float32))

    def transcribe(self, audio, init_prompt=""):
        segments, info = self.model.transcribe(audio,
                                               language=self.original_language,
                                               initial_prompt=init_prompt,
                                               beam_size=5,
                                               word_timestamps=True,
                                               condition_on_previous_text=True,
                                               **self.transcribe_kargs
                                               )
        # return list(segments)
        words, ends = [], []
        for segment in segments:
            # if segment.text.lower() in self.STOP_SEGMENTS:
            #     continue
            # Не работает, надо удалять фразы из объединённого текста,
            # т.к. в сегменты попадают и отдельные слова и лишние слова
            for word in segment.words:
                words.append(Word(**asdict(word)))
            ends.append(segment.end)
        return words, ends
