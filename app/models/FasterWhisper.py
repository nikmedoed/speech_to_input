import numpy as np
from .types import Word


class FasterWhisperASR:
    sep = ""  # join transcribe words with this character "" for faster-whisper because it emits the spaces when neeeded)

    STOP_PHRASES = {
        'Субтитры сделал DimaTorzok',
        'Субтитры создавал DimaTorzok',
        'Продолжение следует...',
        'Редактор субтитров А.Семкин Корректор А.Егорова',
        'Спасибо за внимание.',
        'Продолжение',
        'Продолжение серии.'
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
                words.append(Word(*word))
            ends.append(segment.end)
        return words, ends
