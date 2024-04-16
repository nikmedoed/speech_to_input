import numpy as np

from app.OutputBuffer import HypothesisBuffer
from app.models.types import Word


class ASRProcessor:

    def __init__(self, asr, sampling_rate):
        self.asr = asr
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset = 0
        self.sampling_rate = sampling_rate
        self.transcript_buffer = HypothesisBuffer()
        self.commited = []

        self.buffer_trimming_sec = 30

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        """Returns a tuple: (prompt, context), where "prompt" is a 200-character suffix of commited text that is inside of the scrolled away part of audio buffer.
        "context" is the commited text that is inside the audio buffer. It is transcribed again and skipped. It is returned only for debugging and logging reasons.
        """
        k = max(0, len(self.commited) - 1)
        while k > 0 and self.commited[k - 1].end > self.buffer_time_offset:
            k -= 1

        p = self.commited[:k]
        p = [t.word for t in p]
        prompt = []
        l = 0
        while p and l < 1000:  # 200 characters prompt size
            x = p.pop(-1)
            l += len(x) + 1
            prompt.append(x)
        non_prompt = self.commited[k:]
        return self.asr.sep.join(prompt[::-1]), self.asr.sep.join(t.word for t in non_prompt)

    def process_iter(self):
        """Runs on the current audio buffer.
        Returns: a tuple (beg_timestamp, end_timestamp, "text"), or (None, None, "").
        The non-emty text is confirmed (committed) partial transcript.
        """
        prompt, non_prompt = self.prompt()
        iteration_words, iteration_ends = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)
        self.transcript_buffer.insert(iteration_words, self.buffer_time_offset)
        o = self.transcript_buffer.flush()
        self.commited.extend(o)

        if len(self.audio_buffer) / self.sampling_rate > self.buffer_trimming_sec:
            if not self.commited: return
            t = self.commited[-1].end
            if len(iteration_ends) > 1:
                e = iteration_ends[-2] + self.buffer_time_offset
                while len(iteration_ends) > 2 and e > t:
                    iteration_ends.pop(-1)
                    e = iteration_ends[-2] + self.buffer_time_offset
                if e <= t:
                    self.chunk_at(e)
            # alternative: on any word
            # l = self.buffer_time_offset + len(self.audio_buffer)/self.sampling_rate - 10
            # let's find commited word that is less
            # k = len(self.commited)-1
            # while k>0 and self.commited[k].end > l:
            #    k -= 1
            # t = self.commited[k].end
            # self.chunk_at(t)
        return self.to_flush(o)

    def chunk_at(self, time):
        """trims the hypothesis and audio buffer at "time"
        """
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds * self.sampling_rate):]
        self.buffer_time_offset = time

    def finish(self):
        """Flush the incomplete text when the whole processing ends.
        Returns: the same format as self.process_iter()
        """
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        return f

    def to_flush(self, sents:list[Word]):
        return self.asr.sep.join(s.word for s in sents)
