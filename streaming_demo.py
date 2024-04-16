import numpy as np
import librosa
import time

from app.ASRProcessor import ASRProcessor
from app.models.FasterWhisper import FasterWhisperASR

if __name__ == "__main__":
    import sounddevice as sd
    import threading

    test_audio = r"D:\Whisper_stream\sss.mp3"
    language = 'ru'
    vad = False
    size = 'large-v3'
    asr_cls = FasterWhisperASR
    test_audio, sampling_rate = librosa.load(test_audio, sr=16000, dtype=np.float32)
    start_at = 0
    logfile = open('log.txt', 'w', encoding='utf8')
    min_chunk = 3.0
    duration = len(test_audio) / sampling_rate


    def play_audio():
        sd.play(test_audio, sampling_rate)
        sd.wait()


    def load_audio_chunk(beg, end):
        beg_s = int(beg * sampling_rate)
        end_s = int(end * sampling_rate)
        return test_audio[beg_s:end_s]


    t = time.time()
    asr = asr_cls(modelsize=size, lan=language, vad=vad)
    e = time.time()
    print(f"ASR {size} loaded for {e - t:.2f} seconds.", file=logfile)

    online = ASRProcessor(asr, sampling_rate)
    beg = start_at
    end = 0
    start = time.time() - beg

    audio_thread = threading.Thread(target=play_audio)
    audio_thread.start()

    while end < duration:
        now = time.time() - start
        if now < end + min_chunk:
            time.sleep(min_chunk + end - now)
        end = time.time() - start
        a = load_audio_chunk(beg, end)
        beg = end
        online.insert_audio_chunk(a)

        try:
            o = online.process_iter()
        except AssertionError:
            print("assertion error", file=logfile)
        else:
            print(o or "", end="")
        now = time.time() - start
        print(f"## last processed {end:.2f} s, now is {now:.2f}, the latency is {now - end:.2f} :: {o}", file=logfile,
              flush=True)

        if end >= duration:
            break

    o = online.finish()
    print(o or "", end="")
