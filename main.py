import pyaudio
import numpy as np
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")


def recognize_stream():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096,
                    input_device_index=1)
    print("Говорите в микрофон...")

    frames_to_buffer = 16000 * 5
    audio_buffer = np.array([], dtype=np.float32)

    try:
        while True:
            data = stream.read(4096)
            current_audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768
            audio_buffer = np.concatenate((audio_buffer, current_audio))

            if len(audio_buffer) >= frames_to_buffer:
                segments, info = model.transcribe(
                    audio_buffer,
                    beam_size=5,
                    language='ru',
                    condition_on_previous_text=True,
                    vad_filter=True,
                    # temperature=0,
                    # vad_parameters=dict(min_silence_duration_ms=500)
                )
                for segment in segments:
                    print(segment.text)
                audio_buffer = np.array([], dtype=np.float32)

    except KeyboardInterrupt:
        print("Распознавание завершено.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


recognize_stream()
