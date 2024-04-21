import pyaudio
import numpy as np
import threading
import time
import keyboard
import queue

from app.ASRP_debug_demo import ASRProcessorDemo
from app.ASRProcessor import ASRProcessor
from app.models.FasterWhisper import FasterWhisperASR
from app.RecordingIndicator import RecordingIndicator
import pynput
import pyperclip


def main(processor: ASRProcessor, sample_rate, selected_device, indicator: RecordingIndicator):
    queue_audio_buffer = queue.Queue()
    record_is_process = threading.Event()

    keyboard_control = pynput.keyboard.Controller()

    def send_text(text):
        # print(text or "", end="")
        keyboard_control.type(text)

    def audio_callback(in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768
        queue_audio_buffer.put(audio_data)
        return in_data, pyaudio.paContinue

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        input_device_index=selected_device,
        frames_per_buffer=1024,
        stream_callback=audio_callback
    )

    stream.stop_stream()

    def handle_recording():
        moment = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        if not record_is_process.is_set():
            record_is_process.set()
            stream.start_stream()
            x, y = pynput.mouse.Controller().position
            indicator.show(x, y)
            time.time()
            print(f"\n{moment} Recording started.")
        else:
            record_is_process.clear()
            stream.stop_stream()
            print(f"\n{moment} Recording stopped.")

    keyboard.add_hotkey('ctrl+alt+r', handle_recording)

    try:
        while True:
            if queue_audio_buffer.empty():
                if record_is_process.is_set():
                    time.sleep(3)
                else:
                    record_is_process.wait()
            else:
                data_list = []
                while not queue_audio_buffer.empty():
                    data_list.append(queue_audio_buffer.get())
                processor.insert_audio_chunk(data_list)
                o = processor.process_iter()
                if queue_audio_buffer.empty() and not record_is_process.is_set():
                    pyperclip.copy(processor.gel_all_text().lstrip())
                    o += processor.finish()
                    indicator.hide()
                send_text(o)

    except KeyboardInterrupt:
        pass
    finally:
        record_is_process.clear()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    language = 'ru'
    vad = False
    size = 'large-v3'
    SAMPLE_RATE = 16000
    selected_device = 1

    asr_cls = FasterWhisperASR
    indicator = RecordingIndicator()
    start_time = time.time()

    processor = ASRProcessor(asr_cls(modelsize=size, lan=language, vad=vad), SAMPLE_RATE)
    # processor = ASRProcessorDemo(None, SAMPLE_RATE)

    duration = time.time() - start_time
    print(f'model loaded {duration:.2f} sec')
    processing_thread = threading.Thread(target=main, args=(processor, SAMPLE_RATE, selected_device, indicator))
    processing_thread.start()

    try:
        indicator.root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        processing_thread.join()
