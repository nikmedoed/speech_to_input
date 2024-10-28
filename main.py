import threading
import time

import keyboard
import pynput
import pyperclip

from app.ASRProcessor import ASRProcessor
from app.AudioStreamManager import AudioStreamManager
from app.RecordingIndicator import RecordingIndicator
from app.models.FasterWhisper import FasterWhisperASR
from app.select_device import select_input_devices
from settings import Settings


def send_text(text):
    # print(text or "", end="")
    while keyboard.is_pressed('shift') or keyboard.is_pressed('ctrl') or keyboard.is_pressed('alt'):
        time.sleep(0.1)
    keyboard.write(text)


def main(processor: ASRProcessor, indicator: RecordingIndicator, settings: Settings):
    record_is_process = threading.Event()
    stream = AudioStreamManager(settings)

    def handle_recording():
        moment = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        if not record_is_process.is_set():
            stream.start_stream()
            x, y = pynput.mouse.Controller().position
            indicator.show(x, y)
            record_is_process.set()
            print(f"\n{moment} Recording started.")
        else:
            stream.stop_stream()
            indicator.stop_recording()
            record_is_process.clear()
            print(f"\n{moment} Recording stopped.")

    keyboard.add_hotkey('ctrl+alt+r', handle_recording)
    try:
        keyboard.add_hotkey('ctrl+`', handle_recording)
    except ValueError:
        keyboard.add_hotkey('ctrl+ё', handle_recording)
    # DoubleKeyPress('alt', handle_recording)

    # try:
    while True:
        progressive_work = record_is_process.is_set() or not settings.stop_immediately
        if stream.empty():
            if record_is_process.is_set():
                time.sleep(3)
            else:
                record_is_process.wait()
        else:
            all_text = ""
            data_list = stream.get_audio_data()

            o = ""
            if progressive_work:
                processor.insert_audio_chunk(data_list)
                o = processor.process_iter()

            if stream.empty() and not record_is_process.is_set():
                all_text = processor.gel_all_text().lstrip()
                o += processor.finish()
                indicator.hide()

            if settings.typewrite and progressive_work:
                o = processor.remove_stop_phrases(o)
                send_text(o)
                time.sleep(0.3)
            if all_text and settings.copy_to_buffer:
                all_text = processor.remove_stop_phrases(all_text)
                pyperclip.copy(all_text)

    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     record_is_process.clear()
    #     if stream is not None:
    #         stream.stop_stream()
    #         stream.close()
    #     p.terminate()


if __name__ == "__main__":
    settings = Settings()

    asr_cls = FasterWhisperASR
    indicator = RecordingIndicator()
    start_time = time.time()

    settings.active_microphone_device = select_input_devices() or 1

    processor = ASRProcessor(asr_cls(modelsize=settings.model_size,
                                     lan=settings.model_language,
                                     vad=settings.model_vad),
                             settings.sample_rate)
    # processor = ASRProcessorDemo(None, settings.sample_rate)

    duration = time.time() - start_time
    print(f'model loaded {duration:.2f} sec')
    processing_thread = threading.Thread(target=main, args=(processor, indicator, settings))
    processing_thread.start()

    indicator.root.mainloop()
    # try:
    #     indicator.root.mainloop()
    # except KeyboardInterrupt:
    #     print("interrupted")
    #     indicator.root.destroy()
    # finally:
    #     processing_thread.join()
