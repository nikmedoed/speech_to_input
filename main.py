import threading
import time
import keyboard
from app.ASRProcessor import ASRProcessor
from app.models.FasterWhisper import FasterWhisperASR
from app.RecordingIndicator import RecordingIndicator
import pynput
import pyperclip
from settings import Settings
from app.AudioStreamManager import AudioStreamManager


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
    settigs = Settings()

    asr_cls = FasterWhisperASR
    indicator = RecordingIndicator()
    start_time = time.time()

    processor = ASRProcessor(asr_cls(modelsize=settigs.model_size,
                                     lan=settigs.model_language,
                                     vad=settigs.model_vad),
                             settigs.sample_rate)
    # processor = ASRProcessorDemo(None, settigs.sample_rate)

    duration = time.time() - start_time
    print(f'model loaded {duration:.2f} sec')
    processing_thread = threading.Thread(target=main, args=(processor, indicator, settigs))
    processing_thread.start()

    try:
        indicator.root.mainloop()
    except KeyboardInterrupt:
        print("interrupted")
        indicator.root.destroy()
    finally:
        processing_thread.join()
