from dataclasses import dataclass


@dataclass
class Settings:
    model_language = 'ru'
    model_vad = False
    model_size = 'large-v3'

    active_microphone_device: int = 1
    sample_rate = 16000

    copy_to_buffer = True
    typewrite = True

    stop_immediately = False

    # Варианты использования:
    # - наговорить и вставить без ввода по дорогое
    # - дождаться окончания и завершить
