from dataclasses import dataclass


@dataclass
class Settings:
    model_language = 'ru'
    model_vad = False
    model_size = 'large-v3'

    active_microphone_device: int = 1
    sample_rate = 16000
