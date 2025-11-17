from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class ModelProfile:
    model_language: str
    model_vad: bool
    model_vad_backend: str | None
    model_size: str


@dataclass
class Settings:
    profile: str = "default"
    model_language: str = "ru"
    model_vad: bool = False
    model_vad_backend: str | None = None
    model_size: str = "large-v3"

    active_microphone_device: int = 1
    active_output_device: int | None = None
    soundcard_loopback_id: str | None = None
    sample_rate: int = 16000

    copy_to_buffer: bool = True
    typewrite: bool = True

    stop_immediately: bool = False

    PROFILES: ClassVar[dict[str, ModelProfile]] = {
        "default": ModelProfile(
            model_language="ru",
            model_vad=False,
            model_vad_backend=None,
            model_size="large-v3",
        ),
        "meeting": ModelProfile(
            model_language="en",
            model_vad=True,
            model_vad_backend="webrtc",  # Быстрый и точный WebRTC VAD отлично отсекает тишину
            model_size="medium",
        ),
    }

    def __post_init__(self):
        self.apply_profile(self.profile)

    def apply_profile(self, profile_name: str):
        profile_key = (profile_name or "default").lower()
        profile = self.PROFILES.get(profile_key)
        if not profile:
            raise ValueError(f"Неизвестный профиль настроек: {profile_name!r}")
        self.profile = profile_key
        self.model_language = profile.model_language
        self.model_vad = profile.model_vad
        self.model_vad_backend = profile.model_vad_backend
        self.model_size = profile.model_size

    # Варианты использования:
    # - наговорить и вставить без ввода по дороге
    # - дождаться окончания и завершить
