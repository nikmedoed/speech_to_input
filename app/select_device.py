import pyaudio


def _normalize_name(name: str) -> str:
    try:
        return name.encode("utf-8").decode("utf-8")
    except UnicodeEncodeError:
        return name.encode("ascii", errors="ignore").decode("utf-8", errors="ignore")


def select_input_devices():
    p = pyaudio.PyAudio()
    input_devices = {}
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info["maxInputChannels"] > 0 and device_info["name"]:
            name = _normalize_name(device_info["name"])
            if name not in input_devices.values():
                input_devices[i] = name
    p.terminate()

    for index, name in input_devices.items():
        print(f"{index:>2}: {name.replace('Microphone ', '')}")

    index = int(input("Enter index or none: ") or 0)
    return index


def select_output_loopback_device():
    """Lists WASAPI loopback devices so the user can pick the desired speaker mix."""
    p = pyaudio.PyAudio()
    loopback_devices = {}
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get("isLoopbackDevice"):
            name = _normalize_name(info["name"])
            loopback_devices[i] = name
    p.terminate()

    if not loopback_devices:
        print("Loopback устройства не найдены. Будет использован вывод по умолчанию.")
        return None

    print("Доступные устройства вывода (loopback):")
    for index, name in loopback_devices.items():
        print(f"{index:>2}: {name}")

    raw = input("Введите индекс устройства или оставьте пустым для значения по умолчанию: ").strip()
    return int(raw) if raw else None


def select_soundcard_loopback_device():
    """Optional helper to pick a specific soundcard loopback microphone for fallback mode."""
    try:
        import soundcard as sc  # type: ignore
    except ImportError:
        print("soundcard не установлен. Fallback сможет использовать только устройство по умолчанию.")
        return None

    loopbacks = [mic for mic in sc.all_microphones(include_loopback=True) if getattr(mic, "isloopback", False)]
    if not loopbacks:
        print("Loopback микрофоны (soundcard) не найдены. Будет использован вывод по умолчанию.")
        return None

    print("Loopback-устройства (soundcard fallback):")
    for idx, mic in enumerate(loopbacks):
        print(f"{idx:>2}: {mic.name}")

    raw = input("Выберите устройство для fallback (Enter — использовать звук по умолчанию): ").strip()
    if not raw:
        return None
    try:
        selection = loopbacks[int(raw)]
    except (ValueError, IndexError):
        print("Некорректный выбор. Используется значение по умолчанию.")
        return None
    return selection.id
