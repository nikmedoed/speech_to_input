import pyaudio


def select_input_devices():
    p = pyaudio.PyAudio()
    input_devices = {}
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info["maxInputChannels"] > 0 and device_info["name"]:
            try:
                name = device_info["name"].encode('utf-8').decode('utf-8')
            except UnicodeEncodeError:
                name = device_info["name"].encode('ascii', errors='ignore').decode('utf-8', errors='ignore')
            if name not in input_devices.values():
                input_devices[i] = name
    p.terminate()

    for index, name in input_devices.items():
        print(f"Index: {index}, Name: {name}")

    index = int(input("Enter index or none: ") or 0)
    return index
