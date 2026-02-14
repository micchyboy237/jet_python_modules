import sounddevice as sd


def get_input_device():
    devices = sd.query_devices()
    input_devices = [
        (idx, dev) for idx, dev in enumerate(devices) if dev["max_input_channels"] > 0
    ]

    if not input_devices:
        raise RuntimeError("No input devices available.")

    return input_devices[0]  # or choose by name


print(sd.query_devices())
device_index, device_info = get_input_device()

print(f"Using device: {device_info['name']}")
print(f"Detected {device_info['max_input_channels']} input channels")
