import sounddevice as sd

print(sd.query_devices())
# Output:
#   0 HDMI Audio, Core Audio (0 in, 2 out)
# > 1 BlackHole 2ch, Core Audio (2 in, 2 out)
# < 2 Mac mini Speakers, Core Audio (0 in, 2 out)

device_info = sd.query_devices(sd.default.device[0], 'input')
channels = device_info['max_input_channels']
print(f"Detected {channels} input channels")