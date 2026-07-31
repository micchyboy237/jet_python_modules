"""
audio_diagnostics.py - Check audio device setup
"""

import pyaudio


def diagnose_audio_setup():
    p = pyaudio.PyAudio()

    print("\n" + "=" * 60)
    print("AUDIO DEVICE DIAGNOSTICS")
    print("=" * 60)

    print(f"\nTotal devices found: {p.get_device_count()}\n")

    blackhole_input = None
    blackhole_output = None
    default_output = p.get_default_output_device_info()

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info["name"]

        print(f"[{i}] {name}")
        print(f"    Input channels:  {info['maxInputChannels']}")
        print(f"    Output channels: {info['maxOutputChannels']}")
        print(f"    Sample rate:     {info['defaultSampleRate']:.0f} Hz")

        if "BlackHole" in name:
            if info["maxInputChannels"] > 0:
                blackhole_input = i
                print(f"    >>> CAN CAPTURE from this device")
            if info["maxOutputChannels"] > 0:
                blackhole_output = i
                print(f"    >>> CAN SEND to this device")
        print()

    print("-" * 60)
    print(
        f"Default output device: [{default_output['index']}] {default_output['name']}"
    )
    print("-" * 60)

    # Analysis
    print("\n🔍 ANALYSIS:")
    if blackhole_input is None or blackhole_output is None:
        print("❌ BlackHole not found or incomplete!")
        print("   Install with: brew install blackhole-2ch")
    else:
        print("✅ BlackHole found:")
        print(f"   Input index (capture):  {blackhole_input}")
        print(f"   Output index (send):    {blackhole_output}")

    # The crucial part - check routing
    print("\n⚠️  IMPORTANT SETUP REQUIRED:")
    print("=" * 60)
    print("Your audio routing is NOT automatically configured!")
    print("You need to either:")
    print()
    print("OPTION 1: Create Multi-Output Device (Recommended)")
    print("  1. Open 'Audio MIDI Setup' app")
    print("  2. Click '+' in bottom left → 'Create Multi-Output Device'")
    print("  3. Check BOTH:")
    print("     - Your speakers/headphones")
    print("     - BlackHole 2ch")
    print("  4. Right-click → 'Use This Device For Sound Output'")
    print()
    print("OPTION 2: Route specific app audio (More complex)")
    print("  Use apps like 'Loopback' or 'SoundSource' for per-app routing")
    print("=" * 60)

    p.terminate()

    return blackhole_input is not None and blackhole_output is not None


if __name__ == "__main__":
    diagnose_audio_setup()
