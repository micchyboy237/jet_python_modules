import shutil
import sys
import time
from pathlib import Path

from jet.audio.audio_waveform.audio_manager import AudioStreamManager
from jet.audio.audio_waveform.speech_handlers.vad_observers import (
    create_original_observers,
)
from jet.audio.audio_waveform.speech_handlers.visualizer_observer import (
    VisualizerObserver,
)

OUTPUT_DIR = Path(__file__).parent / "generated" / "speech_tracking"
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

CONFIG = {
    "samplerate": 16000,
    "tracker": {"speech_threshold": 0.3},
    "firered": {
        "smooth_window_size": 5,
        "speech_threshold": 0.3,
        "pad_start_frame": 5,
        "min_speech_frame": 8,
        "soft_max_speech_frame": 450,
        "hard_max_speech_frame": 2000,
        "min_silence_frame": 90,
        "search_window": 250,
        "valley_threshold": 0.65,
        "chunk_max_frame": 30000,
        "min_valley_consecutive_frames": 5,
    },
}


def main():
    manager = AudioStreamManager(samplerate=CONFIG["samplerate"])

    # 1. Instantiate Core Logic Observers
    obs_dict = create_original_observers(
        samplerate=CONFIG["samplerate"],
        tracker_params=CONFIG["tracker"],
        firered_params=CONFIG["firered"],
        output_dir=OUTPUT_DIR,
    )

    # 2. Instantiate UI Observer
    viz = VisualizerObserver(display_points=200)

    # 3. Create a Coordinator Function
    # This ensures all analysis happens before we push to the UI buffer
    def coordinated_callback(samples):
        # Update Analysis
        obs_dict["waveform"](samples)
        obs_dict["silero"](samples)
        obs_dict["speechbrain"](samples)
        obs_dict["firered"](samples)
        obs_dict["ten_vad"](samples)
        obs_dict["tracker"](samples)

        # Feed the hybrid observer the VAD probability that matches whatever
        # the user has selected in the dropdown right now.
        _vad_prob_map = {
            "fr": obs_dict["firered"].probability,
            "silero": obs_dict["silero"].probability,
            "sb": obs_dict["speechbrain"].probability,
            "ten_vad": obs_dict["ten_vad"].probability,
        }
        obs_dict["hybrid"].vad_probability = _vad_prob_map.get(
            viz.current_vad, obs_dict["firered"].probability
        )
        obs_dict["hybrid"](samples)  # now computes .value correctly

        # Sync to UI
        viz.push_data(
            wave=obs_dict["waveform"].value,
            silero=obs_dict["silero"].probability,
            sb=obs_dict["speechbrain"].probability,
            fr=obs_dict["firered"].probability,
            ten_vad=obs_dict["ten_vad"].probability,
            hybrid=obs_dict["hybrid"].value,  # now non-zero and VAD-aware
        )

    manager.add_observer(coordinated_callback)

    try:
        manager.start()
        print("\n--- Audio Engine Running ---")
        print("Live Analysis Active. Press Ctrl+C to shutdown safely.")

        # Main Loop: Keep the script alive and handle Qt events
        while True:
            # Process Qt events so the window remains responsive
            viz.app.processEvents()

            # Print status to console
            print(
                f" VAD Probs | Silero: {obs_dict['silero'].probability:.2f} | "
                f"SB: {obs_dict['speechbrain'].probability:.2f} | "
                f"RMS: {obs_dict['waveform'].value:.2f} (raw: {obs_dict['waveform'].raw_rms:.3f}) | "
                f"Hybrid: {obs_dict['hybrid'].value:.2f}",
                end="\r",
            )
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\n[Shutdown] Keyboard interrupt received.")
    finally:
        print("[Shutdown] Closing stream and UI...")
        manager.stop()
        viz.close()
        print("[Shutdown] Cleanup complete.")
        sys.exit(0)


if __name__ == "__main__":
    main()
