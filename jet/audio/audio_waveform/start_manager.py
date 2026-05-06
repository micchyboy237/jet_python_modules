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
        "hard_max_speech_frame": 1200,
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

    tracker_observer = obs_dict["tracker"]
    # Map from internal key to the VADObserver that should run for that key.
    # FireRed is excluded here because it always runs unconditionally.
    _SKIPPABLE_VAD_MAP = {
        "silero": obs_dict["silero"],
        "sb": obs_dict["speechbrain"],
        "ten_vad": obs_dict["ten_vad"],
    }

    def _on_vad_selection_changed(vad_key: str) -> None:
        tracker_observer.set_active_vad(vad_key)
        obs_dict["hybrid"].vad_probability = 0.0

    viz.add_vad_changed_callback(_on_vad_selection_changed)
    tracker_observer.set_active_vad(viz.current_vad)

    def coordinated_callback(samples):
        obs_dict["waveform"](samples)

        # FireRed always runs — it is the sole source of on_frame() boundary
        # events (speech start / end) that the tracker depends on.
        obs_dict["firered"](samples)

        # Only run the currently selected non-FireRed VAD.
        # This avoids paying the inference cost of idle neural nets.
        active_key = tracker_observer.active_vad
        if active_key in _SKIPPABLE_VAD_MAP:
            _SKIPPABLE_VAD_MAP[active_key](samples)

        obs_dict["tracker"](samples)

        # Build the probability map. Inactive models keep their .probability
        # at the last value they produced, but since they are no longer called
        # those slots will be stale. Use 0.0 for any model that didn't run
        # this frame so plots don't show phantom signals.
        _vad_prob_map = {
            "fr": obs_dict["firered"].probability,
            "silero": obs_dict["silero"].probability if active_key == "silero" else 0.0,
            "sb": obs_dict["speechbrain"].probability if active_key == "sb" else 0.0,
            "ten_vad": obs_dict["ten_vad"].probability
            if active_key == "ten_vad"
            else 0.0,
        }

        active_prob = _vad_prob_map.get(
            active_key,
            obs_dict["firered"].probability,
        )
        obs_dict["tracker"].tracker.add_prob(active_prob)

        obs_dict["hybrid"].vad_probability = _vad_prob_map.get(
            viz.current_vad, obs_dict["firered"].probability
        )
        obs_dict["hybrid"](samples)

        viz.push_data(
            wave=obs_dict["waveform"].value,
            silero=_vad_prob_map["silero"],
            sb=_vad_prob_map["sb"],
            fr=_vad_prob_map["fr"],
            ten_vad=_vad_prob_map["ten_vad"],
            hybrid=obs_dict["hybrid"].value,
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
