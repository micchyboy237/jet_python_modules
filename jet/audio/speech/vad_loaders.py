from collections import OrderedDict

import numpy as np
from jet.audio.audio_waveform.vad._types import AudioInput
from jet.audio.audio_waveform.vad.vad_config import (
    DEFAULT_MAX_BUFFER_SEC,
    DEFAULT_MAX_SPEECH_SEC,
    DEFAULT_MIN_SILENCE_SEC,
    DEFAULT_MIN_SPEECH_SEC,
    DEFAULT_SMOOTH_WINDOW_SIZE,
    DEFAULT_THRESHOLD,
    SAVE_DIR,
)
from jet.audio.audio_waveform.vad.vad_firered_hybrid import FireRedVAD
from jet.audio.helpers.config import SAMPLE_RATE
from jet.audio.utils.loader import load_audio
from jet.data.utils import generate_key
from rich.console import Console

console = Console()


# Single global cache supporting up to 5 different VAD configurations
_global_vad_cache: OrderedDict[str, FireRedVAD] = OrderedDict()


def get_global_vad(
    threshold: float = DEFAULT_THRESHOLD,
    min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_SEC,
    min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_SEC,
    max_speech_duration_sec: float = DEFAULT_MAX_SPEECH_SEC,
    smooth_window_size: int = DEFAULT_SMOOTH_WINDOW_SIZE,
    max_buffer_sec: float = DEFAULT_MAX_BUFFER_SEC,
    **model_kwargs,
) -> FireRedVAD:
    """Get or create a globally cached VAD instance (max 5 configurations)."""
    global _global_vad_cache

    # Generate deterministic cache key
    cache_key = generate_key(
        threshold=threshold,
        min_silence_duration_sec=min_silence_duration_sec,
        min_speech_duration_sec=min_speech_duration_sec,
        max_speech_duration_sec=max_speech_duration_sec,
        smooth_window_size=smooth_window_size,
        max_buffer_sec=max_buffer_sec,
        **model_kwargs,
    )

    # Return cached instance if available
    if cache_key in _global_vad_cache:
        _global_vad_cache.move_to_end(cache_key)
        return _global_vad_cache[cache_key]

    # === New instance creation ===
    current_count = len(_global_vad_cache)

    with console.status(
        f"[bold blue]Loading FireRedVAD model...[/bold blue] "
        f"[cyan](Cache: {current_count}/5)[/cyan]"
    ):
        vad_instance = FireRedVAD(
            model_dir=SAVE_DIR,
            threshold=threshold,
            min_silence_duration_sec=min_silence_duration_sec,
            min_speech_duration_sec=min_speech_duration_sec,
            max_speech_duration_sec=max_speech_duration_sec,
            smooth_window_size=smooth_window_size,
            max_buffer_sec=max_buffer_sec,
            **model_kwargs,
        )

    _global_vad_cache[cache_key] = vad_instance
    _global_vad_cache.move_to_end(cache_key)  # Mark as most recently used

    # Enforce maximum cache size
    if len(_global_vad_cache) > 5:
        evicted_key = _global_vad_cache.popitem(last=False)[0]  # Remove oldest
        console.log(
            f"[yellow]VAD cache limit reached (5). Evicted oldest key.[/yellow] "
            f"Current size: {len(_global_vad_cache)}"
        )
    else:
        console.log(
            f"[green]VAD model cached successfully[/green] "
            f"[dim](Cache size: {len(_global_vad_cache)}/5)[/dim]"
        )

    return vad_instance


def load_vad_hybrid_probs(
    audio: AudioInput,
    threshold: float = DEFAULT_THRESHOLD,
    min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_SEC,
    min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_SEC,
    max_speech_duration_sec: float = DEFAULT_MAX_SPEECH_SEC,
    smooth_window_size: int = DEFAULT_SMOOTH_WINDOW_SIZE,
    max_buffer_sec: float = DEFAULT_MAX_BUFFER_SEC,
    sample_rate: float = SAMPLE_RATE,
    **kwargs,
) -> tuple[list[float], np.ndarray, dict]:
    audio_np, _ = load_audio(audio, sample_rate)

    vad = get_global_vad(
        threshold=threshold,
        min_silence_duration_sec=min_silence_duration_sec,
        min_speech_duration_sec=min_speech_duration_sec,
        max_speech_duration_sec=max_speech_duration_sec,
        smooth_window_size=smooth_window_size,
        max_buffer_sec=max_buffer_sec,
        **kwargs,
    )
    # Note: We do NOT call update_parameters since it doesn't exist.
    # The first set of parameters used will be "frozen" for the cache.

    frame_results, result = vad.detect_full(audio_np)

    probs = [r.smoothed_prob for r in frame_results]

    data = {
        "result": result,
        "frame_results": frame_results,
        "speech_segments": vad.get_speech_segments(),
        "audio": audio_np,
    }

    return probs, data
