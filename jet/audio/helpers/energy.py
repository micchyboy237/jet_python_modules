# jet_python_modules/jet/audio/utils.py   (new file or add to existing utils)
from pathlib import Path
from typing import List, Dict

import numpy as np
import soundfile as sf
from tqdm import tqdm

from jet.audio.record_mic import SAMPLE_RATE  # 16000
from jet.logger import logger

def compute_energy(audio_frame: np.ndarray) -> float:
    """Return mean absolute amplitude of an audio frame."""
    return float(np.mean(np.abs(audio_frame)))

def compute_energies(
    file_path: str | Path,
    chunk_duration: float = 0.25,
    silence_threshold: float | None = None,
) -> List[Dict[str, float]]:
    """
    Load a WAV file and compute the same per-chunk energy that the live streaming pipeline uses.

    Parameters
    ----------
    file_path : str | Path
        Path to a mono or stereo WAV file (any sample rate – will be resampled to 16 kHz).
    chunk_duration : float, default 0.5
        Size of each analysis block in seconds (matches the live stream default).
    silence_threshold : float | None
        If provided, also returns whether each chunk is considered silent.

    Returns
    -------
    List[Dict[str, float]]
        Each dict contains:
        - start_s : start time of the chunk (seconds)
        - end_s   : end time of the chunk (seconds)
        - energy  : mean absolute amplitude of the chunk
        - is_silent (optional) : True if energy < silence_threshold
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    audio, sr = sf.read(file_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)               # downmix to mono
    if sr != SAMPLE_RATE:
        logger.debug(f"Resampling {file_path.name} from {sr} → {SAMPLE_RATE} Hz")
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    audio = audio.astype(np.float32)
    if np.abs(audio).max() > 1.0:                # safety normalize
        audio /= np.abs(audio).max()

    chunk_samples = int(SAMPLE_RATE * chunk_duration)
    total_chunks = int(np.ceil(len(audio) / chunk_samples))

    results: List[Dict[str, float]] = []

    with tqdm(total=total_chunks, desc="Energy chunks", unit="chunk", leave=False) as pbar:
        for i in range(total_chunks):
            start_sample = i * chunk_samples
            end_sample = min((i + 1) * chunk_samples, len(audio))
            chunk = audio[start_sample:end_sample]

            energy = compute_energy(chunk)

            item: Dict[str, float] = {
                "start_s": round(start_sample / SAMPLE_RATE, 3),
                "end_s":   round(end_sample   / SAMPLE_RATE, 3),
                "energy":  round(energy, 6),
            }
            if silence_threshold is not None:
                item["is_silent"] = energy < silence_threshold

            results.append(item)
            pbar.update(1)

    logger.info(f"Computed energy for {len(results)} chunks from {file_path.name}")
    return results

if __name__ == "__main__":
    from jet.audio.helpers.silence import calibrate_silence_threshold

    audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/speech/silero/generated/silero_vad_stream/segment_001/sound.wav"

    threshold = calibrate_silence_threshold()
    energies = compute_energies(audio_file, silence_threshold=threshold)

    for e in energies[:10]:
        print(e)
    # → {'start_s': 0.0, 'end_s': 0.5, 'energy': 0.012345, 'is_silent': True}