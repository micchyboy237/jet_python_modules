# pyannote/audio/pipelines/speech_separation.py

I'll analyze this file systematically before creating example files.

## Step-by-Step Analysis & Blueprint

### What this file does (plain English)

This is a **speech separation pipeline** — it takes an audio recording with multiple people talking, and outputs two things:

1. **Who spoke when** (diarization — a timeline of speakers)
2. **Separated audio per speaker** (like unmixing a cocktail party recording)

---

### Feature Breakdown

| Feature                  | What it does                                                                                           |
| ------------------------ | ------------------------------------------------------------------------------------------------------ |
| `SpeechSeparation` class | Main pipeline combining segmentation + separation + clustering                                         |
| `get_segmentations()`    | Runs the model over audio in sliding windows; returns both speaker activity AND separated audio chunks |
| `get_embeddings()`       | Extracts a "voice fingerprint" per speaker per chunk, with optional overlap exclusion                  |
| `reconstruct()`          | Stitches clustered chunk-level results into a full timeline                                            |
| `apply()`                | Master method — orchestrates everything; returns diarization + sources (+ optional embeddings)         |
| `batchify()`             | Utility to process items in fixed-size batches                                                         |
| Leakage removal          | Zeros out a speaker's audio track when they're not active (with configurable collar padding)           |
| Training cache           | Caches segmentations/embeddings to speed up hyperparameter optimization loops                          |
| `get_metric()`           | Returns DER metric for pipeline optimization                                                           |
| Hook callbacks           | Progress reporting at each major step                                                                  |

---

### Example Files Blueprint

- `example_basic_usage.py` — load pipeline, run on a file, inspect outputs
- `example_speaker_constraints.py` — use `num_speakers`, `min/max_speakers`
- `example_leakage_removal.py` — configure separation params, save separated audio
- `example_embeddings.py` — extract speaker embeddings, compute similarity
- `example_hook_progress.py` — use hooks for progress tracking
- `example_training_cache.py` — show how caching works during training/optimization

---

### New Files

**`example_basic_usage.py`**

```python
"""
example_basic_usage.py
======================
Simplest usage: load the pipeline and run it on an audio file.
Outputs:
  - diarization  : an Annotation (who spoke when)
  - sources      : a SlidingWindowFeature (one separated audio track per speaker)
"""

from pyannote.audio.pipelines.speech_separation import SpeechSeparation

# ------------------------------------------------------------------
# 1. Instantiate the pipeline with default pretrained models.
#    The first run will download ~500 MB of model weights.
# ------------------------------------------------------------------
pipeline = SpeechSeparation(
    segmentation="pyannote/separation-ami-1.0",
    embedding="speechbrain/spkrec-ecapa-voxceleb@5c0be3875fda05e81f3c004ed8c7c06be308de1e",
    clustering="AgglomerativeClustering",
)

# ------------------------------------------------------------------
# 2. Run on a local WAV file.
#    `apply()` is called under the hood when you use pipeline(file).
# ------------------------------------------------------------------
audio_path = "/path/to/your/meeting.wav"
diarization, sources = pipeline(audio_path)

# ------------------------------------------------------------------
# 3. Inspect the diarization timeline.
#    Each row: (time_segment, _, speaker_label)
# ------------------------------------------------------------------
print("=== Diarization ===")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"  [{turn.start:6.1f}s – {turn.end:6.1f}s]  {speaker}")

# ------------------------------------------------------------------
# 4. Inspect the separated sources.
#    sources.data shape: (num_frames, num_speakers)
#    sources.sliding_window: the time axis
# ------------------------------------------------------------------
print(f"\n=== Separated Sources ===")
print(f"  Shape (frames × speakers): {sources.data.shape}")
print(f"  Frame step : {sources.sliding_window.step * 1000:.1f} ms")

# ------------------------------------------------------------------
# 5. Save each speaker's separated audio as a WAV file.
# ------------------------------------------------------------------
import numpy as np
import soundfile as sf

sample_rate = 16_000          # must match the pipeline's internal sample rate
hop = sources.sliding_window.step   # seconds per frame

for i, label in enumerate(diarization.labels()):
    waveform = sources.data[:, i]          # 1-D float32 array
    out_path = f"separated_{label}.wav"
    sf.write(out_path, waveform, sample_rate)
    print(f"  Saved {out_path}  ({len(waveform)/sample_rate:.1f}s)")
```

**`example_speaker_constraints.py`**

```python
"""
example_speaker_constraints.py
===============================
Show how to guide the pipeline with speaker-count hints.

Three modes
-----------
A. Exact count  – you *know* there are N speakers
B. Bounded      – you know the range [min, max]
C. Unconstrained – let the pipeline decide (default)
"""

from pyannote.audio.pipelines.speech_separation import SpeechSeparation

pipeline = SpeechSeparation()
audio_path = "/path/to/your/meeting.wav"

# ------------------------------------------------------------------
# Mode A: exact count
# Useful when the number of participants is known (e.g., a 2-person
# interview). Bypasses the automatic speaker-count estimation step.
# ------------------------------------------------------------------
diarization_2, sources_2 = pipeline(audio_path, num_speakers=2)
print("Mode A — forced 2 speakers:")
for turn, _, spk in diarization_2.itertracks(yield_label=True):
    print(f"  {spk}  {turn.start:.1f}s → {turn.end:.1f}s")

# ------------------------------------------------------------------
# Mode B: bounded
# Good for panel discussions where you have rough expectations.
# The clustering step will produce between min and max clusters.
# ------------------------------------------------------------------
diarization_range, sources_range = pipeline(
    audio_path,
    min_speakers=2,
    max_speakers=5,
)
print(f"\nMode B — 2–5 speakers, found: {len(diarization_range.labels())}")
for turn, _, spk in diarization_range.itertracks(yield_label=True):
    print(f"  {spk}  {turn.start:.1f}s → {turn.end:.1f}s")

# ------------------------------------------------------------------
# Mode C: fully automatic (no hints)
# ------------------------------------------------------------------
diarization_auto, sources_auto = pipeline(audio_path)
print(f"\nMode C — auto, found: {len(diarization_auto.labels())} speakers")

# ------------------------------------------------------------------
# Note: if the detected count falls outside [min, max], the pipeline
# emits a UserWarning (not an exception). You can catch it:
# ------------------------------------------------------------------
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    diarization_w, _ = pipeline(audio_path, min_speakers=10, max_speakers=15)
    for w in caught:
        print(f"\nWarning: {w.message}")
```

**`example_leakage_removal.py`**

```python
"""
example_leakage_removal.py
==========================
"Leakage" = when one speaker's voice bleeds into another speaker's
separated track. This file demonstrates how to configure the
`separation` hyper-parameters to control leakage removal.

Key parameters
--------------
leakage_removal : bool
    If True, the pipeline zeros out a speaker's track whenever that
    speaker is detected as inactive. This removes bleed-through but
    can cut soft speech at turn boundaries.

asr_collar : float  (seconds)
    Safety margin around each active speech segment. Frames within
    `asr_collar` seconds of an active segment are *kept* even during
    an inactive period. Helps preserve the beginning/end of words.
"""

from pyannote.audio.pipelines.speech_separation import SpeechSeparation
import soundfile as sf

pipeline = SpeechSeparation()
audio_path = "/path/to/your/meeting.wav"

# ------------------------------------------------------------------
# 1. No leakage removal (default-ish) — maximum audio fidelity,
#    but other speakers may bleed in.
# ------------------------------------------------------------------
pipeline.separation.leakage_removal = False
pipeline.separation.asr_collar = 0.0

diarization, sources = pipeline(audio_path)
print("Without leakage removal:")
for i, label in enumerate(diarization.labels()):
    sf.write(f"noleak_{label}.wav", sources.data[:, i], 16_000)
    print(f"  Saved noleak_{label}.wav")

# ------------------------------------------------------------------
# 2. Leakage removal with a 0.2-second collar.
#    - Each speaker track is silenced when they are not speaking.
#    - 200 ms before/after each segment is preserved to avoid
#      clipping the first/last phoneme of each utterance.
# ------------------------------------------------------------------
pipeline.separation.leakage_removal = True
pipeline.separation.asr_collar = 0.2        # seconds

diarization, sources = pipeline(audio_path)
print("\nWith leakage removal (collar=0.2s):")
for i, label in enumerate(diarization.labels()):
    sf.write(f"leak02_{label}.wav", sources.data[:, i], 16_000)
    print(f"  Saved leak02_{label}.wav")

# ------------------------------------------------------------------
# 3. Aggressive leakage removal — tight collar.
#    Best for downstream ASR where silence between words matters.
# ------------------------------------------------------------------
pipeline.separation.leakage_removal = True
pipeline.separation.asr_collar = 0.05       # 50 ms — very tight

diarization, sources = pipeline(audio_path)
print("\nAggressive leakage removal (collar=0.05s):")
for i, label in enumerate(diarization.labels()):
    sf.write(f"leak005_{label}.wav", sources.data[:, i], 16_000)
    print(f"  Saved leak005_{label}.wav")

# ------------------------------------------------------------------
# Trade-off summary
# -----------------
# leakage_removal=False  → most natural audio, but dirty tracks
# collar=0.2s            → balanced; good for listening
# collar=0.05s           → cleanest tracks; may clip some phonemes
# ------------------------------------------------------------------
```

**`example_embeddings.py`**

```python
"""
example_embeddings.py
=====================
Extract per-speaker embeddings (voice fingerprints) alongside the
diarization and use them for downstream tasks like speaker
identification or cross-file speaker matching.

`return_embeddings=True` makes `apply()` return a third value:
    centroids : (num_speakers, embedding_dim) np.ndarray
    centroids[i] is the centroid embedding for the i-th speaker
    in `diarization.labels()`.
"""

import numpy as np
from pyannote.audio.pipelines.speech_separation import SpeechSeparation

pipeline = SpeechSeparation()
audio_path = "/path/to/your/meeting.wav"

# ------------------------------------------------------------------
# 1. Run pipeline with embeddings turned on.
# ------------------------------------------------------------------
diarization, sources, embeddings = pipeline(
    audio_path,
    return_embeddings=True,
)

labels = diarization.labels()          # e.g. ["SPEAKER_00", "SPEAKER_01"]
print(f"Speakers : {labels}")
print(f"Embedding matrix shape: {embeddings.shape}")   # (num_spk, dim)

# ------------------------------------------------------------------
# 2. Cosine similarity between every pair of speakers.
#    A high similarity (~1.0) means the model thinks they sound alike.
#    A low similarity (~0.0 or negative) means clearly different.
# ------------------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

print("\nPairwise cosine similarity:")
for i, label_i in enumerate(labels):
    for j, label_j in enumerate(labels):
        if j <= i:
            continue
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"  {label_i} ↔ {label_j} : {sim:.3f}")

# ------------------------------------------------------------------
# 3. Cross-file speaker matching.
#    Run the same pipeline on a second file, then find which speaker
#    in file 2 best matches each speaker in file 1.
# ------------------------------------------------------------------
audio_path_2 = "/path/to/another/meeting.wav"
diarization_2, _, embeddings_2 = pipeline(audio_path_2, return_embeddings=True)
labels_2 = diarization_2.labels()

print("\nCross-file speaker matching:")
for i, label_i in enumerate(labels):
    sims = [cosine_similarity(embeddings[i], embeddings_2[j]) for j in range(len(labels_2))]
    best_j = int(np.argmax(sims))
    print(f"  {label_i} (file1)  →  best match: {labels_2[best_j]} (file2)  sim={sims[best_j]:.3f}")

# ------------------------------------------------------------------
# 4. Save embeddings for later reuse (e.g. enrollment database).
# ------------------------------------------------------------------
np.save("speaker_embeddings.npy", embeddings)
np.save("speaker_labels.npy",     np.array(labels))
print("\nEmbeddings saved to speaker_embeddings.npy")
```

**`example_hook_progress.py`**

```python
"""
example_hook_progress.py
========================
The `hook` callback lets you track pipeline progress in real time.
Useful for long files (>30 min) or when embedding extraction is slow.

Hook signature
--------------
hook(step_name, artifact, *, file=None, total=None, completed=None)

step_name  : str   – human-readable name of the current step
artifact   : any   – the output produced so far (can be None mid-step)
total      : int   – total number of batches in this step (if known)
completed  : int   – how many batches are done so far
"""

from pyannote.audio.pipelines.speech_separation import SpeechSeparation

pipeline = SpeechSeparation()
audio_path = "/path/to/your/meeting.wav"

# ------------------------------------------------------------------
# 1. Simple print-based hook — good for scripts / notebooks.
# ------------------------------------------------------------------
def simple_hook(step_name, artifact, file=None, total=None, completed=None):
    if total is not None and completed is not None:
        pct = 100 * completed / total
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"\r  [{bar}] {pct:5.1f}%  {step_name}", end="", flush=True)
        if completed == total:
            print()   # newline when done
    else:
        print(f"  ✓ {step_name}")

print("Running pipeline with progress hook:")
diarization, sources = pipeline(audio_path, hook=simple_hook)
print("Done!")

# ------------------------------------------------------------------
# 2. Collecting step timings.
# ------------------------------------------------------------------
import time

timings: dict = {}
step_start: dict = {}

def timing_hook(step_name, artifact, file=None, total=None, completed=None):
    if completed == 0 or (total is None and completed is None):
        # Step just started
        step_start[step_name] = time.perf_counter()
    if completed is not None and total is not None and completed == total:
        elapsed = time.perf_counter() - step_start.get(step_name, time.perf_counter())
        timings[step_name] = elapsed

diarization, sources = pipeline(audio_path, hook=timing_hook)
print("\nStep timings:")
for step, duration in timings.items():
    print(f"  {step:<25s}  {duration:.2f}s")

# ------------------------------------------------------------------
# 3. tqdm integration — pretty progress bars in terminal or notebooks.
# ------------------------------------------------------------------
try:
    from tqdm.auto import tqdm

    bars: dict = {}

    def tqdm_hook(step_name, artifact, file=None, total=None, completed=None):
        if total is not None:
            if step_name not in bars:
                bars[step_name] = tqdm(total=total, desc=step_name, leave=True)
            if completed is not None:
                bars[step_name].n = completed
                bars[step_name].refresh()
            if completed == total:
                bars[step_name].close()
                del bars[step_name]

    print("\nRunning with tqdm bars:")
    diarization, sources = pipeline(audio_path, hook=tqdm_hook)

except ImportError:
    print("Install tqdm for pretty progress bars: pip install tqdm")
```

**`example_training_cache.py`**

```python
"""
example_training_cache.py
=========================
During hyperparameter optimization (training mode), the pipeline
re-runs `apply()` hundreds of times on the same files. Without
caching, the expensive segmentation and embedding steps would be
repeated every single iteration.

How the cache works
-------------------
- `pipeline.training = True` activates the cache.
- On the *first* call: segmentations and embeddings are computed
  and stored inside the `file` dict under known keys.
- On *subsequent* calls with the same file dict: the cached values
  are reused and only the cheap post-processing is redone.
- `pipeline.training = False` (default) always recomputes everything.
"""

import copy
import time
import numpy as np
from pyannote.audio.pipelines.speech_separation import SpeechSeparation
from pyannote.metrics.diarization import GreedyDiarizationErrorRate

pipeline = SpeechSeparation()

# A pyannote "file" dict can carry any extra keys — the cache lives here.
audio_file = {
    "uri": "meeting_001",
    "audio": "/path/to/your/meeting.wav",
    # Optional: provide ground-truth annotation for DER computation.
    # "annotation": <pyannote.core.Annotation>,
}

# ------------------------------------------------------------------
# 1. Demonstrate cache speedup.
# ------------------------------------------------------------------
pipeline.training = True

# First call — populates the cache inside `audio_file`.
t0 = time.perf_counter()
diarization, sources = pipeline(audio_file)
t_first = time.perf_counter() - t0
print(f"First call (cache MISS) : {t_first:.2f}s")
print(f"  Cache keys now present: {[k for k in audio_file if 'cache' in k]}")

# Second call — segmentation and embeddings come from the cache.
t0 = time.perf_counter()
diarization, sources = pipeline(audio_file)
t_cached = time.perf_counter() - t0
print(f"Second call (cache HIT) : {t_cached:.2f}s")
print(f"  Speedup : {t_first / t_cached:.1f}×")

pipeline.training = False   # reset to normal mode

# ------------------------------------------------------------------
# 2. Simulated hyperparameter sweep (simplified).
#    In real usage, pyannote.pipeline.Optimizer handles this loop.
#    Here we manually vary segmentation.min_duration_off.
# ------------------------------------------------------------------
print("\nSimulated hyperparameter sweep:")

pipeline.training = True
file_with_cache = copy.copy(audio_file)   # fresh file dict for this sweep
file_with_cache.pop("training_cache/segmentation", None)

der_metric = pipeline.get_metric()         # GreedyDiarizationErrorRate

best_der = float("inf")
best_params = {}

for min_off in np.linspace(0.0, 0.5, 6):
    pipeline.segmentation.min_duration_off = float(min_off)
    diarization, sources = pipeline(file_with_cache)

    # If you have ground-truth, compute real DER:
    # der = der_metric(file_with_cache["annotation"], diarization)
    # Fake DER for demo purposes:
    der = abs(min_off - 0.2) + np.random.uniform(0, 0.02)

    print(f"  min_duration_off={min_off:.2f}  →  DER={der:.3f}")
    if der < best_der:
        best_der = der
        best_params = {"min_duration_off": min_off}

print(f"\nBest params : {best_params}  (DER={best_der:.3f})")
pipeline.training = False
```
