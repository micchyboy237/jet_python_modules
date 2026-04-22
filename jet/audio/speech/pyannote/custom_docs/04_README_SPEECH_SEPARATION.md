# pyannote/audio/pipelines/speech_separation.py

## Step-by-Step Analysis & Blueprint

### What This File Does

`speech_separation.py` solves a harder problem than plain diarization: **"Who spoke when, AND give me their isolated audio."** It jointly produces:

1. A speaker timeline (diarization annotation)
2. Separated audio streams — one clean waveform per speaker

### How It Differs From Speaker Diarization

| Aspect              | SpeakerDiarization           | SpeechSeparation                               |
| ------------------- | ---------------------------- | ---------------------------------------------- |
| Output              | Timeline only                | Timeline + separated audio                     |
| Model               | Segmentation only            | Segmentation + separation (joint)              |
| Segmentation output | `(chunks, frames, speakers)` | Same + `(chunks, samples, speakers)` waveforms |
| Post-processing     | Reconstruct diarization      | Reconstruct both diarization AND sources       |
| Extra step          | —                            | Leakage removal (zero-out inactive speakers)   |

### The 4-Stage Pipeline

```
Audio File
    │
    ▼
┌──────────────────────────────────────────┐
│  Stage 1: SEGMENTATION + SEPARATION      │
│  Joint model (PixIT-style)               │
│  → segmentations: (chunks, frames, spk)  │
│  → separations:   (chunks, samples, spk) │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│  Stage 2: EMBEDDING                      │
│  Voice fingerprint per (chunk, speaker)  │
│  Output: (chunks, speakers, dim)         │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│  Stage 3: CLUSTERING                     │
│  Group fingerprints by identity          │
│  Output: hard_clusters array             │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│  Stage 4: RECONSTRUCT + LEAKAGE REMOVAL  │
│  Build diarization timeline              │
│  Aggregate separated waveforms           │
│  Zero-out inactive speaker channels      │
│  Normalize sources                       │
└──────────────────────────────────────────┘
```

### Feature Map / Blueprint

```
speech_separation.py
│
├── batchify()                    → Utility: split iterable into fixed-size batches
│
└── SpeechSeparation (Pipeline)
    ├── __init__()                → Load joint seg+sep model, embedding, clustering
    ├── segmentation_batch_size   → Property with getter/setter
    ├── default_parameters()      → Not implemented (must be tuned/loaded)
    ├── classes()                 → Infinite generator: SPEAKER_00, SPEAKER_01, ...
    ├── CACHED_SEGMENTATION       → Cache key for training mode
    ├── get_segmentations()       → Stage 1: run joint model → seg + sep tensors
    ├── get_embeddings()          → Stage 2: extract per-(chunk,speaker) embeddings
    ├── reconstruct()             → Merge clustered chunks into one feature map
    ├── apply()                   → Master orchestrator — runs all 4 stages
    └── get_metric()              → Returns DER metric for evaluation
```

Now here's the full README:

---

```markdown
# pyannote/audio/pipelines/speech_separation.py

## Step-by-Step Analysis & Blueprint

### What This File Does (Plain English)

This file answers two questions at once:

1. **"Who spoke when?"** — the standard speaker diarization problem
2. **"Give me each speaker's voice as a separate audio stream"** — speech source separation

Given an audio file where multiple people talk (possibly at the same time), it produces:

- A **diarization timeline**: "Speaker 0 spoke 0–5s, Speaker 1 spoke 4–9s…"
- **Separated audio sources**: one clean waveform per speaker, with other speakers zeroed out

This is harder than plain diarization because it requires a **joint segmentation + separation
model** (the PixIT architecture) that does both jobs in a single forward pass.

---

### How It Differs From Speaker Diarization

| Aspect                  | `SpeakerDiarization`         | `SpeechSeparation`                        |
| ----------------------- | ---------------------------- | ----------------------------------------- |
| **Output**              | Timeline only                | Timeline **+** separated audio            |
| **Model**               | Segmentation only            | Joint segmentation + separation (PixIT)   |
| **Segmentation output** | `(chunks, frames, speakers)` | Same, plus `(chunks, samples, speakers)`  |
| **Post-processing**     | Reconstruct diarization      | Reconstruct diarization AND audio sources |
| **Extra step**          | —                            | Leakage removal: zero-out silent speakers |
| **Reference paper**     | Various                      | PixIT (Odyssey 2024)                      |

---

### The 4-Stage Pipeline
```

Audio File
│
▼
┌──────────────────────────────────────────────┐
│ Stage 1: SEGMENTATION + SEPARATION │
│ Sliding-window joint model (PixIT-style) │
│ Two outputs per chunk: │
│ segmentations: (chunks, frames, speakers) │
│ separations: (chunks, samples, speakers)│
└──────────────────────┬───────────────────────┘
│
▼
┌──────────────────────────────────────────────┐
│ Stage 2: EMBEDDING │
│ Extract voice fingerprint for each │
│ (chunk, speaker) pair │
│ Output: (chunks, speakers, dim) array │
└──────────────────────┬───────────────────────┘
│
▼
┌──────────────────────────────────────────────┐
│ Stage 3: CLUSTERING │
│ Group fingerprints by speaker identity │
│ "These 5 chunks are all Speaker 0" │
│ Output: hard_clusters array │
└──────────────────────┬───────────────────────┘
│
▼
┌──────────────────────────────────────────────┐
│ Stage 4: RECONSTRUCT + LEAKAGE REMOVAL │
│ Build diarization Annotation timeline │
│ Aggregate separated waveforms per speaker │
│ Zero-out channels where speaker is silent │
│ Normalize each source to [-1, 1] │
└──────────────────────────────────────────────┘

```

---

### Feature Map / Blueprint

```

speech_separation.py
│
├── batchify() → Utility: chop any iterable into fixed-size batches
│
└── SpeechSeparation (Pipeline)
├── **init**() → Load joint seg+sep model, embedding, clustering
├── segmentation_batch_size → Property with getter/setter
├── default_parameters() → Not implemented — must be tuned externally
├── classes() → Infinite generator: SPEAKER_00, SPEAKER_01, ...
├── CACHED_SEGMENTATION → Cache key string for training-time caching
├── get_segmentations() → Stage 1: run joint model → seg + sep tensors
├── get_embeddings() → Stage 2: per-(chunk,speaker) voice fingerprints
├── reconstruct() → Merge clustered chunk outputs into one feature map
├── apply() → Master orchestrator: runs all 4 stages
└── get_metric() → Returns GreedyDiarizationErrorRate for evaluation

````

---

### Key Concepts Explained

| Concept                   | Plain English                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------- |
| **PixIT**                 | The joint model architecture: one network that does segmentation AND separation together|
| **Segmentation**          | Sliding window over audio: "who is speaking in this 5-second chunk?"                   |
| **Separation**            | The same model also outputs isolated waveforms, one channel per local speaker          |
| **Powerset mode**         | Segmentation output is already binarized (on/off); no threshold needed                 |
| **Binarization**          | Converts soft probabilities (0.73) to hard on/off (1/0) using a threshold              |
| **Embedding**             | A voice fingerprint — a list of numbers unique to each speaker's voice                 |
| **Clustering**            | Groups fingerprints by identity: "these all sound like the same person"                |
| **Hard clusters**         | An array saying which global speaker ID each local (chunk, speaker) slot belongs to    |
| **reconstruct()**         | Applies the cluster assignments to build one big feature map from many chunk outputs   |
| **Leakage removal**       | Even after separation, a speaker's channel may "leak" audio when they are silent. This step zeros it out |
| **ASR collar**            | When doing leakage removal, keep a small buffer (N seconds) around each active turn so the starts/ends are not clipped |
| **`binary_dilation`**     | Scipy function used to expand the "active" mask by the collar amount on both sides     |
| **Source aggregation**    | Overlap-and-add with a Hamming window to stitch separated chunk waveforms into one continuous stream |
| **Normalization**         | Each source channel is divided by its peak absolute value so output is in [-1, 1]      |
| **DER**                   | Diarization Error Rate — the standard benchmark for timeline quality (lower = better)  |
| **`exclude_overlap`**     | Only use non-overlapping speech for embedding (cleaner signal, may be too short)       |
| **`OracleClustering`**    | Uses ground-truth speaker labels instead of predicted embeddings — for upper-bound research |
| **`return_embeddings`**   | When True, `apply()` also returns the centroid embedding for each discovered speaker   |

---

### Files to Create

1. `example_separation_basic.py` — Simple end-to-end separation: audio in, timeline + waveforms out
2. `example_separation_advanced.py` — Speaker constraints, progress hooks, leakage removal tuning
3. `example_separation_sources.py` — Working with the separated `SlidingWindowFeature` sources
4. `example_separation_evaluation.py` — DER evaluation, batchify utility, parameter sweeping

---

```python
# example_separation_basic.py
"""
Basic Speech Separation — Who Spoke When + Isolated Audio
==========================================================
Goal: Take an audio file where multiple people are talking →
      get back both a speaker timeline AND a clean audio stream
      per speaker.

This is more powerful than plain diarization: instead of just
knowing "Speaker 0 spoke from 3–7s", you can actually extract
what Speaker 0 said as a standalone audio signal.

The pipeline uses the PixIT architecture — a joint model that
does segmentation (who/when) and separation (isolated audio)
in one forward pass.

Install requirements:
  pip install pyannote.audio scipy soundfile

You'll need a HuggingFace token for gated models:
  https://huggingface.co/settings/tokens
"""

import numpy as np
import soundfile as sf
import torch

from pyannote.audio.pipelines.speech_separation import SpeechSeparation

# ------------------------------------------------------------------
# 1. Build the pipeline
#    Default model: "pyannote/separation-ami-1.0"
#    Default embedding: speechbrain ECAPA-TDNN
#    Default clustering: AgglomerativeClustering
# ------------------------------------------------------------------
pipeline = SpeechSeparation(
    # token="hf_your_token_here",   # needed for gated HuggingFace models
    # cache_dir="/tmp/pyannote",     # where to store downloaded models
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)

print("Pipeline ready.")
print(f"  Segmentation model : {pipeline.segmentation_model}")
print(f"  Clustering         : {pipeline.klustering}")
print(f"  Embedding model    : {pipeline.embedding}")
print(f"  Segmentation step  : {pipeline.segmentation_step} "
      f"(= {(1 - pipeline.segmentation_step) * 100:.0f}% window overlap)\n")

# ------------------------------------------------------------------
# 2. Process an audio file
#    apply() returns TWO things (unlike SpeakerDiarization):
#      diarization  → Annotation (speaker timeline)
#      sources      → SlidingWindowFeature (separated waveforms)
#
#    AudioFile can be:
#      - a plain path string:         "meeting.wav"
#      - a dict with waveform tensor: {"waveform": tensor, "sample_rate": 16000}
#      - a dict with uri + audio:     {"uri": "my_file", "audio": "meeting.wav"}
# ------------------------------------------------------------------
audio_file = "meeting.wav"   # ← replace with your file

diarization, sources = pipeline(audio_file)

# ------------------------------------------------------------------
# 3. Read the diarization timeline
# ------------------------------------------------------------------
print("=== Speaker Timeline ===")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"  [{turn.start:6.1f}s → {turn.end:6.1f}s]  {speaker}")

speakers = diarization.labels()
print(f"\nTotal speakers found : {len(speakers)}")
print(f"Speaker labels       : {speakers}\n")

# ------------------------------------------------------------------
# 4. Inspect the separated sources
#    sources is a SlidingWindowFeature with:
#      .data  → numpy array of shape (num_samples, num_speakers)
#      .sliding_window → frame-rate metadata
#
#    Each column is the isolated audio for one speaker.
#    Column order matches diarization.labels().
# ------------------------------------------------------------------
print("=== Separated Sources ===")
print(f"  sources.data.shape   : {sources.data.shape}")
print(f"  Columns (speakers)   : {sources.data.shape[1]}")
print(f"  Audio samples        : {sources.data.shape[0]}")

sr = round(1.0 / sources.sliding_window.step)   # inferred sample rate
print(f"  Inferred sample rate : {sr} Hz")
print(f"  Duration             : {sources.data.shape[0] / sr:.2f}s\n")

# ------------------------------------------------------------------
# 5. Save each speaker's isolated audio to a WAV file
# ------------------------------------------------------------------
print("=== Saving Speaker WAV Files ===")
for i, speaker in enumerate(speakers):
    speaker_audio = sources.data[:, i]   # shape: (num_samples,)
    out_path = f"{speaker}.wav"
    sf.write(out_path, speaker_audio, sr)
    peak = np.max(np.abs(speaker_audio))
    print(f"  Saved {out_path}  (peak amplitude: {peak:.4f})")

print("\nDone. Each speaker's audio is in a separate WAV file.")
````

---

```python
# example_separation_advanced.py
"""
Advanced Speech Separation — Constraints, Hooks & Leakage Removal
==================================================================
Goal: Show how to control the pipeline more precisely:
  1. Speaker count constraints (exact / min / max)
  2. Progress hooks — monitor each pipeline stage in real time
  3. Leakage removal — zero-out channels when speakers are silent
  4. ASR collar — keep a buffer around each active turn
  5. Batch sizes for GPU throughput
  6. Getting speaker embeddings alongside the sources
  7. Overlap exclusion for cleaner embeddings
"""

import numpy as np
import torch
import soundfile as sf

from pyannote.audio.pipelines.speech_separation import SpeechSeparation

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
pipeline = SpeechSeparation(
    # token="hf_your_token_here",
)

audio_file = "meeting.wav"   # ← replace with your file


# ==================================================================
# FEATURE 1: Speaker count constraints
# ==================================================================
print("=== Feature 1: Speaker Constraints ===\n")

# When you KNOW exactly how many people are in the recording
diarization, sources = pipeline(audio_file, num_speakers=3)
print(f"Exact 3 speakers → found: {len(diarization.labels())}")

# When you only know the rough range
diarization, sources = pipeline(audio_file, min_speakers=2, max_speakers=5)
n = len(diarization.labels())
print(f"Range 2–5 speakers → found: {n}  (within bounds: {2 <= n <= 5})")

# Unconstrained — let the pipeline decide
diarization, sources = pipeline(audio_file)
print(f"Unconstrained → found: {len(diarization.labels())} speakers\n")


# ==================================================================
# FEATURE 2: Progress hook
# ==================================================================
print("=== Feature 2: Progress Hook ===\n")

# The hook is called after each major internal step with:
#   step_name      → which stage just ran (str)
#   step_artifact  → what was produced (array / SlidingWindowFeature / etc.)
#   completed      → batches done so far (int, only during batch steps)
#   total          → total batches in this step (int, only during batch steps)

def my_hook(step_name, step_artifact, file=None, completed=None, total=None):
    if completed is not None and total is not None:
        pct = 100 * completed / total
        print(f"  [{step_name}]  batch {completed}/{total}  ({pct:.0f}%)")
    else:
        shape_info = ""
        if hasattr(step_artifact, "data"):
            shape_info = f"data.shape={step_artifact.data.shape}"
        elif hasattr(step_artifact, "shape"):
            shape_info = f"shape={step_artifact.shape}"
        print(f"  ✓ {step_name:30s}  {shape_info}")

diarization, sources = pipeline(audio_file, hook=my_hook)
print()


# ==================================================================
# FEATURE 3: Leakage removal
# ==================================================================
print("=== Feature 3: Leakage Removal ===\n")

# Even after source separation, a speaker's channel often "leaks" —
# there is a faint copy of other speakers bleeding through, especially
# during silent stretches.
#
# Leakage removal zeroes out any channel during frames where that
# speaker is inactive according to the diarization timeline.
# This produces much cleaner audio at the cost of hard cut-offs.
#
# The separation.leakage_removal parameter is a Categorical True/False.
# You set it by instantiating the pipeline with your chosen parameters.

# Disable leakage removal (keep all audio, even during silences)
params_no_removal = {
    "segmentation": {"min_duration_off": 0.1},
    "clustering":   {"threshold": 0.5, "Fa": 0.07, "Fb": 0.8},
    "separation":   {"leakage_removal": False, "asr_collar": 0.0},
}
pipeline.instantiate(params_no_removal)
diarization_leaky, sources_leaky = pipeline(audio_file)

# Enable leakage removal
params_with_removal = {
    "segmentation": {"min_duration_off": 0.1},
    "clustering":   {"threshold": 0.5, "Fa": 0.07, "Fb": 0.8},
    "separation":   {"leakage_removal": True, "asr_collar": 0.05},
}
pipeline.instantiate(params_with_removal)
diarization_clean, sources_clean = pipeline(audio_file)

# Compare RMS energy during silent frames
spk_idx = 0
leaky_rms = np.sqrt(np.mean(sources_leaky.data[:, spk_idx] ** 2))
clean_rms = np.sqrt(np.mean(sources_clean.data[:, spk_idx] ** 2))
print(f"Speaker 0 RMS — without leakage removal : {leaky_rms:.6f}")
print(f"Speaker 0 RMS — with leakage removal    : {clean_rms:.6f}")
print("(Lower RMS with removal = less background bleed-through)\n")


# ==================================================================
# FEATURE 4: ASR collar — buffering around speaker turns
# ==================================================================
print("=== Feature 4: ASR Collar ===\n")

# When leakage_removal=True, the pipeline zeros out a speaker's channel
# the instant they stop talking. This can clip the very start and end
# of their words — bad for Automatic Speech Recognition (ASR).
#
# asr_collar adds N seconds of "grace period" before AND after each
# active turn, so the channel stays open a little longer at boundaries.
#
# asr_collar is in seconds. Typical values: 0.05 – 0.25 s.

params_collar = {
    "segmentation": {"min_duration_off": 0.1},
    "clustering":   {"threshold": 0.5, "Fa": 0.07, "Fb": 0.8},
    "separation":   {"leakage_removal": True, "asr_collar": 0.2},
}
pipeline.instantiate(params_collar)
diarization_asr, sources_asr = pipeline(audio_file)

print("Applied ASR collar of 0.2s around each speaker turn.")
print(f"Speakers found: {diarization_asr.labels()}\n")


# ==================================================================
# FEATURE 5: Batch sizes — faster processing on GPU
# ==================================================================
print("=== Feature 5: Batch Sizes ===\n")

# Two independent batch sizes control GPU throughput:
#   segmentation_batch_size → how many audio windows to run at once
#   embedding_batch_size    → how many (chunk, speaker) pairs to embed at once
#
# Larger values = faster, but require more GPU memory.

pipeline.segmentation_batch_size = 32
pipeline.embedding_batch_size    = 16

print(f"Segmentation batch size : {pipeline.segmentation_batch_size}")
print(f"Embedding batch size    : {pipeline.embedding_batch_size}\n")


# ==================================================================
# FEATURE 6: Return speaker embeddings alongside sources
# ==================================================================
print("=== Feature 6: Return Speaker Embeddings ===\n")

# Passing return_embeddings=True makes apply() return a third value:
# the centroid embedding (voice fingerprint) for each discovered speaker.
# Shape: (num_speakers, embedding_dim)
# This is useful for cross-meeting speaker identification.

result = pipeline(audio_file, return_embeddings=True)
diarization, sources, embeddings = result

if embeddings is not None:
    print(f"Embedding matrix shape : {embeddings.shape}")
    for i, speaker in enumerate(diarization.labels()):
        norm = np.linalg.norm(embeddings[i])
        print(f"  {speaker}: embedding L2-norm = {norm:.4f}")
else:
    print("No embeddings returned (OracleClustering mode or silent file).")
print()


# ==================================================================
# FEATURE 7: Overlap exclusion for cleaner embeddings
# ==================================================================
print("=== Feature 7: Overlap Exclusion ===\n")

# When two speakers talk simultaneously, the mixed audio makes
# a noisy voice fingerprint. embedding_exclude_overlap=True tells
# the embedder to only use frames where exactly one speaker is active.
#
# Trade-off: if a speaker rarely speaks alone, the pipeline falls
# back to the mixed frames anyway.

pipeline_clean = SpeechSeparation(
    embedding_exclude_overlap=True,
    # token="hf_your_token_here",
)
pipeline_clean.instantiate(params_with_removal)
diarization_c, sources_c = pipeline_clean(audio_file)

print(f"With overlap exclusion → {len(diarization_c.labels())} speakers found.\n")
```

---

```python
# example_separation_sources.py
"""
Working with Separated Sources
================================
Goal: Deep-dive into the `sources` SlidingWindowFeature returned by
      the pipeline, and show practical things you can do with it:

  1. Reading shape and frame-rate metadata
  2. Extracting per-speaker numpy arrays
  3. Saving to WAV files with soundfile
  4. Aligning sources with the diarization timeline
  5. Measuring per-speaker energy
  6. Mixing speakers back together (sanity check)
  7. Using the speaker label generator (classes())
"""

import numpy as np
import soundfile as sf
import torch

from pyannote.core import Segment
from pyannote.audio.pipelines.speech_separation import SpeechSeparation

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
pipeline = SpeechSeparation(
    # token="hf_your_token_here",
)
# In real usage you would load tuned parameters; here we use a simple dict.
pipeline.instantiate({
    "segmentation": {"min_duration_off": 0.1},
    "clustering":   {"threshold": 0.5, "Fa": 0.07, "Fb": 0.8},
    "separation":   {"leakage_removal": True, "asr_collar": 0.05},
})

audio_file = "meeting.wav"   # ← replace with your file

diarization, sources = pipeline(audio_file)
speakers = diarization.labels()


# ==================================================================
# FEATURE 1: Shape and frame-rate metadata
# ==================================================================
print("=== Feature 1: sources Shape & Metadata ===\n")

# sources is a pyannote SlidingWindowFeature.
# .data         → numpy array  (num_audio_frames, num_speakers)
# .sliding_window.step      → seconds per frame
# .sliding_window.duration  → frame duration in seconds
# .sliding_window.start     → timestamp of the very first frame

data = sources.data                         # (num_frames, num_speakers)
sw   = sources.sliding_window

num_frames, num_speakers = data.shape
step_seconds   = sw.step
sample_rate_hz = round(1.0 / step_seconds)  # each "frame" is one audio sample

print(f"data.shape             : {data.shape}")
print(f"  num_audio_frames     : {num_frames}")
print(f"  num_speakers         : {num_speakers}")
print(f"sliding_window.step    : {step_seconds:.6f}s  (~{sample_rate_hz} Hz)")
print(f"sliding_window.start   : {sw.start:.4f}s")
total_duration = num_frames * step_seconds
print(f"Total audio duration   : {total_duration:.2f}s\n")


# ==================================================================
# FEATURE 2: Per-speaker numpy arrays
# ==================================================================
print("=== Feature 2: Per-Speaker Arrays ===\n")

# Column i of sources.data is the isolated waveform for speaker i.
# Column order matches diarization.labels() exactly.

for i, speaker in enumerate(speakers):
    waveform = sources.data[:, i]            # 1-D numpy array
    peak     = np.max(np.abs(waveform))
    rms      = np.sqrt(np.mean(waveform**2))
    print(f"  {speaker}:  shape={waveform.shape}  peak={peak:.4f}  RMS={rms:.6f}")
print()


# ==================================================================
# FEATURE 3: Save to WAV files
# ==================================================================
print("=== Feature 3: Save to WAV Files ===\n")

# soundfile.write(path, data, samplerate)
# The data must be in the range [-1.0, 1.0] — the pipeline already
# normalises sources.data to this range before returning.

sr = sample_rate_hz
for i, speaker in enumerate(speakers):
    waveform  = sources.data[:, i]
    out_path  = f"separated_{speaker}.wav"
    sf.write(out_path, waveform, sr)
    print(f"  Saved: {out_path}")
print()


# ==================================================================
# FEATURE 4: Align sources with the diarization timeline
# ==================================================================
print("=== Feature 4: Align Sources with Timeline ===\n")

# You can extract the exact audio slice for a given speaker turn
# by converting timestamps to sample indices.

for turn, _, speaker in diarization.itertracks(yield_label=True):
    spk_idx    = speakers.index(speaker)
    start_idx  = int(turn.start * sr)
    end_idx    = int(turn.end   * sr)
    # Clip to valid range (rounding can push end_idx slightly over)
    end_idx    = min(end_idx, num_frames)
    turn_audio = sources.data[start_idx:end_idx, spk_idx]

    duration_samples = end_idx - start_idx
    print(f"  {speaker}  [{turn.start:.2f}s → {turn.end:.2f}s]  "
          f"→ {duration_samples} samples")

print()


# ==================================================================
# FEATURE 5: Per-speaker energy during active segments
# ==================================================================
print("=== Feature 5: Per-Speaker Energy ===\n")

# Compute RMS energy ONLY during frames where the speaker is active.
# This is more meaningful than the full-file RMS, which includes
# silent stretches.

for i, speaker in enumerate(speakers):
    # Collect active frame indices from diarization timeline
    active_indices = []
    for turn, _, spk_label in diarization.itertracks(yield_label=True):
        if spk_label != speaker:
            continue
        s = int(turn.start * sr)
        e = min(int(turn.end * sr), num_frames)
        active_indices.extend(range(s, e))

    if not active_indices:
        print(f"  {speaker}: no active frames")
        continue

    active_audio = sources.data[active_indices, i]
    active_rms   = np.sqrt(np.mean(active_audio**2))
    active_peak  = np.max(np.abs(active_audio))
    active_dur   = len(active_indices) / sr
    print(f"  {speaker}:  active={active_dur:.2f}s  RMS={active_rms:.6f}  peak={active_peak:.4f}")

print()


# ==================================================================
# FEATURE 6: Mix speakers back (sanity check)
# ==================================================================
print("=== Feature 6: Mix Speakers Back Together ===\n")

# Sum all speaker channels to reconstruct an approximation of the
# original mixture. Perfect reconstruction is not guaranteed because
# the model is lossy, but the result should sound similar to the input.

mix = np.sum(sources.data, axis=1)          # shape: (num_frames,)
# Re-normalise
peak = np.max(np.abs(mix))
if peak > 0:
    mix = mix / peak

sf.write("reconstructed_mix.wav", mix, sr)
print(f"  Saved reconstructed_mix.wav  (shape={mix.shape}, peak={np.max(np.abs(mix)):.4f})\n")


# ==================================================================
# FEATURE 7: Speaker label generator
# ==================================================================
print("=== Feature 7: Speaker Label Generator ===\n")

# SpeechSeparation.classes() is an infinite generator producing
# SPEAKER_00, SPEAKER_01, SPEAKER_02, ... on demand.
# The pipeline uses this internally to assign standardised labels
# to the clusters it discovers.

gen = pipeline.classes()
first_8 = [next(gen) for _ in range(8)]
print(f"First 8 labels from classes(): {first_8}\n")
```

---

```python
# example_separation_evaluation.py
"""
Evaluating Speech Separation Quality
======================================
Goal: Measure how good the diarization part of the pipeline is using
      the standard Diarization Error Rate (DER) metric, and show how
      to sweep parameters to find better settings.

DER = Diarization Error Rate (lower is better, 0% = perfect).
It adds up three types of errors:
  1. Miss         — a speaker was talking but the pipeline missed them
  2. False Alarm  — the pipeline detected speech where there was silence
  3. Confusion    — right time range, but the wrong speaker label

Note: There is no standard single-number metric for the QUALITY of the
separated audio itself (that would need Signal-to-Distortion Ratio or
similar), so this file focuses on the diarization half of the output.

Covers:
  1. DER on a single file
  2. DER across multiple files
  3. DER variants: collar and skip_overlap
  4. Sweeping parameters to lower DER
  5. The batchify() utility used inside get_embeddings()
"""

import numpy as np
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import GreedyDiarizationErrorRate

from pyannote.audio.pipelines.speech_separation import (
    SpeechSeparation,
    batchify,
)

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
pipeline = SpeechSeparation(
    # token="hf_your_token_here",
)
pipeline.instantiate({
    "segmentation": {"min_duration_off": 0.1},
    "clustering":   {"threshold": 0.5, "Fa": 0.07, "Fb": 0.8},
    "separation":   {"leakage_removal": True, "asr_collar": 0.05},
})


# ==================================================================
# FEATURE 1: DER on a single file
# ==================================================================
print("=== Feature 1: DER on One File ===\n")

# To compute DER you need a reference annotation — a ground-truth
# transcript with accurate timestamps and speaker labels.
# In practice this comes from a labelled dataset or human transcript.

# Fake reference annotation for demonstration
reference = Annotation(uri="demo_file")
reference[Segment(0.0, 5.0)]  = "Alice"
reference[Segment(5.5, 10.0)] = "Bob"
reference[Segment(9.0, 15.0)] = "Alice"

# Run the pipeline
audio_file = {"uri": "demo_file", "audio": "demo.wav"}   # ← replace
diarization, sources = pipeline(audio_file)

# get_metric() returns the GreedyDiarizationErrorRate configured
# with this pipeline's der_variant settings.
metric = pipeline.get_metric()

# The metric object finds the best label mapping automatically.
der = metric(reference, diarization)
print(f"DER: {der * 100:.2f}%")

# You can also get a detailed breakdown of each error type:
detail = metric(reference, diarization, detailed=True)
print(f"  Miss          : {detail['miss'] * 100:.2f}%")
print(f"  False Alarm   : {detail['false alarm'] * 100:.2f}%")
print(f"  Confusion     : {detail['confusion'] * 100:.2f}%\n")


# ==================================================================
# FEATURE 2: DER across multiple files
# ==================================================================
print("=== Feature 2: Multi-File DER ===\n")

# GreedyDiarizationErrorRate accumulates results across many calls.
# Call abs() at the end to get the overall corpus-level DER.

test_set = [
    {"uri": "file1", "audio": "file1.wav"},
    {"uri": "file2", "audio": "file2.wav"},
]

# Fake ground-truth references
fake_refs = {
    "file1": Annotation(uri="file1"),
    "file2": Annotation(uri="file2"),
}
fake_refs["file1"][Segment(0, 3)] = "Alice"
fake_refs["file1"][Segment(3, 6)] = "Bob"
fake_refs["file2"][Segment(0, 4)] = "Carol"
fake_refs["file2"][Segment(4, 8)] = "Dave"

accumulated = pipeline.get_metric()   # fresh accumulator

for file in test_set:
    diarization, sources = pipeline(file)
    reference            = fake_refs[file["uri"]]
    accumulated(reference, diarization)

overall_der = abs(accumulated)
print(f"Overall DER across {len(test_set)} files: {overall_der * 100:.2f}%\n")


# ==================================================================
# FEATURE 3: DER variants — collar and skip_overlap
# ==================================================================
print("=== Feature 3: DER Variants ===\n")

# collar : ignore errors within N seconds of each speaker-change boundary.
#          Common in academic papers — it forgives small timing imprecision.
#          collar=0.25 means ±250ms around each boundary is ignored.
#
# skip_overlap : ignore time regions where 2+ speakers talk simultaneously.
#               Useful when you only care about single-speaker accuracy.

pipeline_collar = SpeechSeparation(
    der_variant={"collar": 0.25, "skip_overlap": False},
    # token="hf_your_token_here",
)
pipeline_skip = SpeechSeparation(
    der_variant={"collar": 0.0,  "skip_overlap": True},
    # token="hf_your_token_here",
)

print(f"Default metric  : collar=0.0,  skip_overlap=False  "
      f"(strictest — counts all errors)")
print(f"Collar metric   : collar=0.25, skip_overlap=False  "
      f"(forgives ±250ms boundary errors)")
print(f"Skip-overlap    : collar=0.0,  skip_overlap=True   "
      f"(ignores overlapping regions)\n")


# ==================================================================
# FEATURE 4: Parameter sweep to lower DER
# ==================================================================
print("=== Feature 4: Parameter Sweep ===\n")

# The pipeline has several tunable parameters.
# You would normally sweep these on a held-out dev set.
#
# Key parameters to tune:
#   segmentation.min_duration_off → merge short intra-speaker silences (seconds)
#   clustering.threshold         → cosine distance cutoff; lower = more speakers
#   separation.leakage_removal   → True/False; True gives cleaner sources
#   separation.asr_collar        → buffer around turns when leakage_removal=True

candidate_configs = [
    {
        "segmentation": {"min_duration_off": 0.0},
        "clustering":   {"threshold": 0.5, "Fa": 0.07, "Fb": 0.8},
        "separation":   {"leakage_removal": True, "asr_collar": 0.0},
    },
    {
        "segmentation": {"min_duration_off": 0.1},
        "clustering":   {"threshold": 0.6, "Fa": 0.07, "Fb": 0.8},
        "separation":   {"leakage_removal": True, "asr_collar": 0.05},
    },
    {
        "segmentation": {"min_duration_off": 0.2},
        "clustering":   {"threshold": 0.7, "Fa": 0.07, "Fb": 0.8},
        "separation":   {"leakage_removal": False, "asr_collar": 0.0},
    },
]

dev_file = {"uri": "dev_file", "audio": "dev.wav"}   # ← replace
dev_ref  = Annotation(uri="dev_file")
dev_ref[Segment(0, 5)]  = "A"
dev_ref[Segment(5, 10)] = "B"

best_der    = float("inf")
best_config = None

print("Sweeping configs on dev file:")
for config in candidate_configs:
    pipeline.instantiate(config)
    diarization, sources = pipeline(dev_file)
    m   = pipeline.get_metric()
    der = m(dev_ref, diarization)
    label = (
        f"min_off={config['segmentation']['min_duration_off']}, "
        f"clust={config['clustering']['threshold']}, "
        f"leakage={config['separation']['leakage_removal']}"
    )
    print(f"  [{label}]  DER = {der * 100:.2f}%")
    if der < best_der:
        best_der    = der
        best_config = config

print(f"\nBest config: {best_config}")
print(f"Best DER   : {best_der * 100:.2f}%\n")

# Apply the best config for subsequent inference
pipeline.instantiate(best_config)


# ==================================================================
# FEATURE 5: batchify() — the internal batching utility
# ==================================================================
print("=== Feature 5: batchify() Utility ===\n")

# batchify() splits any iterable into fixed-size tuples.
# Internally the pipeline uses it to group (waveform, mask) pairs
# before feeding them to the speaker embedding model in bulk.
# The last batch is padded with fillvalue if it is not full.

items   = list(range(10))   # [0, 1, 2, ..., 9]
batched = list(batchify(items, batch_size=3, fillvalue=-1))

print(f"Input       : {items}")
print(f"Batch size  : 3")
print(f"Output      : {batched}")
# Expected: [(0,1,2), (3,4,5), (6,7,8), (9,-1,-1)]
print()

# In pipeline code, None-padded entries are filtered out before processing:
#   filter(lambda b: b[0] is not None, batch)
# That's why fillvalue matters — the filter uses b[0] is not None,
# so fillvalue=(None, None) means the last partial batch is safely skipped.

print("Inside get_embeddings(), the pipeline does:")
print("  batches = batchify(iter_waveform_and_mask(),")
print("                     batch_size=self.embedding_batch_size,")
print("                     fillvalue=(None, None))")
print("  for batch in batches:")
print("      waveforms, masks = zip(*filter(lambda b: b[0] is not None, batch))")
```
