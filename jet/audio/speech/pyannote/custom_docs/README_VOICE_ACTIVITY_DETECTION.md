# pyannote/audio/pipelines/voice_activity_detection.py

## Step-by-Step Analysis & Blueprint

### What this file does (plain English)

This file detects **when someone is speaking** in an audio recording — nothing more, nothing less. It answers: _"Is there speech happening right now?"_ and outputs a timeline of speech/non-speech regions. It has two classes:

- **`OracleVoiceActivityDetection`** — a cheat/test version that just reads the ground-truth annotation directly (useful for upper-bound benchmarks)
- **`VoiceActivityDetection`** — the real pipeline that runs a neural model

---

### Feature Breakdown

| Feature                                       | What it does                                                                                                                   |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `OracleVoiceActivityDetection`                | Returns perfect VAD from a ground-truth file — no model needed; used for testing pipelines downstream                          |
| `VoiceActivityDetection`                      | Neural VAD: runs a segmentation model, collapses multi-speaker scores → single speech probability via `np.max`, then binarizes |
| `pre_aggregation_hook`                        | Takes the max across all speaker channels so a multi-speaker model becomes a single speech/non-speech detector                 |
| `default_parameters()`                        | Hardcoded good starting thresholds for known model checkpoints — skip optimization for quick use                               |
| `powerset` branch                             | If the model uses powerset encoding, onset/offset are fixed at 0.5 (not tunable)                                               |
| `initialize()`                                | Builds a `Binarize` object from current params — must be called after changing thresholds manually                             |
| `apply()`                                     | Runs inference → binarizes → renames all labels to `"SPEECH"` → returns `Annotation`                                           |
| Training cache                                | Stores raw inference scores in `file` dict under `"cache/segmentation/inference"`                                              |
| `fscore` / `get_metric()` / `get_direction()` | Switches optimization target between Detection Error Rate and F-score                                                          |

---

### Blueprint

- `example_basic_vad.py` — run VAD, inspect speech timeline, compute total speech duration
- `example_oracle_vad.py` — use oracle pipeline for benchmarking and upper-bound analysis
- `example_default_parameters.py` — use `default_parameters()` to skip optimization
- `example_thresholds.py` — manually tune onset/offset/min_duration and see the effect
- `example_fscore_vs_der.py` — compare the two optimization metrics
- `example_training_cache.py` — cache speedup during hyperparameter sweeps

---

### New Files

**`example_basic_vad.py`**

```python
"""
example_basic_vad.py
====================
Simplest VAD usage: detect when speech is happening in an audio file.

Output
------
An `Annotation` where every segment is labeled "SPEECH".
Non-speech regions simply have no entry.
"""

from pyannote.audio.pipelines.voice_activity_detection import VoiceActivityDetection

# ------------------------------------------------------------------
# 1. Load the pipeline with the default pretrained model.
#    First run downloads ~200 MB of weights from Hugging Face.
# ------------------------------------------------------------------
pipeline = VoiceActivityDetection(
    segmentation="pyannote/segmentation",   # default
)

# Apply default known-good parameters so we skip optimization.
pipeline.instantiate(pipeline.default_parameters())

# ------------------------------------------------------------------
# 2. Run on an audio file.
#    Returns pyannote.core.Annotation — a timeline of SPEECH segments.
# ------------------------------------------------------------------
audio_path = "/path/to/your/audio.wav"
speech = pipeline(audio_path)

# ------------------------------------------------------------------
# 3. Inspect results.
# ------------------------------------------------------------------
print("Speech regions:")
for segment, _, label in speech.itertracks(yield_label=True):
    print(f"  [{segment.start:7.3f}s – {segment.end:7.3f}s]  {label}")

# ------------------------------------------------------------------
# 4. Compute total speech duration and speech ratio.
# ------------------------------------------------------------------
total_speech = sum(seg.duration for seg, _, _ in speech.itertracks())
print(f"\nTotal speech : {total_speech:.2f}s")

# Get total audio duration to compute ratio.
import soundfile as sf
info = sf.info(audio_path)
ratio = total_speech / info.duration
print(f"Audio length : {info.duration:.2f}s")
print(f"Speech ratio : {ratio:.1%}")

# ------------------------------------------------------------------
# 5. Convert the Annotation to a plain list of (start, end) tuples
#    — useful for feeding into downstream tools.
# ------------------------------------------------------------------
speech_segments = [
    (round(seg.start, 3), round(seg.end, 3))
    for seg, _, _ in speech.itertracks()
]
print(f"\nAs (start, end) list: {speech_segments[:5]} ...")

# ------------------------------------------------------------------
# 6. Get the non-speech (silence) gaps.
#    The timeline's gaps() method yields everything NOT in the timeline.
# ------------------------------------------------------------------
from pyannote.core import Timeline
speech_timeline = speech.get_timeline()
audio_extent    = speech_timeline.extent()

print("\nSilence gaps:")
for gap in speech_timeline.gaps(support=audio_extent):
    print(f"  [{gap.start:.3f}s – {gap.end:.3f}s]  ({gap.duration:.3f}s)")
```

**`example_oracle_vad.py`**

```python
"""
example_oracle_vad.py
=====================
`OracleVoiceActivityDetection` returns PERFECT speech regions by
reading the ground-truth annotation directly — no model is run.

When is this useful?
--------------------
1. Upper-bound benchmarking: what's the best DER a downstream
   diarization system could achieve if VAD were perfect?
2. Debugging: rule out VAD as the source of errors.
3. Fast iteration: skip the slow inference step during development.
"""

from pyannote.audio.pipelines.voice_activity_detection import (
    OracleVoiceActivityDetection,
    VoiceActivityDetection,
)
from pyannote.core import Annotation, Segment
from pyannote.metrics.detection import DetectionErrorRate

# ------------------------------------------------------------------
# 1. Build a fake "file" dict with ground-truth annotation.
#    In real use this would come from an RTTM or a dataset.
# ------------------------------------------------------------------
reference = Annotation(uri="demo")
reference[Segment(0.5,  4.2)] = "speaker_A"
reference[Segment(5.0,  9.8)] = "speaker_B"
reference[Segment(10.1, 14.0)] = "speaker_A"

audio_file = {
    "uri":        "demo",
    "audio":      "/path/to/demo.wav",
    "annotation": reference,        # <-- oracle reads this
}

# ------------------------------------------------------------------
# 2. Run Oracle VAD.
#    Returns the union (support) of all speaker segments, labeled SPEECH.
# ------------------------------------------------------------------
oracle = OracleVoiceActivityDetection()
perfect_speech = oracle.apply(audio_file)

print("Oracle VAD output:")
for seg, _, lbl in perfect_speech.itertracks(yield_label=True):
    print(f"  [{seg.start:.2f}s – {seg.end:.2f}s]  {lbl}")

# ------------------------------------------------------------------
# 3. Compare against real VAD to measure the VAD error contribution.
# ------------------------------------------------------------------
real_pipeline = VoiceActivityDetection(segmentation="pyannote/segmentation")
real_pipeline.instantiate(real_pipeline.default_parameters())
predicted_speech = real_pipeline(audio_file)

# Build single-class reference from oracle output (already SPEECH labels).
der_metric = DetectionErrorRate()

# Oracle DER should be ~0.
oracle_der = der_metric(perfect_speech, perfect_speech)
print(f"\nOracle DER  : {oracle_der:.3f}  (should be 0.0)")

# Real pipeline DER vs ground truth.
real_der = der_metric(perfect_speech, predicted_speech)
print(f"Real VAD DER: {real_der:.3f}")
print(f"VAD contributes {real_der:.1%} error to the downstream system.")

# ------------------------------------------------------------------
# 4. Note: OracleVoiceActivityDetection is a @staticmethod —
#    you can call it without instantiating, but the instance form
#    is more consistent with the rest of the pipeline API.
# ------------------------------------------------------------------
speech_direct = OracleVoiceActivityDetection.apply(audio_file)
print(f"\nDirect static call works too: {len(list(speech_direct.itertracks()))} segments")
```

**`example_default_parameters.py`**

```python
"""
example_default_parameters.py
==============================
`default_parameters()` returns pre-tuned threshold values for known
model checkpoints. Use these to skip the optimizer entirely when you
just want reasonable out-of-the-box results.

Pre-tuned values bundled in the code
--------------------------------------
"pyannote/segmentation"
    onset=0.767, offset=0.377, min_duration_on=0.136, min_duration_off=0.067

"pyannote/segmentation-3.0.0"   (powerset model)
    min_duration_on=0.0, min_duration_off=0.0
    (onset/offset are fixed at 0.5 for powerset models — not tunable)

Any other model checkpoint → raises NotImplementedError.
"""

from pyannote.audio.pipelines.voice_activity_detection import VoiceActivityDetection

AUDIO = "/path/to/your/audio.wav"

# ------------------------------------------------------------------
# 1. Load with the classic segmentation model and apply defaults.
# ------------------------------------------------------------------
pipeline_v1 = VoiceActivityDetection(segmentation="pyannote/segmentation")

defaults_v1 = pipeline_v1.default_parameters()
print("Default params for pyannote/segmentation:")
for k, v in defaults_v1.items():
    print(f"  {k:<20s} = {v}")

pipeline_v1.instantiate(defaults_v1)   # applies params + calls initialize()
speech_v1 = pipeline_v1(AUDIO)
print(f"  → {len(list(speech_v1.itertracks()))} speech segments detected\n")

# ------------------------------------------------------------------
# 2. Load with the 3.0.0 powerset model and apply its defaults.
#    Powerset models output exclusive class combinations, so onset/
#    offset are fixed at 0.5 and NOT returned by default_parameters().
# ------------------------------------------------------------------
pipeline_v3 = VoiceActivityDetection(segmentation="pyannote/segmentation-3.0.0")

defaults_v3 = pipeline_v3.default_parameters()
print("Default params for pyannote/segmentation-3.0.0:")
for k, v in defaults_v3.items():
    print(f"  {k:<20s} = {v}")
# Note: onset/offset are absent — they are hardcoded to 0.5 internally.

pipeline_v3.instantiate(defaults_v3)
speech_v3 = pipeline_v3(AUDIO)
print(f"  → {len(list(speech_v3.itertracks()))} speech segments detected\n")

# ------------------------------------------------------------------
# 3. What happens with an unsupported checkpoint.
# ------------------------------------------------------------------
pipeline_custom = VoiceActivityDetection(segmentation="my-org/my-vad-model")
try:
    pipeline_custom.default_parameters()
except NotImplementedError:
    print("Custom model has no bundled defaults — run the optimizer instead.")
    print("See example_training_cache.py for how to do that efficiently.")

# ------------------------------------------------------------------
# 4. Inspect which parameters are tunable (Uniform) vs fixed (float).
#    For powerset models onset/offset are plain floats (fixed at 0.5).
# ------------------------------------------------------------------
from pyannote.pipeline.parameter import Uniform

print("\nTunable parameters in pyannote/segmentation:")
for attr in ["onset", "offset", "min_duration_on", "min_duration_off"]:
    val = getattr(pipeline_v1, attr)
    kind = "tunable (Uniform)" if isinstance(val, Uniform) else f"fixed = {val}"
    print(f"  {attr:<20s}  {kind}")

print("\nTunable parameters in pyannote/segmentation-3.0.0:")
for attr in ["onset", "offset", "min_duration_on", "min_duration_off"]:
    val = getattr(pipeline_v3, attr)
    kind = "tunable (Uniform)" if isinstance(val, Uniform) else f"fixed = {val}"
    print(f"  {attr:<20s}  {kind}")
```

**`example_thresholds.py`**

```python
"""
example_thresholds.py
=====================
Manually tune the four binarization parameters and observe how
the output speech timeline changes. Useful for understanding the
trade-offs before or instead of running the full optimizer.

The four parameters
-------------------
onset          : probability must RISE above this → speech starts
offset         : probability must FALL below this → speech ends
                 (onset should always be >= offset)

min_duration_on  : discard speech blips shorter than this (seconds)
                   → removes false alarms from brief noise spikes

min_duration_off : fill silence gaps shorter than this (seconds)
                   → joins segments split by brief pauses

Visual intuition
----------------
  probability
  1.0 ┤
      │        ╭──────────────╮
  0.7 ┄┄┄┄┄┄┄┄┤ onset=0.7    │
      │        │              │
  0.4 ┄┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┄┄┄┄┄┄┤ offset=0.4
      │                       │
  0.0 ┴─────────────────────────── time
        silent  ←  SPEECH  →  silent
"""

from pyannote.audio.pipelines.voice_activity_detection import VoiceActivityDetection

AUDIO = "/path/to/your/audio.wav"

pipeline = VoiceActivityDetection(segmentation="pyannote/segmentation")

def run_and_summarize(label: str):
    """Apply current params, print a one-line summary."""
    pipeline.initialize()   # rebuild Binarize with current params
    speech = pipeline(AUDIO)
    segments  = list(speech.itertracks())
    total_dur = sum(s.duration for s, _, _ in segments)
    print(f"  {label:<35s}  segs={len(segments):3d}  speech={total_dur:.1f}s")
    return speech

# ------------------------------------------------------------------
# 1. Baseline — default known-good values.
# ------------------------------------------------------------------
params = pipeline.default_parameters()
pipeline.onset            = params["onset"]
pipeline.offset           = params["offset"]
pipeline.min_duration_on  = params["min_duration_on"]
pipeline.min_duration_off = params["min_duration_off"]
run_and_summarize("Baseline (default params)")

# ------------------------------------------------------------------
# 2. Lower onset → catch more speech (higher recall, more false alarms).
# ------------------------------------------------------------------
pipeline.onset  = 0.4
pipeline.offset = 0.3
run_and_summarize("Low onset=0.4 / offset=0.3")

# ------------------------------------------------------------------
# 3. Higher onset → only very confident speech (higher precision).
# ------------------------------------------------------------------
pipeline.onset  = 0.85
pipeline.offset = 0.70
run_and_summarize("High onset=0.85 / offset=0.70")

# ------------------------------------------------------------------
# 4. Large min_duration_on → suppress short noise bursts.
# ------------------------------------------------------------------
pipeline.onset            = params["onset"]
pipeline.offset           = params["offset"]
pipeline.min_duration_on  = 0.5   # ignore speech < 500 ms
pipeline.min_duration_off = params["min_duration_off"]
run_and_summarize("min_duration_on=0.5s  (suppress blips)")

# ------------------------------------------------------------------
# 5. Large min_duration_off → merge segments separated by brief pauses.
# ------------------------------------------------------------------
pipeline.min_duration_on  = params["min_duration_on"]
pipeline.min_duration_off = 0.4   # fill silences < 400 ms
run_and_summarize("min_duration_off=0.4s (merge short gaps)")

# ------------------------------------------------------------------
# 6. Hard rule: onset >= offset — otherwise segments never close.
# ------------------------------------------------------------------
print("\nSanity check:")
print(f"  onset ({pipeline.onset:.3f}) >= offset ({pipeline.offset:.3f}):",
      pipeline.onset >= pipeline.offset)
```

**`example_fscore_vs_der.py`**

```python
"""
example_fscore_vs_der.py
========================
Two metrics are available for optimizing / evaluating VAD:

  fscore=False  →  Detection Error Rate (DER)          [minimize]
    DER = (false alarm + missed speech) / total reference speech
    Penalizes both false alarms and misses proportionally to duration.
    Standard metric for VAD benchmarks.

  fscore=True   →  DetectionPrecisionRecallFMeasure     [maximize]
    F = 2 × precision × recall / (precision + recall)
    Balances not firing when silent (precision) vs catching all speech
    (recall). Better when reference speech is sparse or imbalanced.
"""

from pyannote.core import Annotation, Segment
from pyannote.audio.pipelines.voice_activity_detection import VoiceActivityDetection
from pyannote.metrics.detection import DetectionErrorRate, DetectionPrecisionRecallFMeasure

AUDIO = "/path/to/your/audio.wav"

# ------------------------------------------------------------------
# 1. Instantiate both variants and inspect metric objects.
# ------------------------------------------------------------------
pipeline_der = VoiceActivityDetection(segmentation="pyannote/segmentation", fscore=False)
pipeline_fsc = VoiceActivityDetection(segmentation="pyannote/segmentation", fscore=True)

metric_der = pipeline_der.get_metric()
metric_fsc = pipeline_fsc.get_metric()

print(f"DER    pipeline — metric: {type(metric_der).__name__:40s} direction: {pipeline_der.get_direction()}")
print(f"Fscore pipeline — metric: {type(metric_fsc).__name__:40s} direction: {pipeline_fsc.get_direction()}")

# ------------------------------------------------------------------
# 2. Compute both metrics on a toy example.
# ------------------------------------------------------------------
reference = Annotation(uri="toy")
reference[Segment(1.0,  5.0)] = "SPEECH"
reference[Segment(8.0, 12.0)] = "SPEECH"

# Hypothesis: misses the tail of the first segment, adds a false alarm.
hypothesis = Annotation(uri="toy")
hypothesis[Segment(1.0,  3.5)] = "SPEECH"   # missed 1.5s at end
hypothesis[Segment(6.0,  7.0)] = "SPEECH"   # 1s false alarm
hypothesis[Segment(8.0, 12.0)] = "SPEECH"   # perfect

der_value = abs(metric_der(reference, hypothesis))
fsc_value = metric_fsc(reference, hypothesis)

print(f"\nToy example:")
print(f"  DER    : {der_value:.3f}  (0.0 = perfect, lower is better)")
print(f"  F-score: {fsc_value:.3f}  (1.0 = perfect, higher is better)")

# ------------------------------------------------------------------
# 3. When reference speech is very sparse, DER can be misleading
#    because a small missed region causes a large percentage error.
#    F-score is more robust in that scenario.
# ------------------------------------------------------------------
sparse_ref = Annotation(uri="sparse")
sparse_ref[Segment(50.0, 50.5)] = "SPEECH"   # only 0.5s of speech in a long file

sparse_hyp = Annotation(uri="sparse")
# Hypothesis misses it entirely.

sparse_der = abs(metric_der(sparse_ref, sparse_hyp))
sparse_fsc = metric_fsc(sparse_ref, sparse_hyp)

print(f"\nSparse reference (0.5s speech missed entirely):")
print(f"  DER    : {sparse_der:.3f}  ← can be very large (100%+ possible)")
print(f"  F-score: {sparse_fsc:.3f}  ← 0.0 = all speech missed")

# ------------------------------------------------------------------
# 4. Guidance summary.
# ------------------------------------------------------------------
print("""
Guidance
--------
  fscore=False (DER)    → standard benchmark comparisons, balanced files
  fscore=True  (Fscore) → sparse speech, unbalanced datasets, recall focus
""")
```

**`example_training_cache.py`**

```python
"""
example_training_cache.py
=========================
VAD's training cache works identically to the one in
MultiLabelSegmentation and SpeechSeparation: the expensive
inference step is run once, stored in the `file` dict, and reused
on every subsequent call within a hyperparameter sweep.

Cache key: "cache/segmentation/inference"
           (VoiceActivityDetection.CACHED_SEGMENTATION)

Workflow
--------
  1. pipeline.training = True
  2. First call → inference runs, scores cached in file dict
  3. Subsequent calls → inference skipped, only binarization reruns
  4. pipeline.training = False  ← always reset when done
"""

import time
import numpy as np
from pyannote.audio.pipelines.voice_activity_detection import VoiceActivityDetection

AUDIO = "/path/to/your/audio.wav"

pipeline = VoiceActivityDetection(segmentation="pyannote/segmentation")

# Confirm the cache key the pipeline uses.
print(f"Cache key: '{VoiceActivityDetection.CACHED_SEGMENTATION}'")

# ------------------------------------------------------------------
# 1. Demonstrate the speedup.
# ------------------------------------------------------------------
audio_file = {"uri": "demo", "audio": AUDIO}

pipeline.training = True
pipeline.instantiate(pipeline.default_parameters())

t0 = time.perf_counter()
speech = pipeline(audio_file)
t_cold = time.perf_counter() - t0
print(f"\nFirst call  (cache MISS): {t_cold:.3f}s")
print(f"  Cached key present: "
      f"{VoiceActivityDetection.CACHED_SEGMENTATION in audio_file}")

t0 = time.perf_counter()
speech = pipeline(audio_file)
t_warm = time.perf_counter() - t0
print(f"Second call (cache HIT ): {t_warm:.3f}s")
print(f"  Speedup: {t_cold / max(t_warm, 1e-9):.1f}×")

# ------------------------------------------------------------------
# 2. Threshold sweep — the main use-case for the cache.
#    Inference runs once; only Binarize re-runs each iteration.
# ------------------------------------------------------------------
sweep_file = {"uri": "sweep", "audio": AUDIO}

print("\nOnset sweep (inference runs only on the FIRST iteration):")
results = []

for onset in np.linspace(0.3, 0.8, 6):
    offset = max(0.0, onset - 0.15)        # keep offset below onset
    pipeline.onset  = float(onset)
    pipeline.offset = float(offset)
    pipeline.min_duration_on  = 0.1
    pipeline.min_duration_off = 0.05
    pipeline.initialize()

    t0 = time.perf_counter()
    speech = pipeline(sweep_file)
    elapsed = time.perf_counter() - t0

    n_segs = len(list(speech.itertracks()))
    total  = sum(s.duration for s, _, _ in speech.itertracks())
    results.append((onset, n_segs, total, elapsed))
    print(f"  onset={onset:.2f} offset={offset:.2f}  "
          f"segs={n_segs:3d}  speech={total:6.1f}s  ({elapsed*1000:.0f}ms)")

print(f"\nSlowest iteration: {max(r[3] for r in results)*1000:.0f}ms  "
      f"(first, cache miss)")
print(f"Fastest iteration: {min(r[3] for r in results)*1000:.0f}ms  "
      f"(subsequent, cache hit)")

# ------------------------------------------------------------------
# 3. Always reset training mode when done.
#    Leaving it True means the stale cache is reused even on new files.
# ------------------------------------------------------------------
pipeline.training = False
print("\nTraining mode reset to False — safe for production use.")
```
