# pyannote/audio/pipelines/multilabel.py

## Step-by-Step Analysis & Blueprint

### What this file does (plain English)

`MultiLabelSegmentation` is a pipeline that listens to audio and detects **multiple things happening at the same time** — each "label" is an independent category (e.g., "music", "speech", "noise", "applause"). Unlike speaker diarization which asks _who_ is speaking, this asks _what_ is happening and _when_.

---

### Feature Breakdown

| Feature                        | What it does                                                                                                                  |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `MultiLabelSegmentation` class | Main pipeline — runs a model that outputs one probability track per label                                                     |
| `share_min_duration`           | When `True`, all labels share the same `min_duration_on/off` params instead of having individual ones — fewer params to tune  |
| `initialize()`                 | Builds one `Binarize` object per label using current threshold params; called before each `apply()`                           |
| `apply()`                      | Runs inference, then slices the output per label, binarizes each slice, and merges into one `Annotation`                      |
| `fscore` flag                  | Switches the optimization target from minimizing error rate to maximizing F-score                                             |
| `get_metric()`                 | Returns the right metric object depending on `fscore` flag                                                                    |
| `get_direction()`              | Tells the optimizer whether to maximize or minimize the metric                                                                |
| Training cache                 | Stores raw segmentation output in `file` dict to avoid re-running the model during hyperparameter sweeps                      |
| `Binarize` per label           | Converts continuous probability scores → binary on/off regions using `onset`, `offset`, `min_duration_on`, `min_duration_off` |

---

### Blueprint

- `example_basic_usage.py` — load, run, inspect the `Annotation` output
- `example_share_min_duration.py` — compare shared vs per-label duration params
- `example_fscore_vs_ier.py` — show how metric choice affects optimization
- `example_custom_thresholds.py` — manually set binarization thresholds per label
- `example_hook_and_cache.py` — progress hooks + training cache speedup

---

### New Files

**`example_basic_usage.py`**

```python
"""
example_basic_usage.py
======================
Simplest usage: load a pretrained multi-label segmentation model,
run it on an audio file, and read the resulting Annotation.

What you get back
-----------------
An `Annotation` object where every detected segment is tagged with
one of the model's labels (e.g. "music", "speech", "noise").
Multiple labels can be active at the same time.
"""

from pyannote.audio.pipelines.multilabel import MultiLabelSegmentation

# ------------------------------------------------------------------
# 1. Load the pipeline.
#    `segmentation` must point to a pretrained multi-label model.
#    Replace the string with any compatible model on Hugging Face.
# ------------------------------------------------------------------
pipeline = MultiLabelSegmentation(
    segmentation="pyannote/segmentation",   # swap with your model
)

# ------------------------------------------------------------------
# 2. Run on an audio file.
#    Returns a pyannote.core.Annotation — a timeline of (segment, label) pairs.
# ------------------------------------------------------------------
audio_path = "/path/to/your/audio.wav"
detection: "Annotation" = pipeline(audio_path)

# ------------------------------------------------------------------
# 3. See which labels the model knows about.
# ------------------------------------------------------------------
print("Model labels:", pipeline.classes())
# e.g. ['music', 'speech', 'noise']

# ------------------------------------------------------------------
# 4. Iterate over every detected event.
#    Each row: time segment + the label active during that segment.
# ------------------------------------------------------------------
print("\nDetected events:")
for segment, track, label in detection.itertracks(yield_label=True):
    print(f"  [{segment.start:6.2f}s – {segment.end:6.2f}s]  {label}")

# ------------------------------------------------------------------
# 5. Filter to one label only.
# ------------------------------------------------------------------
target_label = "speech"
speech_only = detection.subset([target_label])   # keep one label
print(f"\nOnly '{target_label}':")
for segment, _, _ in speech_only.itertracks():
    print(f"  {segment.start:.2f}s → {segment.end:.2f}s")

# ------------------------------------------------------------------
# 6. Total active duration per label.
# ------------------------------------------------------------------
print("\nTotal duration per label:")
for label in detection.labels():
    total = sum(s.duration for s, _, _ in detection.itertracks(yield_label=True)
                if _ == label or True)   # re-iterate cleanly below
total_by_label = {
    label: sum(seg.duration for seg, _, lbl
               in detection.itertracks(yield_label=True) if lbl == label)
    for label in detection.labels()
}
for label, dur in total_by_label.items():
    print(f"  {label:<20s}  {dur:.2f}s")
```

**`example_share_min_duration.py`**

```python
"""
example_share_min_duration.py
==============================
`share_min_duration` controls whether every label gets its OWN
`min_duration_on` / `min_duration_off` params, or whether ALL labels
share a single pair of values.

Why it matters
--------------
Hyper-parameter optimization (e.g. via pyannote.pipeline.Optimizer)
searches over all free parameters. Fewer parameters → faster search.

share_min_duration=False  (default)
  Each label has: onset, offset, min_duration_on, min_duration_off
  Total params = 4 × num_labels

share_min_duration=True
  Each label has: onset, offset
  Plus two global params: min_duration_on, min_duration_off
  Total params = 2 × num_labels + 2   ← fewer to search
"""

from pyannote.audio.pipelines.multilabel import MultiLabelSegmentation

MODEL = "pyannote/segmentation"
AUDIO = "/path/to/your/audio.wav"

# ------------------------------------------------------------------
# 1. Per-label duration params (default).
#    Each label can have a different "minimum event length".
#    Best when labels have very different natural durations.
#    e.g. music events are long; clap events are short.
# ------------------------------------------------------------------
pipeline_per_label = MultiLabelSegmentation(
    segmentation=MODEL,
    share_min_duration=False,   # default
)

# Manually set thresholds for a label called "music" as an example.
# (Normally these are tuned automatically by the optimizer.)
if "music" in pipeline_per_label.classes():
    pipeline_per_label.thresholds["music"]["onset"]           = 0.5
    pipeline_per_label.thresholds["music"]["offset"]          = 0.4
    pipeline_per_label.thresholds["music"]["min_duration_on"] = 0.3   # 300 ms
    pipeline_per_label.thresholds["music"]["min_duration_off"]= 0.1

pipeline_per_label.initialize()   # must call after changing params manually
detection_a = pipeline_per_label(AUDIO)
print("Per-label params — events detected:", len(list(detection_a.itertracks())))

# ------------------------------------------------------------------
# 2. Shared duration params.
#    All labels share min_duration_on / min_duration_off.
#    Best when you don't have enough data to tune per-label durations.
# ------------------------------------------------------------------
pipeline_shared = MultiLabelSegmentation(
    segmentation=MODEL,
    share_min_duration=True,
)

# The shared params live directly on the pipeline object.
pipeline_shared.min_duration_on  = 0.2   # 200 ms minimum event length
pipeline_shared.min_duration_off = 0.1   # fill gaps shorter than 100 ms

# Per-label onset/offset are still independent.
for label in pipeline_shared.classes():
    pipeline_shared.thresholds[label]["onset"]  = 0.5
    pipeline_shared.thresholds[label]["offset"] = 0.4

pipeline_shared.initialize()
detection_b = pipeline_shared(AUDIO)
print("Shared params     — events detected:", len(list(detection_b.itertracks())))

# ------------------------------------------------------------------
# 3. Compare parameter counts (useful before starting optimization).
# ------------------------------------------------------------------
def count_params(p):
    """Count tunable float parameters in a pipeline."""
    from pyannote.pipeline.parameter import Uniform
    count = 0
    def _walk(obj):
        nonlocal count
        if isinstance(obj, Uniform):
            count += 1
        elif hasattr(obj, "__dict__"):
            for v in vars(obj).values():
                _walk(v)
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
    _walk(p)
    return count

print(f"\nParameter count — per-label : {count_params(pipeline_per_label)}")
print(f"Parameter count — shared    : {count_params(pipeline_shared)}")
```

**`example_fscore_vs_ier.py`**

```python
"""
example_fscore_vs_ier.py
========================
The pipeline can be optimized toward two different goals:

  fscore=False (default)  →  minimize Identification Error Rate (IER)
    IER penalizes all errors equally: missed detections, false alarms,
    and confusions between labels all count the same way.
    Good when you care about overall timeline accuracy.

  fscore=True  →  maximize macro-average F-score across labels
    F-score balances precision (don't fire when nothing is there) and
    recall (don't miss things that are there).
    Good when label classes are imbalanced or when some labels are rare.

This file shows how to instantiate each variant and inspect the metric.
"""

from pyannote.audio.pipelines.multilabel import MultiLabelSegmentation
from pyannote.core import Annotation, Segment

MODEL = "pyannote/segmentation"
AUDIO = "/path/to/your/audio.wav"

# ------------------------------------------------------------------
# 1. IER mode (default).
# ------------------------------------------------------------------
pipeline_ier = MultiLabelSegmentation(
    segmentation=MODEL,
    fscore=False,
)
metric_ier = pipeline_ier.get_metric()
direction_ier = pipeline_ier.get_direction()
print(f"IER mode   — metric: {type(metric_ier).__name__}, direction: {direction_ier}")
# Output: IdentificationErrorRate, minimize

# ------------------------------------------------------------------
# 2. F-score mode.
# ------------------------------------------------------------------
pipeline_f = MultiLabelSegmentation(
    segmentation=MODEL,
    fscore=True,
)
metric_f  = pipeline_f.get_metric()
direction_f = pipeline_f.get_direction()
print(f"F-score mode — metric: {type(metric_f).__name__}, direction: {direction_f}")
# Output: MacroAverageFMeasure, maximize

# ------------------------------------------------------------------
# 3. Manually compute the metric on a pair of annotations.
#    (Normally the optimizer does this automatically.)
# ------------------------------------------------------------------
# Build a tiny fake ground-truth annotation.
reference = Annotation(uri="demo")
reference["speech"] = {Segment(0.0, 3.0): "speech", Segment(5.0, 8.0): "speech"}

# Simulate a hypothesis (what the pipeline outputs).
hypothesis = Annotation(uri="demo")
hypothesis["speech"] = {Segment(0.5, 3.0): "speech", Segment(5.0, 7.5): "speech"}

# Compute IER.
ier_value = abs(metric_ier(reference, hypothesis))
print(f"\nIER on toy example  : {ier_value:.3f}  (lower is better)")

# Compute F-score.  MacroAverageFMeasure expects classes to be present.
# We use a simpler direct calculation here for illustration:
from pyannote.audio.utils.metric import MacroAverageFMeasure
fscore_metric = MacroAverageFMeasure(classes=["speech"])
fscore_value = fscore_metric(reference, hypothesis)
print(f"F-score on toy example: {fscore_value:.3f}  (higher is better)")

# ------------------------------------------------------------------
# 4. Practical guidance
# ------------------------------------------------------------------
print("""
When to choose each:
  fscore=False (IER)     → balanced multi-label audio (all labels common)
  fscore=True  (F-score) → imbalanced labels, or when recall matters most
                           e.g. rare event detection (gunshot, baby cry)
""")
```

**`example_custom_thresholds.py`**

```python
"""
example_custom_thresholds.py
=============================
Every label has four binarization parameters that turn the model's
continuous probability output into hard on/off segments:

  onset          – probability must RISE above this to start a segment
  offset         – probability must FALL below this to end a segment
  min_duration_on  – discard segments shorter than this (seconds)
  min_duration_off – fill gaps shorter than this (seconds)

                probability
                    │        onset=0.6
              0.6 ──┤─────────────────────────────
                    │   ╭───────────╮   ╭──────╮
              0.4 ──┤───┤  offset   ├───┤      │  offset=0.4
                    │                           │
                    └────────────────────────────── time
                         active segment

Setting these manually lets you trade off precision vs. recall
without running the full optimizer.
"""

from pyannote.audio.pipelines.multilabel import MultiLabelSegmentation

MODEL = "pyannote/segmentation"
AUDIO = "/path/to/your/audio.wav"

pipeline = MultiLabelSegmentation(segmentation=MODEL, share_min_duration=False)
labels = pipeline.classes()
print("Labels:", labels)

# ------------------------------------------------------------------
# 1. Conservative thresholds — high precision, lower recall.
#    Only fire when the model is very confident. Fewer false alarms,
#    but might miss soft/brief events.
# ------------------------------------------------------------------
for label in labels:
    pipeline.thresholds[label]["onset"]            = 0.7   # high bar to start
    pipeline.thresholds[label]["offset"]           = 0.6   # stays on unless clearly gone
    pipeline.thresholds[label]["min_duration_on"]  = 0.5   # ignore blips < 500 ms
    pipeline.thresholds[label]["min_duration_off"] = 0.1

pipeline.initialize()
detection_conservative = pipeline(AUDIO)
print("\nConservative (high precision):")
for seg, _, lbl in detection_conservative.itertracks(yield_label=True):
    print(f"  {lbl:<15s}  {seg.start:.2f}s → {seg.end:.2f}s")

# ------------------------------------------------------------------
# 2. Sensitive thresholds — high recall, lower precision.
#    Fire early and hold on. Catches more events but adds false alarms.
# ------------------------------------------------------------------
for label in labels:
    pipeline.thresholds[label]["onset"]            = 0.3   # fire quickly
    pipeline.thresholds[label]["offset"]           = 0.2   # needs to drop low to stop
    pipeline.thresholds[label]["min_duration_on"]  = 0.05  # keep even very short events
    pipeline.thresholds[label]["min_duration_off"] = 0.3   # fill gaps up to 300 ms

pipeline.initialize()
detection_sensitive = pipeline(AUDIO)
print("\nSensitive (high recall):")
for seg, _, lbl in detection_sensitive.itertracks(yield_label=True):
    print(f"  {lbl:<15s}  {seg.start:.2f}s → {seg.end:.2f}s")

# ------------------------------------------------------------------
# 3. Per-label fine-tuning — different settings per label.
#    Example: speech events are long; clap events are short bursts.
# ------------------------------------------------------------------
custom = {
    "speech": dict(onset=0.5, offset=0.4, min_duration_on=0.3, min_duration_off=0.2),
    "music":  dict(onset=0.4, offset=0.3, min_duration_on=1.0, min_duration_off=0.5),
    "noise":  dict(onset=0.6, offset=0.5, min_duration_on=0.1, min_duration_off=0.05),
}

for label in labels:
    params = custom.get(label, dict(onset=0.5, offset=0.4,
                                     min_duration_on=0.2, min_duration_off=0.1))
    for key, val in params.items():
        pipeline.thresholds[label][key] = val

pipeline.initialize()
detection_custom = pipeline(AUDIO)
print("\nPer-label custom thresholds:")
for seg, _, lbl in detection_custom.itertracks(yield_label=True):
    print(f"  {lbl:<15s}  {seg.start:.2f}s → {seg.end:.2f}s")

# ------------------------------------------------------------------
# 4. Onset must always be >= offset, otherwise the binarizer
#    can produce degenerate (empty or never-ending) segments.
# ------------------------------------------------------------------
print("\nThreshold sanity check:")
for label in labels:
    on  = pipeline.thresholds[label]["onset"]
    off = pipeline.thresholds[label]["offset"]
    ok  = "✓" if on >= off else "✗ PROBLEM"
    print(f"  {label:<15s}  onset={on:.2f}  offset={off:.2f}  {ok}")
```

**`example_hook_and_cache.py`**

```python
"""
example_hook_and_cache.py
=========================
Two independent but complementary features:

HOOK
----
A callback you provide to get notified as the pipeline runs.
Useful for showing progress on long files.

CACHE (training mode)
---------------------
When `pipeline.training = True`, the raw segmentation scores are
stored inside the `file` dict. Subsequent calls re-use the stored
scores and only redo the cheap binarization step. This makes
hyperparameter sweeps ~10-100× faster.
"""

import time
from functools import partial
from pyannote.audio.pipelines.multilabel import MultiLabelSegmentation

MODEL = "pyannote/segmentation"
AUDIO = "/path/to/your/audio.wav"

# ══════════════════════════════════════════════════════════════════
# PART A — Hook
# ══════════════════════════════════════════════════════════════════

pipeline = MultiLabelSegmentation(segmentation=MODEL)

# ------------------------------------------------------------------
# A1. Minimal hook — just prints each step name as it completes.
# ------------------------------------------------------------------
def simple_hook(step_name, artifact, file=None, total=None, completed=None):
    if total is not None and completed is not None:
        print(f"  {step_name}  {completed}/{total}", end="\r", flush=True)
        if completed == total:
            print(f"  {step_name}  done          ")
    else:
        print(f"  ✓ {step_name}")

print("Running with simple hook:")
detection = pipeline(AUDIO, hook=simple_hook)

# ------------------------------------------------------------------
# A2. Hook that records timings for each step.
# ------------------------------------------------------------------
step_times: dict = {}
_step_t0: dict = {}

def timing_hook(step_name, artifact, file=None, total=None, completed=None):
    if step_name not in _step_t0:
        _step_t0[step_name] = time.perf_counter()
    if completed is not None and total is not None and completed == total:
        step_times[step_name] = time.perf_counter() - _step_t0[step_name]

detection = pipeline(AUDIO, hook=timing_hook)
print("\nStep timings:")
for step, dur in step_times.items():
    print(f"  {step:<25s}  {dur:.2f}s")

# ══════════════════════════════════════════════════════════════════
# PART B — Training cache
# ══════════════════════════════════════════════════════════════════

# The cache key the pipeline uses internally.
CACHE_KEY = MultiLabelSegmentation.CACHED_SEGMENTATION   # "cache/segmentation"

# A pyannote "file" dict — the pipeline reads audio and also stores
# the cache as extra keys in this same dict.
audio_file = {
    "uri":   "my_audio",
    "audio": AUDIO,
}

pipeline.training = True   # activate cache

# ------------------------------------------------------------------
# B1. First call: cache is empty → model runs, result is stored.
# ------------------------------------------------------------------
t0 = time.perf_counter()
detection = pipeline(audio_file)
t_cold = time.perf_counter() - t0
print(f"\nFirst call  (cache MISS): {t_cold:.2f}s")
print(f"  Cache key present: {CACHE_KEY in audio_file}")

# ------------------------------------------------------------------
# B2. Second call: cache is warm → model is skipped entirely.
#     Only binarization (very fast) runs again.
# ------------------------------------------------------------------
t0 = time.perf_counter()
detection = pipeline(audio_file)
t_warm = time.perf_counter() - t0
print(f"Second call (cache HIT ): {t_warm:.2f}s")
print(f"  Speedup: {t_cold / max(t_warm, 1e-6):.1f}×")

# ------------------------------------------------------------------
# B3. Simulate a threshold sweep — the whole point of the cache.
#     Only binarization changes each iteration; inference is free.
# ------------------------------------------------------------------
import numpy as np

pipeline.training = True
sweep_file = {"uri": "sweep_demo", "audio": AUDIO}

print("\nThreshold sweep (binarization only after first call):")
for onset in np.linspace(0.3, 0.7, 5):
    for label in pipeline.classes():
        pipeline.thresholds[label]["onset"]  = float(onset)
        pipeline.thresholds[label]["offset"] = float(onset) - 0.1

    pipeline.initialize()
    t0 = time.perf_counter()
    det = pipeline(sweep_file)
    elapsed = time.perf_counter() - t0
    n_events = len(list(det.itertracks()))
    print(f"  onset={onset:.1f}  →  {n_events:3d} events  ({elapsed*1000:.0f}ms)")

pipeline.training = False   # always reset when done
```
