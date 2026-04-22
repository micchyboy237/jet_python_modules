# pyannote/audio/pipelines/multilabel.py

## Step-by-Step Analysis & Blueprint

### What This File Does

`multilabel.py` answers: **"Is [thing X] happening in this audio, and when?"** — for any number of things at once. Unlike speaker diarization (which finds _people_), this pipeline detects **named audio events or classes** — e.g. music, laughter, overlapping speech, noise — whatever the model was trained to recognise.

### How It Differs From the Other Pipelines

| Aspect            | SpeakerDiarization / SpeechSeparation | MultiLabelSegmentation                        |
| ----------------- | ------------------------------------- | --------------------------------------------- |
| Question answered | "Who spoke when?"                     | "Is [class X] present, and when?"             |
| Classes           | Discovered dynamically from audio     | Fixed — defined by the model at training time |
| Output            | Speaker-labelled timeline             | Class-labelled timeline (one track per class) |
| Embedding stage   | Yes                                   | No — no speaker identity needed               |
| Clustering stage  | Yes                                   | No — labels are already known                 |
| Binarization      | One threshold (global)                | Per-class onset + offset thresholds           |
| Metric            | DER                                   | F-measure or Identification Error Rate        |

### The 2-Stage Pipeline

```
Audio File
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Stage 1: SEGMENTATION                           │
│  Sliding window over audio                       │
│  Model outputs soft probabilities per class      │
│  Output: SlidingWindowFeature                    │
│          shape: (num_frames, num_classes)         │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  Stage 2: BINARIZATION (per class)               │
│  For each class independently:                   │
│    - Apply onset/offset hysteresis thresholds    │
│    - Remove regions shorter than min_duration_on │
│    - Fill gaps shorter than min_duration_off     │
│  Output: Annotation with one track per class     │
└──────────────────────────────────────────────────┘
```

### Key Design Decisions

**Onset + Offset (hysteresis binarization):** Instead of one threshold, the pipeline uses two:

- `onset`: probability must **rise above** this to mark a class as active
- `offset`: probability must **fall below** this to mark it inactive
- This prevents rapid on/off flickering near the boundary

**Per-class vs shared `min_duration`:** With `share_min_duration=False` (default), each class has its own `min_duration_on` and `min_duration_off`. With `True`, all classes share a single pair — fewer parameters, faster tuning.

**Two metrics:** The pipeline can optimise for either:

- `IdentificationErrorRate` (minimize) — penalises wrong class labels in each frame
- `MacroAverageFMeasure` (maximize) — average precision/recall F-score across all classes

### Feature Map / Blueprint

```
multilabel.py
│
└── MultiLabelSegmentation (Pipeline)
    ├── __init__()          → Load model, build per-class threshold params
    ├── classes()           → Return fixed list of class names from model
    ├── initialize()        → Build one Binarize object per class using current params
    ├── CACHED_SEGMENTATION → Cache key string for training-time caching
    ├── apply()             → Run model → binarize each class → combine into Annotation
    ├── get_metric()        → MacroAverageFMeasure or IdentificationErrorRate
    └── get_direction()     → "maximize" (fscore) or "minimize" (IER)
```

### Files to Create

1. `example_multilabel_basic.py` — Simple end-to-end detection on a file
2. `example_multilabel_advanced.py` — Hooks, shared durations, parameter tuning
3. `example_multilabel_evaluation.py` — F-measure vs IER, sweeping thresholds

---

```markdown
# pyannote/audio/pipelines/multilabel.py

## Step-by-Step Analysis & Blueprint

### What This File Does (Plain English)

This file answers: _"Is [X] happening right now, and when does it start and stop?"_ — for
any number of named audio classes at once.

Unlike speaker diarization (which discovers _who_ is speaking), this pipeline detects
**fixed, known categories** — whatever the underlying model was trained to recognise.
Examples include: music, laughter, applause, overlapping speech, background noise, a
specific sound event, or any custom class your model defines.

The result is a labelled timeline like:

- `music` active from 0–12s, 45–60s
- `laughter` active from 3–5s, 22–24s
- `overlapping_speech` active from 8–10s

---

### How It Differs From the Other Pipelines

| Aspect               | `SpeakerDiarization` / `SpeechSeparation` | `MultiLabelSegmentation`                      |
| -------------------- | ----------------------------------------- | --------------------------------------------- |
| **Question**         | "Who spoke when?"                         | "Is [class X] present, and when?"             |
| **Classes**          | Discovered dynamically from audio         | Fixed — defined by the model at training time |
| **Output**           | Speaker-labelled timeline                 | Class-labelled timeline (one track per class) |
| **Embedding stage**  | Yes — to fingerprint speakers             | No — labels are already known                 |
| **Clustering stage** | Yes — to group fingerprints by identity   | No — no identity grouping needed              |
| **Binarization**     | Single global threshold                   | Per-class onset + offset hysteresis           |
| **Metric**           | DER                                       | F-measure or Identification Error Rate        |

---

### The 2-Stage Pipeline
```

Audio File
│
▼
┌──────────────────────────────────────────────────────┐
│ Stage 1: SEGMENTATION │
│ Sliding-window inference over the full audio │
│ Model outputs a soft probability per class per frame│
│ Output: SlidingWindowFeature │
│ shape: (num_frames, num_classes) │
└──────────────────────┬───────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────┐
│ Stage 2: BINARIZATION (run independently per class)│
│ For each class: │
│ 1. Apply onset/offset hysteresis thresholds │
│ → converts soft scores to hard on/off │
│ 2. Remove active regions shorter than │
│ min_duration_on (too short to be real) │
│ 3. Fill inactive gaps shorter than │
│ min_duration_off (too short to be a real break)│
│ Output: Annotation with one labelled track per class│
└──────────────────────────────────────────────────────┘

```

---

### Feature Map / Blueprint

```

multilabel.py
│
└── MultiLabelSegmentation (Pipeline)
├── **init**() → Load model, discover class names, build threshold params
├── classes() → Return fixed list of class names from the model spec
├── initialize() → Build one Binarize object per class using current params
├── CACHED_SEGMENTATION → Cache key string for reuse during training/tuning
├── apply() → Run model → binarize each class → merge into Annotation
├── get_metric() → MacroAverageFMeasure or IdentificationErrorRate
└── get_direction() → "maximize" (fscore) or "minimize" (IER)

````

---

### Key Concepts Explained

| Concept                      | Plain English                                                                                        |
| ---------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Multi-label**              | Multiple classes can be active at the same time (music AND laughter simultaneously is fine)          |
| **Fixed classes**            | The set of detectable things is baked into the model — you can't add new classes at runtime         |
| **Onset threshold**          | Probability must **rise above** this value to mark a class as starting                              |
| **Offset threshold**         | Probability must **fall below** this value to mark a class as ending                                |
| **Hysteresis**               | Using two different thresholds (onset > offset) avoids rapid flickering near the boundary            |
| **`min_duration_on`**        | Discard any detected region shorter than this many seconds (removes blips)                          |
| **`min_duration_off`**       | Fill any gap between two active regions shorter than this (merges near-continuous events)            |
| **`share_min_duration`**     | When True, all classes share one `min_duration_on` and `min_duration_off` — fewer params to tune    |
| **`Binarize`**               | Internal helper that applies onset/offset/min-duration logic to a single class probability curve    |
| **`SlidingWindowFeature`**   | A numpy array paired with timing metadata — tells you which time each row corresponds to            |
| **`Annotation`**             | pyannote's labelled timeline object — stores (start, end, label) triples                            |
| **`initialize()`**           | Called automatically before `apply()` after parameters change — rebuilds the Binarize objects       |
| **`F-measure`**              | Harmonic mean of precision and recall — good when you care equally about missing events and false alarms |
| **`MacroAverageFMeasure`**   | Compute F-score per class, then average — each class counts equally regardless of frequency         |
| **`IdentificationErrorRate`**| Frame-level error rate — penalises any frame where the wrong set of labels is active                |
| **`get_direction()`**        | Tells the tuner whether to maximise or minimise the metric — flips depending on `fscore` flag       |
| **Training cache**           | During parameter search, the segmentation output is saved in `file["cache/segmentation"]` so the    |
|                              | model only runs once per file even when parameters are swept many times                              |

---

### Files to Create

1. `example_multilabel_basic.py` — Simple end-to-end detection on a file
2. `example_multilabel_advanced.py` — Hooks, shared durations, per-class inspection
3. `example_multilabel_evaluation.py` — F-measure vs IER, parameter sweeping

---

```python
# example_multilabel_basic.py
"""
Basic Multi-Label Segmentation — Detect Named Audio Events
===========================================================
Goal: Take an audio file → get back a timeline showing when each
      known audio class is active.

Unlike speaker diarization, this pipeline does NOT discover new
categories — it only detects the classes the model was trained on.
Examples of what a model might detect:
  - music, speech, noise, silence
  - laughter, applause, overlapping_speech
  - any custom audio event class

The output is a pyannote Annotation: a labelled timeline where
each entry says "[class] was active from Xs to Ys".

Install requirements:
  pip install pyannote.audio

You'll need a HuggingFace token for gated models:
  https://huggingface.co/settings/tokens
"""

import torch
from pyannote.audio.pipelines.multilabel import MultiLabelSegmentation

# ------------------------------------------------------------------
# 1. Build the pipeline
#    You MUST provide a segmentation model — there is no default.
#    The model defines which classes can be detected.
# ------------------------------------------------------------------
pipeline = MultiLabelSegmentation(
    segmentation="your/multilabel-model",   # ← replace with your model
    # token="hf_your_token_here",           # needed for gated HuggingFace models
    # cache_dir="/tmp/pyannote",            # where to store downloaded models
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)

print("Pipeline ready.")
print(f"Detectable classes : {pipeline.classes()}")
print(f"Optimising for     : {'F-measure' if pipeline.fscore else 'Identification Error Rate'}")
print(f"Shared durations   : {pipeline.share_min_duration}\n")

# ------------------------------------------------------------------
# 2. Instantiate parameters before running
#    Unlike SpeakerDiarization, MultiLabelSegmentation has no
#    default_parameters() — you must provide tuned values or
#    sensible starting points manually.
#
#    Parameter structure depends on share_min_duration:
#
#    share_min_duration=False (default):
#      {
#        "thresholds": {
#          "music":   {"onset": 0.5, "offset": 0.4,
#                      "min_duration_on": 0.1, "min_duration_off": 0.1},
#          "laughter": { ... },
#        }
#      }
#
#    share_min_duration=True:
#      {
#        "min_duration_on": 0.1,
#        "min_duration_off": 0.1,
#        "thresholds": {
#          "music":   {"onset": 0.5, "offset": 0.4},
#          "laughter": { ... },
#        }
#      }
# ------------------------------------------------------------------
classes = pipeline.classes()   # e.g. ["music", "laughter", "noise"]

params = {
    "thresholds": {
        label: {
            "onset": 0.5,
            "offset": 0.4,
            "min_duration_on":  0.1,
            "min_duration_off": 0.1,
        }
        for label in classes
    }
}
pipeline.instantiate(params)

# ------------------------------------------------------------------
# 3. Process an audio file
#    apply() returns a single Annotation (not a tuple).
#    AudioFile can be:
#      - a plain path string:  "audio.wav"
#      - a dict:               {"uri": "my_file", "audio": "audio.wav"}
# ------------------------------------------------------------------
audio_file = "audio.wav"   # ← replace with your file

detection = pipeline(audio_file)

# ------------------------------------------------------------------
# 4. Read the results
#    detection is a pyannote Annotation.
#    Each entry has: segment (start/end), track id, and label (class name).
# ------------------------------------------------------------------
print("=== Detected Events ===")
for segment, _, label in detection.itertracks(yield_label=True):
    print(f"  [{segment.start:6.2f}s → {segment.end:6.2f}s]  {label}")

print(f"\nActive classes found : {detection.labels()}")
print(f"Total detected turns : {len(list(detection.itertracks()))}\n")

# ------------------------------------------------------------------
# 5. Filter by a specific class
# ------------------------------------------------------------------
target_class = classes[0]   # e.g. "music"
class_timeline = detection.label_timeline(target_class)

print(f"=== Timeline for '{target_class}' only ===")
for segment in class_timeline:
    print(f"  [{segment.start:.2f}s → {segment.end:.2f}s]  "
          f"duration={segment.duration:.2f}s")

total = sum(seg.duration for seg in class_timeline)
print(f"  Total active time: {total:.2f}s\n")
````

---

```python
# example_multilabel_advanced.py
"""
Advanced Multi-Label Segmentation — Hooks, Shared Durations & Per-Class Inspection
====================================================================================
Goal: Show how to control the pipeline more precisely:
  1. Progress hooks — monitor inference in real time
  2. Shared min_duration — fewer parameters, faster tuning
  3. Per-class threshold inspection
  4. Accessing the raw segmentation probabilities
  5. Running on a waveform tensor (no file on disk needed)
  6. Combining multiple classes into a composite event
"""

import numpy as np
import torch

from pyannote.audio.pipelines.multilabel import MultiLabelSegmentation
from pyannote.core import Annotation, Segment

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
CLASSES = ["music", "laughter", "noise"]   # adjust to match your model

pipeline = MultiLabelSegmentation(
    segmentation="your/multilabel-model",
    # token="hf_your_token_here",
)
pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

audio_file = "audio.wav"   # ← replace


# ==================================================================
# FEATURE 1: Progress hook
# ==================================================================
print("=== Feature 1: Progress Hook ===\n")

# The hook is called after each major internal step.
# For MultiLabelSegmentation the main step is "segmentation".
# During long files it is also called multiple times with
# completed/total to show batch progress.

def my_hook(step_name, step_artifact, file=None, completed=None, total=None):
    if completed is not None and total is not None:
        pct = 100 * completed / total
        print(f"  [{step_name}] batch {completed}/{total}  ({pct:.0f}%)")
    else:
        info = ""
        if hasattr(step_artifact, "data"):
            info = f"data.shape={step_artifact.data.shape}"
        print(f"  ✓ {step_name:20s}  {info}")

params = {
    "thresholds": {
        label: {"onset": 0.5, "offset": 0.4,
                "min_duration_on": 0.1, "min_duration_off": 0.1}
        for label in pipeline.classes()
    }
}
pipeline.instantiate(params)
detection = pipeline(audio_file, hook=my_hook)
print()


# ==================================================================
# FEATURE 2: Shared min_duration — fewer parameters
# ==================================================================
print("=== Feature 2: Shared min_duration ===\n")

# With share_min_duration=True, all classes share one min_duration_on
# and one min_duration_off.  Only the onset/offset thresholds remain
# per-class.  This halves the number of duration parameters to tune.

pipeline_shared = MultiLabelSegmentation(
    segmentation="your/multilabel-model",
    share_min_duration=True,
    # token="hf_your_token_here",
)

classes = pipeline_shared.classes()

# Parameter structure is different when share_min_duration=True
shared_params = {
    "min_duration_on":  0.1,   # applies to ALL classes
    "min_duration_off": 0.1,   # applies to ALL classes
    "thresholds": {
        label: {"onset": 0.5, "offset": 0.4}
        for label in classes
    },
}
pipeline_shared.instantiate(shared_params)

detection_shared = pipeline_shared(audio_file)
print(f"Shared-duration pipeline → {detection_shared.labels()} detected.\n")

# Compare parameter counts:
#   share_min_duration=False: 4 params × num_classes
#   share_min_duration=True:  2 shared + 2 × num_classes
n = len(classes)
print(f"  Params without sharing : {4 * n}  ({n} classes × 4)")
print(f"  Params with sharing    : {2 + 2 * n}  (2 shared + {n} classes × 2)\n")


# ==================================================================
# FEATURE 3: Per-class threshold inspection
# ==================================================================
print("=== Feature 3: Per-Class Threshold Inspection ===\n")

# After instantiate(), the _binarize dict holds one Binarize object
# per class. You can inspect the thresholds that were actually applied.

pipeline.instantiate(params)   # re-instantiate the non-shared version

print("Binarizer thresholds per class:")
for label, binarizer in pipeline._binarize.items():
    print(f"  {label:20s}  onset={binarizer.onset:.2f}  "
          f"offset={binarizer.offset:.2f}  "
          f"min_on={binarizer.min_duration_on:.2f}s  "
          f"min_off={binarizer.min_duration_off:.2f}s")
print()


# ==================================================================
# FEATURE 4: Accessing raw segmentation probabilities
# ==================================================================
print("=== Feature 4: Raw Segmentation Probabilities ===\n")

# The pipeline's internal _segmentation object is a pyannote Inference.
# You can call it directly to get the soft probability matrix BEFORE
# binarization — useful for debugging or building custom logic.

raw: "SlidingWindowFeature" = pipeline._segmentation(audio_file)

print(f"Raw segmentation shape : {raw.data.shape}")
print(f"  Rows (frames)        : {raw.data.shape[0]}")
print(f"  Cols (classes)       : {raw.data.shape[1]}  → {pipeline.classes()}")
print(f"  Frame step           : {raw.sliding_window.step:.4f}s")

# Show the time-averaged probability for each class
print("\nMean probability per class:")
for i, label in enumerate(pipeline.classes()):
    mean_prob = np.mean(raw.data[:, i])
    print(f"  {label:20s}: {mean_prob:.4f}")
print()


# ==================================================================
# FEATURE 5: Running on an in-memory waveform tensor
# ==================================================================
print("=== Feature 5: In-Memory Waveform ===\n")

# If you already have audio loaded as a tensor (e.g. from torchaudio),
# pass it as a dict instead of a file path.
# Required keys: "waveform" (tensor) and "sample_rate" (int).
# Optional key:  "uri" (a string name for the file, used in the Annotation).

sample_rate = 16000
duration_s  = 10
fake_waveform = torch.zeros(1, sample_rate * duration_s)   # 10s of silence

file_dict = {
    "uri":         "in_memory_audio",
    "waveform":    fake_waveform,
    "sample_rate": sample_rate,
}

detection_mem = pipeline(file_dict)
print(f"Detected from in-memory tensor: {detection_mem.labels()}")
print(f"(Silent audio → likely no events detected)\n")


# ==================================================================
# FEATURE 6: Combining multiple classes into a composite event
# ==================================================================
print("=== Feature 6: Composite Events ===\n")

# Sometimes you want to know: "is EITHER music OR laughter happening?"
# You can merge multiple class timelines into one.

from pyannote.core import Timeline

detection = pipeline(audio_file)

# Get the timeline for each target class and union them
target_classes = ["music", "laughter"]   # adjust to your model's classes
composite = Timeline(uri=audio_file)

for label in target_classes:
    for segment in detection.label_timeline(label):
        composite.add(segment)

# support() merges overlapping segments into the minimal covering set
composite_support = composite.support()

print(f"Composite event (music OR laughter):")
for segment in composite_support:
    print(f"  [{segment.start:.2f}s → {segment.end:.2f}s]  "
          f"duration={segment.duration:.2f}s")
total_composite = sum(seg.duration for seg in composite_support)
print(f"  Total active time: {total_composite:.2f}s\n")
```

---

```python
# example_multilabel_evaluation.py
"""
Evaluating Multi-Label Segmentation Quality
=============================================
Goal: Measure how accurate the pipeline is against ground-truth
      annotations, and sweep parameters to find better settings.

Two metrics are available:
  1. IdentificationErrorRate (IER) — frame-level, lower is better
     Penalises any frame where the wrong set of labels is active.

  2. MacroAverageFMeasure — precision/recall F-score, higher is better
     Computes F-score per class, then averages — each class counts equally.

Covers:
  1. Evaluating with IER on a single file
  2. Evaluating with F-measure on a single file
  3. Multi-file evaluation (corpus-level metric)
  4. Parameter sweep to find better thresholds
  5. Understanding get_direction() for tuning loops
"""

import numpy as np
from pyannote.core import Annotation, Segment
from pyannote.audio.pipelines.multilabel import MultiLabelSegmentation
from pyannote.audio.utils.metric import MacroAverageFMeasure
from pyannote.metrics.identification import IdentificationErrorRate

# ------------------------------------------------------------------
# Setup — two pipelines, one per metric type
# ------------------------------------------------------------------
CLASSES = ["music", "laughter"]   # adjust to match your model

def make_pipeline(fscore: bool) -> MultiLabelSegmentation:
    p = MultiLabelSegmentation(
        segmentation="your/multilabel-model",
        fscore=fscore,
        # token="hf_your_token_here",
    )
    p.instantiate({
        "thresholds": {
            label: {"onset": 0.5, "offset": 0.4,
                    "min_duration_on": 0.1, "min_duration_off": 0.1}
            for label in p.classes()
        }
    })
    return p

pipeline_ier     = make_pipeline(fscore=False)
pipeline_fscore  = make_pipeline(fscore=True)

# Fake ground-truth reference (replace with real labelled data)
reference = Annotation(uri="demo")
reference[Segment(0.0,  8.0)]  = "music"
reference[Segment(3.0,  5.0)]  = "laughter"
reference[Segment(10.0, 15.0)] = "music"

audio_file = {"uri": "demo", "audio": "demo.wav"}   # ← replace


# ==================================================================
# FEATURE 1: Identification Error Rate on one file
# ==================================================================
print("=== Feature 1: Identification Error Rate (IER) ===\n")

# IER measures the fraction of frames where the predicted set of
# active labels does NOT exactly match the reference set.
# Lower is better. 0% means every frame was perfectly labelled.

detection_ier = pipeline_ier(audio_file)
metric_ier    = pipeline_ier.get_metric()   # IdentificationErrorRate instance

ier = metric_ier(reference, detection_ier)
print(f"IER: {ier * 100:.2f}%")

detail = metric_ier(reference, detection_ier, detailed=True)
for key, val in detail.items():
    if isinstance(val, float):
        print(f"  {key:25s}: {val * 100:.2f}%")
print()


# ==================================================================
# FEATURE 2: Macro-Average F-measure on one file
# ==================================================================
print("=== Feature 2: Macro-Average F-measure ===\n")

# F-measure = 2 * (precision * recall) / (precision + recall)
# Macro-average: compute F-score per class, then take the mean.
# Higher is better. 1.0 (100%) = perfect precision and recall.
#
# This metric is better than IER when:
#   - classes are rare and you care about recall
#   - you want to know per-class performance independently

detection_f  = pipeline_fscore(audio_file)
metric_f     = pipeline_fscore.get_metric()   # MacroAverageFMeasure instance

fscore = metric_f(reference, detection_f)
print(f"Macro F-measure: {fscore:.4f}  ({fscore * 100:.2f}%)")

# The metric also tracks per-class breakdown internally
# (access via the metric object's internal state if needed)
print(f"Optimisation direction: {pipeline_fscore.get_direction()}")   # "maximize"
print(f"IER      direction    : {pipeline_ier.get_direction()}\n")     # "minimize"


# ==================================================================
# FEATURE 3: Multi-file corpus evaluation
# ==================================================================
print("=== Feature 3: Multi-File Corpus Evaluation ===\n")

# Both metric objects are stateful — calling them multiple times
# accumulates results. Use abs() at the end for the corpus total.

test_set = [
    {"uri": "file1", "audio": "file1.wav"},
    {"uri": "file2", "audio": "file2.wav"},
]

fake_refs = {}
ref1 = Annotation(uri="file1")
ref1[Segment(0, 5)]  = "music"
ref1[Segment(5, 8)]  = "laughter"
fake_refs["file1"] = ref1

ref2 = Annotation(uri="file2")
ref2[Segment(0, 3)]  = "music"
ref2[Segment(6, 10)] = "music"
fake_refs["file2"] = ref2

accumulated_ier = pipeline_ier.get_metric()
accumulated_f   = pipeline_fscore.get_metric()

for file in test_set:
    det_ier = pipeline_ier(file)
    det_f   = pipeline_fscore(file)
    ref     = fake_refs[file["uri"]]
    accumulated_ier(ref, det_ier)
    accumulated_f(ref, det_f)

overall_ier = abs(accumulated_ier)
overall_f   = abs(accumulated_f)

print(f"Corpus IER           : {overall_ier * 100:.2f}%")
print(f"Corpus F-measure     : {overall_f:.4f}  ({overall_f * 100:.2f}%)\n")


# ==================================================================
# FEATURE 4: Parameter sweep — find better thresholds
# ==================================================================
print("=== Feature 4: Parameter Sweep ===\n")

# Key parameters to tune per class:
#   onset            → higher = more conservative (fewer detections)
#   offset           → lower = events end earlier
#   min_duration_on  → higher = ignore short blips
#   min_duration_off → higher = merge nearby events

# Here we sweep onset/offset jointly for all classes at once.
# In production you would sweep per-class on a held-out dev set.

sweep_configs = [
    {"onset": 0.3, "offset": 0.2, "min_duration_on": 0.05, "min_duration_off": 0.05},
    {"onset": 0.5, "offset": 0.4, "min_duration_on": 0.10, "min_duration_off": 0.10},
    {"onset": 0.7, "offset": 0.6, "min_duration_on": 0.20, "min_duration_off": 0.20},
]

dev_file = {"uri": "dev", "audio": "dev.wav"}   # ← replace
dev_ref  = Annotation(uri="dev")
dev_ref[Segment(0, 6)]  = "music"
dev_ref[Segment(4, 7)]  = "laughter"

# Use F-measure pipeline for the sweep (fscore=True → higher is better)
pipeline_sweep = make_pipeline(fscore=True)
classes_sweep  = pipeline_sweep.classes()

best_score  = -1.0
best_config = None

print("Sweeping onset/offset configs (dev file):")
for cfg in sweep_configs:
    params = {
        "thresholds": {
            label: {
                "onset":            cfg["onset"],
                "offset":           cfg["offset"],
                "min_duration_on":  cfg["min_duration_on"],
                "min_duration_off": cfg["min_duration_off"],
            }
            for label in classes_sweep
        }
    }
    pipeline_sweep.instantiate(params)
    det   = pipeline_sweep(dev_file)
    m     = pipeline_sweep.get_metric()
    score = m(dev_ref, det)

    label = (f"onset={cfg['onset']}, offset={cfg['offset']}, "
             f"min_on={cfg['min_duration_on']}")
    direction = "↑" if score > best_score else " "
    print(f"  {direction} [{label}]  F={score:.4f}")

    if score > best_score:
        best_score  = score
        best_config = params

print(f"\nBest F-measure : {best_score:.4f}")
print(f"Best config    : onset={sweep_configs[0]['onset']} ...")

# Apply best config for final inference
pipeline_sweep.instantiate(best_config)


# ==================================================================
# FEATURE 5: Understanding get_direction() in a tuning loop
# ==================================================================
print("\n=== Feature 5: get_direction() for Tuning Loops ===\n")

# get_direction() is used by automatic optimisers (like Optuna) to
# know whether they should maximise or minimise the metric.
# You can use it yourself to write direction-agnostic sweep code.

for name, p in [("IER pipeline", pipeline_ier), ("F-score pipeline", pipeline_fscore)]:
    direction = p.get_direction()
    metric    = p.get_metric()
    print(f"  {name:20s}  direction={direction:8s}  "
          f"metric_class={type(metric).__name__}")

print()

# Example direction-agnostic comparison:
def is_better(new_score, old_score, direction):
    """Return True if new_score is an improvement given the direction."""
    if direction == "maximize":
        return new_score > old_score
    return new_score < old_score   # minimize

direction   = pipeline_fscore.get_direction()
scores      = [0.62, 0.71, 0.68, 0.75, 0.70]
best        = scores[0]
best_idx    = 0

for i, s in enumerate(scores[1:], 1):
    if is_better(s, best, direction):
        best, best_idx = s, i

print(f"Direction: {direction}")
print(f"Scores   : {scores}")
print(f"Best     : {best} at index {best_idx}")
```
