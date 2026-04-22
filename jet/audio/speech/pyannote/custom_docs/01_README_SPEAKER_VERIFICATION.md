# pyannote/audio/pipelines/speaker_verification.py

### Key Concepts Explained

| Concept                            | Plain English                                                                                                                                   |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Voice Activity Detection (VAD)** | Detecting _when_ speech happens — ignores _who_ is speaking                                                                                     |
| **Segmentation model**             | A neural network that processes short audio windows and outputs per-frame probabilities                                                         |
| **SlidingWindowFeature**           | A data structure holding frame-level scores, plus timing info for each window                                                                   |
| **`np.max` over speaker axis**     | The segmentation model produces one score per speaker — VAD only needs the loudest one                                                          |
| **Binarize**                       | Converts continuous probabilities (0.0–1.0) into discrete on/off decisions                                                                      |
| **onset**                          | The probability threshold you must _cross going up_ to trigger "speech started"                                                                 |
| **offset**                         | The probability threshold you must _drop below_ to trigger "speech ended" — usually lower than onset to avoid rapid toggling                    |
| **`min_duration_on`**              | Delete any speech region shorter than this many seconds (removes blips)                                                                         |
| **`min_duration_off`**             | Fill any silence gap shorter than this many seconds (removes stutters)                                                                          |
| **Powerset mode**                  | A special segmentation model variant whose output is already binarized — onset/offset are fixed at 0.5                                          |
| **Uniform**                        | A hyper-parameter type that tells the optimizer: "search this parameter between 0.0 and 1.0"                                                    |
| **hook**                           | A progress-callback function you can pass to `apply()` — it fires after each major step                                                         |
| **Training cache**                 | During training, `apply()` saves segmentation output to `file["cache/segmentation/inference"]` to avoid recomputing it on every optimizer trial |
| **DER**                            | Detection Error Rate — penalises missed speech and false alarms. Lower is better.                                                               |
| **F-measure**                      | Precision × Recall metric. Higher is better. Used when `fscore=True`.                                                                           |

---

### Files to Create

1. `example_vad_basic.py` — Simplest end-to-end VAD on one file
2. `example_vad_advanced.py` — Hooks, parameter tuning, fscore mode
3. `example_vad_oracle.py` — Oracle pipeline and ground-truth comparison
4. `example_vad_evaluation.py` — Benchmarking with DER and F-score

---

```python
# example_vad_basic.py
"""
Basic Voice Activity Detection — Is Anyone Speaking?
=====================================================
Goal: Take an audio file → get back a timeline of when speech occurs.

This is the simplest possible usage. The pipeline handles everything:
  - loading audio
  - running the segmentation model
  - binarizing scores into SPEECH / SILENCE regions

Install requirements:
  pip install pyannote.audio

You may need a HuggingFace token for gated models:
  https://huggingface.co/settings/tokens
"""

import torch
from pyannote.audio.pipelines.voice_activity_detection import VoiceActivityDetection

# ------------------------------------------------------------------
# 1. Build the pipeline
#    The default model ("pyannote/segmentation") is downloaded
#    automatically from HuggingFace on first use.
# ------------------------------------------------------------------
pipeline = VoiceActivityDetection(
    segmentation="pyannote/segmentation",
    # token="hf_your_token_here",   # needed for gated HuggingFace models
    # cache_dir="/tmp/pyannote",     # where to store downloaded model files
)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)
print("Pipeline ready.")

# ------------------------------------------------------------------
# 2. Set parameters — onset, offset, and duration filters
#
#    default_parameters() returns values tuned on a standard benchmark.
#    These are a good starting point.
# ------------------------------------------------------------------
pipeline.instantiate(pipeline.default_parameters())
print(f"Parameters in use: {pipeline.default_parameters()}\n")

# ------------------------------------------------------------------
# 3. Process an audio file
#    AudioFile formats accepted:
#      - plain path string:          "speech.wav"
#      - Path object:                Path("speech.wav")
#      - dict with waveform tensor:  {"waveform": tensor, "sample_rate": 16000}
#      - dict with audio path:       {"uri": "my_file", "audio": "speech.wav"}
# ------------------------------------------------------------------
audio_file = "speech.wav"   # ← replace with your file

speech = pipeline(audio_file)

# ------------------------------------------------------------------
# 4. Read the results
#    speech is a pyannote.core.Annotation with label "SPEECH".
#    Each track represents one detected speech region.
# ------------------------------------------------------------------
print("=== Detected Speech Regions ===")
for turn, _, label in speech.itertracks(yield_label=True):
    print(f"  [{turn.start:7.3f}s → {turn.end:7.3f}s]  {label}  "
          f"(duration: {turn.duration:.3f}s)")

total_speech = sum(seg.duration for seg, *_ in speech.itertracks())
print(f"\nTotal speech duration : {total_speech:.3f}s")
print(f"Number of regions     : {len(list(speech.itertracks()))}")
print(f"Labels in output      : {speech.labels()}")   # always ["SPEECH"]
```

---

```python
# example_vad_advanced.py
"""
Advanced VAD — Hooks, Parameter Tuning, and F-score Mode
=========================================================
Goal: Show the less-obvious controls available in VoiceActivityDetection.

Covers:
  1. Progress hook — watch segmentation unfold in real time
  2. Manual parameter tuning — onset, offset, duration filters
  3. fscore=True — switch the optimisation target from DER to F-measure
  4. Using a newer model (pyannote/segmentation-3.0.0)
  5. Batch size control for faster GPU processing
  6. Passing a raw waveform tensor instead of a file path
"""

import numpy as np
import torch
from pyannote.audio.pipelines.voice_activity_detection import VoiceActivityDetection

audio_file = "speech.wav"   # ← replace with your file

# ==================================================================
# FEATURE 1: Progress hook
# ==================================================================
print("=== Feature 1: Progress Hook ===\n")

# The hook is called after each major pipeline step with:
#   step_name     → name of the stage (str)
#   step_artifact → what was produced (SlidingWindowFeature, Annotation, ...)
#   completed     → batches done so far (int, only during long steps)
#   total         → total batches in this step (int, only during long steps)

def my_hook(step_name, step_artifact, file=None, completed=None, total=None):
    if completed is not None and total is not None:
        pct = 100 * completed / total
        print(f"  [{step_name}] {completed}/{total} ({pct:.0f}%)")
    else:
        info = ""
        if hasattr(step_artifact, "data"):
            info = f"shape={step_artifact.data.shape}"
        elif hasattr(step_artifact, "shape"):
            info = f"shape={step_artifact.shape}"
        print(f"  ✓ {step_name}  {info}")

pipeline = VoiceActivityDetection(segmentation="pyannote/segmentation")
pipeline.instantiate(pipeline.default_parameters())

speech = pipeline(audio_file, hook=my_hook)
print(f"\nFound {len(list(speech.itertracks()))} speech regions with hook.\n")


# ==================================================================
# FEATURE 2: Manual parameter tuning
# ==================================================================
print("=== Feature 2: Parameter Tuning ===\n")

# Parameters and what they control:
#
#   onset  (float 0–1):
#     The model's speech probability must RISE ABOVE this to start a region.
#     Higher onset → stricter, fewer false alarms, may miss soft speech.
#
#   offset (float 0–1):
#     The model's speech probability must FALL BELOW this to end a region.
#     Lower offset → speech regions extend further into quiet sections.
#     offset < onset is normal — creates hysteresis (avoids rapid switching).
#
#   min_duration_on (float, seconds):
#     Delete any detected speech region shorter than this.
#     Useful for removing noise spikes labelled as speech.
#
#   min_duration_off (float, seconds):
#     Fill any silence gap shorter than this.
#     Useful for merging closely-spaced words into one region.

custom_params = {
    "onset": 0.8,            # stricter — only high-confidence speech
    "offset": 0.4,           # generous end — don't cut off word endings
    "min_duration_on": 0.2,  # ignore blips shorter than 0.2s
    "min_duration_off": 0.1, # fill pauses shorter than 0.1s
}

pipeline.instantiate(custom_params)
speech_custom = pipeline(audio_file)

print("Default params speech regions :", len(list(
    VoiceActivityDetection(segmentation="pyannote/segmentation")
    .apply.__func__  # just counting — pipeline already set above
    if False else speech.itertracks()  # use earlier result
)))
print("Custom params  speech regions :", len(list(speech_custom.itertracks())))
print()


# ==================================================================
# FEATURE 3: fscore=True — optimise for F-measure instead of DER
# ==================================================================
print("=== Feature 3: F-score Mode ===\n")

# By default the pipeline minimises DER (Detection Error Rate).
# Setting fscore=True switches the optimisation target to F-measure
# (precision × recall harmonic mean), which is better when you care
# equally about not missing speech AND not adding false alarms.

pipeline_fscore = VoiceActivityDetection(
    segmentation="pyannote/segmentation",
    fscore=True,
    # token="hf_your_token_here",
)
pipeline_fscore.instantiate(pipeline_fscore.default_parameters())

speech_fscore = pipeline_fscore(audio_file)

print(f"Optimisation target : {'F-measure (maximize)' if pipeline_fscore.fscore else 'DER (minimize)'}")
print(f"get_direction()     : {pipeline_fscore.get_direction()}")
print(f"get_metric() type   : {type(pipeline_fscore.get_metric()).__name__}")
print(f"Regions found       : {len(list(speech_fscore.itertracks()))}\n")


# ==================================================================
# FEATURE 4: Newer model — pyannote/segmentation-3.0.0
# ==================================================================
print("=== Feature 4: Newer Segmentation Model ===\n")

# The 3.0.0 model uses powerset mode — its output is already binarized,
# so onset and offset are fixed at 0.5 and cannot be tuned.
# Only min_duration_on and min_duration_off remain as free parameters.

pipeline_v3 = VoiceActivityDetection(
    segmentation="pyannote/segmentation-3.0.0",
    # token="hf_your_token_here",
)

defaults_v3 = pipeline_v3.default_parameters()
print(f"v3.0 default params: {defaults_v3}")
# → {'min_duration_on': 0.0, 'min_duration_off': 0.0}
# onset and offset are NOT tunable in powerset mode

pipeline_v3.instantiate(defaults_v3)
speech_v3 = pipeline_v3(audio_file)
print(f"v3.0 regions found : {len(list(speech_v3.itertracks()))}\n")


# ==================================================================
# FEATURE 5: Batch size — faster on GPU
# ==================================================================
print("=== Feature 5: Batch Size Control ===\n")

# inference_kwargs are forwarded straight to pyannote.audio.Inference.
# The most useful one is batch_size — how many audio windows to process
# in parallel. Larger = faster on GPU, but uses more VRAM.

pipeline_batched = VoiceActivityDetection(
    segmentation="pyannote/segmentation",
    batch_size=32,   # ← passed to Inference via **inference_kwargs
    # token="hf_your_token_here",
)
pipeline_batched.instantiate(pipeline_batched.default_parameters())
speech_batched = pipeline_batched(audio_file)
print(f"Batched (bs=32) regions: {len(list(speech_batched.itertracks()))}\n")


# ==================================================================
# FEATURE 6: Raw waveform tensor as input
# ==================================================================
print("=== Feature 6: Waveform Tensor Input ===\n")

# Instead of a file path you can pass a dict with:
#   "waveform"    → torch.Tensor of shape (channels, samples)
#   "sample_rate" → int, e.g. 16000

sample_rate = 16000
duration_sec = 5
fake_waveform = torch.randn(1, sample_rate * duration_sec)   # 5 s of noise

audio_dict = {
    "uri": "synthetic_audio",       # optional but useful for logging
    "waveform": fake_waveform,
    "sample_rate": sample_rate,
}

pipeline_w = VoiceActivityDetection(segmentation="pyannote/segmentation")
pipeline_w.instantiate(pipeline_w.default_parameters())
speech_waveform = pipeline_w(audio_dict)

print(f"Input  : waveform tensor {fake_waveform.shape}, sr={sample_rate}")
print(f"Output : {len(list(speech_waveform.itertracks()))} speech regions")
for turn, _, label in speech_waveform.itertracks(yield_label=True):
    print(f"  [{turn.start:.3f}s → {turn.end:.3f}s]  {label}")
```

---

```python
# example_vad_oracle.py
"""
Oracle VAD — Ground-Truth Speech Regions
=========================================
Goal: Understand OracleVoiceActivityDetection and use it to build
      a reference baseline for evaluation.

The Oracle pipeline does NOT run any model. It reads the "annotation"
key directly from the AudioFile dict — the human-labelled ground truth.

This is useful for:
  1. Checking your annotation format is correct
  2. Getting an upper-bound DER of 0% for sanity checking your eval loop
  3. Comparing oracle VAD output to a real model side by side

Covers:
  1. Building and running the oracle pipeline
  2. Constructing a proper AudioFile dict with an annotation
  3. Comparing oracle output to model output
  4. Understanding the .support() + .to_annotation() chain
"""

import torch
from pyannote.core import Annotation, Segment, Timeline

from pyannote.audio.pipelines.voice_activity_detection import (
    OracleVoiceActivityDetection,
    VoiceActivityDetection,
)

# ------------------------------------------------------------------
# 1. Build a fake AudioFile with a ground-truth annotation
#
#    An AudioFile is just a plain Python dict.
#    For OracleVoiceActivityDetection it MUST contain "annotation".
#    For VoiceActivityDetection it MUST contain "audio" (or waveform).
# ------------------------------------------------------------------

# Reference: who spoke and when (this is our "ground truth")
reference = Annotation(uri="example_file")
reference[Segment(0.5, 3.2)]  = "SPEAKER_A"
reference[Segment(3.0, 6.5)]  = "SPEAKER_B"   # overlaps slightly with A
reference[Segment(7.0, 10.0)] = "SPEAKER_A"
reference[Segment(10.5, 13.0)] = "SPEAKER_B"

audio_file = {
    "uri": "example_file",
    "audio": "example_file.wav",    # ← path to audio (used by VoiceActivityDetection)
    "annotation": reference,        # ← ground truth (used by OracleVoiceActivityDetection)
}

# ------------------------------------------------------------------
# 2. Run the Oracle pipeline
#    It calls:  file["annotation"].get_timeline().support()
#    Which means: take all labelled segments → merge overlapping ones
#    → return a Timeline of contiguous speech blocks.
#    Then wraps that as an Annotation with label "speech".
# ------------------------------------------------------------------
oracle = OracleVoiceActivityDetection()
oracle_speech = oracle.apply(audio_file)

print("=== Oracle Speech Regions ===")
print("(These are the ground-truth speech regions, speaker identity removed)")
for turn, _, label in oracle_speech.itertracks(yield_label=True):
    print(f"  [{turn.start:.3f}s → {turn.end:.3f}s]  {label}")

# ------------------------------------------------------------------
# 3. What .support() does — merging overlapping segments
# ------------------------------------------------------------------
print("\n=== Original Reference Segments (with speaker labels) ===")
for turn, _, speaker in reference.itertracks(yield_label=True):
    print(f"  [{turn.start:.3f}s → {turn.end:.3f}s]  {speaker}")

raw_timeline: Timeline = reference.get_timeline()
supported_timeline: Timeline = raw_timeline.support()

print("\n=== After .support() — overlapping segments merged ===")
for seg in supported_timeline:
    print(f"  [{seg.start:.3f}s → {seg.end:.3f}s]")

# ------------------------------------------------------------------
# 4. Run the real VAD model on the same file
# ------------------------------------------------------------------
print("\n=== Real Model Speech Regions ===")
pipeline = VoiceActivityDetection(
    segmentation="pyannote/segmentation",
    # token="hf_your_token_here",
)
pipeline.instantiate(pipeline.default_parameters())
model_speech = pipeline(audio_file)

for turn, _, label in model_speech.itertracks(yield_label=True):
    print(f"  [{turn.start:.3f}s → {turn.end:.3f}s]  {label}")

# ------------------------------------------------------------------
# 5. Side-by-side summary
# ------------------------------------------------------------------
oracle_dur = sum(s.duration for s, *_ in oracle_speech.itertracks())
model_dur  = sum(s.duration for s, *_ in model_speech.itertracks())

print(f"\n{'':30} Oracle     Model")
print(f"{'Total speech duration':30} {oracle_dur:6.3f}s   {model_dur:6.3f}s")
print(f"{'Number of regions':30} "
      f"{len(list(oracle_speech.itertracks())):6}     "
      f"{len(list(model_speech.itertracks())):6}")
```

---

```python
# example_vad_evaluation.py
"""
Evaluating VAD Quality — DER and F-score
==========================================
Goal: Measure how well VoiceActivityDetection performs against
      a human-labelled reference.

Two metrics are supported:

  DetectionErrorRate (DER) — lower is better (0 = perfect)
    = (missed speech + false alarm speech) / total reference speech

  DetectionPrecisionRecallFMeasure — higher is better (1.0 = perfect)
    = 2 × precision × recall / (precision + recall)

Covers:
  1. DER on a single file
  2. F-score on a single file
  3. Accumulating metrics over a dataset
  4. Detailed metric breakdown (miss / false alarm)
  5. Sweeping parameters to find the best config
  6. Printing a confusion matrix of speech vs. silence
"""

from pyannote.core import Annotation, Segment
from pyannote.metrics.detection import (
    DetectionErrorRate,
    DetectionPrecisionRecallFMeasure,
)

from pyannote.audio.pipelines.voice_activity_detection import VoiceActivityDetection

# ------------------------------------------------------------------
# Helper: build a simple reference annotation
# ------------------------------------------------------------------
def make_reference(uri: str) -> Annotation:
    ref = Annotation(uri=uri)
    ref[Segment(0.5,  3.2)]  = "speech"
    ref[Segment(4.0,  6.5)]  = "speech"
    ref[Segment(7.0, 10.0)]  = "speech"
    ref[Segment(11.0, 13.5)] = "speech"
    return ref

# ------------------------------------------------------------------
# Build pipelines (DER-mode and F-score-mode)
# ------------------------------------------------------------------
pipeline_der = VoiceActivityDetection(
    segmentation="pyannote/segmentation",
    fscore=False,   # optimise DER (default)
    # token="hf_your_token_here",
)
pipeline_der.instantiate(pipeline_der.default_parameters())

pipeline_f = VoiceActivityDetection(
    segmentation="pyannote/segmentation",
    fscore=True,    # optimise F-measure
    # token="hf_your_token_here",
)
pipeline_f.instantiate(pipeline_f.default_parameters())

audio_file = {"uri": "test_file", "audio": "test.wav"}   # ← replace
reference  = make_reference("test_file")

# ==================================================================
# FEATURE 1: DER on a single file
# ==================================================================
print("=== Feature 1: Detection Error Rate ===\n")

hypothesis = pipeline_der(audio_file)

# get_metric() returns a new DetectionErrorRate instance
metric_der = pipeline_der.get_metric()

# Calling the metric computes the score (finds the best label mapping)
der = metric_der(reference, hypothesis)
print(f"Detection Error Rate: {der * 100:.2f}%  (lower is better)\n")

# ==================================================================
# FEATURE 2: F-score on a single file
# ==================================================================
print("=== Feature 2: Detection F-score ===\n")

hypothesis_f = pipeline_f(audio_file)
metric_f     = pipeline_f.get_metric()   # DetectionPrecisionRecallFMeasure

fscore = metric_f(reference, hypothesis_f)
print(f"Detection F-score: {fscore:.4f}  (higher is better, max = 1.0)\n")

# ==================================================================
# FEATURE 3: Detailed breakdown — miss / false alarm
# ==================================================================
print("=== Feature 3: Detailed DER Breakdown ===\n")

# detailed=True returns a dict with per-component scores
detail = metric_der(reference, hypothesis, detailed=True)

# The keys depend on the metric class.
# DetectionErrorRate provides these components:
print("Metric components:")
for key, value in detail.items():
    if isinstance(value, float):
        print(f"  {key:30s}: {value * 100:.2f}%")
    else:
        print(f"  {key:30s}: {value}")

# Compute precision and recall manually from the detail dict
# (miss rate = missed speech / reference speech)
# (false alarm rate = false alarm / reference speech)
ref_duration = detail.get("total", None)
if ref_duration:
    miss_pct = 100 * detail.get("miss", 0) / ref_duration
    fa_pct   = 100 * detail.get("false alarm", 0) / ref_duration
    print(f"\nMiss rate       : {miss_pct:.2f}%  (speech that was not detected)")
    print(f"False alarm rate: {fa_pct:.2f}%  (silence that was labelled as speech)")
print()

# ==================================================================
# FEATURE 4: Accumulating over a dataset
# ==================================================================
print("=== Feature 4: Dataset-Level Evaluation ===\n")

# Build a small test set (replace with a real pyannote.database protocol)
test_set = [
    {"uri": "file1", "audio": "file1.wav"},
    {"uri": "file2", "audio": "file2.wav"},
    {"uri": "file3", "audio": "file3.wav"},
]
references = {f["uri"]: make_reference(f["uri"]) for f in test_set}

# Accumulate DER across all files
accumulated_metric = pipeline_der.get_metric()   # fresh instance

for file in test_set:
    hyp = pipeline_der(file)
    ref = references[file["uri"]]
    accumulated_metric(ref, hyp)   # adds to internal running total

# abs() on the metric returns the final accumulated score
overall_der = abs(accumulated_metric)
print(f"Overall DER across {len(test_set)} files: {overall_der * 100:.2f}%\n")

# ==================================================================
# FEATURE 5: Parameter sweep to find best config
# ==================================================================
print("=== Feature 5: Parameter Sweep ===\n")

# Sweep over onset/offset combinations on a dev file.
# In practice: run this on a held-out development set, NOT the test set.

dev_file      = {"uri": "dev_file", "audio": "dev.wav"}   # ← replace
dev_reference = make_reference("dev_file")

candidates = [
    {"onset": 0.5, "offset": 0.3, "min_duration_on": 0.0, "min_duration_off": 0.0},
    {"onset": 0.7, "offset": 0.4, "min_duration_on": 0.1, "min_duration_off": 0.05},
    {"onset": 0.8, "offset": 0.5, "min_duration_on": 0.2, "min_duration_off": 0.1},
]

best_der    = float("inf")
best_config = None

print(f"{'Config':55s}  DER")
print("-" * 65)

for cfg in candidates:
    pipeline_der.instantiate(cfg)
    hyp = pipeline_der(dev_file)
    m   = DetectionErrorRate(collar=0.0, skip_overlap=False)
    der = m(dev_reference, hyp)

    label = (f"onset={cfg['onset']:.1f}, offset={cfg['offset']:.1f}, "
             f"on={cfg['min_duration_on']:.2f}, off={cfg['min_duration_off']:.2f}")
    print(f"  {label:53s}  {der * 100:.2f}%")

    if der < best_der:
        best_der    = der
        best_config = cfg

print(f"\nBest config : {best_config}")
print(f"Best DER    : {best_der * 100:.2f}%\n")

# Apply best config for final inference
pipeline_der.instantiate(best_config)

# ==================================================================
# FEATURE 6: get_direction() — tells the optimizer which way is better
# ==================================================================
print("=== Feature 6: Optimization Direction ===\n")

# The pipeline exposes get_direction() so that external hyperparameter
# optimizers (like Optuna or pyannote's built-in optimizer) know whether
# to minimize or maximize the metric.

pipeline_min = VoiceActivityDetection(fscore=False)
pipeline_max = VoiceActivityDetection(fscore=True)

print(f"fscore=False  → get_direction() = '{pipeline_min.get_direction()}'  "
      f"(minimize DER)")
print(f"fscore=True   → get_direction() = '{pipeline_max.get_direction()}'  "
      f"(maximize F-score)")
```

---

### How All the Pieces Fit Together

```text
┌──────────────────────────────────────────────┐
│ VoiceActivityDetection.__init__()            │
│                                              │
│   segmentation="pyannote/segmentation"       │
│   ↓                                          │
│   get_model() → Inference object             │
│   pre_aggregation_hook: np.max()             │
│   (collapse speaker scores → 1 curve)        │
│                                              │
│   If powerset model:                         │
│     onset = offset = 0.5 (fixed)             │
│   Else:                                      │
│     onset ~ Uniform(0, 1)                    │
│     offset ~ Uniform(0, 1)                   │
│     min_duration_on ~ Uniform(0, 1)          │
│     min_duration_off ~ Uniform(0, 1)         │
└───────────────────────┬──────────────────────┘
                        │ pipeline.instantiate(params)
                        ▼
┌──────────────────────────────────────────────┐
│ initialize()                                 │
│   Builds Binarize(                           │
│     onset, offset,                           │
│     min_duration_on,                         │
│     min_duration_off                         │
│   )                                          │
└───────────────────────┬──────────────────────┘
                        │ pipeline(audio_file)
                        ▼
┌──────────────────────────────────────────────┐
│ apply()                                      │
│                                              │
│  1. setup_hook()                             │
│  2. If training → check cache                │
│     Else → run _segmentation()               │
│            → SlidingWindowFeature            │
│  3. hook("segmentation", ...)                │
│  4. _binarize(segmentations)                 │
│            → Annotation                      │
│  5. rename all labels → "SPEECH"             │
│  6. return Annotation                        │
└──────────────────────────────────────────────┘
```

---

### Model Versions at a Glance

| Model string                  | Powerset | Tunable onset/offset | Default onset | Default offset |
| ----------------------------- | -------- | -------------------- | ------------- | -------------- |
| `pyannote/segmentation`       | No       | Yes                  | 0.767         | 0.377          |
| `pyannote/segmentation-3.0.0` | Yes      | No (fixed 0.5)       | 0.5           | 0.5            |

---

### Common Mistakes

**1. Forgetting `pipeline.instantiate()`**

```python
# ❌ Wrong — pipeline has no parameters set, will raise AttributeError
speech = pipeline(audio_file)

# ✅ Correct
pipeline.instantiate(pipeline.default_parameters())
speech = pipeline(audio_file)
```

**2. Setting onset < offset**

```python
# ❌ Unusual — means speech starts at a LOWER threshold than it ends
#    This removes the hysteresis that prevents rapid on/off switching
pipeline.instantiate({"onset": 0.3, "offset": 0.6, ...})

# ✅ Typical — onset > offset creates stable, hysteretic detection
pipeline.instantiate({"onset": 0.7, "offset": 0.4, ...})
```

**3. Calling `default_parameters()` on a model it doesn't know**

```python
# ❌ Will raise NotImplementedError for custom/fine-tuned model paths
pipeline = VoiceActivityDetection(segmentation="my_org/my_model")
params = pipeline.default_parameters()   # raises NotImplementedError

# ✅ Provide your own parameters
pipeline.instantiate({
    "onset": 0.5, "offset": 0.5,
    "min_duration_on": 0.1, "min_duration_off": 0.05
})
```

**4. Expecting speaker labels in the output**

```python
# ❌ VAD does not identify speakers — there is only one label
for turn, _, label in speech.itertracks(yield_label=True):
    print(label)   # always "SPEECH" — never "SPEAKER_00"

# ✅ For speaker identity, use SpeakerDiarization instead
```
