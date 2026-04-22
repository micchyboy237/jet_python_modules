# pyannote/audio/pipelines/speaker_diarization.py

## Step-by-Step Analysis & Blueprint

### What This File Does (Plain English)

This file answers: _"Who spoke when?"_ — the classic **speaker diarization** problem. Given an audio file with multiple people talking, it produces a timeline like: "Speaker 0 spoke from 0–5s, Speaker 1 from 4–9s, Speaker 0 again from 10–15s…"

---

### The 3-Stage Pipeline (How It Works)

```
Audio File
    │
    ▼
┌─────────────────────────────────────┐
│  Stage 1: SEGMENTATION              │
│  Sliding window over audio          │
│  → "Who is speaking in each chunk?" │
│  Output: (chunks, frames, speakers) │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Stage 2: EMBEDDING                 │
│  Extract voice fingerprint per      │
│  (chunk, speaker) pair              │
│  Output: (chunks, speakers, dim)    │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Stage 3: CLUSTERING                │
│  Group fingerprints by identity     │
│  "These 5 chunks are all Speaker 0" │
│  Output: hard_clusters array        │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  RECONSTRUCT + LABEL                │
│  Build final Annotation timeline    │
│  → DiarizeOutput                    │
└─────────────────────────────────────┘
```

---

### Feature Map / Blueprint

```
speaker_diarization.py
│
├── batchify()                    → Utility: split any iterable into fixed-size batches
│
├── DiarizeOutput (dataclass)     → Result container
│   ├── speaker_diarization       → Full timeline (includes overlapping speech)
│   ├── exclusive_speaker_diarization → Timeline with NO overlapping speech
│   ├── speaker_embeddings        → One voice fingerprint per speaker (centroid)
│   └── serialize()               → Convert to plain JSON-friendly dict
│
└── SpeakerDiarization (Pipeline)
    ├── __init__()                → Load segmentation + embedding + clustering models
    ├── segmentation_batch_size   → Property with getter/setter
    ├── default_parameters()      → Sensible defaults for tunable params
    ├── classes()                 → Infinite generator: SPEAKER_00, SPEAKER_01, ...
    ├── get_segmentations()       → Stage 1: run segmentation model (with caching)
    ├── get_embeddings()          → Stage 2: extract per-(chunk,speaker) embeddings
    ├── reconstruct()             → Merge clustered chunks into one diarization
    ├── apply()                   → Master orchestrator — runs all 3 stages
    └── get_metric()              → Returns DER metric for evaluation
```

---

### Key Concepts Explained

| Concept                   | Plain English                                                               |
| ------------------------- | --------------------------------------------------------------------------- |
| **Segmentation**          | Chops audio into overlapping windows, asks "who's speaking here?"           |
| **Powerset mode**         | Special segmentation where output is already binarized (on/off)             |
| **Binarization**          | Converts soft probabilities (0.7) to hard on/off (1/0) using a threshold    |
| **Embedding**             | A voice fingerprint — a list of numbers unique to each speaker              |
| **Clustering**            | Groups fingerprints by identity — "these all sound like the same person"    |
| **VBxClustering**         | The default: uses PLDA (a probabilistic model) for smarter grouping         |
| **Hard clusters**         | An array saying which global speaker ID each local chunk-speaker belongs to |
| **Centroids**             | Average embedding per speaker — one fingerprint to represent each person    |
| **DER**                   | Diarization Error Rate — the standard benchmark (lower = better)            |
| **Exclusive diarization** | Same as diarization but overlapping regions are removed                     |
| **`exclude_overlap`**     | Only embed non-overlapping speech (cleaner signal, may be too short)        |
| **`legacy` mode**         | Returns bare `Annotation` instead of full `DiarizeOutput`                   |

---

### Files to Create

1. `example_diarization_basic.py` — Simple end-to-end diarization on a file
2. `example_diarization_advanced.py` — Speaker constraints, hooks, output inspection
3. `example_diarization_output.py` — Working with `DiarizeOutput` and serialization
4. `example_diarization_evaluation.py` — Benchmarking with DER metric

---

```python
# example_diarization_basic.py
"""
Basic Speaker Diarization — Who Spoke When?
============================================
Goal: Take an audio file → get back a labelled timeline of speakers.

This is the simplest possible usage. The pipeline handles everything:
  - loading audio
  - detecting speech segments
  - clustering speakers
  - labelling them SPEAKER_00, SPEAKER_01, etc.

Install requirements:
  pip install pyannote.audio

You'll need a HuggingFace token for gated models:
  https://huggingface.co/settings/tokens
"""

import torch
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization

# ------------------------------------------------------------------
# 1. Build the pipeline
#    The defaults pull community models from HuggingFace automatically.
# ------------------------------------------------------------------
pipeline = SpeakerDiarization(
    # token="hf_your_token_here",   # needed for gated HuggingFace models
    # cache_dir="/tmp/pyannote",     # where to store downloaded models
)

# Move everything to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)

print("Pipeline ready.")
print(f"Clustering algorithm : {pipeline.klustering}")
print(f"Embedding batch size : {pipeline.embedding_batch_size}")
print(f"Segmentation step    : {pipeline.segmentation_step} "
      f"(= {pipeline.segmentation_step * 100:.0f}% overlap between windows)\n")

# ------------------------------------------------------------------
# 2. Process an audio file
#    AudioFile can be:
#      - a plain path string:         "meeting.wav"
#      - a Path object:               Path("meeting.wav")
#      - a dict with waveform tensor: {"waveform": tensor, "sample_rate": 16000}
#      - a dict with audio path:      {"uri": "my_file", "audio": "meeting.wav"}
# ------------------------------------------------------------------
audio_file = "meeting.wav"   # ← replace with your file

output = pipeline(audio_file)

# ------------------------------------------------------------------
# 3. Read the results
#    output is a DiarizeOutput dataclass with three fields:
#      .speaker_diarization          → full timeline
#      .exclusive_speaker_diarization → overlap-free timeline
#      .speaker_embeddings           → one embedding per speaker
# ------------------------------------------------------------------
print("=== Speaker Timeline ===")
for turn, _, speaker in output.speaker_diarization.itertracks(yield_label=True):
    print(f"  [{turn.start:6.1f}s → {turn.end:6.1f}s]  {speaker}")

print(f"\nTotal speakers found: {len(output.speaker_diarization.labels())}")
print(f"Speaker labels      : {output.speaker_diarization.labels()}")

# ------------------------------------------------------------------
# 4. Exclusive diarization — no overlapping speech regions
#    Useful when you need clean, non-ambiguous speaker turns
# ------------------------------------------------------------------
print("\n=== Exclusive (no-overlap) Timeline ===")
for turn, _, speaker in output.exclusive_speaker_diarization.itertracks(yield_label=True):
    print(f"  [{turn.start:6.1f}s → {turn.end:6.1f}s]  {speaker}")

# ------------------------------------------------------------------
# 5. Speaker embeddings — one voice fingerprint per speaker
#    Shape: (num_speakers, embedding_dimension)
#    Useful for: identifying speakers across different meetings
# ------------------------------------------------------------------
if output.speaker_embeddings is not None:
    import numpy as np
    print(f"\n=== Speaker Embeddings ===")
    print(f"Shape: {output.speaker_embeddings.shape}")
    for i, label in enumerate(output.speaker_diarization.labels()):
        emb = output.speaker_embeddings[i]
        print(f"  {label}: embedding norm = {np.linalg.norm(emb):.4f}")
```

---

```python
# example_diarization_advanced.py
"""
Advanced Diarization — Constraints, Hooks & Tuning
====================================================
Goal: Show how to control the pipeline more precisely:
  1. Speaker count constraints (exact / min / max)
  2. Progress hooks — monitor each pipeline stage in real time
  3. Tuning segmentation & clustering parameters
  4. Batch sizes for GPU throughput
  5. Overlap exclusion for cleaner embeddings
  6. Legacy mode for backward-compatible output

Most real-world usage needs at least one of these techniques.
"""

import numpy as np
import torch
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
pipeline = SpeakerDiarization(
    # token="hf_your_token_here",
)

audio_file = "meeting.wav"   # ← replace with your file


# ==================================================================
# FEATURE 1: Speaker count constraints
# ==================================================================
print("=== Feature 1: Speaker Constraints ===\n")

# When you KNOW exactly how many people are in the recording
output_exact = pipeline(audio_file, num_speakers=3)
print(f"Exact 3 speakers → found: {len(output_exact.speaker_diarization.labels())}")

# When you only know the range
output_range = pipeline(audio_file, min_speakers=2, max_speakers=5)
n = len(output_range.speaker_diarization.labels())
print(f"Range 2–5 speakers → found: {n} (within bounds: {2 <= n <= 5})")

# Unconstrained — pipeline decides on its own
output_auto = pipeline(audio_file)
print(f"Auto speakers → found: {len(output_auto.speaker_diarization.labels())}\n")


# ==================================================================
# FEATURE 2: Progress hook — watch each stage as it runs
# ==================================================================
print("=== Feature 2: Progress Hook ===\n")

# The hook is called after each major step with:
#   step_name     → which stage just finished (str)
#   step_artifact → what was produced (tensor / array / etc.)
#   completed     → how many batches done so far (int, optional)
#   total         → total batches in this step (int, optional)

def my_progress_hook(step_name, step_artifact, file=None, completed=None, total=None):
    if completed is not None and total is not None:
        pct = 100 * completed / total
        print(f"  [{step_name}] {completed}/{total} batches ({pct:.0f}%)")
    else:
        artifact_info = ""
        if hasattr(step_artifact, "shape"):
            artifact_info = f"shape={step_artifact.shape}"
        elif hasattr(step_artifact, "data"):
            artifact_info = f"data.shape={step_artifact.data.shape}"
        print(f"  ✓ Stage done: '{step_name}'  {artifact_info}")

output = pipeline(audio_file, hook=my_progress_hook)
print()


# ==================================================================
# FEATURE 3: Tuning pipeline parameters
# ==================================================================
print("=== Feature 3: Parameter Tuning ===\n")

# You can set internal parameters directly on the pipeline object.
# These control the segmentation binarization and clustering sensitivity.
#
# default_parameters() shows the recommended starting values:
defaults = pipeline.default_parameters()
print("Default parameters:")
for section, params in defaults.items():
    for key, val in params.items():
        print(f"  {section}.{key} = {val}")

# Apply defaults (or your custom values after tuning)
pipeline.instantiate(defaults)

# Alternatively, tune a specific parameter:
# pipeline.segmentation.min_duration_off = 0.2  # seconds of silence before splitting turn
# pipeline.clustering.threshold = 0.5           # lower = more aggressive splitting into speakers

output_tuned = pipeline(audio_file)
print(f"\nTuned pipeline → {len(output_tuned.speaker_diarization.labels())} speakers\n")


# ==================================================================
# FEATURE 4: Batch size — faster processing on GPU
# ==================================================================
print("=== Feature 4: Batch Sizes ===\n")

# Segmentation batch size: how many audio windows to process at once
# Embedding batch size: how many (chunk, speaker) pairs to embed at once
# Larger = faster on GPU, but uses more memory

pipeline.segmentation_batch_size = 32   # uses the setter → updates internal inference
pipeline.embedding_batch_size    = 16   # set directly

print(f"Segmentation batch size: {pipeline.segmentation_batch_size}")
print(f"Embedding batch size   : {pipeline.embedding_batch_size}\n")


# ==================================================================
# FEATURE 5: Overlap exclusion — cleaner embeddings
# ==================================================================
print("=== Feature 5: Overlap-Excluded Embeddings ===\n")

# When multiple speakers talk at the same time, the mixed audio
# makes a noisy embedding. This option tells the pipeline to
# ONLY use the parts where exactly one person is speaking.
# Trade-off: may fall back to overlapping regions if there's not enough clean speech.

pipeline_clean = SpeakerDiarization(
    embedding_exclude_overlap=True,
    # token="hf_your_token_here",
)
pipeline_clean.instantiate(pipeline_clean.default_parameters())

output_clean = pipeline_clean(audio_file)
print(f"With overlap exclusion → {len(output_clean.speaker_diarization.labels())} speakers\n")


# ==================================================================
# FEATURE 6: Legacy mode — returns plain Annotation (old API)
# ==================================================================
print("=== Feature 6: Legacy Mode ===\n")

# Before DiarizeOutput was introduced, pipeline returned a bare Annotation.
# legacy=True restores that behavior for code written against the old API.

pipeline_legacy = SpeakerDiarization(legacy=True)
pipeline_legacy.instantiate(pipeline_legacy.default_parameters())

annotation = pipeline_legacy(audio_file)   # returns Annotation, not DiarizeOutput

from pyannote.core import Annotation
assert isinstance(annotation, Annotation), "Legacy mode should return an Annotation"

print("Legacy output type:", type(annotation).__name__)
for turn, _, speaker in annotation.itertracks(yield_label=True):
    print(f"  [{turn.start:.1f}s → {turn.end:.1f}s]  {speaker}")
```

---

```python
# example_diarization_output.py
"""
Working with DiarizeOutput
============================
Goal: Deep-dive into what DiarizeOutput contains and how to use each field.

Covers:
  1. Reading the full diarization timeline
  2. Reading the exclusive (no-overlap) timeline
  3. Speaker embeddings — cross-meeting identification
  4. JSON serialization — save results to disk
  5. Comparing speakers across two different recordings
  6. The infinite speaker label generator (classes())
"""

import json
import numpy as np
import torch
from pathlib import Path
from scipy.spatial.distance import cdist

from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization, DiarizeOutput

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
pipeline = SpeakerDiarization(
    # token="hf_your_token_here",
)
pipeline.instantiate(pipeline.default_parameters())

audio_file_1 = "meeting_day1.wav"   # ← replace with your files
audio_file_2 = "meeting_day2.wav"

output1: DiarizeOutput = pipeline(audio_file_1)
output2: DiarizeOutput = pipeline(audio_file_2)


# ==================================================================
# FEATURE 1: Full diarization — includes overlapping speech
# ==================================================================
print("=== Full Diarization (with overlaps) ===")

# itertracks(yield_label=True) gives you: (segment, track_id, label)
# segment has .start and .end in seconds
for turn, track, speaker in output1.speaker_diarization.itertracks(yield_label=True):
    print(f"  {speaker}:  {turn.start:6.2f}s → {turn.end:6.2f}s  (duration: {turn.duration:.2f}s)")

# You can also get just a specific speaker's segments:
target_speaker = output1.speaker_diarization.labels()[0]
speaker_timeline = output1.speaker_diarization.label_timeline(target_speaker)
total_speech = sum(seg.duration for seg in speaker_timeline)
print(f"\n{target_speaker} total speech time: {total_speech:.2f}s\n")


# ==================================================================
# FEATURE 2: Exclusive diarization — overlaps removed
# ==================================================================
print("=== Exclusive Diarization (no overlaps) ===")

# This is useful for:
#   - Word-level alignment (transcript + speaker)
#   - Clean speech extraction per speaker
#   - Metrics that penalise overlap attribution

for turn, _, speaker in output1.exclusive_speaker_diarization.itertracks(yield_label=True):
    print(f"  {speaker}:  {turn.start:6.2f}s → {turn.end:6.2f}s")

# Measure how much speech is overlap
full_duration     = sum(seg.duration for seg, *_ in output1.speaker_diarization.itertracks())
exclusive_duration= sum(seg.duration for seg, *_ in output1.exclusive_speaker_diarization.itertracks())
overlap_pct = 100 * (1 - exclusive_duration / full_duration) if full_duration > 0 else 0
print(f"\nOverlapping speech: {overlap_pct:.1f}% of total labelled speech\n")


# ==================================================================
# FEATURE 3: Speaker embeddings — voice fingerprints
# ==================================================================
print("=== Speaker Embeddings ===")

# speaker_embeddings shape: (num_speakers, embedding_dim)
# One centroid embedding per discovered speaker.
# The order matches diarization.labels() — first label → row 0, etc.

if output1.speaker_embeddings is not None:
    embs  = output1.speaker_embeddings
    labels = output1.speaker_diarization.labels()

    print(f"Embedding matrix shape: {embs.shape}")
    print(f"Speakers ({len(labels)}): {labels}\n")

    # Cosine distance between all speaker pairs within the same meeting
    dist_matrix = cdist(embs, embs, metric="cosine")
    print("Pairwise speaker distances (cosine):")
    header = "       " + "  ".join(f"{l:>10}" for l in labels)
    print(header)
    for i, label_i in enumerate(labels):
        row = "  ".join(f"{dist_matrix[i, j]:10.4f}" for j in range(len(labels)))
        print(f"  {label_i}: {row}")
    print()


# ==================================================================
# FEATURE 4: JSON serialization — save to disk
# ==================================================================
print("=== JSON Serialization ===")

# serialize() converts the DiarizeOutput to a plain dict
# with "diarization" and "exclusive_diarization" lists
serialized = output1.serialize()

print("Serialized structure:")
print(f"  Keys: {list(serialized.keys())}")
print(f"  First 2 turns:")
for turn in serialized["diarization"][:2]:
    print(f"    {turn}")

# Save to JSON file
output_path = Path("diarization_result.json")
output_path.write_text(json.dumps(serialized, indent=2))
print(f"\nSaved to: {output_path.resolve()}\n")


# ==================================================================
# FEATURE 5: Cross-meeting speaker comparison
# ==================================================================
print("=== Cross-Meeting Speaker Comparison ===")

# "Is the SPEAKER_00 from meeting 1 the same person as SPEAKER_01 from meeting 2?"
# Compare their centroid embeddings with cosine distance.

THRESHOLD = 0.7   # tune this based on your embedding model

if output1.speaker_embeddings is not None and output2.speaker_embeddings is not None:
    embs1  = output1.speaker_embeddings
    labels1 = output1.speaker_diarization.labels()
    embs2  = output2.speaker_embeddings
    labels2 = output2.speaker_diarization.labels()

    cross_distances = cdist(embs1, embs2, metric="cosine")

    print("Cross-meeting cosine distances (< threshold → likely same person):")
    for i, l1 in enumerate(labels1):
        for j, l2 in enumerate(labels2):
            d = cross_distances[i, j]
            match = "← MATCH" if d < THRESHOLD else ""
            print(f"  Meeting1/{l1} vs Meeting2/{l2}: {d:.4f}  {match}")
    print()


# ==================================================================
# FEATURE 6: The speaker label generator
# ==================================================================
print("=== Speaker Label Generator ===")

# SpeakerDiarization.classes() is an infinite generator producing
# SPEAKER_00, SPEAKER_01, SPEAKER_02, ... on demand.
# The pipeline uses this to assign standardised names to clusters.

gen = pipeline.classes()
first_10 = [next(gen) for _ in range(10)]
print(f"First 10 labels from classes(): {first_10}")
```

---

```python
# example_diarization_evaluation.py
"""
Evaluating Diarization Quality with DER
=========================================
Goal: Measure how accurate our diarization is compared to a ground-truth
      transcript (called a "reference annotation").

DER = Diarization Error Rate (lower is better, 0% = perfect).
It sums three types of errors:
  1. Miss       — speaker was talking but not detected
  2. False Alarm — detected speech where there was silence
  3. Confusion  — right region, wrong speaker label

Covers:
  1. Computing DER on one file
  2. Evaluating across multiple files
  3. Tuning parameters to lower DER
  4. Understanding DER variants (collar, skip_overlap)
  5. The batchify() utility used internally
"""

import numpy as np
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import GreedyDiarizationErrorRate

from pyannote.audio.pipelines.speaker_diarization import (
    SpeakerDiarization,
    batchify,
)

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
pipeline = SpeakerDiarization(
    # token="hf_your_token_here",
)
pipeline.instantiate(pipeline.default_parameters())


# ==================================================================
# FEATURE 1: DER on a single file
# ==================================================================
print("=== Feature 1: DER on One File ===\n")

# To compute DER you need a reference annotation — a ground-truth
# transcript of who spoke when. This would normally come from a
# labelled dataset or a human transcript.

# Fake reference annotation for demonstration
reference = Annotation(uri="demo_file")
reference[Segment(0.0, 5.0)]  = "Alice"
reference[Segment(5.5, 10.0)] = "Bob"
reference[Segment(9.0, 15.0)] = "Alice"

# Run pipeline on the same file
audio_file = {"uri": "demo_file", "audio": "demo.wav"}   # ← replace
output = pipeline(audio_file)
hypothesis = output.speaker_diarization

# get_metric() returns the metric configured at pipeline init time
metric = pipeline.get_metric()   # GreedyDiarizationErrorRate

# Calling the metric object computes DER (it finds the best label mapping)
der = metric(reference, hypothesis)
print(f"DER: {der * 100:.2f}%")

# You can also get a detailed breakdown
detail = metric(reference, hypothesis, detailed=True)
print(f"Miss          : {detail['miss'] * 100:.2f}%")
print(f"False Alarm   : {detail['false alarm'] * 100:.2f}%")
print(f"Speaker Conf. : {detail['confusion'] * 100:.2f}%\n")


# ==================================================================
# FEATURE 2: Evaluating across a dataset (multiple files)
# ==================================================================
print("=== Feature 2: Multi-File Evaluation ===\n")

# Real evaluation loops over a test set and accumulates the metric.
# The GreedyDiarizationErrorRate object is stateful — each call adds
# to its running total.

# Simulated test set (replace with real data from pyannote.database)
test_set = [
    {"uri": "file1", "audio": "file1.wav"},
    {"uri": "file2", "audio": "file2.wav"},
]
fake_references = {
    "file1": Annotation(uri="file1"),
    "file2": Annotation(uri="file2"),
}
fake_references["file1"][Segment(0, 3)] = "Alice"
fake_references["file1"][Segment(3, 6)] = "Bob"
fake_references["file2"][Segment(0, 4)] = "Carol"
fake_references["file2"][Segment(4, 8)] = "Dave"

accumulated_metric = pipeline.get_metric()

for file in test_set:
    output = pipeline(file)
    hypothesis = output.speaker_diarization
    reference  = fake_references[file["uri"]]
    accumulated_metric(reference, hypothesis)

# abs() on the metric returns the overall accumulated DER
overall_der = abs(accumulated_metric)
print(f"Overall DER across {len(test_set)} files: {overall_der * 100:.2f}%\n")


# ==================================================================
# FEATURE 3: DER variants — collar and skip_overlap
# ==================================================================
print("=== Feature 3: DER Variants ===\n")

# collar : ignore errors within N seconds of a speaker change boundary
#          (e.g. 0.25 means ±250ms around transitions don't count)
#          Common in academic papers to be more forgiving of timing errors.
#
# skip_overlap : ignore regions where 2+ speakers talk simultaneously
#               (makes DER easier — useful if overlap detection is out of scope)

pipeline_collar = SpeakerDiarization(
    der_variant={"collar": 0.25, "skip_overlap": False},
    # token="hf_your_token_here",
)
pipeline_skip = SpeakerDiarization(
    der_variant={"collar": 0.0, "skip_overlap": True},
    # token="hf_your_token_here",
)

metric_collar = pipeline_collar.get_metric()
metric_skip   = pipeline_skip.get_metric()

print(f"Standard metric class : {type(pipeline.get_metric()).__name__}")
print(f"Collar metric details : collar=0.25, skip_overlap=False")
print(f"Skip-overlap metric   : collar=0.0,  skip_overlap=True\n")


# ==================================================================
# FEATURE 4: Tuning to lower DER
# ==================================================================
print("=== Feature 4: Parameter Tuning for Lower DER ===\n")

# The pipeline has tunable parameters exposed via pipeline.segmentation
# and pipeline.clustering. In practice you'd use a held-out dev set
# and sweep over these. Here we show manual tuning.

# Key parameters to tune:
#   segmentation.min_duration_off → merge short silences (seconds)
#     too small → noisy splits; too large → merges different turns
#   segmentation.threshold       → (non-powerset only) binarization cutoff
#     higher → less speech detected; lower → more speech (more false alarms)
#   clustering.threshold         → cosine distance cutoff for merging speakers
#     lower → more speakers detected; higher → fewer (more confusion)

candidate_configs = [
    {"segmentation": {"min_duration_off": 0.0}, "clustering": {"threshold": 0.5, "Fa": 0.07, "Fb": 0.8}},
    {"segmentation": {"min_duration_off": 0.1}, "clustering": {"threshold": 0.6, "Fa": 0.07, "Fb": 0.8}},
    {"segmentation": {"min_duration_off": 0.2}, "clustering": {"threshold": 0.7, "Fa": 0.07, "Fb": 0.8}},
]

print("Sweeping over parameter configs (on one dev file):")

dev_file      = {"uri": "dev_file", "audio": "dev.wav"}   # ← replace
dev_reference = Annotation(uri="dev_file")
dev_reference[Segment(0, 5)]  = "A"
dev_reference[Segment(5, 10)] = "B"

best_der    = float("inf")
best_config = None

for config in candidate_configs:
    pipeline.instantiate(config)
    output    = pipeline(dev_file)
    m         = pipeline.get_metric()
    der       = m(dev_reference, output.speaker_diarization)
    label     = f"min_off={config['segmentation']['min_duration_off']}, "   \
                f"clust_thresh={config['clustering']['threshold']}"
    print(f"  Config [{label}] → DER = {der * 100:.2f}%")
    if der < best_der:
        best_der    = der
        best_config = config

print(f"\nBest config: {best_config}  (DER = {best_der * 100:.2f}%)\n")
# Apply best config for inference
pipeline.instantiate(best_config)


# ==================================================================
# FEATURE 5: batchify() — the utility used inside get_embeddings()
# ==================================================================
print("=== Feature 5: batchify() Utility ===\n")

# batchify splits any iterable into fixed-size chunks.
# Internally used to batch (waveform, mask) pairs before feeding to the embedder.
# Leftover slots are padded with fillvalue (default None).

items = list(range(10))   # [0, 1, 2, ..., 9]
batched = list(batchify(items, batch_size=3, fillvalue=-1))

print(f"Input  : {items}")
print(f"Batch size = 3")
print(f"Output : {batched}")
# → [(0,1,2), (3,4,5), (6,7,8), (9,-1,-1)]

# In pipeline code, None-filled entries are filtered out before processing:
# filter(lambda b: b[0] is not None, batch)
```
