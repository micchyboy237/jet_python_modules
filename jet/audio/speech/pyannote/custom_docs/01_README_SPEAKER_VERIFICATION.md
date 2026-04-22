# pyannote/audio/pipelines/speaker_verification.py

## Step-by-Step Analysis & Blueprint

### What This File Does (Plain English)

This file answers one question: _"Do these two audio clips belong to the same person?"_

It does this by turning each clip into a **speaker embedding** — a compact list of numbers
that acts like a voice fingerprint. Two clips from the same person produce embeddings that
are close together. Two clips from different people produce embeddings that are far apart.
The distance is measured with **cosine distance** (0 = identical, 2 = complete opposite).

There are four different neural-network backends that can produce these embeddings,
plus one smart router function that picks the right backend automatically from a model name string.

---

### The Core Idea in One Picture

```
Audio clip 1  ──▶  [embedding model]  ──▶  [ 0.12, -0.45, 0.88, ... ]  (512 numbers)
                                                        │
                                               cosine_distance()
                                                        │
Audio clip 2  ──▶  [embedding model]  ──▶  [ 0.11, -0.44, 0.90, ... ]  ──▶  0.03  ← same person

Audio clip 3  ──▶  [embedding model]  ──▶  [-0.80,  0.22, -0.15, ... ]  ──▶  1.72  ← different person
```

---

### Four Backends + One Router

```
speaker_verification.py
│
├── PyannoteAudioPretrainedSpeakerEmbedding   ← default, uses pyannote.audio model
├── SpeechBrainPretrainedSpeakerEmbedding     ← uses SpeechBrain (ECAPA-TDNN, etc.)
├── NeMoPretrainedSpeakerEmbedding            ← uses NVIDIA NeMo (TitaNet, etc.)
├── ONNXWeSpeakerPretrainedSpeakerEmbedding   ← uses WeSpeaker .onnx model file
│
├── PretrainedSpeakerEmbedding()    ← ROUTER: picks the right class from model name
│
├── SpeakerEmbedding (Pipeline)     ← high-level: file → one embedding
│   ├── __init__()                  → load embedding + optional segmentation model
│   └── apply()                     → read audio → optional VAD weights → embed
│
└── main()                          ← CLI evaluator: computes EER on VoxCeleb trials
```

---

### Shared Interface (All Four Backends)

Every backend inherits from `BaseInference` and exposes the same properties and method:

| Property / Method            | What it returns           | Notes                                  |
| ---------------------------- | ------------------------- | -------------------------------------- |
| `sample_rate`                | int (e.g. 16000)          | Audio must be resampled to this rate   |
| `dimension`                  | int (e.g. 512)            | Length of the output embedding vector  |
| `metric`                     | str (`"cosine"`)          | How to compare two embeddings          |
| `min_num_samples`            | int                       | Shortest waveform the model can handle |
| `to(device)`                 | self                      | Move model to CPU / GPU                |
| `__call__(waveforms, masks)` | `np.ndarray (batch, dim)` | The actual embedding extraction        |

---

### Key Concepts Explained

| Concept                           | Plain English                                                                                                                                                                            |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Speaker embedding**             | A fixed-length list of numbers that uniquely represents a voice, like a fingerprint                                                                                                      |
| **Cosine distance**               | How different two embeddings are. 0 = identical direction, 2 = opposite. Lower = more similar.                                                                                           |
| **`waveforms` shape `(B, 1, N)`** | Batch of B mono audio clips, each N samples long                                                                                                                                         |
| **`masks` shape `(B, frames)`**   | Per-frame weights — 1 = use this frame, 0 = ignore (e.g. silence). Lets the model focus only on speech.                                                                                  |
| **`min_num_samples`**             | The model needs a minimum amount of audio to work. Too-short clips return `NaN` embeddings.                                                                                              |
| **`cached_property`**             | A property computed only once on first access, then stored. `sample_rate` and `dimension` are computed by probing the model once.                                                        |
| **`too_short` mask**              | When a masked clip is shorter than `min_num_samples`, its row in the output is filled with `NaN` instead of crashing.                                                                    |
| **fbank features**                | Filter bank features — a frequency-based representation of audio. WeSpeaker needs these as input instead of raw waveforms.                                                               |
| **ONNX**                          | A portable model format. WeSpeaker models are exported to `.onnx` so they can run without the full WeSpeaker Python library.                                                             |
| **`SpeakerEmbedding` pipeline**   | A higher-level wrapper: takes a full audio **file** → internally loads the waveform → returns one embedding                                                                              |
| **VAD weights (segmentation)**    | When `segmentation` is set in `SpeakerEmbedding`, speech-probability scores are cubed (`**3`) and used as per-frame weights. This makes the embedding focus on confident speech regions. |
| **EER**                           | Equal Error Rate — the threshold where false accepts = false rejects. Lower is better. Standard benchmark for speaker verification.                                                      |
| **DET curve**                     | Detection Error Tradeoff — a plot of false alarm vs. miss rate across all thresholds.                                                                                                    |

---

### The Router: `PretrainedSpeakerEmbedding()`

```
embedding string
       │
       ├── "pyannote/..." or Model object  →  PyannoteAudioPretrainedSpeakerEmbedding
       ├── "speechbrain/..."               →  SpeechBrainPretrainedSpeakerEmbedding
       ├── "nvidia/..."                    →  NeMoPretrainedSpeakerEmbedding
       ├── "wespeaker/..." or .onnx path   →  ONNXWeSpeakerPretrainedSpeakerEmbedding
       └── anything else                   →  PyannoteAudioPretrainedSpeakerEmbedding
```

This is the function you should call in practice — you never need to instantiate the
backend classes directly unless you need fine-grained control.

---

### Files to Create

1. `example_speaker_verification_basic.py` — Compare two speakers, understand cosine distance
2. `example_speaker_verification_backends.py` — All 4 backends side by side
3. `example_speaker_verification_pipeline.py` — `SpeakerEmbedding` pipeline + VAD masking
4. `example_speaker_verification_evaluation.py` — EER, DET curve, batch trial evaluation

---

```python
# example_speaker_verification_basic.py
"""
Basic Speaker Verification — Are These Two Clips the Same Person?
=================================================================
Goal: Load two audio files → extract embeddings → compare with cosine distance.

This is the core use case for speaker_verification.py.
A small cosine distance (< ~0.5) usually means same speaker.
A large cosine distance (> ~1.0) usually means different speakers.
(The exact threshold depends on the model and your False Accept tolerance.)

Install requirements:
  pip install pyannote.audio scipy

You may need a HuggingFace token:
  https://huggingface.co/settings/tokens
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

# ------------------------------------------------------------------
# 1. Build the embedding extractor
#    PretrainedSpeakerEmbedding() is the smart router — it picks the
#    right backend class automatically from the model name string.
# ------------------------------------------------------------------
get_embedding = PretrainedSpeakerEmbedding(
    "pyannote/embedding",
    # token="hf_your_token_here",   # needed for gated HuggingFace models
    # cache_dir="/tmp/pyannote",     # where to store downloaded model files
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
get_embedding.to(device)

print(f"Backend        : {type(get_embedding).__name__}")
print(f"Sample rate    : {get_embedding.sample_rate} Hz")
print(f"Embedding size : {get_embedding.dimension} dimensions")
print(f"Distance metric: {get_embedding.metric}")
print(f"Min audio len  : {get_embedding.min_num_samples} samples "
      f"({get_embedding.min_num_samples / get_embedding.sample_rate * 1000:.1f} ms)\n")

# ------------------------------------------------------------------
# 2. Load audio waveforms
#    The embedding model expects:
#      waveforms shape: (batch_size, num_channels, num_samples)
#      - batch_size  = number of clips processed at once
#      - num_channels = 1  (mono only)
#      - num_samples = length of the audio clip
# ------------------------------------------------------------------
import torchaudio

def load_mono(path: str, sample_rate: int) -> torch.Tensor:
    """Load an audio file and return a (1, 1, num_samples) tensor."""
    waveform, sr = torchaudio.load(path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.unsqueeze(0)   # → (1, 1, num_samples)

sr = get_embedding.sample_rate

# Replace these with real audio files
clip_A1 = load_mono("speaker_A_clip1.wav", sr)   # Alice, recording 1
clip_A2 = load_mono("speaker_A_clip2.wav", sr)   # Alice, recording 2 (same person)
clip_B  = load_mono("speaker_B_clip1.wav", sr)   # Bob   (different person)

print(f"Clip A1 shape: {clip_A1.shape}  ({clip_A1.shape[-1] / sr:.2f}s)")
print(f"Clip A2 shape: {clip_A2.shape}  ({clip_A2.shape[-1] / sr:.2f}s)")
print(f"Clip B  shape: {clip_B.shape}   ({clip_B.shape[-1] / sr:.2f}s)\n")

# ------------------------------------------------------------------
# 3. Extract embeddings
#    Returns np.ndarray of shape (batch_size, dimension)
# ------------------------------------------------------------------
emb_A1 = get_embedding(clip_A1)   # shape: (1, 512)
emb_A2 = get_embedding(clip_A2)   # shape: (1, 512)
emb_B  = get_embedding(clip_B)    # shape: (1, 512)

print(f"Embedding shape: {emb_A1.shape}")
print(f"Embedding norm (A1): {np.linalg.norm(emb_A1):.4f}\n")

# ------------------------------------------------------------------
# 4. Compare embeddings using cosine distance
#    cdist returns a (1, 1) matrix — we take [0][0] for the scalar
# ------------------------------------------------------------------
dist_same    = cdist(emb_A1, emb_A2, metric="cosine")[0][0]
dist_diff    = cdist(emb_A1, emb_B,  metric="cosine")[0][0]

print("=== Cosine Distance Results ===")
print(f"  Alice vs Alice (same person) : {dist_same:.4f}  ← should be LOW")
print(f"  Alice vs Bob   (different)   : {dist_diff:.4f}  ← should be HIGH\n")

# ------------------------------------------------------------------
# 5. Make a verification decision
#    This threshold (0.5) is a starting point — tune it on your data.
# ------------------------------------------------------------------
THRESHOLD = 0.5

def verify(dist: float, threshold: float = THRESHOLD) -> str:
    return "✓ SAME speaker" if dist < threshold else "✗ DIFFERENT speaker"

print("=== Verification Decisions ===")
print(f"  Alice vs Alice : {verify(dist_same)}")
print(f"  Alice vs Bob   : {verify(dist_diff)}")

# ------------------------------------------------------------------
# 6. Batch processing — compare many clips at once
#    Stack clips along the batch dimension for GPU efficiency
# ------------------------------------------------------------------
print("\n=== Batch Embedding Extraction ===")

# Pad clips to the same length before stacking
max_len = max(clip_A1.shape[-1], clip_A2.shape[-1], clip_B.shape[-1])

def pad_to(tensor: torch.Tensor, length: int) -> torch.Tensor:
    pad_size = length - tensor.shape[-1]
    return torch.nn.functional.pad(tensor, (0, pad_size))

batch = torch.cat([
    pad_to(clip_A1, max_len),
    pad_to(clip_A2, max_len),
    pad_to(clip_B,  max_len),
], dim=0)   # shape: (3, 1, max_len)

batch_embs = get_embedding(batch)   # shape: (3, 512)
print(f"Batch embeddings shape: {batch_embs.shape}")

dist_matrix = cdist(batch_embs, batch_embs, metric="cosine")
labels = ["Alice-1", "Alice-2", "Bob"]

print("\nPairwise cosine distance matrix:")
header = "          " + "  ".join(f"{l:>9}" for l in labels)
print(header)
for i, li in enumerate(labels):
    row = "  ".join(f"{dist_matrix[i,j]:9.4f}" for j in range(len(labels)))
    print(f"  {li:>8}: {row}")
```

---

```python
# example_speaker_verification_backends.py
"""
All Four Embedding Backends Side by Side
=========================================
Goal: Show how to load and use each of the four backend classes,
      and understand when to choose each one.

Quick comparison:
┌──────────────────────────────────────────┬──────────────────────────────────────┐
│ Backend                                  │ When to use                          │
├──────────────────────────────────────────┼──────────────────────────────────────┤
│ PyannoteAudioPretrainedSpeakerEmbedding  │ Default — tight pyannote integration │
│ SpeechBrainPretrainedSpeakerEmbedding    │ ECAPA-TDNN, strong SOTA model        │
│ NeMoPretrainedSpeakerEmbedding           │ NVIDIA TitaNet, large-scale models   │
│ ONNXWeSpeakerPretrainedSpeakerEmbedding  │ Portable ONNX, no heavy framework    │
└──────────────────────────────────────────┴──────────────────────────────────────┘

All four share the same __call__ signature:
  embeddings = backend(waveforms)            → shape (batch, dim)
  embeddings = backend(waveforms, masks=m)   → shape (batch, dim), masked
"""

import numpy as np
import torch

SR = 16000
DURATION = 3   # seconds
BATCH    = 2

# Synthetic waveforms: (batch_size=2, channels=1, samples=48000)
fake_waveforms = torch.randn(BATCH, 1, SR * DURATION)

# Soft masks: (batch_size=2, num_frames)
# Values between 0 and 1 — 1 means "this frame is definitely speech"
NUM_FRAMES = 300
fake_masks = torch.ones(BATCH, NUM_FRAMES)
fake_masks[0, 200:] = 0.0   # last third of clip 1 is silence


def probe_backend(name, backend):
    """Print key properties and run a quick embedding extraction."""
    print(f"\n{'─' * 55}")
    print(f"  Backend : {name}")
    print(f"{'─' * 55}")
    print(f"  sample_rate      : {backend.sample_rate}")
    print(f"  dimension        : {backend.dimension}")
    print(f"  metric           : {backend.metric}")
    print(f"  min_num_samples  : {backend.min_num_samples}  "
          f"({backend.min_num_samples / backend.sample_rate * 1000:.1f} ms)")

    embs = backend(fake_waveforms)
    print(f"  Output shape     : {embs.shape}")
    print(f"  Any NaN?         : {np.any(np.isnan(embs))}")
    print(f"  Norm (clip 0)    : {np.linalg.norm(embs[0]):.4f}")

    # With masks
    embs_masked = backend(fake_waveforms, masks=fake_masks)
    print(f"  Masked shape     : {embs_masked.shape}")
    nan_masked = np.any(np.isnan(embs_masked))
    print(f"  Any NaN (masked) : {nan_masked}")


# ==================================================================
# BACKEND 1: PyannoteAudio
# ==================================================================
print("=" * 55)
print("BACKEND 1: PyannoteAudioPretrainedSpeakerEmbedding")
print("=" * 55)
print("""
Model source  : HuggingFace ("pyannote/embedding" or custom fine-tune)
Framework     : PyTorch (pyannote.audio Model class)
Masking       : Passes weights= to the model's forward() call
Best for      : Tight pyannote ecosystem integration, fine-tuning
Install       : pip install pyannote.audio
""")
try:
    from pyannote.audio.pipelines.speaker_verification import (
        PyannoteAudioPretrainedSpeakerEmbedding,
    )
    backend_pya = PyannoteAudioPretrainedSpeakerEmbedding(
        "pyannote/embedding",
        # token="hf_your_token_here",
    )
    probe_backend("PyannoteAudio", backend_pya)
except Exception as e:
    print(f"  (Skipped: {type(e).__name__}: {e})")


# ==================================================================
# BACKEND 2: SpeechBrain
# ==================================================================
print("\n" + "=" * 55)
print("BACKEND 2: SpeechBrainPretrainedSpeakerEmbedding")
print("=" * 55)
print("""
Model source  : HuggingFace ("speechbrain/spkrec-ecapa-voxceleb", etc.)
Framework     : SpeechBrain (EncoderClassifier)
Masking       : Passes wav_lens= (relative lengths 0–1) to encode_batch()
Best for      : ECAPA-TDNN — strong SOTA speaker model
Install       : pip install speechbrain

Tip: Use "@revision" suffix to pin a specific model version:
     "speechbrain/spkrec-ecapa-voxceleb@v3.0.0"
""")
try:
    from pyannote.audio.pipelines.speaker_verification import (
        SpeechBrainPretrainedSpeakerEmbedding,
    )
    backend_sb = SpeechBrainPretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb",
        # token="hf_your_token_here",
        # cache_dir="/tmp/speechbrain",
    )
    probe_backend("SpeechBrain", backend_sb)
except Exception as e:
    print(f"  (Skipped: SpeechBrain not installed or model unavailable: {e})")


# ==================================================================
# BACKEND 3: NeMo
# ==================================================================
print("\n" + "=" * 55)
print("BACKEND 3: NeMoPretrainedSpeakerEmbedding")
print("=" * 55)
print("""
Model source  : NVIDIA NGC / HuggingFace ("nvidia/speakerverification_en_titanet_large")
Framework     : NVIDIA NeMo (EncDecSpeakerLabelModel)
Masking       : Interpolates mask to waveform length, pads masked sequences
Best for      : Large-scale NVIDIA models, TitaNet architecture
Install       : pip install nemo_toolkit[asr]

Note: NeMo's .to(device) reloads from scratch — use device= at init time
      instead of calling .to() afterwards when possible.
""")
try:
    from pyannote.audio.pipelines.speaker_verification import (
        NeMoPretrainedSpeakerEmbedding,
    )
    backend_nemo = NeMoPretrainedSpeakerEmbedding(
        "nvidia/speakerverification_en_titanet_large",
        device=torch.device("cpu"),
    )
    probe_backend("NeMo", backend_nemo)
except Exception as e:
    print(f"  (Skipped: NeMo not installed or model unavailable: {e})")


# ==================================================================
# BACKEND 4: WeSpeaker (ONNX)
# ==================================================================
print("\n" + "=" * 55)
print("BACKEND 4: ONNXWeSpeakerPretrainedSpeakerEmbedding")
print("=" * 55)
print("""
Model source  : HuggingFace ("hbredin/wespeaker-voxceleb-resnet34-LM")
                or local path to a .onnx file
Framework     : ONNX Runtime (no PyTorch needed at inference time)
Input         : fbank features (not raw waveforms) — computed internally
Masking       : Masks are applied to the fbank feature frames
Best for      : Portable deployment, no heavy framework dependencies
Install       : pip install onnxruntime

Quirk: compute_fbank() is called on raw waveforms first,
       then the ONNX session receives (batch, frames, 80) fbank features.
""")
try:
    from pyannote.audio.pipelines.speaker_verification import (
        ONNXWeSpeakerPretrainedSpeakerEmbedding,
    )
    backend_onnx = ONNXWeSpeakerPretrainedSpeakerEmbedding(
        "hbredin/wespeaker-voxceleb-resnet34-LM",
        # token="hf_your_token_here",
        # cache_dir="/tmp/wespeaker",
    )
    probe_backend("WeSpeaker ONNX", backend_onnx)

    # Show fbank feature extraction explicitly
    print(f"\n  Fbank feature demo:")
    fbank = backend_onnx.compute_fbank(fake_waveforms[:1])
    print(f"    Input waveform : {fake_waveforms[:1].shape}")
    print(f"    fbank output   : {fbank.shape}  (batch, frames, 80 mel bins)")
    print(f"    min_num_frames : {backend_onnx.min_num_frames}")
except Exception as e:
    print(f"  (Skipped: onnxruntime not installed or model unavailable: {e})")


# ==================================================================
# ROUTER: PretrainedSpeakerEmbedding()
# ==================================================================
print("\n" + "=" * 55)
print("ROUTER: PretrainedSpeakerEmbedding()")
print("=" * 55)
print("""
This is the function you should use in most cases.
It inspects the model name string and returns the right backend.
""")

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

routing_table = [
    ("pyannote/embedding",                          "PyannoteAudio"),
    ("speechbrain/spkrec-ecapa-voxceleb",           "SpeechBrain"),
    ("nvidia/speakerverification_en_titanet_large", "NeMo"),
    ("hbredin/wespeaker-voxceleb-resnet34-LM",      "WeSpeaker ONNX"),
]

print(f"  {'Model string':50s}  →  Expected backend")
for model_str, expected in routing_table:
    # Show routing logic without actually downloading
    if "pyannote" in model_str:
        resolved = "PyannoteAudioPretrainedSpeakerEmbedding"
    elif "speechbrain" in model_str:
        resolved = "SpeechBrainPretrainedSpeakerEmbedding"
    elif "nvidia" in model_str:
        resolved = "NeMoPretrainedSpeakerEmbedding"
    elif "wespeaker" in model_str:
        resolved = "ONNXWeSpeakerPretrainedSpeakerEmbedding"
    else:
        resolved = "PyannoteAudioPretrainedSpeakerEmbedding (fallback)"
    print(f"  {model_str:50s}  →  {resolved}")
```

---

```python
# example_speaker_verification_pipeline.py
"""
SpeakerEmbedding Pipeline — Audio File → One Embedding
=========================================================
Goal: Use the high-level SpeakerEmbedding Pipeline class, which
      accepts an audio file dict instead of raw waveform tensors.

This is more convenient than using the raw backend classes directly
when you're working with files rather than pre-loaded tensors.

It also supports optional VAD (voice activity detection) weighting:
  - Without segmentation: embed the entire audio uniformly
  - With    segmentation: focus the embedding on speech frames only
    (speech probability is cubed → strongly down-weights near-silence)

Covers:
  1. Basic usage — file path → embedding
  2. With segmentation model — VAD-weighted embedding
  3. AudioFile dict formats accepted
  4. Comparing two speakers with the pipeline
  5. The VAD weight formula (scores**3)
  6. Running the CLI evaluator (main())
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist

from pyannote.audio.pipelines.speaker_verification import SpeakerEmbedding


# ==================================================================
# FEATURE 1: Basic usage — file path → one embedding
# ==================================================================
print("=== Feature 1: Basic SpeakerEmbedding Pipeline ===\n")

pipeline = SpeakerEmbedding(
    embedding="pyannote/embedding",
    segmentation=None,    # no VAD — embed the whole file uniformly
    # token="hf_your_token_here",
)

# SpeakerEmbedding.apply() returns np.ndarray of shape (1, dimension)
emb_alice = pipeline("alice.wav")    # ← replace with your file
emb_bob   = pipeline("bob.wav")

print(f"Embedding shape : {emb_alice.shape}   (1 × {emb_alice.shape[-1]} dims)")

dist = cdist(emb_alice, emb_bob, metric="cosine")[0][0]
print(f"Cosine distance : {dist:.4f}")
print(f"Decision        : {'same speaker' if dist < 0.5 else 'different speaker'}\n")


# ==================================================================
# FEATURE 2: With segmentation — VAD-weighted embedding
# ==================================================================
print("=== Feature 2: VAD-Weighted Embedding ===\n")
print("""
When you add a segmentation model, the pipeline:
  1. Runs the segmentation model on the file
  2. Takes per-frame speech probability scores
  3. Cubes them: weights = scores ** 3
     → prob=0.9 → weight=0.73  (keep)
     → prob=0.5 → weight=0.13  (down-weight heavily)
     → prob=0.1 → weight=0.001 (nearly ignore)
  4. Passes these weights to the embedding model

Effect: the embedding becomes less polluted by silence and background noise.
This typically improves verification accuracy on noisy recordings.
""")

pipeline_vad = SpeakerEmbedding(
    embedding="pyannote/embedding",
    segmentation="pyannote/segmentation",   # adds VAD weighting
    # token="hf_your_token_here",
)

emb_alice_vad = pipeline_vad("alice.wav")
emb_alice_plain = pipeline("alice.wav")    # no VAD (from Feature 1)

# Both are (1, 512) — same shape, different content
diff = cdist(emb_alice_vad, emb_alice_plain, metric="cosine")[0][0]
print(f"Without VAD weights embedding norm : {np.linalg.norm(emb_alice_plain):.4f}")
print(f"With    VAD weights embedding norm : {np.linalg.norm(emb_alice_vad):.4f}")
print(f"Distance between the two           : {diff:.4f}  (small = similar)\n")

# Visualise the weight formula
print("VAD weight formula  scores**3:")
for prob in [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]:
    weight = prob ** 3
    bar = "█" * int(weight * 30)
    print(f"  prob={prob:.2f}  weight={weight:.4f}  {bar}")
print()


# ==================================================================
# FEATURE 3: AudioFile dict formats
# ==================================================================
print("=== Feature 3: AudioFile Dict Formats ===\n")
print("""
SpeakerEmbedding.apply() accepts several AudioFile formats.
All are equivalent — use whichever is most convenient.
""")

# Format A: plain file path string
file_a = "alice.wav"

# Format B: Path object
from pathlib import Path
file_b = Path("alice.wav")

# Format C: dict with audio path (allows adding metadata like uri)
file_c = {
    "uri": "alice_recording_01",   # optional identifier
    "audio": "alice.wav",
}

# Format D: dict with pre-loaded waveform tensor
import torchaudio
waveform, sr = torchaudio.load("alice.wav")
if waveform.shape[0] > 1:
    waveform = waveform.mean(0, keepdim=True)
file_d = {
    "waveform": waveform,
    "sample_rate": sr,
}

for fmt, f in [("string path", file_a), ("Path object", file_b),
               ("dict + audio", file_c), ("dict + waveform", file_d)]:
    emb = pipeline(f)
    print(f"  {fmt:20s}  → embedding shape {emb.shape}")
print()


# ==================================================================
# FEATURE 4: Comparing multiple speakers with the pipeline
# ==================================================================
print("=== Feature 4: Multi-Speaker Comparison ===\n")

audio_files = {
    "Alice" : "alice.wav",
    "Bob"   : "bob.wav",
    "Carol" : "carol.wav",
}

# Extract embeddings (cache them to avoid re-running the model)
embeddings = {name: pipeline(path) for name, path in audio_files.items()}

# Build pairwise distance matrix
names = list(embeddings.keys())
emb_matrix = np.vstack([embeddings[n] for n in names])   # (3, 512)
dist_matrix = cdist(emb_matrix, emb_matrix, metric="cosine")

print("Pairwise cosine distances:")
header = "         " + "  ".join(f"{n:>7}" for n in names)
print(header)
for i, ni in enumerate(names):
    row = "  ".join(f"{dist_matrix[i,j]:7.4f}" for j in range(len(names)))
    print(f"  {ni:>6}: {row}")
print()


# ==================================================================
# FEATURE 5: The VAD weight formula in detail
# ==================================================================
print("=== Feature 5: VAD Weight Formula (scores**3) ===\n")
print("""
Inside SpeakerEmbedding.apply():

  1. self._segmentation(file).data
     → SlidingWindowFeature.data  shape: (num_frames, 1)
        Values are speech probabilities in [0, 1]

  2. weights[np.isnan(weights)] = 0.0
     → Replace NaN frames (no prediction) with 0 weight

  3. weights = torch.from_numpy(weights ** 3)[None, :, 0]
     → Cube the probabilities: this aggressively penalises
        uncertain frames without completely ignoring them.
     → Shape becomes (1, num_frames) — ready for the embedding model

  4. self.embedding_model_(waveform, weights=weights)
     → The model uses these weights to produce a weighted mean
        embedding across frames (only confident speech contributes)
""")

# Demonstrate the cubing effect numerically
import numpy as np
probs = np.array([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 1.0])
weights_cubed = probs ** 3

print(f"  {'prob':>6}  {'weight (prob**3)':>18}  {'relative contribution':>22}")
for p, w in zip(probs, weights_cubed):
    bar = "▉" * int(w * 25)
    print(f"  {p:6.2f}  {w:18.4f}  {bar}")
print()


# ==================================================================
# FEATURE 6: Running the CLI evaluator
# ==================================================================
print("=== Feature 6: CLI Evaluator (main()) ===\n")
print("""
The module includes a main() function that computes EER on VoxCeleb trial lists.
Run it from the command line:

  python speaker_verification.py \\
    --protocol VoxCeleb.SpeakerVerification.VoxCeleb1 \\
    --subset test \\
    --embedding pyannote/embedding \\
    --segmentation pyannote/segmentation

What it does:
  1. Loads all trial pairs (two audio files + same/different label)
  2. Extracts embeddings for each unique audio file (with caching)
  3. Computes cosine distance for each trial pair
  4. Runs det_curve() to find the EER threshold
  5. Prints: protocol | subset | embedding | segmentation | EER = X.XXX%

EER (Equal Error Rate) — the operating point where:
  False Accept Rate  (wrong "same") == False Reject Rate (wrong "different")
  Lower EER = better verification system.
  State-of-the-art systems achieve < 1% EER on VoxCeleb1.
""")
```

---

```python
# example_speaker_verification_evaluation.py
"""
Speaker Verification Evaluation — EER, DET Curve, Batch Trials
================================================================
Goal: Measure how accurately an embedding model verifies speakers.

The standard protocol:
  1. You have a list of "trials": pairs of audio clips with a label
     (1 = same speaker, 0 = different speaker)
  2. For each pair: extract embeddings → compute cosine distance
  3. At every possible threshold: count false accepts and false rejects
  4. Find the EER — the threshold where both error types are equal

Covers:
  1. Building and evaluating a trial list
  2. The DET curve and EER
  3. Precision / Recall at a fixed threshold
  4. Embedding caching for efficient evaluation
  5. min_num_samples guard — handling very short clips
  6. NaN embedding detection and filtering
"""

import numpy as np
import torch
import torchaudio
from pathlib import Path
from scipy.spatial.distance import cdist

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding


# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
get_embedding = PretrainedSpeakerEmbedding(
    "pyannote/embedding",
    # token="hf_your_token_here",
)
SR = get_embedding.sample_rate
MIN_SAMPLES = get_embedding.min_num_samples

print(f"Model           : pyannote/embedding")
print(f"Dimension       : {get_embedding.dimension}")
print(f"Min audio length: {MIN_SAMPLES} samples ({MIN_SAMPLES/SR*1000:.1f} ms)\n")


def load_clip(path: str) -> torch.Tensor:
    """Load audio → (1, 1, num_samples) tensor at model's sample rate."""
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    return wav.unsqueeze(0)   # (1, 1, N)


# ==================================================================
# FEATURE 1: Building and evaluating a trial list
# ==================================================================
print("=== Feature 1: Trial List Evaluation ===\n")

# A trial list is a list of dicts, each with:
#   "file1"     → path to clip 1
#   "file2"     → path to clip 2
#   "reference" → 1 if same speaker, 0 if different

# Replace with your actual trial list (e.g. from VoxCeleb or AMI)
trials = [
    {"file1": "alice_1.wav", "file2": "alice_2.wav", "reference": 1},
    {"file1": "alice_1.wav", "file2": "bob_1.wav",   "reference": 0},
    {"file1": "bob_1.wav",   "file2": "bob_2.wav",   "reference": 1},
    {"file1": "carol_1.wav", "file2": "alice_1.wav", "reference": 0},
    {"file1": "carol_1.wav", "file2": "carol_2.wav", "reference": 1},
]

# Embed unique files only (avoid re-computing the same file twice)
embedding_cache = {}

def get_cached_embedding(path: str) -> np.ndarray:
    if path not in embedding_cache:
        clip = load_clip(path)
        embedding_cache[path] = get_embedding(clip)   # (1, dim)
    return embedding_cache[path]

y_true, y_pred = [], []

for trial in trials:
    emb1 = get_cached_embedding(trial["file1"])
    emb2 = get_cached_embedding(trial["file2"])
    distance = cdist(emb1, emb2, metric="cosine")[0][0]
    y_pred.append(distance)
    y_true.append(trial["reference"])
    same = "SAME" if trial["reference"] == 1 else "DIFF"
    print(f"  [{same}]  {Path(trial['file1']).stem:15s} vs {Path(trial['file2']).stem:15s}"
          f"  dist={distance:.4f}")

print(f"\n  Cached {len(embedding_cache)} unique embeddings for {len(trials)} trials\n")


# ==================================================================
# FEATURE 2: EER and DET curve
# ==================================================================
print("=== Feature 2: EER (Equal Error Rate) ===\n")
print("""
EER is the threshold where:
  False Accept Rate (FAR) = False Reject Rate (FRR)

  FAR = (different-speaker pairs accepted as same) / (all different pairs)
  FRR = (same-speaker pairs rejected as different) / (all same pairs)

Lower EER = better model.
""")

try:
    from pyannote.metrics.binary_classification import det_curve
    _, _, _, eer = det_curve(y_true, np.array(y_pred), distances=True)
    print(f"  EER = {eer * 100:.3f}%")
    print(f"  (On VoxCeleb1, state-of-the-art models achieve < 1% EER)\n")
except ImportError:
    print("  (pyannote.metrics not installed — install with: pip install pyannote.metrics)")

    # Manual EER approximation without pyannote.metrics
    thresholds = np.linspace(0, 2, 1000)
    same_mask = np.array(y_true) == 1
    diff_mask = ~same_mask
    dists = np.array(y_pred)

    best_eer, best_threshold = float("inf"), 0.5
    for t in thresholds:
        far = np.mean(dists[diff_mask] < t) if diff_mask.any() else 0
        frr = np.mean(dists[same_mask] >= t) if same_mask.any() else 0
        eer_approx = abs(far - frr)
        if eer_approx < best_eer:
            best_eer = eer_approx
            best_threshold = t

    print(f"  Approximate EER threshold : {best_threshold:.4f}")
    print(f"  (Install pyannote.metrics for exact EER)\n")


# ==================================================================
# FEATURE 3: Precision and Recall at a fixed threshold
# ==================================================================
print("=== Feature 3: Precision / Recall at Fixed Threshold ===\n")

THRESHOLD = 0.5   # tune on a development set
dists  = np.array(y_pred)
labels = np.array(y_true)
preds  = (dists < THRESHOLD).astype(int)   # 1 = predicted same, 0 = predicted different

tp = np.sum((preds == 1) & (labels == 1))
fp = np.sum((preds == 1) & (labels == 0))
tn = np.sum((preds == 0) & (labels == 0))
fn = np.sum((preds == 0) & (labels == 1))

precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")

print(f"  Threshold  : {THRESHOLD}")
print(f"  Precision  : {precision:.4f}  (of predicted SAME, fraction truly SAME)")
print(f"  Recall     : {recall:.4f}  (of all SAME pairs, fraction correctly found)")
print(f"  F1-score   : {f1:.4f}")
print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}\n")


# ==================================================================
# FEATURE 4: min_num_samples guard — short clip handling
# ==================================================================
print("=== Feature 4: min_num_samples Guard ===\n")
print(f"""
If a clip is shorter than {MIN_SAMPLES} samples ({MIN_SAMPLES/SR*1000:.1f} ms),
the model cannot produce a valid embedding.

All four backends handle this the same way:
  - The output row for that clip is filled with NaN
  - No exception is raised — the batch continues processing
  - You must check for NaN before using the embeddings

The exact minimum is found at init time by binary search:
  try increasingly small inputs until the model raises RuntimeError,
  then take the smallest input that succeeded.
""")

SR = get_embedding.sample_rate
MIN = get_embedding.min_num_samples

short_clip = torch.randn(1, 1, max(1, MIN - 100))    # intentionally too short
valid_clip = torch.randn(1, 1, MIN + 1000)           # safely above minimum

emb_short = get_embedding(short_clip)
emb_valid = get_embedding(valid_clip)

print(f"  Short clip  ({short_clip.shape[-1]:6d} samples) → NaN? {np.any(np.isnan(emb_short))}")
print(f"  Valid clip  ({valid_clip.shape[-1]:6d} samples) → NaN? {np.any(np.isnan(emb_valid))}\n")


# ==================================================================
# FEATURE 5: NaN detection and filtering in batch evaluation
# ==================================================================
print("=== Feature 5: Filtering NaN Embeddings ===\n")
print("""
When processing many clips in a batch, some may be too short or
contain no speech (after masking). Their rows will be NaN.
You must filter these out before computing distances.
""")

# Simulate a batch where one embedding is NaN (too-short clip)
batch_embs = np.random.randn(5, get_embedding.dimension)
batch_embs[2, :] = np.nan   # simulate a too-short clip

valid_mask = ~np.any(np.isnan(batch_embs), axis=1)
valid_embs = batch_embs[valid_mask]
valid_idxs = np.where(valid_mask)[0]

print(f"  Batch size          : {len(batch_embs)}")
print(f"  NaN rows            : {(~valid_mask).sum()}  (indices: {np.where(~valid_mask)[0].tolist()})")
print(f"  Valid rows kept     : {valid_mask.sum()}  (indices: {valid_idxs.tolist()})")
print(f"  Valid embedding shape: {valid_embs.shape}\n")

# Safe distance computation
if len(valid_embs) >= 2:
    dist_mat = cdist(valid_embs, valid_embs, metric="cosine")
    print(f"  Distance matrix shape (valid only): {dist_mat.shape}")
else:
    print("  Not enough valid embeddings to compute distances.")


# ==================================================================
# FEATURE 6: EER threshold sweep — find the best operating point
# ==================================================================
print("\n=== Feature 6: Threshold Sweep ===\n")
print("""
In production you need to pick a fixed operating threshold.
Sweep over thresholds on a development set to find the best trade-off.
""")

# Use the y_true / y_pred from Feature 1
dists  = np.array(y_pred)
labels = np.array(y_true)

print(f"  {'Threshold':>10}  {'FAR':>8}  {'FRR':>8}  {'|FAR-FRR|':>10}")
print("  " + "-" * 44)

same_mask = labels == 1
diff_mask = ~same_mask

for t in np.arange(0.2, 1.4, 0.2):
    far = float(np.mean(dists[diff_mask] < t)) if diff_mask.any() else 0.0
    frr = float(np.mean(dists[same_mask] >= t)) if same_mask.any() else 0.0
    gap = abs(far - frr)
    marker = " ← EER point" if gap < 0.1 else ""
    print(f"  {t:10.2f}  {far:8.3f}  {frr:8.3f}  {gap:10.4f}{marker}")
```

---

### How All the Pieces Fit Together

```
                ┌─────────────────────────────────────────────────────┐
                │  PretrainedSpeakerEmbedding("some/model")           │
                │  (router function — returns the right backend)      │
                │                                                     │
                │  "pyannote/..." → PyannoteAudioPretrainedSpeakerEmb │
                │  "speechbrain/..." → SpeechBrainPretrainedSpeakerEmb│
                │  "nvidia/..."  → NeMoPretrainedSpeakerEmbedding     │
                │  "wespeaker/..."→ ONNXWeSpeakerPretrainedSpeakerEmb │
                └────────────────────────┬────────────────────────────┘
                                         │
                       backend.__call__(waveforms, masks=...)
                                         │
                                         ▼
                            np.ndarray  (batch_size, dim)
                                         │
                         ┌───────────────┤
                         │               │
              ┌──────────▼──────────┐   cdist(emb1, emb2, metric="cosine")
              │  SpeakerEmbedding   │         │
              │  (Pipeline)         │         ▼
              │                     │   distance  ──▶  threshold  ──▶  same / different
              │  file → waveform    │
              │  optional VAD mask  │
              │  scores ** 3        │
              └─────────────────────┘
                         │
                  main() CLI loop
                         │
              y_true, y_pred lists
                         │
                    det_curve()
                         │
                  EER printed to stdout
```

---

### Backend Quirks at a Glance

| Backend           | Masking method                          | Special input                            | `to(device)` behaviour                   |
| ----------------- | --------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| **PyannoteAudio** | `weights=` passed to `model_.forward()` | Raw waveform                             | Move model in-place                      |
| **SpeechBrain**   | `wav_lens=` (relative 0–1 lengths)      | Raw waveform                             | Reload classifier from disk              |
| **NeMo**          | Interpolate mask → pad sequences        | Raw waveform                             | Move model in-place                      |
| **WeSpeaker**     | Mask applied to fbank frames            | **fbank features** (computed internally) | Re-create ONNX session with new provider |

---

### Common Mistakes

**1. Wrong waveform shape**

```python
# ❌ Wrong — missing the channel dimension
waveform = torch.randn(batch_size, num_samples)   # (B, N)

# ✅ Correct — must be (batch, channels, samples) with channels=1
waveform = torch.randn(batch_size, 1, num_samples)   # (B, 1, N)
```

**2. Not checking for NaN embeddings**

```python
# ❌ Wrong — silently produces NaN distances for short clips
embs = get_embedding(short_clips)
distances = cdist(embs, embs, metric="cosine")

# ✅ Correct — filter NaN rows before computing distances
valid = ~np.any(np.isnan(embs), axis=1)
embs_clean = embs[valid]
distances = cdist(embs_clean, embs_clean, metric="cosine")
```

**3. Using `SpeakerEmbedding` when you need the raw backend**

```python
# SpeakerEmbedding assumes ONE speaker per file.
# ❌ Wrong for multi-speaker recordings — it embeds everything together
pipeline = SpeakerEmbedding()
emb = pipeline("meeting_with_multiple_speakers.wav")   # mixed embedding

# ✅ For multi-speaker files, use SpeakerDiarization instead
```

**4. Forgetting to resample**

```python
# ❌ Wrong — model expects 16 kHz but file is 44.1 kHz
waveform, sr = torchaudio.load("music.wav")   # sr=44100
emb = get_embedding(waveform.unsqueeze(0))    # wrong sample rate

# ✅ Correct
waveform = torchaudio.functional.resample(waveform, sr, get_embedding.sample_rate)
emb = get_embedding(waveform.unsqueeze(0))
```
