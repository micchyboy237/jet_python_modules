# pyannote/audio/pipelines/clustering.py

## Step-by-Step Analysis & Blueprint

### What This File Does (Plain English)

After speaker embeddings are extracted, we need to answer: _"Which of these voice fingerprints belong to the same person?"_ That's **clustering** — grouping similar embeddings together. This file provides 4 different algorithms to do that, all sharing a common base.

---

### The Clustering Flow (How Every Algorithm Works)

```
Embeddings (num_chunks, num_speakers, dimension)
        │
        ▼
┌───────────────────────────────────┐
│  filter_embeddings()              │
│  Remove: NaN, too-short speakers  │
│  → train_embeddings (clean set)   │
└──────────────────┬────────────────┘
                   │
                   ▼
┌───────────────────────────────────┐
│  cluster()  ← algorithm-specific │
│  Groups train_embeddings into K   │
│  → train_clusters (int labels)    │
└──────────────────┬────────────────┘
                   │
                   ▼
┌───────────────────────────────────┐
│  assign_embeddings()              │
│  Compute centroids per cluster    │
│  Assign ALL embeddings (incl.     │
│  filtered-out ones) to nearest    │
│  centroid                         │
└──────────────────┬────────────────┘
                   │
                   ▼
        hard_clusters  (chunk, speaker) → cluster_id
        soft_clusters  (chunk, speaker, cluster) → confidence
        centroids      (num_clusters, dimension)
```

---

### Feature Map / Blueprint

```
clustering.py
│
├── BaseClustering (Pipeline)           → Shared logic for all algorithms
│   ├── set_num_clusters()              → Clamp/validate K within [min, max]
│   ├── filter_embeddings()             → Remove NaN + inactive speakers
│   ├── constrained_argmax()            → Hungarian algorithm assignment
│   ├── assign_embeddings()             → Centroid computation + full assignment
│   └── __call__()                      → Orchestrates filter → cluster → assign
│
├── AgglomerativeClustering             → Bottom-up tree merging (no K needed)
│   ├── Hyper-params: threshold, method, min_cluster_size
│   └── cluster()                       → linkage() → fcluster() → merge small clusters
│
├── KMeansClustering                    → Classic K-Means (K required)
│   └── cluster()                       → sklearn KMeans, cosine-aware
│
├── VBxClustering                       → Bayesian VB + PLDA (most sophisticated)
│   ├── Hyper-params: threshold, Fa, Fb
│   └── __call__()                      → AHC init → VBx refine → fallback KMeans
│
├── OracleClustering                    → Cheats using ground-truth (for testing)
│   └── __call__()                      → Reads reference annotation directly
│
└── Clustering (Enum)                   → Registry: maps name → class
```

---

### Key Concepts Explained

| Concept                    | Plain English                                                                                                             |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Hard cluster**           | A definitive answer: "chunk 3, speaker 1 IS cluster 0"                                                                    |
| **Soft cluster**           | A confidence score: "chunk 3, speaker 1 is 87% likely cluster 0"                                                          |
| **Centroid**               | The average embedding of all speakers assigned to a cluster — the "representative voice"                                  |
| **`filter_embeddings`**    | Throws out speakers who are barely active or have broken (NaN) embeddings                                                 |
| **`min_active_ratio`**     | A speaker must be active in ≥20% of a chunk's frames to count                                                             |
| **Constrained assignment** | Uses Hungarian algorithm — ensures each speaker gets a _unique_ cluster per chunk                                         |
| **Agglomerative**          | Starts with every embedding as its own cluster, merges closest pairs repeatedly                                           |
| **Linkage / dendrogram**   | The merge tree built by agglomerative clustering                                                                          |
| **`min_cluster_size`**     | Small clusters get absorbed into the nearest big one                                                                      |
| **VBx**                    | Variational Bayes — a probabilistic method that refines cluster assignments iteratively                                   |
| **PLDA**                   | Probabilistic Linear Discriminant Analysis — scores how likely two embeddings are from the same speaker                   |
| **Oracle**                 | "Cheating" mode — uses the ground-truth transcript to get perfect cluster assignments. Used for benchmarking upper bounds |
| **`expects_num_clusters`** | Flag that tells the diarization pipeline whether K must be supplied                                                       |

---

### Files to Create

1. `example_clustering_basic.py` — Core concepts: filter, assign, hard/soft clusters
2. `example_clustering_algorithms.py` — Side-by-side comparison of all 4 algorithms
3. `example_clustering_advanced.py` — Constrained assignment, tuning, the Enum registry

---

```python
# example_clustering_basic.py
"""
Clustering Basics — Understanding the Shared Foundation
=========================================================
Goal: Understand what BaseClustering does before looking at specific algorithms.

All 4 clustering algorithms in this file share the same 3-step flow:
  1. filter_embeddings() → remove noisy/inactive speakers
  2. cluster()           → group clean embeddings (algorithm-specific)
  3. assign_embeddings() → assign ALL embeddings to computed centroids

We'll walk through each step manually using AgglomerativeClustering
so you can see exactly what's happening inside __call__().
"""

import numpy as np
from pyannote.core import SlidingWindow, SlidingWindowFeature

from pyannote.audio.pipelines.clustering import AgglomerativeClustering

# ------------------------------------------------------------------
# Synthetic data setup
# We simulate a 3-chunk audio file with 2 local speakers per chunk
# and 256-dimensional embeddings (like a real embedding model would give).
#
# Think of it as: 3 overlapping windows of audio, each window has
# been segmented into 2 possible speaker tracks.
# ------------------------------------------------------------------
np.random.seed(42)

NUM_CHUNKS   = 6      # audio windows
NUM_SPEAKERS = 2      # local speakers per window (from segmentation)
NUM_FRAMES   = 100    # frames per window (for segmentation mask)
DIMENSION    = 64     # embedding size (smaller for demo)
NUM_CLUSTERS = 2      # we know there are 2 real speakers

# Create two clearly-separated speaker "identities"
# Speaker A lives around [1, 0, 0, ...]  Speaker B around [-1, 0, 0, ...]
speaker_A = np.array([1.0] + [0.0] * (DIMENSION - 1))
speaker_B = np.array([-1.0] + [0.0] * (DIMENSION - 1))

# Fill embedding array: alternate A and B across chunks
embeddings = np.zeros((NUM_CHUNKS, NUM_SPEAKERS, DIMENSION))
for c in range(NUM_CHUNKS):
    # Speaker slot 0 is always A, slot 1 is always B (with some noise)
    embeddings[c, 0] = speaker_A + np.random.randn(DIMENSION) * 0.05
    embeddings[c, 1] = speaker_B + np.random.randn(DIMENSION) * 0.05

# Inject one NaN embedding to show filtering (chunk 2, speaker 0)
embeddings[2, 0, :] = np.nan

# ------------------------------------------------------------------
# Build segmentation mask
# Shape: (num_chunks, num_frames, num_speakers)
# 1 = this speaker is active in this frame, 0 = silent
# ------------------------------------------------------------------
seg_data = np.zeros((NUM_CHUNKS, NUM_FRAMES, NUM_SPEAKERS))
# Both speakers active for most frames
seg_data[:, :80, 0] = 1.0   # speaker 0 active in first 80% of frames
seg_data[:, :80, 1] = 1.0   # speaker 1 active in first 80% of frames
# Make chunk 4, speaker 1 barely active (< 20% → should be filtered)
seg_data[4, :, 1] = 0.0
seg_data[4, :10, 1] = 1.0   # only 10% active → below min_active_ratio

window = SlidingWindow(start=0.0, duration=2.0, step=0.2)
segmentations = SlidingWindowFeature(seg_data, window)

# ------------------------------------------------------------------
# Build the clustering object
# We set parameters directly (normally optimized on a dev set)
# ------------------------------------------------------------------
clustering = AgglomerativeClustering(metric="cosine")
clustering.instantiate({
    "threshold": 0.5,
    "method": "average",
    "min_cluster_size": 1,
})

print("=" * 55)
print("STEP 1: filter_embeddings()")
print("=" * 55)

# This removes:
#   - NaN embeddings (chunk 2, speaker 0 — we injected this)
#   - Speakers active in < 20% of frames (chunk 4, speaker 1)
filtered, chunk_idx, speaker_idx = clustering.filter_embeddings(
    embeddings,
    segmentations=segmentations,
    min_active_ratio=0.2,   # default
)

print(f"Original embedding grid : {NUM_CHUNKS} chunks × {NUM_SPEAKERS} speakers "
      f"= {NUM_CHUNKS * NUM_SPEAKERS} total slots")
print(f"After filtering         : {len(filtered)} valid embeddings kept")
print(f"\nFiltered-OUT slots:")
all_slots  = set((c, s) for c in range(NUM_CHUNKS) for s in range(NUM_SPEAKERS))
kept_slots = set(zip(chunk_idx.tolist(), speaker_idx.tolist()))
for c, s in sorted(all_slots - kept_slots):
    reason = "NaN embedding" if np.isnan(embeddings[c, s, 0]) else "too inactive"
    print(f"  chunk={c}, speaker={s}  ({reason})")

print("\nKept slot indices (chunk_idx, speaker_idx):")
for c, s in zip(chunk_idx, speaker_idx):
    print(f"  chunk={c}, speaker={s}")


print("\n" + "=" * 55)
print("STEP 2: cluster()  [AgglomerativeClustering-specific]")
print("=" * 55)

# cluster() only sees the CLEAN embeddings and assigns integer labels
train_clusters = clustering.cluster(
    filtered,
    min_clusters=NUM_CLUSTERS,
    max_clusters=NUM_CLUSTERS,
    num_clusters=NUM_CLUSTERS,
)
print(f"Train cluster labels (one per kept embedding): {train_clusters}")
print(f"Unique clusters found: {np.unique(train_clusters)}")


print("\n" + "=" * 55)
print("STEP 3: assign_embeddings()")
print("=" * 55)

# Now ALL embedding slots (including filtered ones) get assigned
# to the nearest centroid computed from the clean training set.
hard_clusters, soft_clusters, centroids = clustering.assign_embeddings(
    embeddings,
    chunk_idx,
    speaker_idx,
    train_clusters,
    constrained=False,
)

print(f"hard_clusters shape : {hard_clusters.shape}  "
      f"(num_chunks={NUM_CHUNKS}, num_speakers={NUM_SPEAKERS})")
print(f"soft_clusters shape : {soft_clusters.shape}  "
      f"(num_chunks, num_speakers, num_clusters={NUM_CLUSTERS})")
print(f"centroids shape     : {centroids.shape}  "
      f"(num_clusters, dimension={DIMENSION})\n")

print("Hard cluster assignments (which global speaker each local speaker maps to):")
for c in range(NUM_CHUNKS):
    for s in range(NUM_SPEAKERS):
        k = hard_clusters[c, s]
        conf = soft_clusters[c, s, k] if k >= 0 else float("nan")
        print(f"  chunk={c}, speaker={s}  → cluster {k}  (confidence={conf:.3f})")


print("\n" + "=" * 55)
print("FULL __call__() pipeline (all 3 steps in one go)")
print("=" * 55)

hard, soft, cents = clustering(
    embeddings,
    segmentations=segmentations,
    num_clusters=NUM_CLUSTERS,
)
print(f"hard_clusters:\n{hard}")
print(f"\ncentroids shape: {cents.shape}")
print(f"centroid[0] norm: {np.linalg.norm(cents[0]):.4f}")
print(f"centroid[1] norm: {np.linalg.norm(cents[1]):.4f}")
```

---

```python
# example_clustering_algorithms.py
"""
Comparing All 4 Clustering Algorithms
=======================================
Goal: Show how to use each algorithm and highlight their differences.

Quick reference:
┌──────────────────────────┬──────────────┬─────────────────────────────────────┐
│ Algorithm                │ Needs K?     │ Best for                            │
├──────────────────────────┼──────────────┼─────────────────────────────────────┤
│ AgglomerativeClustering  │ No           │ Unknown # speakers, flexible        │
│ KMeansClustering         │ Yes          │ Known # speakers, fast              │
│ VBxClustering            │ No (+ PLDA)  │ Best accuracy, speaker diarization  │
│ OracleClustering         │ Yes          │ Research only — needs ground truth  │
└──────────────────────────┴──────────────┴─────────────────────────────────────┘
"""

import numpy as np
from pyannote.core import SlidingWindow, SlidingWindowFeature

# ------------------------------------------------------------------
# Shared synthetic data (same setup as example_clustering_basic.py)
# ------------------------------------------------------------------
np.random.seed(0)

NUM_CHUNKS, NUM_SPEAKERS, NUM_FRAMES, DIMENSION = 8, 2, 100, 64
TRUE_NUM_SPEAKERS = 2

spk_A = np.array([1.0] + [0.0] * (DIMENSION - 1))
spk_B = np.array([-1.0] + [0.0] * (DIMENSION - 1))

embeddings = np.zeros((NUM_CHUNKS, NUM_SPEAKERS, DIMENSION))
for c in range(NUM_CHUNKS):
    embeddings[c, 0] = spk_A + np.random.randn(DIMENSION) * 0.05
    embeddings[c, 1] = spk_B + np.random.randn(DIMENSION) * 0.05

seg_data = np.ones((NUM_CHUNKS, NUM_FRAMES, NUM_SPEAKERS))
window = SlidingWindow(start=0.0, duration=2.0, step=0.2)
segmentations = SlidingWindowFeature(seg_data, window)

def print_result(name, hard, soft, centroids):
    unique = np.unique(hard[hard >= 0])
    print(f"\n  Algorithm     : {name}")
    print(f"  Clusters found: {len(unique)}  (labels: {unique.tolist()})")
    print(f"  Centroids     : {centroids.shape}")
    print(f"  Hard clusters (chunk × speaker):\n{hard}")


# ==================================================================
# ALGORITHM 1: AgglomerativeClustering
# "Bottom-up tree merging"
# ==================================================================
print("=" * 60)
print("ALGORITHM 1: AgglomerativeClustering")
print("=" * 60)
print("""
How it works:
  1. Start: each embedding is its own cluster
  2. Repeatedly merge the two closest clusters
  3. Stop when distance between clusters exceeds `threshold`
  4. Small clusters (< min_cluster_size) are absorbed into larger ones

When to use:
  - You DON'T know the number of speakers
  - You want a threshold-based cutoff
  - Typical default for pyannote diarization
""")

from pyannote.audio.pipelines.clustering import AgglomerativeClustering

ahc = AgglomerativeClustering(metric="cosine")
ahc.instantiate({
    "threshold": 0.5,       # cosine distance cutoff for merging
    "method": "average",    # average linkage — uses mean distance between clusters
    "min_cluster_size": 1,  # clusters with fewer members merge into nearest large one
})

hard, soft, cents = ahc(
    embeddings,
    segmentations=segmentations,
    # No num_clusters needed — threshold decides
    min_clusters=1,
    max_clusters=10,
)
print_result("AgglomerativeClustering", hard, soft, cents)

# Linkage method comparison:
print("\n  Effect of linkage method on cluster count:")
for method in ["single", "average", "complete", "ward"]:
    ahc2 = AgglomerativeClustering(metric="cosine")
    # ward requires euclidean internally (cosine embeddings get normalised first)
    ahc2.instantiate({"threshold": 0.5, "method": method, "min_cluster_size": 1})
    h, _, _ = ahc2(embeddings, segmentations=segmentations, min_clusters=1, max_clusters=10)
    n_clusters = len(np.unique(h[h >= 0]))
    print(f"    method={method:<10}  → {n_clusters} clusters")


# ==================================================================
# ALGORITHM 2: KMeansClustering
# "Assign to nearest of K fixed centroids"
# ==================================================================
print("\n" + "=" * 60)
print("ALGORITHM 2: KMeansClustering")
print("=" * 60)
print("""
How it works:
  1. Randomly initialise K centroids
  2. Assign each embedding to nearest centroid
  3. Recompute centroids as mean of assigned embeddings
  4. Repeat until stable

When to use:
  - You KNOW the number of speakers (num_clusters is required)
  - Fast and simple
  - Good for short recordings with clear speaker counts

Limitation: Only supports 'cosine' or 'euclidean' metric.
""")

from pyannote.audio.pipelines.clustering import KMeansClustering

kmeans = KMeansClustering(metric="cosine")
# KMeans REQUIRES num_clusters — it will raise ValueError without it
hard, soft, cents = kmeans(
    embeddings,
    segmentations=segmentations,
    num_clusters=TRUE_NUM_SPEAKERS,   # must provide this
)
print_result("KMeansClustering", hard, soft, cents)

# What happens if num_embeddings < num_clusters:
print("\n  Edge case: fewer embeddings than requested clusters")
tiny = np.random.randn(2, 1, DIMENSION)   # only 2 embedding slots
tiny_seg = SlidingWindowFeature(np.ones((2, NUM_FRAMES, 1)), window)
kmeans2 = KMeansClustering(metric="cosine")
h2, _, c2 = kmeans2(tiny, segmentations=tiny_seg, num_clusters=5)
print(f"  Requested 5 clusters from 2 embeddings → got {len(np.unique(h2[h2>=0]))} clusters")
print("  (KMeans falls back to assigning each embedding its own cluster)")


# ==================================================================
# ALGORITHM 3: VBxClustering
# "Variational Bayes + PLDA — the most accurate"
# ==================================================================
print("\n" + "=" * 60)
print("ALGORITHM 3: VBxClustering")
print("=" * 60)
print("""
How it works:
  1. Run AHC (agglomerative) with cosine threshold to get initial clusters
  2. Transform embeddings through PLDA (makes same-speaker pairs score higher)
  3. Run VBx — iterative Bayesian refinement of cluster assignments
  4. If result violates num_speakers bounds → fallback to KMeans

When to use:
  - Best accuracy for speaker diarization (default in pyannote pipeline)
  - When you have a PLDA model (trained on speaker data)
  - Does NOT strictly require num_clusters

Requires: a pretrained PLDA model (from pyannote/speaker-diarization-community-1)
""")

# VBxClustering needs a real PLDA model — load from the diarization bundle
# In practice this is handled automatically by SpeakerDiarization.__init__()
try:
    from pyannote.audio.pipelines.utils import get_plda
    plda = get_plda({
        "checkpoint": "pyannote/speaker-diarization-community-1",
        "subfolder": "plda",
    })
    from pyannote.audio.pipelines.clustering import VBxClustering

    vbx = VBxClustering(plda=plda, metric="cosine", constrained_assignment=True)
    vbx.instantiate({
        "threshold": 0.65,  # AHC init threshold
        "Fa": 0.07,         # VBx acoustic weight — higher = more confident in embeddings
        "Fb": 0.8,          # VBx prior weight — higher = more conservative (fewer speakers)
    })

    hard, soft, cents = vbx(
        embeddings,
        segmentations=segmentations,
        min_clusters=1,
        max_clusters=10,
    )
    print_result("VBxClustering", hard, soft, cents)

except Exception as e:
    print(f"  (Skipped — PLDA model unavailable: {type(e).__name__}: {e})")
    print("  In real usage: VBxClustering is instantiated by SpeakerDiarization pipeline")


# ==================================================================
# ALGORITHM 4: OracleClustering
# "Cheating with ground truth — for research/benchmarking"
# ==================================================================
print("\n" + "=" * 60)
print("ALGORITHM 4: OracleClustering")
print("=" * 60)
print("""
How it works:
  - Reads the ground-truth reference annotation from the file dict
  - Aligns predicted segmentation to oracle using permutation matching
  - Returns perfect cluster assignments (as good as the segmentation model allows)

When to use:
  - Benchmarking: "what's the best DER we can achieve if clustering is perfect?"
  - Isolating segmentation errors from clustering errors
  - NEVER use in production — requires the answer to be known in advance

Requires: file["annotation"] to contain the reference transcript
""")

from pyannote.core import Annotation, Segment

# Simulate a file dict with ground-truth annotation
reference = Annotation(uri="demo")
reference[Segment(0.0, 5.0)] = "Alice"
reference[Segment(5.0, 10.0)] = "Bob"
reference[Segment(10.0, 15.0)] = "Alice"

oracle_file = {"uri": "demo", "audio": "demo.wav", "annotation": reference}

from pyannote.audio.pipelines.clustering import OracleClustering

oracle = OracleClustering()
try:
    hard, soft, cents = oracle(
        embeddings=embeddings,
        segmentations=segmentations,
        file=oracle_file,
        frames=window,
    )
    print_result("OracleClustering", hard, soft, cents)
    print(f"\n  centroids: {'computed from embeddings' if cents is not None else 'None (no embeddings passed)'}")
except Exception as e:
    print(f"  (Requires real audio + annotation alignment: {type(e).__name__})")
```

---

```python
# example_clustering_advanced.py
"""
Advanced Clustering — Constrained Assignment, Tuning & the Enum Registry
==========================================================================
Goal: Explore the less-obvious but important features:
  1. Constrained vs unconstrained assignment (Hungarian algorithm)
  2. set_num_clusters() — how K bounds are validated and clamped
  3. The Clustering Enum — how to look up algorithms by name
  4. min_cluster_size — absorbing tiny clusters
  5. expects_num_clusters flag — pipeline compatibility check
  6. Soft clusters — reading confidence scores
"""

import numpy as np
from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.audio.pipelines.clustering import (
    AgglomerativeClustering,
    KMeansClustering,
    Clustering,   # the Enum registry
)

# ------------------------------------------------------------------
# Shared data
# ------------------------------------------------------------------
np.random.seed(7)
NUM_CHUNKS, NUM_SPEAKERS, NUM_FRAMES, DIM = 6, 3, 80, 32

# 3 well-separated speaker identities
A = np.eye(DIM)[0]    # [1, 0, 0, ...]
B = np.eye(DIM)[1]    # [0, 1, 0, ...]
C = np.eye(DIM)[2]    # [0, 0, 1, ...]

embeddings = np.zeros((NUM_CHUNKS, NUM_SPEAKERS, DIM))
for c in range(NUM_CHUNKS):
    embeddings[c, 0] = A + np.random.randn(DIM) * 0.02
    embeddings[c, 1] = B + np.random.randn(DIM) * 0.02
    embeddings[c, 2] = C + np.random.randn(DIM) * 0.02

seg_data = np.ones((NUM_CHUNKS, NUM_FRAMES, NUM_SPEAKERS))
window   = SlidingWindow(start=0.0, duration=2.0, step=0.2)
segmentations = SlidingWindowFeature(seg_data, window)


# ==================================================================
# FEATURE 1: Constrained vs Unconstrained Assignment
# ==================================================================
print("=" * 60)
print("FEATURE 1: Constrained vs Unconstrained Assignment")
print("=" * 60)
print("""
Unconstrained (default):
  Each (chunk, speaker) slot independently picks the closest cluster.
  Multiple speakers in the same chunk CAN map to the same cluster.
  → Can happen when two speakers sound similar.

Constrained (constrained_assignment=True):
  Uses the Hungarian algorithm (linear_sum_assignment).
  Guarantees each speaker in a chunk gets a UNIQUE cluster.
  → Prevents the same cluster being assigned twice in one window.
  → Used by VBxClustering by default.
""")

ahc_unconstrained = AgglomerativeClustering(metric="cosine", constrained_assignment=False)
ahc_unconstrained.instantiate({"threshold": 0.5, "method": "average", "min_cluster_size": 1})

ahc_constrained = AgglomerativeClustering(metric="cosine", constrained_assignment=True)
ahc_constrained.instantiate({"threshold": 0.5, "method": "average", "min_cluster_size": 1})

hard_u, soft_u, _ = ahc_unconstrained(embeddings, segmentations=segmentations, num_clusters=3)
hard_c, soft_c, _ = ahc_constrained(  embeddings, segmentations=segmentations, num_clusters=3)

print("Unconstrained hard clusters (chunk × speaker):")
print(hard_u)
print("\nConstrained hard clusters (chunk × speaker):")
print(hard_c)

# Check for duplicate cluster assignments in each chunk
print("\nDuplicate cluster assignments per chunk:")
for c in range(NUM_CHUNKS):
    row_u = hard_u[c]
    row_c = hard_c[c]
    dup_u = len(row_u) != len(set(row_u))
    dup_c = len(row_c) != len(set(row_c))
    print(f"  chunk={c}  unconstrained duplicates={dup_u}  constrained duplicates={dup_c}")


# ==================================================================
# FEATURE 2: set_num_clusters() — bounds validation
# ==================================================================
print("\n" + "=" * 60)
print("FEATURE 2: set_num_clusters() — Bounds Clamping")
print("=" * 60)
print("""
This utility method makes sure the requested K is sensible:
  - Clamps min/max to [1, num_embeddings]
  - If min == max → that becomes the fixed K
  - If num_clusters is provided → overrides min and max

It prevents edge cases like "give me 100 speakers from 3 embeddings".
""")

ahc = AgglomerativeClustering()

examples = [
    dict(num_embeddings=10, num_clusters=None, min_clusters=2, max_clusters=5),
    dict(num_embeddings=10, num_clusters=4,    min_clusters=2, max_clusters=8),
    dict(num_embeddings=3,  num_clusters=None, min_clusters=1, max_clusters=100),
    dict(num_embeddings=10, num_clusters=None, min_clusters=6, max_clusters=6),
]

print(f"  {'num_emb':>7}  {'num_k':>5}  {'min_k':>5}  {'max_k':>5}  →  result")
for ex in examples:
    num_k, min_k, max_k = ahc.set_num_clusters(**ex)
    print(f"  {ex['num_embeddings']:>7}  "
          f"{str(ex['num_clusters']):>5}  "
          f"{str(ex['min_clusters']):>5}  "
          f"{str(ex['max_clusters']):>5}  →  "
          f"num_clusters={num_k}, min={min_k}, max={max_k}")


# ==================================================================
# FEATURE 3: The Clustering Enum — look up algorithms by name
# ==================================================================
print("\n" + "=" * 60)
print("FEATURE 3: The Clustering Enum Registry")
print("=" * 60)
print("""
SpeakerDiarization uses Clustering[name] to instantiate algorithms
from a string parameter. This is how the diarization pipeline lets
you swap algorithms via a single config string.
""")

print("Available algorithms in the Clustering enum:")
for name, member in Clustering.__members__.items():
    cls = member.value
    needs_k = cls.expects_num_clusters
    print(f"  Clustering['{name}']  →  {cls.__name__}  (expects_num_clusters={needs_k})")

# Instantiate from string — exactly how SpeakerDiarization does it
algorithm_name = "AgglomerativeClustering"
ClusteringClass = Clustering[algorithm_name].value
instance = ClusteringClass(metric="cosine")
print(f"\nInstantiated '{algorithm_name}' from enum: {type(instance).__name__}")

# Error handling for unknown names
try:
    _ = Clustering["UnknownMethod"]
except KeyError:
    valid = list(Clustering.__members__.keys())
    print(f"\nKeyError for unknown name → valid options: {valid}")


# ==================================================================
# FEATURE 4: min_cluster_size — absorbing tiny clusters
# ==================================================================
print("\n" + "=" * 60)
print("FEATURE 4: min_cluster_size — Absorbing Tiny Clusters")
print("=" * 60)
print("""
After agglomerative clustering, some clusters may have very few members.
min_cluster_size sets the minimum. Clusters below it get merged into the
nearest large cluster (by centroid distance).

This prevents spurious micro-clusters from noise or brief sounds.
""")

# Create data where one embedding is clearly an outlier
noisy_embeddings = embeddings.copy()
noisy_embeddings[0, 0] = np.random.randn(DIM) * 2   # outlier

for min_size in [1, 2, 3]:
    ahc_test = AgglomerativeClustering(metric="cosine")
    ahc_test.instantiate({"threshold": 0.3, "method": "average", "min_cluster_size": min_size})
    h, _, _ = ahc_test(noisy_embeddings, segmentations=segmentations, min_clusters=1, max_clusters=10)
    n = len(np.unique(h[h >= 0]))
    print(f"  min_cluster_size={min_size}  →  {n} cluster(s) found")


# ==================================================================
# FEATURE 5: expects_num_clusters flag
# ==================================================================
print("\n" + "=" * 60)
print("FEATURE 5: expects_num_clusters Flag")
print("=" * 60)
print("""
SpeakerDiarization checks this flag to know if the algorithm REQUIRES
you to provide num_speakers. If True and num_speakers is None, the
pipeline raises a ValueError before even starting.
""")

for name, member in Clustering.__members__.items():
    cls = member.value
    print(f"  {cls.__name__:<35}  expects_num_clusters = {cls.expects_num_clusters}")


# ==================================================================
# FEATURE 6: Reading soft_clusters — confidence scores
# ==================================================================
print("\n" + "=" * 60)
print("FEATURE 6: Soft Clusters — Confidence Scores")
print("=" * 60)
print("""
soft_clusters[chunk, speaker, k] = 2 - cosine_distance(embedding, centroid_k)

Range interpretation:
  ~2.0  → embedding is nearly identical to centroid k (very confident)
  ~1.0  → moderate similarity
  ~0.0  → very different from centroid k

The highest value across k gives the hard cluster assignment.
NaN embeddings give low soft scores (treated as confidence = 0).
""")

ahc_final = AgglomerativeClustering(metric="cosine")
ahc_final.instantiate({"threshold": 0.5, "method": "average", "min_cluster_size": 1})
hard, soft, cents = ahc_final(embeddings, segmentations=segmentations, num_clusters=3)

print(f"\nSoft cluster shape: {soft.shape}  (chunks × speakers × clusters)\n")
print("Chunk 0 soft scores (speaker × cluster):")
print(np.round(soft[0], 4))
print("\nChunk 0 hard assignments (argmax over clusters):")
print(hard[0])
print("\nChunk 0 winning confidence per speaker:")
for s in range(NUM_SPEAKERS):
    k    = hard[0, s]
    conf = soft[0, s, k] if k >= 0 else float("nan")
    print(f"  speaker={s}  →  cluster={k}  confidence={conf:.4f}")
```
