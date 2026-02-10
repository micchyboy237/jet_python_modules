import io
from pathlib import Path

import torch
import torchaudio
from jet.utils.inspect_utils import get_entry_file_dir, get_entry_file_name
from rich.console import Console
from rich.table import Table
from speechbrain.inference import EncoderClassifier
from tqdm import tqdm

console = Console()

DATA_DIR = (
    Path(get_entry_file_dir()) / "generated" / Path(get_entry_file_name()).stem / "data"
)

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(DATA_DIR / "pretrained_ecapa_tdnn"),  # Auto-downloads on first run
)


def get_embedding(
    audio_bytes: bytes, target_sr: int = 16000, min_samples: int = 3200
) -> torch.Tensor:
    """Extract normalized embedding from raw bytes (any format torchaudio supports).

    Handles very short or empty audio by padding with zeros to a minimum length.
    This prevents convolution padding errors in ECAPA-TDNN for ultra-short clips.
    """
    waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:  # Stereo to mono
        waveform = waveform.mean(0, keepdim=True)

    # Pad short waveforms to avoid conv padding > input size errors
    if waveform.shape[1] < min_samples:
        padding = min_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    with torch.no_grad():
        embedding = classifier.encode_batch(waveform)
    return embedding.squeeze()


def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """
    Compute cosine similarity between two 1D embedding tensors.
    Both tensors are expected to be 1D (embedding_dim,).
    Returns a float in range [-1.0, 1.0].
    """
    # Ensure 1D tensors
    if emb1.dim() > 1:
        emb1 = emb1.squeeze()
    if emb2.dim() > 1:
        emb2 = emb2.squeeze()

    return torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()


def compute_similarity_matrix(audio_bytes_list: list[bytes]) -> list[list[float]]:
    embeddings = [
        get_embedding(ab) for ab in tqdm(audio_bytes_list, desc="Extracting embeddings")
    ]
    n = len(embeddings)
    sim_matrix = []
    for i in tqdm(range(n), desc="Cosine comparisons"):
        row = []
        for j in range(n):
            score = cosine_similarity(embeddings[i], embeddings[j])
            row.append(score)
        sim_matrix.append(row)
    return sim_matrix


def print_similarity_table(
    sim_matrix: list[list[float]], labels: list[str] | None = None
) -> None:
    table = Table(
        title="Audio Similarity Matrix (Cosine, -1 to 1)",
        show_header=True,
        header_style="bold magenta",
    )
    if labels is None:
        labels = [f"Audio {i + 1}" for i in range(len(sim_matrix))]

    # Add a corner cell + column headers
    table.add_column("", style="dim")  # Empty corner for row labels
    for label in labels:
        # Truncate long labels to avoid breaking the table layout
        truncated = label[-30:] if len(label) > 30 else label
        table.add_column(truncated, justify="right")

    # Add rows with row label in first column
    for i, row in enumerate(sim_matrix):
        truncated_row_label = labels[i][-30:] if len(labels[i]) > 30 else labels[i]
        table.add_row(truncated_row_label, *[f"{val:+.3f}" for val in row])

    console.print(table)


def find_similar_groups(
    sim_matrix: list[list[float]], labels: list[str], threshold: float = 0.95
) -> list[list[str]]:
    """
    Find groups of audio files that are mutually similar above the given threshold.

    Returns a list of groups (each group is a list of labels).
    Groups are sorted by size descending. Singleton groups are excluded.
    """
    n = len(sim_matrix)
    visited = [False] * n
    groups: list[list[str]] = []

    for i in range(n):
        if visited[i]:
            continue

        # Start a new group with i
        group_indices = [i]
        visited[i] = True

        # Find all j where sim(i,j) >= threshold AND sim(j,i) >= threshold
        for j in range(i + 1, n):
            if (
                not visited[j]
                and sim_matrix[i][j] >= threshold
                and sim_matrix[j][i] >= threshold
            ):
                group_indices.append(j)
                visited[j] = True

        if len(group_indices) > 1:
            group_labels = [labels[idx] for idx in group_indices]
            groups.append(sorted(group_labels))

    # Sort groups by size descending
    groups.sort(key=len, reverse=True)
    return groups


def print_similar_groups(groups: list[list[str]]) -> None:
    """Pretty-print the detected similar/duplicate audio groups."""
    if not groups:
        console.print("[green]No similar audio groups found (all unique).[/green]")
        return

    console.print(
        f"[bold green]Found {len(groups)} group(s) of similar audio files:[/bold green]"
    )
    for idx, group in enumerate(groups, 1):
        table = Table(title=f"Group {idx} ({len(group)} files)")
        table.add_column("File Label", style="cyan")
        for label in group:
            table.add_row(label)
        console.print(table)
        console.print()  # Empty line between groups
