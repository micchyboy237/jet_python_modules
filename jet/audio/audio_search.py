from pathlib import Path
from typing import List
import io  # For BytesIO in tests, but safe to add

import chromadb
import torch
import torchaudio
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor

console = Console()

# Load pre-trained CLAP model (audio-to-embedding)
model_name = "laion/clap-htsat-unfused"  # Modern, strong general audio embeddings
processor = ClapProcessor.from_pretrained(model_name, local_files_only=True)
model = ClapModel.from_pretrained(model_name, local_files_only=True)
model.eval()  # Inference mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def load_audio_segment(audio_path: str | bytes, start_sec: float = 0.0, duration_sec: float = 10.0) -> torch.Tensor:
    """
    Load an audio segment (file path or raw bytes) and return waveform tensor.
    Resamples to 48kHz (CLAP requirement).
    The default duration is 10s, with padding as needed for short clips.
    """
    if isinstance(audio_path, bytes):
        waveform, sr = torchaudio.load(io.BytesIO(audio_path))  # Use io.BytesIO for bytes input
    else:
        waveform, sr = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != 48000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)
        waveform = resampler(waveform)

    # Extract segment
    start_sample = int(start_sec * 48000)
    end_sample = int((start_sec + duration_sec) * 48000)
    segment = waveform[:, start_sample:end_sample]

    # Pad if too short (use updated default duration for expected short queries)
    if segment.shape[1] < 48000 * duration_sec:
        pad_length = int(48000 * duration_sec - segment.shape[1])
        pad = torch.zeros((1, pad_length))
        segment = torch.cat([segment, pad], dim=1)

    return segment.squeeze(0)

class AudioSegmentDatabase:
    """Generic, reusable class for storing and searching audio segments by content similarity."""

    def __init__(self, persist_dir: str = "./audio_vector_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="audio_segments",
            metadata={"hnsw:space": "cosine"}  # No embedding_function
        )

    def _compute_embeddings(self, audios: List[torch.Tensor]) -> List[List[float]]:
        console.print("[yellow][DEBUG] Computing embeddings for {} audio segments[/yellow]".format(len(audios)))
        audios_np = [a.cpu().numpy() for a in audios]
        console.print("[yellow][DEBUG] Audio types: {}[/yellow]".format(type(audios_np[0])))
        with torch.no_grad():
            inputs = processor(audios=audios_np, return_tensors="pt", sampling_rate=48000).to(device)
            console.print("[yellow][DEBUG] Inputs keys: {}[/yellow]".format(list(inputs.keys())))
            embeddings = model.get_audio_features(**inputs)
            console.print("[yellow][DEBUG] Embeddings shape: {}[/yellow]".format(embeddings.shape))
            return embeddings.cpu().tolist()

    def add_segments_from_file(
        self,
        audio_path: str,
        segment_duration_sec: float = 30.0,
        overlap_sec: float = 10.0,
        metadata_base: dict | None = None
    ):
        """
        Chunk a long audio file into overlapping segments, embed, and store.
        """
        waveform, sr = torchaudio.load(audio_path)
        total_samples = waveform.shape[1]
        sample_rate = 48000  # After potential resample in load_audio_segment

        step = int((segment_duration_sec - overlap_sec) * sample_rate)
        segments = []
        metadatas = []
        ids = []

        start_sec = 0.0
        pbar = tqdm(desc=f"Chunking {Path(audio_path).name}")
        while start_sec * sample_rate < total_samples:
            segment = load_audio_segment(audio_path, start_sec=start_sec, duration_sec=segment_duration_sec)
            segments.append(segment)

            meta = {"file": audio_path, "start_sec": start_sec, "end_sec": start_sec + segment_duration_sec}
            if metadata_base:
                meta.update(metadata_base)
            metadatas.append(meta)

            ids.append(f"{Path(audio_path).stem}_{start_sec:.1f}")

            start_sec += (segment_duration_sec - overlap_sec)
            pbar.update(1)

        pbar.close()

        embeddings = self._compute_embeddings(segments)

        self.collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        console.print(f"[green]Added {len(segments)} segments from {audio_path}[/green]")

    def search_similar(
        self,
        query_audio: str | bytes,
        top_k: int = 10,
        duration_sec: float = 30.0
    ) -> List[dict]:
        """
        Search by a query audio segment (file path or raw bytes).
        Returns list of results with metadata and distance.
        Handles empty collection gracefully.
        """
        query_waveform = load_audio_segment(query_audio, duration_sec=duration_sec)

        query_embedding = self._compute_embeddings([query_waveform])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"]  # Removed "ids" – IDs are always returned by default
        )

        # Handle empty collection or no results
        if not results["ids"][0]:
            console.print("[bold red]No similar segments found (database empty or no matches).[/bold red]")
            return []

        actual_results = len(results["ids"][0])

        formatted = []
        for i in range(actual_results):
            formatted.append({
                "id": results["ids"][0][i],
                "file": results["metadatas"][0][i]["file"],
                "start_sec": results["metadatas"][0][i]["start_sec"],
                "end_sec": results["metadatas"][0][i]["end_sec"],
                "distance": results["distances"][0][i]
            })

        return formatted

    def print_results(self, results: List[dict]):
        if not results:
            console.print("[bold yellow]No results to display.[/bold yellow]")
            return

        table = Table(title="Similar Audio Segments")
        table.add_column("Rank")
        table.add_column("File")
        table.add_column("Time Range")
        table.add_column("Distance")

        for rank, res in enumerate(results, 1):
            time_range = f"{res['start_sec']:.1f}s - {res['end_sec']:.1f}s"
            table.add_row(str(rank), Path(res["file"]).name, time_range, f"{res['distance']:.4f}")

        console.print(table)

# Usage Examples

if __name__ == "__main__":
    db = AudioSegmentDatabase(persist_dir="./my_audio_db")

    # Example 1: Index some audio files (run once)
    audio_files = ["path/to/song1.wav", "path/to/song2.mp3"]  # Add your files
    for file in audio_files:
        db.add_segments_from_file(file, segment_duration_sec=30.0, overlap_sec=10.0)

    # Example 2: Search with a query file
    query_path = "path/to/query_segment.wav"
    results = db.search_similar(query_path, top_k=10)
    db.print_results(results)

    # Example 3: Search with raw audio bytes (e.g., from API upload)
    with open("path/to/query_segment.wav", "rb") as f:
        query_bytes = f.read()

    results_bytes = db.search_similar(query_bytes, top_k=5)
    db.print_results(results_bytes)