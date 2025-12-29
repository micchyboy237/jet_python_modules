from hashlib import sha256
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

console = Console()  # noqa: F841 (used in functions)

# Load pre-trained CLAP model (audio-to-embedding)
model_name = "laion/clap-htsat-unfused"  # Modern, strong general audio embeddings
processor = ClapProcessor.from_pretrained(model_name, local_files_only=True)
model = ClapModel.from_pretrained(model_name, local_files_only=True)
model.eval()  # Inference mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def load_audio_segment(
    audio_input: str | Path | bytes,
    start_sec: float = 0.0,
    duration_sec: float = 10.0
) -> torch.Tensor:
    """
    Load an audio segment (file path, Path object or raw bytes) and return waveform tensor.
    Resamples to 48kHz (CLAP requirement).
    """
    if isinstance(audio_input, bytes):
        waveform, sr = torchaudio.load(io.BytesIO(audio_input))
    else:
        waveform, sr = torchaudio.load(str(audio_input))   # Path → str, str stays str

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 48000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)
        waveform = resampler(waveform)

    start_sample = int(start_sec * 48000)
    end_sample = int((start_sec + duration_sec) * 48000)
    segment = waveform[:, start_sample:end_sample]

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
            metadata={"hnsw:space": "cosine"}
        )  # noqa: E501

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

    def add_segments(
        self,
        audio_input: str | Path | bytes,
        audio_name: str | None = None,
        segment_duration_sec: float | None = None,
        overlap_sec: float = 10.0,
        metadata_base: dict | None = None,
    ):
        """
        Generic method to chunk audio (path, Path or raw bytes) into segments and store them.

        Uses a short content hash (first 10s) in the ID to prevent duplicates from identical audio content.
        Logs whether segments were newly added or already indexed (updated via upsert).
        """

        # ── Load waveform and determine base name / metadata file ───────────────────────────────
        if isinstance(audio_input, (str, Path)):
            audio_path = Path(audio_input)
            waveform, original_sr = torchaudio.load(str(audio_path))
            base_name = audio_name or audio_path.stem
            file_for_meta = str(audio_path.resolve())
        else:
            waveform_io = io.BytesIO(audio_input)
            waveform, original_sr = torchaudio.load(waveform_io)
            base_name = audio_name or "in_memory"
            file_for_meta = base_name

        # ── Mono + resample ───────────────────────────────────────────────────────────────
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        total_samples = waveform.shape[1]
        total_duration_sec = total_samples / original_sr

        if original_sr != 48000:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=48000)
            waveform = resampler(waveform)
            total_duration_sec = waveform.shape[1] / 48000.0

        # ── Compute short content hash for deduplication ──────────────────────────────────────
        sample_duration = min(10.0, total_duration_sec)
        sample_waveform = load_audio_segment(
            audio_input,
            start_sec=0.0,
            duration_sec=sample_duration
        )
        # Fix: convert to numpy before .tobytes()
        waveform_bytes = sample_waveform.cpu().numpy().tobytes()
        content_hash = sha256(waveform_bytes).hexdigest()[:16]

        # ── Prepare segments, metadata, IDs ───────────────────────────────────────────────────
        segments = []
        metadatas = []
        ids = []

        if segment_duration_sec is None:
            # Whole file as single segment
            segment = load_audio_segment(audio_input, start_sec=0.0, duration_sec=total_duration_sec)
            segments.append(segment)

            meta = {
                "file": file_for_meta,
                "start_sec": 0.0,
                "end_sec": round(total_duration_sec, 3),
                "source_type": "file" if isinstance(audio_input, (str, Path)) else "bytes",
            }
            if metadata_base:
                meta.update(metadata_base)
            metadatas.append(meta)

            segment_id = f"{base_name}_{content_hash}_full"
            ids.append(segment_id)

            # Existence check + smart logging
            existing = self.collection.get(ids=[segment_id], include=[])
            if existing["ids"]:
                console.print(
                    f"[dim]Already indexed (updated): {base_name} "
                    f"({total_duration_sec:.2f}s, hash={content_hash})[/dim]"
                )
            else:
                console.print(
                    f"[green]Added new full segment: {base_name} "
                    f"({total_duration_sec:.2f}s, hash={content_hash})[/green]"
                )

        else:
            # Chunked mode
            if overlap_sec >= segment_duration_sec:
                raise ValueError("overlap_sec must be less than segment_duration_sec when chunking")

            step_sec = segment_duration_sec - overlap_sec
            start_sec = 0.0
            pbar = tqdm(desc=f"Chunking {base_name}")

            while start_sec < total_duration_sec:
                current_duration = min(segment_duration_sec, total_duration_sec - start_sec)
                segment = load_audio_segment(
                    audio_input,
                    start_sec=start_sec,
                    duration_sec=current_duration
                )
                segments.append(segment)

                meta = {
                    "file": file_for_meta,
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(start_sec + current_duration, 3),
                    "source_type": "file" if isinstance(audio_input, (str, Path)) else "bytes",
                }
                if metadata_base:
                    meta.update(metadata_base)
                metadatas.append(meta)

                segment_id = f"{base_name}_{content_hash}_{start_sec:.1f}"
                ids.append(segment_id)

                start_sec += step_sec
                pbar.update(1)

            pbar.close()
            console.print(f"[green]Processed {len(segments)} chunks for {base_name} (hash={content_hash})[/green]")

        # ── Upsert to collection ───────────────────────────────────────────────────────────────
        if segments:
            embeddings = self._compute_embeddings(segments)
            self.collection.upsert(
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

    def search_similar(
        self,
        query_audio: str | bytes,
        top_k: int = 10,
        duration_sec: float | None = None
    ) -> List[dict]:
        """
        Search by a query audio segment (file path or raw bytes).

        If duration_sec is None:
          - For file path: use the actual full duration of the file
          - For bytes: fallback to 10.0 seconds

        Always returns a normalized similarity score (1.0 = identical, 0.0 = completely different).
        """
        # Determine actual duration to use
        if duration_sec is None:
            if isinstance(query_audio, str):
                waveform, sr = torchaudio.load(query_audio)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if sr != 48000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)
                    waveform = resampler(waveform)
                actual_duration = waveform.shape[1] / 48000.0
            else:
                actual_duration = 10.0
        else:
            actual_duration = duration_sec

        query_waveform = load_audio_segment(query_audio, duration_sec=actual_duration)

        query_embedding = self._compute_embeddings([query_waveform])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"]
        )

        if not results["ids"][0]:
            console.print("[bold red]No similar segments found (database empty or no matches).[/bold red]")
            return []

        actual_results = len(results["ids"][0])

        formatted = []
        for i in range(actual_results):
            raw_distance = results["distances"][0][i]
            score = 1.0 - raw_distance  # Normalized similarity score

            formatted.append({
                "id": results["ids"][0][i],
                "file": results["metadatas"][0][i]["file"],
                "start_sec": results["metadatas"][0][i]["start_sec"],
                "end_sec": results["metadatas"][0][i]["end_sec"],
                "score": score
            })

        return formatted

    def print_results(self, results: List[dict]):
        if not results:
            console.print("[bold yellow]No results to display.[/bold yellow]")
            return

        table = Table(title="Most Similar Audio Segments")
        table.add_column("Rank", justify="right")
        table.add_column("ID", style="cyan")           # ← new column
        table.add_column("File")
        table.add_column("Time Range")
        table.add_column("Similarity", justify="right")

        for rank, res in enumerate(results, 1):
            time_range = f"{res['start_sec']:.1f}s – {res['end_sec']:.1f}s"
            table.add_row(
                str(rank),
                res["id"],                             # ← added
                Path(res["file"]).name if res["file"] != "<bytes>" else "<bytes>",
                time_range,
                f"{res['score']:.4f}"
            )

        console.print(table)

# Usage Examples

if __name__ == "__main__":
    db = AudioSegmentDatabase(persist_dir="./my_audio_db")

    # Example 1: Index some audio files (run once)
    audio_files = ["path/to/song1.wav", "path/to/song2.mp3"]  # Add your files
    for file in audio_files:
        db.add_segments(file)

    # Example 2: Search with a query file
    query_path = "path/to/query_segment.wav"
    results = db.search_similar(query_path, top_k=10)
    db.print_results(results)

    # Example 3: Search with raw audio bytes (e.g., from API upload)
    with open("path/to/query_segment.wav", "rb") as f:
        query_bytes = f.read()

    results_bytes = db.search_similar(query_bytes, top_k=5)
    db.print_results(results_bytes)
