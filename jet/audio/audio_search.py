from hashlib import sha256
from pathlib import Path
from typing import List, Dict, Any
import io  # For BytesIO in tests, but safe to add

import numpy as np
from numpy.typing import NDArray

import chromadb
import torch
import torchaudio
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor

from jet.utils.collection_utils import growing_windows

console = Console()  # noqa: F841 (used in functions)

# Load pre-trained CLAP model (audio-to-embedding)
model_name = "laion/clap-htsat-unfused"  # Modern, strong general audio embeddings
processor = ClapProcessor.from_pretrained(model_name, local_files_only=True)
model = ClapModel.from_pretrained(model_name, local_files_only=True)
model.eval()  # Inference mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

AudioSearchInput = str | Path | bytes | NDArray[np.float32] | NDArray[np.float64]

def load_audio_segment(
    audio_input: AudioSearchInput,
    start_sec: float = 0.0,
    duration_sec: float = 10.0
) -> torch.Tensor:
    """
    Load an audio segment (file path, Path object, raw bytes, or numpy array) and return waveform tensor.
    Resamples to 48kHz (CLAP requirement).
    """
    if isinstance(audio_input, bytes):
        waveform, sr = torchaudio.load(io.BytesIO(audio_input))
    elif isinstance(audio_input, np.ndarray):
        # Convert numpy array to torch tensor
        waveform = torch.from_numpy(audio_input.copy())
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # (1, samples)
        elif waveform.ndim != 2:
            raise ValueError("NumPy audio array must be 1D or 2D (channels x samples)")
        sr = 48000  # Assume 48kHz if provided as ndarray (common for CLAP pipelines)
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
        audio_input: AudioSearchInput,
        audio_name: str | None = None,
        segment_duration_sec: float | None = None,
        overlap_sec: float = 10.0,
        metadata_base: dict | None = None,
    ):
        """
        Generic method to chunk audio (path, Path, raw bytes, or numpy array) into segments and store them.

        Uses a short content hash (first 10s) in the ID to prevent duplicates from identical audio content.
        Logs whether segments were newly added or already indexed (updated via upsert).
        """

        # ── Load waveform and determine base name / metadata file ───────────────────────────────
        if isinstance(audio_input, (str, Path)):
            audio_path = Path(audio_input)
            waveform, original_sr = torchaudio.load(str(audio_path))
            base_name = audio_name or audio_path.stem
            file_for_meta = str(audio_path.resolve())
        elif isinstance(audio_input, np.ndarray):
            waveform = torch.from_numpy(audio_input.copy())
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.ndim != 2:
                raise ValueError("NumPy audio array must be 1D or 2D (channels x samples)")
            original_sr = 48000
            base_name = audio_name or "in_memory"
            file_for_meta = base_name
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
        waveform_bytes = sample_waveform.cpu().numpy().tobytes()
        content_hash = sha256(waveform_bytes).hexdigest()[:16]

        # ── Prepare segments, metadata, IDs ───────────────────────────────────────────────────
        segments = []
        metadatas = []
        ids = []

        new_segments = []
        new_metadatas = []
        new_ids = []

        if segment_duration_sec is None:
            # Whole file as single segment
            segment = load_audio_segment(audio_input, start_sec=0.0, duration_sec=total_duration_sec)
            meta = {
                "file": file_for_meta,
                "start_sec": 0.0,
                "end_sec": round(total_duration_sec, 3),
                "source_type": (
                    "file" if isinstance(audio_input, (str, Path))
                    else "ndarray" if isinstance(audio_input, np.ndarray)
                    else "bytes"
                ),
            }
            if metadata_base:
                meta.update(metadata_base)

            segment_id = f"{base_name}_{content_hash}_full"
            existing = self.collection.get(ids=[segment_id], include=[])

            if not existing["ids"]:
                new_segments.append(segment)
                new_metadatas.append(meta)
                new_ids.append(segment_id)
                console.print(
                    f"[green]Added new full segment: {base_name} "
                    f"({total_duration_sec:.2f}s, hash={content_hash})[/green]"
                )
            else:
                # console.print(
                #     f"[dim]Already present (no recompute): {base_name} "
                #     f"({total_duration_sec:.2f}s)[/dim]"
                # )
                pass

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
                meta = {
                    "file": file_for_meta,
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(start_sec + current_duration, 3),
                    "source_type": (
                        "file" if isinstance(audio_input, (str, Path))
                        else "ndarray" if isinstance(audio_input, np.ndarray)
                        else "bytes"
                    ),
                }
                if metadata_base:
                    meta.update(metadata_base)

                segment_id = f"{base_name}_{content_hash}_{start_sec:.1f}"

                existing = self.collection.get(ids=[segment_id], include=[])
                if not existing["ids"]:
                    new_segments.append(segment)
                    new_metadatas.append(meta)
                    new_ids.append(segment_id)
                    # Optional: log added chunk

                start_sec += step_sec
                pbar.update(1)

            pbar.close()

        if new_segments:
            embeddings = self._compute_embeddings(new_segments)
            self.collection.upsert(
                embeddings=embeddings,
                metadatas=new_metadatas,
                ids=new_ids
            )
            console.print(f"[green]Processed {len(new_segments)} new/updated segments[/green]")
        else:
            # console.print("[dim]All segments already present — no computation needed[/dim]")
            pass

    def search_similar(
        self,
        query_audio: "AudioSearchInput",
        localize_in_query: bool = False,
        use_growing_short_segments: bool = False,
        top_k: int = 10,
        duration_sec: float | None = None
    ) -> List[dict]:
        """
        Search by a query audio segment (file path, raw bytes, or numpy array).

        If duration_sec is None:
          - Use the actual duration of the provided audio

        If localize_in_query=True:
          - Chunk query into overlapping 10s windows (5s overlap)
          - Return best match per DB segment + query time range where it matched

        If use_growing_short_segments=True:
          - Splits query into 0.1s segments and forms growing windows from those segments, computes similarity scores per window.

        Always returns a normalized similarity score (1.0 = identical, 0.0 = completely different).
        """
        import io

        if duration_sec is None:
            # Compute actual duration for all input types
            if isinstance(query_audio, bytes):
                waveform, sr = torchaudio.load(io.BytesIO(query_audio))
            elif isinstance(query_audio, np.ndarray):
                waveform = torch.from_numpy(query_audio.copy())
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                sr = 48000
            else:
                waveform, sr = torchaudio.load(str(query_audio))

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if not (isinstance(query_audio, np.ndarray)):
                if sr != 48000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)
                    waveform = resampler(waveform)

            actual_duration = waveform.shape[1] / 48000.0
        else:
            actual_duration = duration_sec

        # GROWING SHORT SEGMENTS MODE
        if use_growing_short_segments:
            # ────────────────────────────────────────────────
            # GROWING PREFIX MODE (0.1s growing windows)
            # ────────────────────────────────────────────────
            segment_dur = 0.1
            full_waveform, sr = self._load_full_waveform(query_audio)
            if full_waveform.shape[0] > 1:
                full_waveform = full_waveform.mean(dim=0, keepdim=True)
            if sr != 48000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)
                full_waveform = resampler(full_waveform)
            total_samples = full_waveform.shape[1]
            total_duration = total_samples / 48000.0

            chunk_samples = int(segment_dur * 48000)
            chunks = []
            for start_sample in range(0, total_samples, chunk_samples):
                end_sample = min(start_sample + chunk_samples, total_samples)
                chunk = full_waveform[:, start_sample:end_sample]
                # Pad short last chunk to exactly 0.1s for consistency
                if chunk.shape[1] < chunk_samples:
                    pad = torch.zeros((1, chunk_samples - chunk.shape[1]))
                    chunk = torch.cat([chunk, pad], dim=1)
                chunks.append(chunk.squeeze(0))

            console.print(f"[yellow]Number of chunks: {len(chunks)}[/yellow]")

            # ── Fallback case: inject prefix info into the normal results structure ──
            if not chunks:
                console.print("[bold yellow]Query too short for growing segments mode[/bold yellow]")
                query_waveform = load_audio_segment(query_audio, duration_sec=actual_duration)
                query_embedding = self._compute_embeddings([query_waveform])[0]
                raw_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["metadatas", "distances"]
                )
                # Build same shape as growing case
                sorted_matches = []
                for i in range(len(raw_results["ids"][0])):
                    dist = raw_results["distances"][0][i]
                    score = 1.0 - dist
                    meta = raw_results["metadatas"][0][i]
                    sorted_matches.append({
                        "id": raw_results["ids"][0][i],
                        "file": meta["file"],
                        "start_sec": meta["start_sec"],
                        "end_sec": meta["end_sec"],
                        "score": score,
                        "prefix_scores": [score],
                        "prefix_durations_sec": [total_duration],
                    })
                results = {
                    "ids": [[m["id"] for m in sorted_matches]],
                    "metadatas": [sorted_matches],
                    "distances": [[1.0 - m["score"] for m in sorted_matches]],
                }
            else:
                # Generate growing windows of chunks (max_size=None means full prefix windows)
                growing = list(growing_windows(chunks, max_size=None))
                prefix_waveforms = []
                for window in growing:
                    # Concatenate chunks in window and pad/truncate to 10s (CLAP standard)
                    concat = torch.cat(window, dim=0)
                    target_samples = int(10.0 * 48000)
                    if concat.shape[0] > target_samples:
                        concat = concat[:target_samples]
                    else:
                        pad = torch.zeros(target_samples - concat.shape[0])
                        concat = torch.cat([concat, pad], dim=0)
                    prefix_waveforms.append(concat)

                prefix_embeddings = self._compute_embeddings(prefix_waveforms)

                # Query with all prefixes
                all_results = self.collection.query(
                    query_embeddings=prefix_embeddings,
                    n_results=top_k,
                    include=["metadatas", "distances"]
                )

                # Aggregate: per DB id, keep the highest score across all prefixes
                best_matches: Dict[str, Dict[str, Any]] = {}
                # Also track all prefix scores for each DB id (for diagnostics)
                prefix_scores_per_id: Dict[str, List[float]] = {}
                for sub_idx in range(len(all_results["ids"])):
                    for i in range(len(all_results["ids"][sub_idx])):
                        db_id = all_results["ids"][sub_idx][i]
                        dist = all_results["distances"][sub_idx][i]
                        score = 1.0 - dist
                        meta = all_results["metadatas"][sub_idx][i]
                        candidate = {
                            "id": db_id,
                            "file": meta["file"],
                            "start_sec": meta["start_sec"],
                            "end_sec": meta["end_sec"],
                            "score": score,
                        }
                        if db_id not in best_matches or score > best_matches[db_id]["score"]:
                            best_matches[db_id] = candidate
                        # Initialize list on first sight
                        if db_id not in prefix_scores_per_id:
                            prefix_scores_per_id[db_id] = [0.0] * len(prefix_embeddings)
                        # Store score at this prefix index
                        prefix_scores_per_id[db_id][sub_idx] = score

                sorted_matches = sorted(
                    best_matches.values(),
                    key=lambda x: x["score"],
                    reverse=True
                )[:top_k]

                # Attach full prefix score history and prefix durations
                prefix_durations = [
                    min((idx + 1) * 0.1, total_duration) for idx in range(len(prefix_embeddings))
                ]
                for match in sorted_matches:
                    db_id = match["id"]
                    match["prefix_scores"] = prefix_scores_per_id.get(db_id, [])
                    match["prefix_durations_sec"] = prefix_durations

                results = {
                    "ids": [[m["id"] for m in sorted_matches]],
                    "metadatas": [sorted_matches],
                    "distances": [[1.0 - m["score"] for m in sorted_matches]],
                }

        # LOCALIZATION MODE
        elif localize_in_query:
            # ────────────────────────────────────────────────
            # LOCALIZE IN QUERY MODE (10s overlapping windows)
            # ────────────────────────────────────────────────
            window_duration = 10.0
            overlap = 5.0
            step = window_duration - overlap

            full_waveform, sr = self._load_full_waveform(query_audio)
            if full_waveform.shape[0] > 1:
                full_waveform = full_waveform.mean(dim=0, keepdim=True)
            if sr != 48000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)
                full_waveform = resampler(full_waveform)

            total_samples = full_waveform.shape[1]
            total_duration = total_samples / 48000.0

            sub_waveforms = []
            sub_starts = []
            start_secs = np.arange(0.0, max(total_duration - window_duration + 1e-6, 0.0), step)
            for start_sec in start_secs:
                end_sec = min(start_sec + window_duration, total_duration)
                seg = load_audio_segment(query_audio, start_sec=start_sec, duration_sec=window_duration)
                sub_waveforms.append(seg)
                sub_starts.append(start_sec)

            if not sub_waveforms:
                # Fallback to global matching for very short queries
                console.print("[bold yellow]Query too short for localization — falling back to global search[/bold yellow]")
                query_waveform = load_audio_segment(query_audio, duration_sec=actual_duration)
                query_embedding = self._compute_embeddings([query_waveform])[0]
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["metadatas", "distances"]
                )
            else:
                sub_embeddings = self._compute_embeddings(sub_waveforms)
                all_results = self.collection.query(
                    query_embeddings=sub_embeddings,
                    n_results=top_k,
                    include=["metadatas", "distances"]
                )

                # Aggregate: best score per unique DB segment
                best_matches: Dict[str, Dict[str, Any]] = {}
                for sub_idx, sub_start in enumerate(sub_starts):
                    for i in range(len(all_results["ids"][sub_idx])):
                        db_id = all_results["ids"][sub_idx][i]
                        dist = all_results["distances"][sub_idx][i]
                        score = 1.0 - dist
                        meta = all_results["metadatas"][sub_idx][i]

                        candidate = {
                            "id": db_id,
                            "file": meta["file"],
                            "start_sec": meta["start_sec"],
                            "end_sec": meta["end_sec"],
                            "score": score,
                            "query_start_sec": sub_start,
                            "query_end_sec": sub_start + window_duration,
                        }

                        if db_id not in best_matches or score > best_matches[db_id]["score"]:
                            best_matches[db_id] = candidate

                # Sort and limit to top_k
                sorted_matches = sorted(
                    best_matches.values(),
                    key=lambda x: x["score"],
                    reverse=True
                )[:top_k]

                # Reconstruct Chroma-like structure for unified post-processing
                results = {
                    "ids": [[m["id"] for m in sorted_matches]],
                    "metadatas": [sorted_matches],
                    "distances": [[1.0 - m["score"] for m in sorted_matches]],
                }

        # NORMAL GLOBAL SEARCH
        else:
            # ────────────────────────────────────────────────
            # NORMAL GLOBAL SEARCH (single embedding of whole query)
            # ────────────────────────────────────────────────
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
            score = 1.0 - raw_distance

            entry = {
                "id": results["ids"][0][i],
                "file": results["metadatas"][0][i]["file"],
                "start_sec": results["metadatas"][0][i]["start_sec"],
                "end_sec": results["metadatas"][0][i]["end_sec"],
                "score": score,
            }
            if localize_in_query and "query_start_sec" in results["metadatas"][0][i]:
                entry["query_start_sec"] = results["metadatas"][0][i]["query_start_sec"]
                entry["query_end_sec"] = results["metadatas"][0][i]["query_end_sec"]
            # Preserve growing prefix data if present (both normal growing and fallback cases)
            meta = results["metadatas"][0][i]
            if "prefix_scores" in meta:
                entry["prefix_scores"] = meta["prefix_scores"]
                entry["prefix_durations_sec"] = meta["prefix_durations_sec"]
            formatted.append(entry)

        return formatted

    def _load_full_waveform(self, audio_input: "AudioSearchInput") -> tuple["torch.Tensor", int]:
        """Helper to load full waveform without segmenting."""
        import io
        if isinstance(audio_input, bytes):
            waveform, sr = torchaudio.load(io.BytesIO(audio_input))
        elif isinstance(audio_input, np.ndarray):
            waveform = torch.from_numpy(audio_input.copy())
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            sr = 48000
        else:
            waveform, sr = torchaudio.load(str(audio_input))
        return waveform, sr

    def print_results(self, results: List[dict]):
        if not results:
            console.print("[bold yellow]No results to display.[/bold yellow]")
            return

        table = Table(title="Most Similar Audio Segments")
        table.add_column("Rank", justify="right")
        table.add_column("ID", style="cyan")
        table.add_column("File")
        table.add_column("Time Range")
        table.add_column("Similarity", justify="right")

        for rank, res in enumerate(results, 1):
            time_range = f"{res['start_sec']:.1f}s – {res['end_sec']:.1f}s"
            table.add_row(
                str(rank),
                res["id"],
                Path(res["file"]).name if res["file"] != "<bytes>" else "<bytes>",
                time_range,
                f"{res['score']:.4f}"
            )

        console.print(table)

        # If growing short segments were used, show score progression per top match
        if results and "prefix_scores" in results[0]:
            console.print("\n[bold magenta]Score progression across growing query prefixes (0.1s steps):[/bold magenta]")
            prog_table = Table(title="Growing Prefix Confidence Curves")
            prog_table.add_column("Rank", justify="right")
            prog_table.add_column("File")
            prog_table.add_column("Prefix Duration", style="cyan")
            prog_table.add_column("Score History", style="green")

            for rank, res in enumerate(results[:5], 1):  # Show top 5 for clarity
                durations = [f"{d:.1f}s" for d in res["prefix_durations_sec"]]
                scores = [f"{s:.3f}" for s in res["prefix_scores"]]
                duration_str = " → ".join(durations)
                score_str = " → ".join(scores)
                prog_table.add_row(
                    str(rank),
                    Path(res["file"]).name if res["file"] != "<bytes>" else "<bytes>",
                    duration_str,
                    score_str,
                )
            console.print(prog_table)

        # Optional: Show deduplicated view (highest score per unique content)
        best_by_content = {}

        for r in results:
            # Primary key: perfect match + duration (most reliable for true duplicates)
            # Fallback: content hash from ID
            is_perfect = abs(r["score"] - 1.0) < 1e-6
            duration = r["end_sec"]

            primary_key = (is_perfect, duration)

            try:
                content_hash = r["id"].split("_")[1]
            except IndexError:
                content_hash = r["id"]

            key = primary_key if is_perfect else content_hash

            # Keep highest score; if equal, prefer real file path
            prefer_new = (
                key not in best_by_content or
                r["score"] > best_by_content[key]["score"] or
                (abs(r["score"] - best_by_content[key]["score"]) < 1e-6 and
                 (Path(r["file"]).is_file() and not Path(best_by_content[key]["file"]).is_file()))
            )
            if prefer_new:
                best_by_content[key] = r

        dedup_results = sorted(best_by_content.values(), key=lambda x: x["score"], reverse=True)
        if len(dedup_results) < len(results):
            console.print("\n[bold green]Deduplicated view (one per unique audio content):[/bold green]")
            # Print directly to avoid recursion and extra headers
            dedup_table = Table(title="Most Similar Audio Segments")
            dedup_table.add_column("Rank", justify="right")
            dedup_table.add_column("ID", style="cyan")
            dedup_table.add_column("File")
            dedup_table.add_column("Time Range")
            dedup_table.add_column("Similarity", justify="right")

            for rank, res in enumerate(dedup_results, 1):
                time_range = f"{res['start_sec']:.1f}s – {res['end_sec']:.1f}s"
                dedup_table.add_row(
                    str(rank),
                    res["id"],
                    Path(res["file"]).name if res["file"] != "<bytes>" else "<bytes>",
                    time_range,
                    f"{res['score']:.4f}"
                )
            console.print(dedup_table)

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
