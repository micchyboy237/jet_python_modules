import os
import tempfile
from pathlib import Path
from typing import List, Literal, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from jet.audio.speech.speechbrain.speech_types import SpeechSegment
from jet.audio.utils import convert_audio_to_tensor, load_audio
from rich.console import Console
from speechbrain.inference.VAD import VAD

console = Console()


def _load_speechbrain_vad() -> VAD:
    """Lazily load the SpeechBrain CRDNN VAD model."""
    with console.status("[bold green]Loading SpeechBrain VAD model...[/bold green]"):
        vad = VAD.from_hparams(
            source="speechbrain/vad-crdnn-libriparty",
            savedir="pretrained_models/vad-crdnn-libriparty",
        )
    console.print("✅ SpeechBrain VAD model ready")
    return vad


vad = _load_speechbrain_vad()


@torch.no_grad()
def extract_speech_timestamps(
    audio: Union[str, Path, np.ndarray, torch.Tensor, list[np.ndarray]],
    threshold: float = 0.5,
    neg_threshold: float = 0.25,
    sampling_rate: int = 16000,
    max_speech_duration_sec: float | None = None,
    return_seconds: bool = False,
    time_resolution: int = 2,
    with_scores: bool = False,
    normalize_loudness: bool = False,
    include_non_speech: bool = False,
    large_chunk_size: int = 30,
    small_chunk_size: int = 10,
    double_check: bool = True,
) -> Union[List[SpeechSegment], tuple[List[SpeechSegment], List[float]]]:
    """
    Extract speech timestamps using SpeechBrain VAD (vad-crdnn-libriparty).
    When include_non_speech=True, returns both speech and non-speech (silence) segments.
    """

    if max_speech_duration_sec is None:
        max_speech_duration_sec = 15.0

    if isinstance(audio, list) and all(isinstance(x, np.ndarray) for x in audio):
        audio = convert_audio_to_tensor(audio)

    audio_np, sr = load_audio(
        audio,
        sr=sampling_rate,
        # normalize_loudness=normalize_loudness,
    )
    waveform = torch.from_numpy(audio_np).unsqueeze(0).clamp(-1.0, 1.0)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    torchaudio.save(tmp.name, waveform, sampling_rate)

    temp_path = tmp.name

    try:
        with console.status(
            "[bold blue]Running SpeechBrain VAD inference...[/bold blue]"
        ):
            boundaries_sec = vad.get_speech_segments(
                temp_path,
                large_chunk_size=large_chunk_size,
                small_chunk_size=small_chunk_size,
                activation_th=threshold,
                deactivation_th=neg_threshold,
                double_check=double_check,
            )

        boundaries_sec = boundaries_sec.view(-1).tolist()
        speech_pairs = list(zip(boundaries_sec[::2], boundaries_sec[1::2]))

        prob_tensor = vad.get_speech_prob_file(
            temp_path,
            large_chunk_size=large_chunk_size,
            small_chunk_size=small_chunk_size,
        )
        probs = prob_tensor.squeeze().cpu().tolist()

        hop_samples = 160
        hop_sec = hop_samples / sr

        def make_segment(
            num: int,
            start_sec: float,
            end_sec: float,
            seg_type: Literal["speech", "non-speech"],
        ) -> SpeechSegment:
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            frame_start = int(start_sec / hop_sec)
            frame_end = int(end_sec / hop_sec)
            segment_probs_slice = probs[frame_start : frame_end + 1]
            avg_prob = np.mean(segment_probs_slice) if segment_probs_slice else 0.0
            duration_sec = end_sec - start_sec

            start_val = start_sec if return_seconds else start_sample
            end_val = end_sec if return_seconds else end_sample

            return SpeechSegment(
                num=num,
                start=start_val,
                end=end_val,
                prob=avg_prob,
                duration=duration_sec,
                frames_length=len(segment_probs_slice),
                frame_start=frame_start,
                frame_end=frame_end,
                type=seg_type,
                segment_probs=segment_probs_slice if with_scores else [],
            )

        def split_long_speech_segment(
            seg: SpeechSegment,
            temp_audio_path: str,
            sr: int,
            max_dur_sec: float,
            min_silence_sec: float = 0.45,
            win_sec: float = 0.30,
            energy_th_rel: float = 0.13,
        ) -> List[SpeechSegment]:
            """
            Iteratively split long speech segments.
            Avoids recursion to prevent maximum recursion depth errors.
            """

            segments_to_process = [seg]
            final_segments: List[SpeechSegment] = []

            while segments_to_process:
                current = segments_to_process.pop(0)
                start_s = float(current["start"])
                end_s = float(current["end"])
                duration = end_s - start_s

                # If already small enough, keep it
                if duration <= max_dur_sec:
                    final_segments.append(current)
                    continue

                beg_sample = int(start_s * sr)
                num_frames = int(duration * sr)

                waveform, _ = torchaudio.load(
                    temp_audio_path,
                    frame_offset=beg_sample,
                    num_frames=num_frames,
                )

                if waveform.shape[0] > 1:
                    waveform = waveform.mean(0, keepdim=True)

                waveform = waveform.squeeze(0)

                if waveform.numel() < 100:
                    final_segments.extend(_force_split_segment(current, max_dur_sec))
                    continue

                win_samples = int(win_sec * sr)
                hop_samples = win_samples // 2

                pad_total = win_samples - hop_samples
                waveform_padded = F.pad(waveform, (0, pad_total))
                windows = waveform_padded.unfold(0, win_samples, hop_samples)

                rms = torch.sqrt((windows**2).mean(dim=1) + 1e-10)

                if rms.numel() <= 1:
                    final_segments.extend(_force_split_segment(current, max_dur_sec))
                    continue

                norm_rms = rms / (rms.max() + 1e-10)
                is_silence = norm_rms < energy_th_rel

                best_len = 0
                best_start_idx = -1
                current_len = 0

                for i, silent in enumerate(is_silence.tolist() + [False]):
                    if silent:
                        current_len += 1
                    else:
                        if current_len > best_len:
                            best_len = current_len
                            best_start_idx = i - current_len
                        current_len = 0

                silence_duration = best_len * (hop_samples / sr)

                # If no good silence, fallback to forced split
                if silence_duration < min_silence_sec or best_start_idx < 0:
                    final_segments.extend(_force_split_segment(current, max_dur_sec))
                    continue

                split_idx = best_start_idx + best_len // 2
                split_sample = beg_sample + split_idx * hop_samples
                split_sec = split_sample / sr

                # Guard: if split doesn't reduce duration meaningfully
                if abs(split_sec - start_s) < 0.05 or abs(end_s - split_sec) < 0.05:
                    final_segments.extend(_force_split_segment(current, max_dur_sec))
                    continue

                left_seg = make_segment(
                    current["num"],
                    start_s,
                    split_sec,
                    current["type"],
                )

                right_seg = make_segment(
                    current["num"] + 1,
                    split_sec,
                    end_s,
                    current["type"],
                )

                segments_to_process.append(left_seg)
                segments_to_process.append(right_seg)

            return final_segments

        def _force_split_segment(
            seg: SpeechSegment, max_dur_sec: float
        ) -> List[SpeechSegment]:
            """Fallback: hard split with small overlap when no good silence found."""
            parts = []
            cur_start = float(seg["start"])
            target_end = float(seg["end"])

            while cur_start + max_dur_sec <= target_end:  # <= to avoid tiny tail
                parts.append(
                    make_segment(
                        seg["num"],
                        cur_start,
                        round(cur_start + max_dur_sec, 3),
                        seg["type"],
                    )
                )
                cur_start += max_dur_sec - 0.25  # overlap
            parts.append(
                make_segment(seg["num"], round(cur_start, 3), target_end, seg["type"])
            )
            return parts

        # Build initial segments list
        enhanced: List[SpeechSegment] = []
        current_time = 0.0
        seg_num = 1

        if include_non_speech and speech_pairs:
            first_start = speech_pairs[0][0]
            if first_start > 0.001:
                enhanced.append(make_segment(seg_num, 0.0, first_start, "non-speech"))
                seg_num += 1
            current_time = first_start

        for start_sec, end_sec in speech_pairs:
            if include_non_speech and (start_sec > current_time + 0.01):
                enhanced.append(
                    make_segment(seg_num, current_time, start_sec, "non-speech")
                )
                seg_num += 1

            enhanced.append(make_segment(seg_num, start_sec, end_sec, "speech"))
            seg_num += 1
            current_time = end_sec

        # Split long speech segments if requested
        if max_speech_duration_sec < float("inf"):
            final_segments: List[SpeechSegment] = []
            current_num = 1
            for seg in enhanced:
                if seg["type"] != "speech":
                    seg["num"] = current_num
                    final_segments.append(seg)
                    current_num += 1
                    continue

                sub_segments = split_long_speech_segment(
                    seg, temp_path, sr, max_speech_duration_sec
                )
                for sub in sub_segments:
                    sub["num"] = current_num
                    final_segments.append(sub)
                    current_num += 1
            enhanced = final_segments

        if include_non_speech:
            total_duration = len(probs) * hop_sec
            if current_time < total_duration - 0.01:
                enhanced.append(
                    make_segment(seg_num, current_time, total_duration, "non-speech")
                )
                seg_num += 1

        if with_scores:
            return enhanced, probs
        return enhanced

    finally:
        os.remove(temp_path)


if __name__ == "__main__":
    audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav"
    console.print(f"[bold cyan]Processing:[/bold cyan] {Path(audio_file).name}")
    segments = extract_speech_timestamps(
        audio_file,
        threshold=0.3,
        neg_threshold=0.1,
        max_speech_duration_sec=45.0,  # example value
        return_seconds=True,
        time_resolution=2,
        normalize_loudness=False,
    )
    console.print(f"\n[bold green]Segments found:[/bold green] {len(segments)}\n")
    for seg in segments:
        console.print(
            f"[yellow][[/yellow] [bold white]{seg['start']:.2f}[/bold white] - [bold white]{seg['end']:.2f}[/bold white] [yellow]][/yellow] "
            f"duration=[bold magenta]{seg['duration']}s[/bold magenta] "
            f"prob=[bold cyan]{seg['prob']:.3f}[/bold cyan]"
        )
