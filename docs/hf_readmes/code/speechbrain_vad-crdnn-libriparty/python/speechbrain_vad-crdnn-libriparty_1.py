import os
import tempfile

import torchaudio
from rich.console import Console
from rich.table import Table
from speechbrain.inference.VAD import VAD

audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_2_speakers_short.wav"

# Load the pre-trained VAD model (load once)
vad = VAD.from_hparams(
    source="speechbrain/vad-crdnn-libriparty",
    savedir="pretrained_models/vad-crdnn-libriparty",
    # run_opts={"device": "cuda"}  # uncomment for GPU on GTX 1660
)

# Load original audio
waveform, sample_rate = torchaudio.load(audio_file)

# Convert to mono if needed (average channels)
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Resample to 16kHz if not already (VAD model trained @16kHz)
target_sr = 16000
if sample_rate != target_sr:
    resampler = torchaudio.transforms.Resample(
        orig_freq=sample_rate, new_freq=target_sr
    )
    waveform = resampler(waveform)
    sample_rate = target_sr

# Create temporary mono file (auto-deleted on context exit)
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
    tmp_path = tmp_file.name
    torchaudio.save(tmp_path, waveform, sample_rate)

    # Now run VAD on the mono temp file
    boundaries = vad.get_speech_segments(tmp_path)

# Clean up temp file explicitly (though delete=False + os.remove is safe)
os.remove(tmp_path)

# Pretty-print results with rich
console = Console()
table = Table(title="Detected Speech Segments")
table.add_column("Segment", justify="right", style="cyan")
table.add_column("Start (s)", justify="right", style="green")
table.add_column("End (s)", justify="right", style="green")
table.add_column("Duration (s)", justify="right", style="magenta")

for i, (start, end) in enumerate(boundaries, 1):
    duration = end - start
    table.add_row(f"{i:03d}", f"{start:.2f}", f"{end:.2f}", f"{duration:.2f}")

console.print(table)

# Save boundaries to text file
vad.save_boundaries(boundaries, save_path="VAD_file.txt")
console.print("\n[bold]Boundaries saved to:[/bold] VAD_file.txt")
