# example_client_usage.py
"""
Example client-side usage for the new batch transcription endpoints:

    POST /batch/transcribe          → transcription only
    POST /batch/transcribe_translate → transcription + translation to English

Both endpoints accept multipart/form-data with any number of audio files
under any field names (file, audio, data, upload, etc.).

Tested with Python 3.10+ and `requests`.
"""

from __future__ import annotations

import pathlib
from typing import List

import requests

from jet.audio.utils import resolve_audio_paths

BASE_URL = "http://shawn-pc.local:8001"  # adjust if your server runs on a different host/port


def batch_transcribe(
    audio_paths: List[str | pathlib.Path],
    translate: bool = False,
) -> requests.Response:
    """
    Send multiple audio files for batch processing.

    Args:
        audio_paths: List of local paths to audio files (any format supported by ffmpeg).
        translate: If True, uses /batch/transcribe_translate (includes English translation).

    Returns:
        requests.Response object (JSON list of TranscriptionResponse objects on success).
    """
    endpoint = "/batch/transcribe_translate" if translate else "/batch/transcribe"
    url = BASE_URL + endpoint

    # Open all files in binary mode and let requests handle multipart encoding
    files = []
    for path in audio_paths:
        path_obj = pathlib.Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        files.append(
            ("files", (path_obj.name, open(path_obj, "rb"), "audio/wav"))  # field name can be anything
        )

    print(f"Sending {len(files)} file(s) to {url} ...")
    response = requests.post(url, files=files)

    # Close file handles
    for _, (_, file_handle, _) in files:
        file_handle.close()

    return response


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    audio_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/speech/generated/run_analyze_speech/raw_segments"
    sample_audio_files = resolve_audio_paths(audio_dir, recursive=True)

    # 1. Batch transcription only
    resp1 = batch_transcribe(sample_audio_files, translate=False)
    print("\n--- /batch/transcribe response ---")
    print(f"Status: {resp1.status_code}")
    if resp1.ok:
        for i, item in enumerate(resp1.json(), 1):
            print(f"\nFile {i}:")
            print(f"  Duration:     {item['duration_sec']}s")
            print(f"  Language:     {item['detected_language']} (prob: {item['detected_language_prob']})")
            print(f"  Transcription:\n    {item['transcription'][:200]}{'...' if len(item['transcription']) > 200 else ''}")
    else:
        print(resp1.text)

    # 2. Batch transcription + translation
    resp2 = batch_transcribe(sample_audio_files, translate=True)
    print("\n--- /batch/transcribe_translate response ---")
    print(f"Status: {resp2.status_code}")
    if resp2.ok:
        for i, item in enumerate(resp2.json(), 1):
            print(f"\nFile {i}:")
            print(f"  Duration:     {item['duration_sec']}s")
            print(f"  Language:     {item['detected_language']} (prob: {item['detected_language_prob']})")
            print(f"  Transcription:\n    {item['transcription'][:200]}{'...' if len(item['transcription']) > 200 else ''}")
            if item["translation"]:
                print(f"  Translation:\n    {item['translation'][:200]}{'...' if len(item['translation']) > 200 else ''}")
    else:
        print(resp2.text)