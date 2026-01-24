def audio_buffer_duration(
    buffer: bytes | bytearray | None,
    sample_rate: int,           # ← still required, no default
) -> float:
    """
    Calculate duration of PCM audio buffer (assumes 16-bit integer samples).
    
    Currently hard-coded to 2 bytes per sample (int16).
    If you ever need to support other formats, add bytes_per_sample parameter back.
    """
    if not buffer:
        return 0.0
    
    BYTES_PER_SAMPLE = 2
    num_samples = len(buffer) // BYTES_PER_SAMPLE
    return num_samples / sample_rate