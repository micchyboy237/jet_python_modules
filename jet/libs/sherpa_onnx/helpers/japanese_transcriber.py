"""
Reusable Japanese Speech Transcriber using Sherpa-ONNX
Default: Zipformer ReazonSpeech (2025-01-17) for best Japanese accuracy
"""

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice: pip install sounddevice")
    sys.exit(-1)

import sherpa_onnx


class JapaneseSpeechTranscriber:
    """
    Reusable class for high-quality Japanese speech transcription.
    Default model: sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17 (Zipformer)
    """

    def __init__(
        self,
        # Zipformer ReazonSpeech (default)
        encoder: Optional[str] = None,
        decoder: Optional[str] = None,
        joiner: Optional[str] = None,
        # SenseVoice (alternative)
        sense_voice: Optional[str] = None,
        tokens: Optional[str] = None,
        silero_vad_model: Optional[str] = None,
        model_type: str = "zipformer_reazon",  # "zipformer_reazon" or "sense_voice"
        num_threads: int = 4,
        sample_rate: int = 16000,
        feature_dim: int = 80,
        debug: bool = False,
    ):
        self.model_type = model_type.lower()
        self.sample_rate = sample_rate
        self.num_threads = num_threads
        self.feature_dim = feature_dim
        self.debug = debug
        self.silero_vad_model = silero_vad_model

        if not tokens:
            raise ValueError("tokens path is required")

        assert Path(tokens).is_file(), f"Tokens file not found: {tokens}"

        print(f"Creating Japanese recognizer using {self.model_type}...")

        if self.model_type == "zipformer_reazon":
            if not all([encoder, decoder, joiner]):
                raise ValueError(
                    "For zipformer_reazon you must provide encoder, decoder, and joiner paths"
                )
            self.recognizer = self._create_zipformer_recognizer(
                encoder, decoder, joiner, tokens
            )
        elif self.model_type == "sense_voice":
            if not sense_voice:
                raise ValueError(
                    "For sense_voice you must provide sense_voice model path"
                )
            self.recognizer = self._create_sensevoice_recognizer(sense_voice, tokens)
        else:
            raise ValueError("model_type must be 'zipformer_reazon' or 'sense_voice'")

        print("Recognizer created successfully!")

        if silero_vad_model:
            assert Path(silero_vad_model).is_file(), (
                f"VAD model not found: {silero_vad_model}"
            )
            self.vad = self._create_vad()
            print("VAD ready for live transcription.")
        else:
            self.vad = None

    def _create_zipformer_recognizer(self, encoder, decoder, joiner, tokens):
        assert Path(encoder).is_file(), f"Encoder not found: {encoder}"
        assert Path(decoder).is_file(), f"Decoder not found: {decoder}"
        assert Path(joiner).is_file(), f"Joiner not found: {joiner}"

        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=self.num_threads,
            sample_rate=self.sample_rate,
            feature_dim=self.feature_dim,
            decoding_method="greedy_search",
            debug=self.debug,
        )

    def _create_sensevoice_recognizer(self, sense_voice, tokens):
        assert Path(sense_voice).is_file(), f"SenseVoice model not found: {sense_voice}"

        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=sense_voice,
            tokens=tokens,
            num_threads=self.num_threads,
            use_itn=True,
            debug=self.debug,
        )

    def _create_vad(self):
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = self.silero_vad_model
        config.silero_vad.min_silence_duration = 0.25
        config.sample_rate = self.sample_rate

        return sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=100)

    def transcribe_japanese(
        self,
        audio_file: Optional[str] = None,
        max_duration: Optional[float] = None,
    ) -> str:
        """
        Main method to transcribe Japanese speech.

        Args:
            audio_file: Path to audio file. If None → live microphone with VAD.
            max_duration: Max seconds for live recording.

        Returns:
            Transcribed text (Japanese)
        """
        if audio_file:
            return self._transcribe_file(audio_file)
        else:
            if not self.vad:
                raise RuntimeError(
                    "silero_vad_model is required for live microphone transcription."
                )
            return self._transcribe_live(max_duration)

    def _transcribe_file(self, audio_file: str) -> str:
        assert Path(audio_file).is_file(), f"Audio file not found: {audio_file}"

        stream = self.recognizer.create_stream()
        samples, sr = sherpa_onnx.read_audio_file(audio_file)

        if sr != self.sample_rate:
            print(f"Warning: Input sample rate {sr}Hz, expected {self.sample_rate}Hz")

        stream.accept_waveform(self.sample_rate, samples)
        self.recognizer.decode_stream(stream)

        text = stream.result.text.strip()
        print(f"File transcription: {text}")
        return text

    def _transcribe_live(self, max_duration: Optional[float] = None) -> str:
        print("🎤 Live Japanese transcription started. Speak now! (Ctrl+C to stop)")

        buffer = []
        texts: List[str] = []
        window_size = self.vad.config.silero_vad.window_size
        samples_per_read = int(0.1 * self.sample_rate)

        try:
            with sd.InputStream(
                channels=1, dtype="float32", samplerate=self.sample_rate
            ) as s:
                while True:
                    samples, _ = s.read(samples_per_read)
                    samples = samples.reshape(-1)
                    buffer = np.concatenate([buffer, samples])

                    while len(buffer) > window_size:
                        self.vad.accept_waveform(buffer[:window_size])
                        buffer = buffer[window_size:]

                    while not self.vad.empty():
                        segment = self.vad.front.samples
                        stream = self.recognizer.create_stream()
                        stream.accept_waveform(self.sample_rate, segment)
                        self.vad.pop()

                        self.recognizer.decode_stream(stream)
                        text = stream.result.text.strip()

                        if text:
                            texts.append(text)
                            print(f"→ {text}")

                    if (
                        max_duration and len(texts) > 0
                    ):  # simple timeout logic can be enhanced
                        pass

        except KeyboardInterrupt:
            print("\nStopped by user.")

        final_text = " ".join(texts).strip()
        print(f"\nFinal Japanese transcription:\n{final_text}")
        return final_text


# Quick demo when running directly
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Japanese Speech Transcriber")
    parser.add_argument(
        "--model-type",
        choices=["zipformer_reazon", "sense_voice"],
        default="zipformer_reazon",
    )
    parser.add_argument("--encoder", help="Path to encoder.onnx (for zipformer)")
    parser.add_argument("--decoder", help="Path to decoder.onnx (for zipformer)")
    parser.add_argument("--joiner", help="Path to joiner.onnx (for zipformer)")
    parser.add_argument("--sense-voice", help="Path to SenseVoice model.onnx")
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--silero-vad-model", help="Path to silero_vad.onnx")
    parser.add_argument("--audio-file", help="Transcribe from file instead of mic")
    parser.add_argument("--num-threads", type=int, default=4)

    args = parser.parse_args()

    transcriber = JapaneseSpeechTranscriber(
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        sense_voice=args.sense_voice,
        tokens=args.tokens,
        silero_vad_model=args.silero_vad_model,
        model_type=args.model_type,
        num_threads=args.num_threads,
    )

    if args.audio_file:
        transcriber.transcribe_japanese(audio_file=args.audio_file)
    else:
        transcriber.transcribe_japanese()
