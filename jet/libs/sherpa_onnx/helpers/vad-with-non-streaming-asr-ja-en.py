"""
Japanese Live Speech Recognition Demo
Uses JapaneseSpeechTranscriber class (Zipformer ReazonSpeech by default)
"""

import argparse
import sys

from jet.libs.sherpa_onnx.helpers.japanese_transcriber import JapaneseSpeechTranscriber


def get_args():
    parser = argparse.ArgumentParser(
        description="Japanese Speech Recognition using Sherpa-ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required for live transcription
    parser.add_argument(
        "--silero-vad-model",
        type=str,
        required=True,
        help="Path to silero_vad.onnx",
    )

    # Zipformer ReazonSpeech model (default)
    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Path to tokens.txt",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="",
        help="Path to encoder-*.int8.onnx (for zipformer_reazon)",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="",
        help="Path to decoder-*.int8.onnx (for zipformer_reazon)",
    )
    parser.add_argument(
        "--joiner",
        type=str,
        default="",
        help="Path to joiner-*.int8.onnx (for zipformer_reazon)",
    )

    # Alternative: SenseVoice model
    parser.add_argument(
        "--sense-voice",
        type=str,
        default="",
        help="Path to SenseVoice model.int8.onnx (use with --model-type sense_voice)",
    )

    # Configuration
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["zipformer_reazon", "sense_voice"],
        default="zipformer_reazon",
        help="Which ASR model to use. Default is best Japanese model (ReazonSpeech)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of threads for neural network computation",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate of the feature extractor",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=80,
        help="Feature dimension (usually 80 for Zipformer)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug messages when loading models",
    )

    return parser.parse_args()


def main():
    args = get_args()

    # Create the reusable transcriber
    transcriber = JapaneseSpeechTranscriber(
        encoder=args.encoder if args.encoder else None,
        decoder=args.decoder if args.decoder else None,
        joiner=args.joiner if args.joiner else None,
        sense_voice=args.sense_voice if args.sense_voice else None,
        tokens=args.tokens,
        silero_vad_model=args.silero_vad_model,
        model_type=args.model_type,
        num_threads=args.num_threads,
        sample_rate=args.sample_rate,
        feature_dim=args.feature_dim,
        debug=args.debug,
    )

    print("\n" + "=" * 60)
    print("🎤 Japanese Speech Recognition Started")
    print(f"Model Type : {args.model_type}")
    print("Speak clearly into your microphone...")
    print("Press Ctrl + C to stop")
    print("=" * 60 + "\n")

    try:
        # This will run live transcription with VAD
        transcriber.transcribe_japanese()
    except KeyboardInterrupt:
        print("\n\nCaught Ctrl + C. Exiting gracefully.")
    except Exception as e:
        print(f"\nError occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
