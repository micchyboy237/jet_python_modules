"""
Japanese Live Speech-to-English Translation Demo

Uses:

  1. JapaneseSpeechTranscriber (Zipformer ReazonSpeech by default) for ASR
  2. JapaneseToEnglishTranslator (llama-cpp-python + GGUF) for translation

Usage:

python /Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/libs/sherpa_onnx/helpers/vad-with-non-streaming-asr-ja-en.py \
  --silero-vad-model /Users/jethroestrada/.cache/pretrained_models/sherpa-onnx/vad/silero_vad.onnx \
  --tokens /Users/jethroestrada/.cache/pretrained_models/sherpa-onnx/sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17/tokens.txt \
  --encoder /Users/jethroestrada/.cache/pretrained_models/sherpa-onnx/sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17/encoder-epoch-35-avg-1.int8.onnx \
  --decoder /Users/jethroestrada/.cache/pretrained_models/sherpa-onnx/sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17/decoder-epoch-35-avg-1.int8.onnx \
  --joiner /Users/jethroestrada/.cache/pretrained_models/sherpa-onnx/sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17/joiner-epoch-35-avg-1.int8.onnx \
  --num-threads 4 \
  --sample-rate 16000 \
  --feature-dim 80

"""

import argparse
import sys

# from jet.libs.sherpa_onnx.helpers.ja_en_translator import JapaneseToEnglishTranslator
from jet.libs.sherpa_onnx.helpers.ja_en_translator_http import translate
from jet.libs.sherpa_onnx.helpers.ja_transcriber import JapaneseSpeechTranscriber


def get_args():
    parser = argparse.ArgumentParser(
        description="Japanese Speech to English Translation (Live)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # === ASR Settings (Sherpa-ONNX) ===
    parser.add_argument(
        "--silero-vad-model",
        type=str,
        required=True,
        help="Path to silero_vad.onnx",
    )
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
    parser.add_argument(
        "--sense-voice",
        type=str,
        default="",
        help="Path to SenseVoice model.int8.onnx",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["zipformer_reazon", "sense_voice"],
        default="zipformer_reazon",
        help="ASR model type. Default = best Japanese model (ReazonSpeech)",
    )

    # === Translation Settings (llama-cpp-python) ===
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=2048,
        help="Context length for KV cache (larger = better for conversation)",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU (-1 = all possible)",
    )
    parser.add_argument(
        "--cache-type-k",
        type=str,
        default="q8_0",
        choices=["f16", "q8_0", "q4_0"],
        help="KV cache quantization for keys",
    )
    parser.add_argument(
        "--cache-type-v",
        type=str,
        default="q8_0",
        choices=["f16", "q8_0", "q4_0"],
        help="KV cache quantization for values",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for translation (lower = more deterministic)",
    )

    # === Common Settings ===
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of threads for ASR",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=80,
        help="Feature dimension",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug messages",
    )

    return parser.parse_args()


def main():
    args = get_args()

    print("Loading Japanese Speech Transcriber...")
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

    # print("\nLoading Japanese-to-English Translator (GGUF)...")
    # translator = JapaneseToEnglishTranslator(
    #     model_path=args.llm_model,
    #     n_ctx=args.n_ctx,
    #     n_gpu_layers=args.n_gpu_layers,
    #     cache_type_k=args.cache_type_k,
    #     cache_type_v=args.cache_type_v,
    #     verbose=False,
    # )

    print("\n" + "=" * 70)
    print("🎤 Japanese Speech → English Translation Started")
    print(f"ASR Model   : {args.model_type} (ReazonSpeech default)")
    # print(f"LLM Model   : {args.llm_model}")
    # print(f"KV Cache    : n_ctx={args.n_ctx} | {args.cache_type_k}/{args.cache_type_v}")
    print("Speak clearly into your microphone...")
    print("Press Ctrl + C to stop")
    print("=" * 70 + "\n")

    try:
        # Override transcribe_japanese to add real-time translation
        # (We do this by monkey-patching or better: modify the live loop if needed.
        # For simplicity, we assume JapaneseSpeechTranscriber can be extended or we run a custom loop.
        # Here we simulate by calling a custom live loop that translates each segment.)

        # Note: For full integration, it's better to modify JapaneseSpeechTranscriber
        # to accept a callback. For now, we'll run a simple custom live loop below.

        # === Custom Live Loop with Translation ===
        import numpy as np
        import sounddevice as sd

        if not transcriber.vad:
            raise RuntimeError("VAD is required for live mode.")

        buffer = []
        window_size = transcriber.vad.config.silero_vad.window_size
        samples_per_read = int(0.1 * transcriber.sample_rate)

        with sd.InputStream(
            channels=1, dtype="float32", samplerate=transcriber.sample_rate
        ) as stream:
            print("Listening... Speak now!\n")

            while True:
                samples, _ = stream.read(samples_per_read)
                samples = samples.reshape(-1)
                buffer = np.concatenate([buffer, samples])

                while len(buffer) > window_size:
                    transcriber.vad.accept_waveform(buffer[:window_size])
                    buffer = buffer[window_size:]

                while not transcriber.vad.empty():
                    segment = transcriber.vad.front.samples
                    stream_asr = transcriber.recognizer.create_stream()
                    stream_asr.accept_waveform(transcriber.sample_rate, segment)
                    transcriber.vad.pop()

                    transcriber.recognizer.decode_stream(stream_asr)
                    japanese_text = stream_asr.result.text.strip()

                    if japanese_text:
                        print(f"🇯🇵 Japanese: {japanese_text}")
                        en_result = translate(japanese_text)
                        english_text = en_result["text"]
                        print(f"🇬🇧 English : {english_text}\n")

    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user. Exiting gracefully.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    finally:
        print("\nSession ended.")


if __name__ == "__main__":
    main()
