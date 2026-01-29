from smolagents.tools import PipelineTool


class SpeechToTextTool(PipelineTool):
    default_checkpoint = "Systran/faster-whisper-large-v3"  # or "openai/whisper-large-v3-turbo" if fallback
    description = "This is a tool that transcribes an audio into text. It returns the transcribed text."
    name = "transcriber"
    inputs = {
        "audio": {
            "type": "audio",
            "description": "The audio to transcribe. Can be a local path, an url, or a tensor.",
        }
    }
    output_type = "string"

    use_faster_whisper = True  # ← toggle to False to fallback to transformers

    def __new__(cls, *args, **kwargs):
        if cls.use_faster_whisper:
            # faster-whisper does not need these imports at class level
            pass
        else:
            from transformers.models.whisper import (
                WhisperForConditionalGeneration,
                WhisperProcessor,
            )

            cls.pre_processor_class = WhisperProcessor
            cls.model_class = WhisperForConditionalGeneration
        return super().__new__(cls)

    def setup(self):
        if not self.is_initialized:
            if self.use_faster_whisper:
                try:
                    import torch
                    from faster_whisper import WhisperModel
                except ImportError:
                    raise ImportError(
                        "faster-whisper is not installed. "
                        "Run: pip install faster-whisper"
                    )
                model_id = (
                    self.model
                    if isinstance(self.model, str)
                    else self.default_checkpoint
                )
                self.model = WhisperModel(
                    model_id,
                    device="cuda"
                    if hasattr(__import__("torch"), "cuda")
                    and __import__("torch").cuda.is_available()
                    else "cpu",
                    compute_type="default",  # "int8" / "float16" / ...
                    download_root=None,  # or custom cache
                )
            else:
                # original transformers loading
                super().setup()
            self.is_initialized = True

    def encode(self, audio):
        from smolagents.agent_types import AgentAudio

        audio = AgentAudio(audio).to_raw()
        # faster-whisper expects numpy array, sample rate 16000
        return {"audio": audio, "sr": 16000}

    def forward(self, inputs):
        if self.use_faster_whisper:
            segments, info = self.model.transcribe(
                inputs["audio"],
                language=None,  # auto-detect
                beam_size=5,
                vad_filter=True,  # very useful for real-world audio
                without_timestamps=False,
            )
            return list(segments), info
        else:
            # original transformers
            return self.model.generate(inputs["input_features"])

    def decode(self, outputs):
        if self.use_faster_whisper:
            segments, _info = outputs
            # Join text from segments
            text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
            return text
        else:
            # original
            return self.pre_processor.batch_decode(outputs, skip_special_tokens=True)[0]
