import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from pathlib import Path

from jp_transcriber import JapaneseTranscriber, AudioConfig, TranscriberConfig


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Provide a temporary output directory."""
    return tmp_path / "output"
    

@pytest.fixture
def audio_config() -> AudioConfig:
    return {
        "device": None,
        "samplerate": 16000,
        "chunk_duration": 3.0,
        "channels": 1,
    }


@pytest.fixture
def trans_config() -> TranscriberConfig:
    return {
        "model_size": "tiny",
        "compute_type": "int8",
        "language": "ja",
        "task": "translate",
    }


@pytest.fixture
def sample_audio_chunk() -> np.ndarray:
    """3 seconds of silence-like audio."""
    return np.zeros(16000 * 3, dtype=np.float32)


class TestJapaneseTranscriberWithFileSaving:

    def test_initialization_creates_output_dir_and_files(self, temp_output_dir, audio_config, trans_config):
        # Given a new output directory path
        # When JapaneseTranscriber is initialized
        transcriber = JapaneseTranscriber(
            audio_cfg=audio_config,
            trans_cfg=trans_config,
            output_dir=temp_output_dir,
            print_to_console=False,
        )

        # Then output directory exists
        assert temp_output_dir.exists()

        # Then .txt and .srt files are created with timestamped names
        txt_files = list(temp_output_dir.glob("translation_*.txt"))
        srt_files = list(temp_output_dir.glob("subtitles_*.srt"))
        assert len(txt_files) == 1
        assert len(srt_files) == 1
        assert txt_files[0].name.startswith("translation_")
        assert srt_files[0].name.startswith("subtitles_")

        # Cleanup
        transcriber.txt_handle.close()
        transcriber.srt_handle.close()

    def test_save_text_writes_to_both_files_correctly(self, temp_output_dir, audio_config, trans_config):
        # Given a transcriber with real file handles
        transcriber = JapaneseTranscriber(
            audio_cfg=audio_config,
            trans_cfg=trans_config,
            output_dir=temp_output_dir,
            print_to_console=False,
        )

        # When _save_text is called
        transcriber._save_text("Hello, this is a test.", start=10.0, end=13.5)

        # Then content is written correctly
        txt_content = transcriber.txt_handle.read()
        expected_txt = "Hello, this is a test.\n"
        assert txt_content == expected_txt

        transcriber.srt_handle.seek(0)
        srt_content = transcriber.srt_handle.read()
        expected_srt = "1\n00:00:10,000 --> 00:00:13,500\nHello, this is a test.\n\n"
        assert srt_content == expected_srt

        # Cleanup
        transcriber.txt_handle.close()
        transcriber.srt_handle.close()

    def test_seconds_to_srt_time_converts_correctly(self, temp_output_dir, audio_config, trans_config):
        transcriber = JapaneseTranscriber(audio_config, trans_config, temp_output_dir, print_to_console=False)

        # Given various float seconds
        cases = [
            (0.0, "00:00:00,000"),
            (5.123, "00:00:05,123"),
            (59.999, "00:00:59,999"),
            (60.0, "00:01:00,000"),
            (3661.555, "01:01:01,555"),
        ]

        # When converted
        for seconds, expected in cases:
            # Then output matches SRT format
            result = transcriber._seconds_to_srt_time(seconds)
            assert result == expected

        transcriber.txt_handle.close()
        transcriber.srt_handle.close()

    @patch("jp_to_en_realtime.WhisperModel")
    def test_transcription_worker_yields_text_with_timing(
        self, mock_model_class, temp_output_dir, audio_config, trans_config, sample_audio_chunk
    ):
        # Given a mocked Whisper model returning one segment
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "This is a test sentence."
        mock_segment.start = 0.5
        mock_segment.end = 2.8
        mock_model.transcribe.return_value = ([mock_segment], MagicMock())

        mock_model_class.return_value = mock_model

        transcriber = JapaneseTranscriber(audio_config, trans_config, temp_output_dir, print_to_console=False)
        transcriber.queue.put(sample_audio_chunk)
        transcriber.running = True

        # When worker processes one chunk
        results = list(transcriber._transcription_worker())

        # Then yields (text, start, end) tuple with correct timing
        assert len(results) == 1
        text, start, end = results[0]
        assert text == "This is a test sentence."
        assert 0.4 < start < 0.6   # approximate due to chunk timing
        assert 2.7 < end < 2.9

        transcriber.txt_handle.close()
        transcriber.srt_handle.close()

    def test_callback_saves_and_optionally_prints(self, temp_output_dir, audio_config, trans_config, capsys):
        transcriber = JapaneseTranscriber(audio_config, trans_config, temp_output_dir, print_to_console=True)

        # When callback is triggered (console enabled)
        transcriber._callback("Printed and saved", start=1.0, end=2.0)

        # Then prints to console
        captured = capsys.readouterr()
        assert "Printed and saved" in captured.out

        # Then saves to files
        assert (temp_output_dir / transcriber.txt_file.name).read_text(encoding="utf-8").strip().endswith("Printed and saved")

        # Now disable console
        transcriber.print_to_console = False
        transcriber._callback("Silent save only", start=5.0, end=6.0)

        captured = capsys.readouterr()
        assert "Silent save only" not in captured.out  # No print

        transcriber.txt_handle.close()
        transcriber.srt_handle.close()

    @patch("sounddevice.InputStream")
    def test_start_creates_files_and_closes_on_exit(self, mock_stream, temp_output_dir, audio_config, trans_config):
        # Given a transcriber
        transcriber = JapaneseTranscriber(audio_config, trans_config, temp_output_dir, print_to_console=False)

        # Patch sleep to trigger KeyboardInterrupt immediately
        with patch("sounddevice.sleep") as mock_sleep:
            mock_sleep.side_effect = KeyboardInterrupt

            # When start() is called and interrupted
            transcriber.start()

        # Then files are closed properly
        assert transcriber.txt_handle.closed
        assert transcriber.srt_handle.closed

        # Then output files exist on disk
        assert len(list(temp_output_dir.glob("translation_*.txt"))) == 1
        assert len(list(temp_output_dir.glob("subtitles_*.srt"))) == 1