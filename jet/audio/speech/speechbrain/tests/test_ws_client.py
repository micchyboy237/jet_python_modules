from unittest.mock import AsyncMock, patch

import pytest
from jet.audio.speech.speechbrain.ws_client import send_audio_to_server


class TestSendAudioToServer:
    @pytest.mark.asyncio
    @patch("websockets.connect", new_callable=AsyncMock)
    async def test_given_audio_bytes_when_send_to_server_then_receives_response(
        self, mock_connect
    ):
        mock_ws = AsyncMock()
        mock_connect.return_value.__aenter__.return_value = mock_ws
        mock_ws.recv.return_value = (
            '{"en_text": "Hello", "confidence": 0.9, "quality": "High"}'
        )
        audio_bytes = bytearray(b"fake_audio")
        result = await send_audio_to_server(
            "ws://test", audio_bytes, "test_client", "utt1", 1
        )
        mock_ws.send.assert_called_once()
        assert result["en_text"] == "Hello"
        assert result["confidence"] == 0.9
