# Jet Audio E2E – Real-time Microphone RTP Streaming (macOS → Any)

Lightweight, zero-dependency (except FFmpeg) end-to-end audio streaming tools for sending macOS microphone input as uncompressed PCM over RTP and receiving it on the same or another machine.

Ideal for low-latency audio pipelines (e.g. AI voice agents, monitoring, intercom).

## Features

- **Uncompressed PCM S16LE (L16)** over RTP → near-zero encoding delay
- Supports **44.1 kHz / 48 kHz**, mono/stereo
- Automatic local IP detection and SDP generation
- Simultaneous local WAV recording (sender side)
- Segmented 5-minute recordings + volume detection insights (advanced receiver)
- Graceful shutdown on Ctrl+C
- Built-in device validation and helpful error messages
- Pure Python + FFmpeg (no compiled dependencies)

## Use Cases

- Real-time voice streaming to a remote server for transcription/synthesis
- Audio monitoring across machines in a studio or lab
- Feeding live microphone into local AI models (Whisper, VALL-E, etc.)
- Debugging audio pipelines with clean, timestamped segments

## Limitations

- **macOS only for sending** (uses `avfoundation` input)
- Receiving works on Linux/Windows/macOS
- **No encryption** – suitable only for trusted local networks
- No forward error correction or jitter buffer tuning beyond FFmpeg defaults
- Requires FFmpeg with RTP support in PATH
- Hardcoded paths in some scripts (intended for development)

## Requirements

- Python 3.8+
- FFmpeg (with `avfoundation` on macOS)
- Microphone access granted to terminal/Python (System Settings → Privacy & Security → Microphone)

Install FFmpeg (macOS recommended via Homebrew):

```bash
brew install ffmpeg
```
