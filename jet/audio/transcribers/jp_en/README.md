# Japanese → English Real-Time Transcription & Translation

**Fully local • Mac M1/M2/M3/M4 optimized • Zero cloud • <1s latency**

Live-captures any Japanese audio from your Mac, instantly transcribes + translates it to natural English text using OpenAI Whisper (large-v3 or turbo) accelerated on Apple Silicon.

Perfect for:

- Japanese dramas, movies, variety shows
- YouTube, Twitch, Netflix (with audio routing)
- Podcasts, interviews, language learning
- Live translation of meetings or calls
- Any Japanese audio source

### One-command setup & run

```bash
# 1. Install dependencies (once)
pip install faster-whisper sounddevice torch torchaudio numpy

# 2. Route your desired audio to BlackHole (or any virtual cable)
#    System Settings → Sound → Input → BlackHole 4ch (or 16ch)

# 3. Start real-time Japanese → English translation
python jp_to_en_realtime.py --device "BlackHole 4ch" --model turbo
```
