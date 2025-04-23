import datetime
import asyncio
import os
import shutil
from typing import Optional
import threading
from gtts import gTTS
from jet.logger.logger import CustomLogger
import pygame

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)


class AdvancedTTSEngine:
    def __init__(self, rate: int = 200, voice_map: dict = None, output_dir: str = None):
        self.rate = rate
        self.voice_map = voice_map or {
            "Emma": "female_voice_id", "Liam": "male_voice_id"}
        self.lock = threading.Lock()
        pygame.mixer.init()
        self.temp_files = []
        self.combined_files = []
        self.output_dir = output_dir if output_dir else script_dir
        self._reset_output_dir()
        os.makedirs(self.output_dir, exist_ok=True)
        self.cache = {}
        self.channel = pygame.mixer.Channel(0)

    def _reset_output_dir(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
            logger.info(f"Cleared output directory: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_audio_filename(self, speaker_name: str, text: str, prefix: str = "tts") -> str:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        safe_text = ''.join(c for c in text[:30] if c.isalnum() or c in (
            ' ', '_')).strip().replace(' ', '_')
        filename = os.path.join(
            self.output_dir, f"{prefix}_{speaker_name}_{timestamp}_{safe_text}.mp3")
        if prefix == "tts":
            self.temp_files.append(filename)
        else:
            self.combined_files.append(filename)
        return filename

    def speak(self, text: str, speaker_name: str = "Agent") -> Optional[str]:
        with self.lock:
            if not text.strip():
                logger.warning(f"Skipping empty TTS text for {speaker_name}")
                return None
            cache_key = f"{speaker_name}:{text}"
            if cache_key in self.cache:
                file_path = self.cache[cache_key]
            else:
                file_path = self._get_audio_filename(speaker_name, text)
                try:
                    tts = gTTS(text=text, lang='en')
                    tts.save(file_path)
                    self.cache[cache_key] = file_path
                except Exception as e:
                    logger.error(f"Error in TTS generation: {e}")
                    raise
            try:
                self.channel.play(pygame.mixer.Sound(file_path))
                return file_path
            except Exception as e:
                logger.error(f"Error in audio playback: {e}")
                raise

    async def speak_async(self, text: str, speaker_name: str = "Agent") -> Optional[str]:
        loop = asyncio.get_event_loop()
        file_path = await loop.run_in_executor(None, self.speak, text, speaker_name)
        await asyncio.sleep(0.01)
        return file_path

    def _cleanup_temp_files(self):
        try:
            if self.lock.acquire(blocking=False):
                try:
                    files_to_delete = []
                    for file_path in self.temp_files:
                        files_to_delete.append(file_path)
                    logger.info(
                        f"Deleting temp files ({len(files_to_delete)})...")
                    for file_path in files_to_delete:
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                        except Exception as e:
                            logger.error(
                                f"Error deleting temp file {file_path}: {e}")
                finally:
                    self.lock.release()
            else:
                logger.warning(
                    "Could not acquire lock for cleanup; skipping temp file deletion")
        except Exception as e:
            logger.error(f"Error during temp file cleanup: {e}")

    def cleanup(self):
        self._cleanup_temp_files()
