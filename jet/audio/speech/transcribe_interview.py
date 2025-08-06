import speech_recognition as sr
from typing import Optional
import time
import logging

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class Transcriber:
    """A class to handle real-time audio transcription from a microphone."""

    def __init__(self, recognizer: Optional[sr.Recognizer] = None):
        """Initialize the transcriber with a recognizer instance."""
        self.recognizer = recognizer if recognizer else sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_running = False

    def adjust_for_ambient_noise(self, duration: float = 5.0) -> None:
        """Adjust the recognizer for ambient noise."""
        try:
            logging.info("Adjusting for ambient noise...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(
                    source, duration=duration)
            logging.info("Ambient noise adjustment completed.")
        except Exception as e:
            logging.error(f"Failed to adjust for ambient noise: {e}")
            raise

    def transcribe_audio(self, audio: sr.AudioData) -> Optional[str]:
        """Transcribe audio data to text using Google's Speech Recognition API."""
        try:
            transcription = self.recognizer.recognize_google(audio)
            return transcription
        except sr.UnknownValueError:
            logging.warning("Could not understand audio.")
            return None
        except sr.RequestError as e:
            logging.error(f"Speech recognition request failed: {e}")
            return None

    def start_transcription(self) -> None:
        """Start real-time transcription from the microphone."""
        self.is_running = True
        self.adjust_for_ambient_noise()

        logging.info(
            "Starting live transcription. Speak into the microphone...")
        with self.microphone as source:
            while self.is_running:
                try:
                    audio = self.recognizer.listen(
                        source, timeout=5, phrase_time_limit=10)
                    transcription = self.transcribe_audio(audio)
                    if transcription:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[{timestamp}] {transcription}")
                except sr.WaitTimeoutError:
                    logging.debug("No speech detected within timeout.")
                    continue
                except KeyboardInterrupt:
                    self.stop_transcription()
                    break
                except Exception as e:
                    logging.error(f"Error during transcription: {e}")
                    continue

    def stop_transcription(self) -> None:
        """Stop the transcription process."""
        self.is_running = False
        logging.info("Transcription stopped.")


def main():
    """Main function to run the transcription program."""
    transcriber = Transcriber()
    try:
        transcriber.start_transcription()
    except KeyboardInterrupt:
        transcriber.stop_transcription()
        logging.info("Program terminated by user.")


if __name__ == "__main__":
    main()
