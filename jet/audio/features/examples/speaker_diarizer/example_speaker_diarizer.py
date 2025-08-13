import os
from pyannote.core import Annotation, Segment
from jet.audio.features.speaker_diarizer import SpeakerDiarizer
from jet.logger import logger


def main():
    # Define directories
    base_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/features/speaker_diarizer"
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "outputs")
    # config_path = os.path.join(base_dir, "config.yaml")
    config_path = None
    audio_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/mocks/sample_16k.wav"

    # Initialize diarizer
    diarizer = SpeakerDiarizer(
        data_dir=data_dir, output_dir=output_dir, config_path=config_path)
    logger.info(f"Processing audio file: {audio_path}")

    # Perform diarization
    try:
        annotation: Annotation = diarizer.diarize(audio_path)
        logger.info(f"Diarization completed. Output saved in {output_dir}")

        # Print results
        print("Diarization Results:")
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            start = segment.start
            end = segment.end
            print(f"Speaker {speaker}: {start:.2f}s - {end:.2f}s")
    except Exception as e:
        logger.error(f"Diarization failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
