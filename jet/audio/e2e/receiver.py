import subprocess
import logging
from typing import List

logging.basicConfig(
    filename='receiver.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def get_receiver_command(sdp_file: str, output_file: str, insights_file: str) -> List[str]:
    return [
        "ffmpeg",
        "-loglevel", "debug",
        "-protocol_whitelist", "file,rtp,udp",
        "-rtbufsize", "100M",
        "-i", sdp_file,
        "-acodec", "pcm_s16le",
        "-ar", "48000",
        "-ac", "2",
        "-t", "300",
        "-y",
        "-filter:a", "volumedetect",
        "-f", "wav",
        output_file,
        "-f", "null",
        "-",
        "-report"
    ]


def run_receiver(
    sdp_file: str = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/e2e/stream.sdp",
    output_file: str = "last5min.wav",
    insights_file: str = "audio_insights.log"
):
    logging.info("Starting receiver with SDP: %s, Output: %s",
                 sdp_file, output_file)
    try:
        cmd = get_receiver_command(sdp_file, output_file, insights_file)
        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        for line in process.stderr:
            logging.debug(line.strip())
            if "volumedetect" in line:
                with open(insights_file, "a") as f:
                    f.write(line + "\n")
    except Exception as e:
        logging.error("Receiver failed: %s", str(e))
        raise


if __name__ == "__main__":
    run_receiver()
