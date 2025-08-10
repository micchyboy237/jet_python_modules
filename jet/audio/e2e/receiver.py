import subprocess


def receive_audio():
    # Define the ffmpeg command
    sdp_file_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/e2e/stream.sdp"
    output_file_format = "last5min_%Y%m%d-%H%M%S.wav"

    command = [
        "ffmpeg",
        "-protocol_whitelist", "file,rtp,udp",
        "-i", sdp_file_path,
        "-acodec", "pcm_s16le",
        "-ar", "48000",
        "-ac", "2",
        "-f", "segment",
        "-segment_time", "300",
        "-reset_timestamps", "1",
        "-strftime", "1",
        output_file_format
    ]

    # Run the ffmpeg command
    subprocess.run(command)


if __name__ == "__main__":
    receive_audio()
