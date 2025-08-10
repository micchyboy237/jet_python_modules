import subprocess


def send_audio():
    # Define the ffmpeg command
    command = [
        "ffmpeg",
        "-f", "avfoundation",
        "-i", ":0",
        "-acodec", "pcm_s16le",
        "-ar", "48000",
        "-ac", "2",
        "-f", "rtp",
        "rtp://192.168.68.104:5004",
        "-sdp_file", "stream.sdp"
    ]

    # Run the ffmpeg command
    subprocess.run(command)


if __name__ == "__main__":
    send_audio()
