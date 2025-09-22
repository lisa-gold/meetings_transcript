import os


def convert_audio_file(file_path: str) -> str:
    from pydub import AudioSegment

    audio = AudioSegment.from_file(file_path)

    base, _ = os.path.splitext(file_path)
    output_path = f"{base}.wav"

    # Export to wav
    audio.export(output_path, format="wav")
    return output_path
