import sys
import os

from commands.transcribe import transcribe_audio
from constants import Models

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        current_path = os.path.dirname(os.path.abspath(__file__))
        audio_file = os.path.join(current_path, 'input', audio_file)
        model = sys.argv[2] if len(sys.argv) > 2 else Models.TINY
        if model not in Models:
            print(f"{model} is a wrong model name. Available models: {'|'.join(model.value for model in Models)}")
            exit(1)
        transcribe_audio(audio_file, model)
    else:
        print(f"Использование: python transcribe.py <путь_к_аудиофайлу> <model: {'|'.join(model.value for model in Models)}>")
