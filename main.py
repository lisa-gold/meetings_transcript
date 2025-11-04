import os

from whisper.tokenizer import LANGUAGES

from commands.transcribe import transcribe_audio
from constants import Models
from cli import parser

if __name__ == "__main__":
    args = parser.parse_args()

    current_path = os.path.dirname(os.path.abspath(__file__))
    audio_file = os.path.join(current_path, 'input', args.audio_file)
    model = args.model
    language = args.lang
    if model not in Models:
        print(f"{model} is a wrong model name. Available models: {'|'.join(model.value for model in Models)}")
        exit(1)
    if language and language not in LANGUAGES:
        print(f"Language {language} is not an option. Available languages: {', '.join(LANGUAGES.keys())}")
        exit(1)
    transcribe_audio(audio_file, model)
