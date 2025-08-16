import os
from dotenv import load_dotenv
from typing import Tuple, List

import whisper
from pathlib import Path

from pyannote.core import Segment
from pyannote.core.utils.types import Label

from constants import Models


def transcribe_audio(file_path: str, model_name: str = Models.TINY):
    """
    Транскрибирует аудиофайл с помощью Whisper.
    """
    print("=== Stage 1. Define speakers with pyannotate. Начало разбивки на спикеров...")
    diarization = define_speakers(file_path)
    print("Stage 1 is finished. Аудио разбито на спикеров.")
    print(f"=== Stage 2. Transcribe audio to text. Загрузка модели Whisper ({model_name})...")
    # Для первого запуска модель будет скачиваться, это может занять время.
    # Модели хранятся в ~/.cache/whisper
    model_transcript = whisper.load_model(model_name)
    print("Модель загружена. Начало транскрибации...")
    result = model_transcript.transcribe(file_path, word_timestamps=True)
    segments = result['segments']
    final_text = ''
    if diarization:
        for segment in segments:
            speaker = define_speaker_by_timestamp(diarization, segment['start'], segment['end'])
            final_text += f"{speaker}: {segment['text']}\n"
    else:
        print('Text with no speakers')
        final_text = result['text']
    print("\n--- Результат транскрибации ---")
    print(final_text)

    # Получаем путь к входному файлу и создаем директорию output на уровень выше
    input_path = Path(file_path)
    output_dir = input_path.parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Получение имени файла и создание нового пути в директории output
    output_filename = output_dir / f"{input_path.name}.txt"

    # Сохранение результата в файл
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(final_text)
    print(f"\nТранскрипция сохранена в файл: {output_filename}")


def define_speakers(file_path: str) -> List[Tuple[Segment, Label]] | None:
    """
    :param file_path:
    :return: a list of speech parts:
        every speech part contains:
         - turn (start, end floats)
         - speaker label
    """
    import torch
    from pyannote.audio import Pipeline

    try:
        load_dotenv()
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.getenv("HG_TOKEN")
        )
    except Exception as e:
        print(f"Error: {e}")
        print("Check huggingface token.")
        return None

    if not pipeline:
        print("Pipeline is not loaded. Check huggingface token.")
        return None

    # send pipeline to GPU (when available)
    pipeline.to(torch.device("cuda"))

    # apply pretrained pipeline
    converted_file_path = convert_audio_file(file_path)
    diarization = pipeline(converted_file_path)

    # print the result
    result = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        result.append((turn, speaker))
    return result


def convert_audio_file(file_path: str) -> str:
    from pydub import AudioSegment

    audio = AudioSegment.from_file(file_path)

    base, _ = os.path.splitext(file_path)
    output_path = f"{base}.wav"

    # Export to wav
    audio.export(output_path, format="wav")
    return output_path


def define_speaker_by_timestamp(diarization: List[Tuple[Segment, Label]],
                                timestamp_start: float, timestamp_end: float) -> str:
    print(f'Start define speaker by timestamps: {timestamp_start} - {timestamp_end}')
    for (turn, speaker) in diarization:
        if turn.start <= timestamp_start <= turn.end or turn.start <= timestamp_end <= turn.end or \
                (timestamp_start < turn.start and timestamp_end > turn.end):
            return str(speaker)
    return 'Unknown'
