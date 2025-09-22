import os
import pickle

from dotenv import load_dotenv
from typing import Tuple, List

import whisper
from pathlib import Path

from pyannote.core import Segment
from pyannote.core.utils.types import Label

from constants import Models
from db_accessor import DBAccessor
from llm_accessor import LlmAccessor
from utils import convert_audio_file

THRESHOLD = 0.6


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
    llm_accessor = LlmAccessor('Vllm')
    summery_prompt = f''' Make summery of the text. 
    For every speaker highlight what important words they said. 
    if tasks were given, write these tasks in bullet points for every speaker who has tasks. Use md formatting.
    <text>{final_text}</text>
    '''
    llm_accessor.generate_response(summery_prompt)


def define_speakers(file_path: str) -> List[Tuple[Segment, Label]] | None:
    """
    :param file_path:
    :return: a list of speech parts:
        every speech part contains:
         - turn (start, end floats)
         - speaker label
    """
    import torch
    from pyannote.audio import Pipeline, Audio
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)

    # apply pretrained pipeline
    converted_file_path = convert_audio_file(file_path)
    diarization = pipeline(converted_file_path)
    audio = Audio()
    embedding_model = PretrainedSpeakerEmbedding(
        "pyannote/embedding",
        device=device,
        use_auth_token=os.getenv("HG_TOKEN")
    )

    # print the result
    result = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # Crop segment audio
        waveform, sample_rate = audio.crop(converted_file_path, turn)
        # Get embedding for the segment
        vector = embedding_model(waveform[None])
        name = map_speaker_name(vector)
        result.append((turn, name or speaker))
    return result


def define_speaker_by_timestamp(diarization: List[Tuple[Segment, Label]],
                                timestamp_start: float, timestamp_end: float) -> str:
    for (turn, speaker) in diarization:
        if turn.start <= timestamp_start <= turn.end or turn.start <= timestamp_end <= turn.end or \
                (timestamp_start < turn.start and timestamp_end > turn.end):
            return str(speaker)
    return 'Unknown'


def map_speaker_name(current_emb: list, threshold: float = THRESHOLD) -> str | None:
    from sklearn.metrics.pairwise import cosine_similarity

    # Get all speakers and their vectors from the database
    db_accessor = DBAccessor('speakers.db')
    speaker_vector = db_accessor.cursor.execute("SELECT name, embedding FROM speakers").fetchall()
    print(speaker_vector)
    print(f'Found {len(speaker_vector)} speakers in the database')
    # Find the closest reference
    best_name = None
    best_score = -1
    for name, ref_emb_bytes in speaker_vector:
        ref_emb = pickle.loads(ref_emb_bytes)
        result = cosine_similarity(current_emb, ref_emb)
        score = result[0][0]
        print(f'  - {name}: {result}')
        if score > best_score:
            best_score = score
            best_name = name

    # Only label if confidence is high enough
    if best_score >= threshold:
        speaker_name = best_name
    else:
        print(f'  - No match found for speaker. Best match: {best_name} - {best_score}')
        speaker_name = None
    return speaker_name
