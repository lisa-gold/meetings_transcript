import logging
import os
import pickle

from fastapi import UploadFile
from typing import List
from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import torch
import numpy as np

from constants import ALLOWED_AUDIO_EXTENSIONS, ALLOWED_AUDIO_MIME_PREFIX
from db_accessor import DBAccessor
from exceptions import SpeakerServiceException, SpeakerServiceWrongFileType
from utils import convert_audio_file
from dotenv import load_dotenv

class SpeakerService:
    def __init__(self, db_accessor: DBAccessor = None, logger: logging.Logger = None):
        load_dotenv()
        self._db_accessor = db_accessor or DBAccessor('speakers.db')
        self._logger = logger or logging.getLogger(__name__)

    async def add_sample_record(self, speaker_name: str, file: UploadFile):
        self._logger.info(f'Start adding voice sample for {speaker_name}')
        await self._validate_audio_file_type(file)
        file_path = await self._save_input_file_speaker_dir(file, speaker_name)

        # Device setup (use GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedding_model = PretrainedSpeakerEmbedding(
            "pyannote/embedding",
            device=device,
            token=os.getenv("HG_TOKEN")
        )

        vector = await self._build_averaged_vector(file_path, embedding_model, speaker_name)
        await self._save_vector_in_db(speaker_name, vector)

    async def get_speakers(self) -> List[str]:
        self._db_accessor.execute_query("SELECT name FROM speakers")
        return [name for (name,) in self._db_accessor.cursor]

    async def clear_speakers_db(self):
        self._db_accessor.execute_query("DELETE FROM speakers")
        self._logger.info(f'DB speakers is cleared')

    async def _validate_audio_file_type(self, file: UploadFile):
        self._logger.info('Start validating file type')
        # Validate MIME
        if not file.content_type.startswith(ALLOWED_AUDIO_MIME_PREFIX):
            raise SpeakerServiceWrongFileType(
                f"File must be an audio file. Received MIME: {file.content_type}"
            )

        # Validate extension
        _, ext = os.path.splitext(file.filename)
        if ext.lower() not in ALLOWED_AUDIO_EXTENSIONS:
            raise SpeakerServiceWrongFileType(
                f"Unsupported audio format '{ext}'. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
            )

    async def _save_input_file_speaker_dir(self, file: UploadFile, speaker_name: str) -> str:
        self._logger.info(f'Start saving uploaded file {file.filename} in {speaker_name} directory')
        try:
            destination_dir = f'voice_samples/{speaker_name}'
            os.makedirs(destination_dir, exist_ok=True)
            file_path = os.path.join(destination_dir, file.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(file.file.read())
            return file_path
        except Exception as e:
            self._logger.exception(e)
            raise SpeakerServiceException(f"Failed to save file: {str(e)}")

    async def _get_embedding(self, file_path: str, embedding_model):
        self._logger.info(f'Get embedding for {file_path}')
        audio = Audio()
        waveform, sample_rate = audio(file_path)
        return embedding_model(waveform[None])


    async def _build_averaged_vector(self, file_path: str, embedding_model, speaker_name: str):
        embeddings = []
        speaker_dir = os.path.dirname(file_path)
        for filename in os.listdir(speaker_dir):
            full_path = os.path.join(speaker_dir, filename)

            # Skip directories
            if os.path.isdir(full_path):
                continue

            if filename.lower().endswith('.wav'):
                emb = await self._get_embedding(full_path, embedding_model)
                embeddings.append(emb)
            else:
                converted_path = convert_audio_file(full_path)
                os.remove(full_path)
                emb = await self._get_embedding(converted_path, embedding_model)
                embeddings.append(emb)

        if embeddings:
            avg_emb = np.mean(embeddings, axis=0)
            self._logger.info(f"  - {speaker_name}: {len(embeddings)} samples averaged")
            return avg_emb
        else:
            self._logger.info(f"  - {speaker_name}: no samples found")
            return None

    async def _save_vector_in_db(self, speaker_name: str, vector: list):
        self._logger.info(f'Save vector in db for {speaker_name}')
        blob = pickle.dumps(vector)
        self._db_accessor.execute_query("""CREATE TABLE IF NOT EXISTS speakers (
id INTEGER PRIMARY KEY AUTOINCREMENT,
name VARCHAR(100) NOT NULL UNIQUE,
embedding BLOB NOT NULL)""")
        speaker = self._db_accessor.cursor.execute(f"SELECT * FROM speakers WHERE name = '{speaker_name}'").fetchone()
        if speaker:
            self._db_accessor.execute_query_with_params("UPDATE speakers SET embedding = ? WHERE name = ?",
                                                  (blob, speaker_name))
        else:
            self._db_accessor.execute_query_with_params("INSERT INTO speakers (name, embedding) VALUES (?, ?)",
                                                  (speaker_name, blob))
