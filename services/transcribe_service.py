import logging
import os
import pickle
import tempfile
import torch

import torchaudio
from dotenv import load_dotenv
from typing import Tuple, List
from fastapi import UploadFile


import whisper

from pyannote.core import Segment
from pyannote.core.utils.types import Label
from pyannote.audio import Pipeline, Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from whisper.tokenizer import LANGUAGES

from constants import Models, ALLOWED_AUDIO_EXTENSIONS, ALLOWED_AUDIO_MIME_PREFIX, DEFAULT_MODEL
from db_accessor import DBAccessor
from dto import TranscribeAudioResponse
from exceptions import TranscribeServiceModelNotFound, TranscribeServiceLanguageNotFound, TranscribeServiceException, \
    TranscribeServiceWrongFileType, TranscribeServiceBadRequest, TranscribeServiceAiModelException
from llm_accessor import LlmAccessor
from utils import convert_audio_file

THRESHOLD = 0.6
SPEAKER_DEFINING_MODEL = 'pyannote/speaker-diarization-3.1'


class TranscribeService:
    def __init__(self, logger: logging.Logger = None, speakers_db: str = None, llm_accessor: LlmAccessor = None):
        self._speakers_db = speakers_db or 'speakers.db'
        self._llm_accessor = llm_accessor or LlmAccessor(DEFAULT_MODEL)
        self._logger = logger or logging.getLogger(__name__)

    async def transcribe_audio(self,
                               file: UploadFile,
                               model_name: str = Models.TINY,
                               language: str | None = None) -> TranscribeAudioResponse:
        temp_audio_path = None
        try:
            await self.validate_params(model_name, language, file)
            temp_audio_path = await self._save_input_file_in_temp_dir(file)
            self._logger.info("=== Stage 1. Define speakers with pyannotate.")
            diarization = await self.define_speakers(temp_audio_path)
            self._logger.info(f"Stage 1 is finished. Speakers are defined.\n"
                              f"=== Stage 2. Transcribe audio to text. Loading model {model_name}...")

            model_transcript = whisper.load_model(model_name)
            self._logger.info("Model is loaded. Start to transcribe...")
            decode_options = {}
            if language:
                decode_options.update({"language": language})
            result = model_transcript.transcribe(audio=temp_audio_path,
                                                 word_timestamps=True,
                                                 **decode_options)
            segments = result['segments']
            final_text = ''
            if diarization:
                for segment in segments:
                    speaker = await self._define_speaker_by_timestamp(diarization, segment['start'], segment['end'])
                    final_text += f"{speaker}: {segment['text']}\n"
            else:
                self._logger.error('Text with no speakers')
                final_text = result['text']

            summary = await self._llm_accessor.generate_response(self._get_summary_prompt(final_text))

            return TranscribeAudioResponse(
                transcription=final_text,
                summary=summary
            )
        except TranscribeServiceBadRequest as e:
            self._logger.error(f'Bad request: {e.description}')
            raise e
        except Exception as e:
            self._logger.exception(e)
            raise TranscribeServiceException(f"Transcription failed: {str(e)}")
        finally:
            if temp_audio_path:
                os.remove(temp_audio_path)

    async def validate_params(self, model: str, language: str, file: UploadFile):
        self._logger.info('Start validating request params')
        if model not in [m.value for m in Models]:
            raise TranscribeServiceModelNotFound(
                f"Invalid model '{model}'. Available: {', '.join(m.value for m in Models)}"
            )

        if language and language not in LANGUAGES:
            raise TranscribeServiceLanguageNotFound(
                f"Invalid language '{language}'. Available: {', '.join(LANGUAGES.keys())}"
            )
        await self._validate_audio_file_type(file)
        self._logger.info('Validation passed successfully')

    async def _validate_audio_file_type(self, file: UploadFile):
        self._logger.info('Start validating file type')
        # Validate MIME
        if not file.content_type.startswith(ALLOWED_AUDIO_MIME_PREFIX):
            raise TranscribeServiceWrongFileType(
                f"File must be an audio file. Received MIME: {file.content_type}"
            )

        # Validate extension
        _, ext = os.path.splitext(file.filename)
        if ext.lower() not in ALLOWED_AUDIO_EXTENSIONS:
            raise TranscribeServiceWrongFileType(
                f"Unsupported audio format '{ext}'. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
            )

    async def _save_input_file_in_temp_dir(self, file: UploadFile) -> str:
        self._logger.info(f'Start saving uploaded file {file.filename} in temp directory')
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                tmp.write(await file.read())
                return tmp.name
        except Exception as e:
            self._logger.exception(e)
            raise TranscribeServiceException(f"Failed to save file: {str(e)}")

    async def define_speakers(self,
                              file_path: str,
                              speakers_model: str = SPEAKER_DEFINING_MODEL) -> List[Tuple[Segment, str]] | None:
        """
        Return: a list of speech parts:
            every speech part contains:
             - turn (start, end floats)
             - speaker label
        """

        try:
            load_dotenv()
            pipeline = Pipeline.from_pretrained(
                speakers_model,
                token=os.getenv("HG_TOKEN")
            )
        except Exception as e:
            self._logger.exception(e)
            raise TranscribeServiceAiModelException(f'Error loading model to define speakers. Error: {str(e)}')

        if not pipeline:
            self._logger.error("Pipeline is not loaded. Check huggingface token.")
            return None

        # send pipeline to GPU (when available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)

        # apply pretrained pipeline
        converted_file_path = convert_audio_file(file_path)
        waveform, sample_rate = torchaudio.load(converted_file_path)
        audio_duration = waveform.shape[1] / sample_rate  # seconds
        diarization = pipeline(converted_file_path)
        audio = Audio(mono=True)
        embedding_model = PretrainedSpeakerEmbedding(
            "pyannote/embedding",
            device=device,
            token=os.getenv("HG_TOKEN")
        )

        # Get all speakers and their vectors from the database
        db_accessor = DBAccessor(self._speakers_db)
        speaker_vectors = db_accessor.cursor.execute("SELECT name, embedding FROM speakers").fetchall()
        self._logger.info(f'Found {len(speaker_vectors)} speakers in the database')

        result = []
        for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
            # Crop segment audio
            end_time = min(turn.end, audio_duration)
            start_time = max(turn.start, 0.0)
            safe_turn = Segment(start_time, end_time)
            waveform, sample_rate = audio.crop(converted_file_path, safe_turn)
            if waveform.shape[1] < sample_rate / 2:
                continue  # skip segment, it is too short
            # Make sure it's mono and shaped (batch, num_samples)
            if waveform.ndim > 1:  # it is stereo
                waveform = waveform.mean(dim=0, keepdim=True)  # force mono

            waveform = waveform.squeeze(0).unsqueeze(0)  # -> (1, num_samples)

            # Get embedding for the segment
            vector = embedding_model(waveform)
            name = await self._map_speaker_name(vector, speaker_vectors)
            result.append((turn, name or speaker))
        return result

    async def _define_speaker_by_timestamp(self, diarization: List[Tuple[Segment, Label]],
                                           timestamp_start: float, timestamp_end: float) -> str:
        for (turn, speaker) in diarization:
            if turn.start <= timestamp_start <= turn.end or turn.start <= timestamp_end <= turn.end or \
                    (timestamp_start < turn.start and timestamp_end > turn.end):
                return str(speaker)
        self._logger.error(f'For period between {timestamp_start} and {timestamp_end}')
        return 'Unknown'

    async def _map_speaker_name(self,
                                current_emb: list,
                                speaker_vectors: list,
                                threshold: float = THRESHOLD) -> str | None:
        from sklearn.metrics.pairwise import cosine_similarity

        # Find the closest reference
        best_name = None
        best_score = -1
        for name, ref_emb_bytes in speaker_vectors:
            ref_emb = pickle.loads(ref_emb_bytes)
            result = cosine_similarity(current_emb, ref_emb)
            score = result[0][0]
            if score > best_score:
                best_score = score
                best_name = name

        # Only label if confidence is high enough
        if best_score >= threshold:
            speaker_name = best_name
        else:
            self._logger.error(f'  - No match found for speaker. Best match: {best_name} - {round(best_score, 3)}')
            speaker_name = None
        return speaker_name

    def _get_summary_prompt(self, text: str):
        return f''' Make summery of the text. For every speaker highlight what important words they said.
If tasks were given, write these tasks in bullet points for every speaker who has tasks. Use md formatting.
<text>{text}</text>'''
