from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from constants import Models
from dto import TranscribeAudioResponse
from exceptions import TranscribeServiceModelNotFound, TranscribeServiceLanguageNotFound, TranscribeServiceBadRequest, \
    TranscribeServiceAiModelException, TranscribeServiceException, SpeakerServiceException, SpeakerServiceBadRequest
from services.speaker_service import SpeakerService
from services.transcribe_service import TranscribeService

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
        file: UploadFile = File(...),
        model: str = Form(Models.TINY.value),
        language: str | None = Form(None),
) -> TranscribeAudioResponse:
    try:
        service = TranscribeService()
        result = await service.transcribe_audio(file, model, language)
        return result
    except (TranscribeServiceModelNotFound, TranscribeServiceLanguageNotFound) as e:
        raise HTTPException(status_code=404, detail=e.description)
    except TranscribeServiceBadRequest as e:
        raise HTTPException(status_code=400, detail=e.description)
    except TranscribeServiceAiModelException as e:
        raise HTTPException(status_code=500, detail=e.description)
    except TranscribeServiceException as e:
        raise HTTPException(status_code=500, detail=e.description)


@app.post("/add_speaker")
async def add_speaker(
        speaker_name: str = Form(),
        file: UploadFile = File(...)
):
    try:
        service = SpeakerService()
        await service.add_sample_record(speaker_name, file)
        return f'Speaker {speaker_name} voice vector is updated'
    except SpeakerServiceBadRequest as e:
        return HTTPException(
            status_code=400,
            detail=e.description
        )
    except SpeakerServiceException as e:
        return HTTPException(
            status_code=500,
            detail=e.description
        )


@app.get("/speakers")
async def get_speakers():
    try:
        service = SpeakerService()
        return await service.get_speakers()
    except SpeakerServiceException as e:
        return HTTPException(
            status_code=500,
            detail=e.description
        )
