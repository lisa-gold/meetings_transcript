from pydantic import BaseModel


class TranscribeAudioResponse(BaseModel):
    transcription: str
    summary: str
