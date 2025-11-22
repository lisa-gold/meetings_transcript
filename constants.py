from enum import Enum


class Models(str, Enum):
    TINY = 'tiny'
    BASE = 'base'
    SMALL = 'small'
    MEDIUM = 'medium'
    LARGE = 'large'
    TURBO = 'turbo'


ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"}
ALLOWED_AUDIO_MIME_PREFIX = "audio/"
DEFAULT_MODEL = 'Vllm'
