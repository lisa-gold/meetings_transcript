class TranscribeServiceException(Exception):
    def __init__(self, description: str):
        self.description = description


class TranscribeServiceBadRequest(TranscribeServiceException):
    pass


class TranscribeServiceModelNotFound(TranscribeServiceBadRequest):
    pass


class TranscribeServiceLanguageNotFound(TranscribeServiceBadRequest):
    pass


class TranscribeServiceWrongFileType(TranscribeServiceBadRequest):
    pass


class TranscribeServiceAiModelException(TranscribeServiceException):
    pass


class SpeakerServiceException(Exception):
    def __init__(self, description: str):
        self.description = description


class SpeakerServiceBadRequest(SpeakerServiceException):
    pass


class SpeakerServiceWrongFileType(SpeakerServiceBadRequest):
    pass
