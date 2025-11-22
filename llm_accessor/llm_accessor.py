import logging


class LlmAccessor:
    def __init__(self, model_name: str, logger: logging.Logger = None):
        self.model_name = model_name
        self._logger = logger or logging.getLogger(__name__)

    async def generate_response(self, prompt: str) -> str:
        # TODO implement
        self._logger.error(f'{self.__class__} NOT IMPLEMENTED')
        return ''