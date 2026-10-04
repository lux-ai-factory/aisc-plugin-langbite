from langbite.utils import clean_string
import time

class PromptResponse:
    
    @property
    def instance(self) -> str:
        return self.__instance
    
    @property
    def response(self) -> str:
        return self.__response
    
    @response.setter
    def response(self, value: str):
        from langbite.llm_services.llm_openai_factory import Refusal
        cleaned = clean_string(value)
        # keep the mark: a refusal is evaluated as its own outcome (Prompt.evaluate)
        self.__response = Refusal(cleaned) if isinstance(value, Refusal) else cleaned
    
    @property
    def execution_time(self):
        return self.__timestamp

    def __init__(self, instance, response):
        self.__instance = instance
        self.response = response
        self.__timestamp = time.localtime()