from langbite.llm_services.llm_service import LLMService
from openai import OpenAI

OPENAI_URL = 'https://api.openai.com/v1'

class OpenAIChatServiceBuilder:
    def __init__(self, model):
        self._instance = None
        self._model = model
    
    def __call__(self, openai_api_key, **_ignored):
        if not self._instance:
            self._instance = OpenAIChatService(openai_api_key, self._model)
        return self._instance


class OpenAIService(LLMService):

    @property
    def api_client(self):
        return self.__api_client

    @property
    def promptSuffix(self):
        return self.__promptSuffix
    
    def __init__(self, openai_api_key, model, base_url=None):
        # base_url given explicitly: the client never falls back to OPENAI_BASE_URL from the environment
        self.__api_client = OpenAI(api_key=openai_api_key, base_url=base_url or OPENAI_URL)
        self.provider = 'OpenAI'
        self.model = model
        self.__promptSuffix = ' Do not use carry returns in your response.'


class OpenAIChatService(OpenAIService):
    def validate_history(self, history):
        pass

    def execute_prompt(self, prompt, history=None):
        messages = [dict(message) for message in history or []]
        messages.append({ "role": "user", "content": prompt + self.promptSuffix })
        arguments = {"model": self.model, "messages": messages}
        if (self.temperature): arguments["temperature"] = self.temperature
        if (self.tokens): arguments["max_tokens"] = self.tokens
        completion = self.api_client.chat.completions.create(**arguments)
        return completion.choices[0].message.content


class AISCTargetServiceBuilder:
    """The evaluation's target, reached through the AISC platform: the platform sets AISC_TARGET_* for
    the length of a run (aisc-plugin-interface's @system_under_test). Never cached: a run key is for one
    run."""

    def __call__(self, aisc_target_base_url='', aisc_target_api_key='', aisc_target_model='', **_ignored):
        missing = [name for name, value in (('AISC_TARGET_BASE_URL', aisc_target_base_url),
                                            ('AISC_TARGET_API_KEY', aisc_target_api_key),
                                            ('AISC_TARGET_MODEL', aisc_target_model)) if not value]
        if missing:
            raise ValueError(f"the target is not set ({', '.join(missing)}): run LangBiTe on a target with an endpoint")
        return AISCTargetService(aisc_target_api_key, aisc_target_model, aisc_target_base_url)


class Refusal(str):
    """The target declined to answer. A str, so everything that reads a response reads its text
    ("Refused: <reason>"); evaluation counts it as its own outcome (Prompt.evaluate)."""


class AISCTargetService(OpenAIChatService):
    """An OpenAI-compatible endpoint that the platform translates to whatever the target is. The
    platform marks a refusal the OpenAI way (message.refusal)."""

    def __init__(self, api_key, model, base_url):
        super().__init__(api_key, model, base_url)
        self.provider = 'AISC target'

    def execute_prompt(self, prompt, history=None):
        messages = [dict(message) for message in history or []]
        messages.append({"role": "user", "content": prompt + self.promptSuffix})
        arguments = {"model": self.model, "messages": messages}
        if self.temperature: arguments["temperature"] = self.temperature
        if self.tokens: arguments["max_tokens"] = self.tokens
        message = self.api_client.chat.completions.create(**arguments).choices[0].message
        if getattr(message, 'refusal', None):
            return Refusal(message.content or f'Refused: {message.refusal}')
        return message.content
