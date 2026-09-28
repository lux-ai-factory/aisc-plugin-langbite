from langbite.llm_services.llm_service import LLMService
from gpt4all import GPT4All

MODEL_NAME = "Phi-3-mini-4k-instruct.Q4_0.gguf"

class GPT4AllServiceBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, **_ignored):
        if not self._instance:
            self._instance = GPT4AllService()
        return self._instance

class GPT4AllService(LLMService):

    def __init__(self):
        self.provider = 'GPT4All'
        self.llm = GPT4All(MODEL_NAME, device="cpu")

    # gpt4all's chat_session() only takes a system prompt; earlier turns cannot be injected.
    def validate_history(self, history):
        if history and (len(history) > 1 or history[0]['role'] != 'system'):
            raise NotImplementedError('GPT4All only supports a history made of a single system message')

    def execute_prompt(self, prompt, history=None):
        self.validate_history(history)
        session_args = {'system_prompt': history[0]['content']} if history else {}
        with self.llm.chat_session(**session_args):
            output = self.llm.generate(prompt)
            output = output.replace("<|assistant|>", "").replace('\n', ' ').strip()
            return output