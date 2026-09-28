import requests
from langbite.llm_services.llm_service import LLMService

# MCAS-lite's /chat limits (mcas/api.py ChatRequest)
MAX_HISTORY_MESSAGES = 20
MAX_TURN_CHARS = 2000
TIMEOUT_SECONDS = 180


class MCASChatServiceBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, mcas_url, **_ignored):
        if not self._instance:
            self._instance = MCASChatService(mcas_url)
        return self._instance


class MCASChatService(LLMService):
    """The MCAS-lite assistant, reached through its HTTP API (POST /chat). MCAS keeps no
    state between calls, so the history is sent with every question. MCAS uses its own
    system prompt, so a history may only hold user/assistant turns."""

    def __init__(self, mcas_url):
        self.provider = 'MCAS'
        self.model = 'mcas-chat'
        self.__url = mcas_url.rstrip('/')

    def validate_history(self, history):
        if not history:
            return
        if any(m['role'] == 'system' for m in history):
            raise NotImplementedError('MCAS uses its own system prompt: use a history without a system message (the *_api templates)')
        if len(history) > MAX_HISTORY_MESSAGES:
            raise NotImplementedError(f'MCAS accepts at most {MAX_HISTORY_MESSAGES} history turns')
        for i, message in enumerate(history):
            if message['role'] != ('user' if i % 2 == 0 else 'assistant'):
                raise NotImplementedError('MCAS needs alternating turns, starting with user and ending with assistant')
            if len(message['content']) > MAX_TURN_CHARS:
                raise NotImplementedError(f'MCAS accepts history turns of at most {MAX_TURN_CHARS} characters')

    def execute_prompt(self, prompt, history=None):
        self.validate_history(history)
        payload = {'question': prompt, 'history': [dict(m) for m in history or []]}
        response = requests.post(f'{self.__url}/chat', json=payload, timeout=TIMEOUT_SECONDS)
        if response.status_code == 200:
            return response.json()['answer']
        # MCAS refuses (HTTP 502) when no policy clause grounds an answer. The refusal is
        # the behaviour under test, so it is returned as the answer, not raised as an error.
        if response.status_code == 502:
            detail = response.json().get('detail', {})
            if isinstance(detail, dict) and detail.get('error') == 'answer_not_grounded':
                return f"Refused by MCAS: {detail.get('reason', 'no policy clause covers the question')}"
        raise RuntimeError(f'MCAS /chat returned HTTP {response.status_code}: {response.text[:300]}')
