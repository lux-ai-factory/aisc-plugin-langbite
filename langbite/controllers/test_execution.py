from langbite.controllers.test_scenario import TestScenario
from langbite.model.prompt import Prompt
from langbite.llm_services import llm_factory
from langbite.llm_services.llm_service import LLMService
from langbite.oracles.sentiment_analyzer_oracle import SentimentAnalyzerOracle
import langbite.io_managers.secrets as Secrets
from langbite.model.view_model import EvaluationView, ResponseView
import time

class TestExecution:

    @property
    def responses(self) -> list[ResponseView]:
        if self.__responses is None: self.__responses = []
        return self.__responses

    @property
    def evaluations(self) -> list[EvaluationView]:
        if self.__evaluations is None: self.__evaluations = []
        return self.__evaluations
    
    def __init__(self, scenario: TestScenario):
        self.__scenario = scenario
        self.__responses = []
        self.__evaluations = []
        self.__config = Secrets.load_api_keys()
        # made before any prompt is sent, so a missing judge key stops the run at the start
        self.__llm_sentiment = (SentimentAnalyzerOracle(scenario.judge_model, scenario.judge_api_key)
                                if scenario.use_llm_eval else None)
    
    def execute_scenario(self):
        for model in self.__scenario.models:
            self.__query_model(model)
    
    def __query_model(self, model: str):
        print(f'querying {model}...')
        llmservice: LLMService = llm_factory.factory.create(model, **self.__config)
        llmservice.temperature = self.__scenario.temperature
        llmservice.tokens = self.__scenario.tokens
        provider = llmservice.provider
        history = self.__scenario.history
        # fail before the loop: inside it every exception is retried and recorded as 'Error'
        if history: llmservice.validate_history(history)
        label = f'{model}+{self.__scenario.history_name}' if history else model
        prompt: Prompt
        print(f'running {len(self.__scenario.prompts)} prompts...')
        for i, prompt in enumerate(self.__scenario.prompts):
            print(f'running prompt {i}')
            # set a number of attempts to retry when the model either:
            # a) raises an exception (connection issue, time out, etc.)
            # b) the model replies in an unexpected format
            n_attempts = self.__scenario.num_retries
            while n_attempts > 0:
                try:
                    prompt.execute(llmservice, history)
                    evaluation = prompt.evaluate(self.__llm_sentiment)
                    self.__update_responses(provider, label, prompt)
                    self.__update_evaluations(provider, label, prompt, evaluation)
                    break
                except Exception as ex:
                    n_attempts = n_attempts - 1
                    if (n_attempts == 0):
                        self.__update_responses_error(provider, label, prompt, ex.args[0])
                        #self.__update_evaluations_error(provider, model, prompt, ex.args[0])
                        self.__update_evaluations(provider, label, prompt, 'Error')
                    else:
                        time.sleep(1) # sleep for 1 second to allow the model to restore
        print('done')
    
    def __update_responses(self, provider, model, prompt: Prompt):
        if len(prompt.responses) == 0:
            self.responses.append(ResponseView(provider, model, 'Template: ' + prompt.template, 'No response provided'))
        else:
            for prompt_response in prompt.responses:
                self.responses.append(ResponseView(provider, model, prompt_response.instance, prompt_response.response))

    def __update_evaluations(self, provider, model, prompt: Prompt, evaluation: str):
        judged = [(str(r.instance), r.raw_response) for r in prompt.responses]
        self.evaluations.append(EvaluationView(provider, model, prompt.concern, prompt.language, prompt.input_type, prompt.reflection_type, prompt.template, prompt.oracle_operation, prompt.oracle_prediction, evaluation, judged))

    def __update_responses_error(self, provider, model, prompt: Prompt, error_msg):
        for prompt_response in prompt.responses:
            self.responses.append(ResponseView(provider, model, prompt_response.instance, 'ERROR: ' + error_msg))

    def __update_evaluations_error(self, provider, model, prompt: Prompt, error_msg):
        self.evaluations.append(EvaluationView(provider, model, prompt.concern, prompt.language, prompt.input_type, prompt.reflection_type, prompt.template, prompt.oracle_operation, prompt.oracle_prediction, 'ERROR: ' + error_msg))
