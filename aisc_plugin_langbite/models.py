from enum import Enum

from pydantic import BaseModel, Field, model_validator


def _factories() -> list[dict]:
    try:
        from langbite.io_managers import json_io_manager
        return json_io_manager.load_factories()
    except Exception:
        return []


def _history_names() -> list[str]:
    try:
        from langbite.io_managers import json_io_manager
        return list(json_io_manager.load_histories())
    except Exception:
        return []


# The judge (LLMEval) is an OpenAI chat model of langbite's factories.json. A member NAME can't hold
# '.', so the name is sanitised and the VALUE keeps the real key.
_JUDGE_KEYS = [f["key"] for f in _factories() if f.get("key") and f.get("provider", "").upper() == "OPENAI"]
JudgeModel = Enum("JudgeModel", {k.replace(".", "_"): k for k in (_JUDGE_KEYS or ["OpenAIGPT4oMini"])}, type=str)
_DEFAULT_JUDGE = JudgeModel("OpenAIGPT4oMini") if "OpenAIGPT4oMini" in _JUDGE_KEYS else next(iter(JudgeModel))

# A fixed conversation sent before every test prompt (langbite/resources/histories.json), or none.
HistoryChoice = Enum("HistoryChoice", {n: n for n in ["none", *_history_names()]}, type=str)


class LanguageEnum(str, Enum):
    en_us = "en_us"
    es_es = "es_es"
    ca_es = "ca_es"
    fr_fr = "fr_fr"


class RequirementInputs(str, Enum):
    constrained = "constrained"
    verbose = "verbose"


class RequirementReflections(str, Enum):
    observational = "observational"
    utopian = "utopian"


class Communities(BaseModel):
    language: LanguageEnum = Field(..., title="Select Language")
    entries: list[str] = Field(
        default_factory=list,
        title="Community Strings"
    )


class RequirementsSchema(BaseModel):
    @model_validator(mode='after')
    def check_lengths(self):
        lengths = [len(community.entries) for community in self.communities]
        if lengths and len(set(lengths)) > 1:
            raise ValueError("Inconsistent list lengths found in communities.")
        return self

    name: str = Field(..., title="Requirement Name")
    rationale: str = Field(..., title="Rationale")
    languages: set[LanguageEnum] = Field(
        default=[],
        title="Supported Languages"
    )
    tolerance: float = Field(0.9, ge=0, le=1)
    delta: float = Field(0.02)
    concern: str = Field(...)
    markup: str = Field(...)
    communities: list[Communities] = Field(
        default_factory=list,
        title="Communities"
    )
    inputs: set[RequirementInputs] = Field(
        default=[input.value for input in RequirementInputs],
        title="Inputs"
    )
    reflections: set[RequirementReflections] = Field(default=[reflection.value for reflection in RequirementReflections], title="Reflections")


class ConfigFormSchema(BaseModel):
    nTemplates: int = Field(default=60, title="Number of templates")
    nRetries: int = Field(default=1, title="Number of retries")
    temperature: float = Field(default=1.0, ge=0, le=2, title="Temperature")
    tokens: int = Field(default=60, title="Number of tokens")
    history: HistoryChoice = Field(
        default=HistoryChoice("none"),
        title="Conversation history",
        description="A fixed conversation sent before every test prompt. For a target that has its own "
                    "system prompt (such as MCAS-lite /chat) choose a *_api history.",
    )
    useLLMEval: bool = Field(
        default=False, title="Use LLMEval",
        description="Ask a judge model to re-check every answer that fails a check. Needs the judge's key.")
    judge_model: JudgeModel = Field(
        default=_DEFAULT_JUDGE, title="Judge model (LLMEval)",
        description="The model that judges the answers: a tool setting, never the system under test.")
    judge_api_key: str = Field(
        default="", title="Judge API key (LLMEval)",
        description="The OpenAI API key for the judge model. Used only with LLMEval.",
        json_schema_extra={"format": "password"},
    )
    requirements: list[RequirementsSchema] = Field(default_factory=list, title="Requirements")
    language: LanguageEnum = Field(
        default=LanguageEnum.en_us,
        title="Language to run"
    )
    