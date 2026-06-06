from enum import Enum

from pydantic import BaseModel, Field, model_validator


def _registered_model_keys() -> list[str]:
    """Model keys registered in langbite/resources/factories.json, so the
    'Select AI Model' dropdown offers every supported model (OpenAI, Ollama,
    HuggingFace, Replicate, GPT4ALL) instead of only GPT4ALL. Falls back to
    GPT4ALL if the registry can't be read."""
    try:
        from langbite.io_managers import json_io_manager
        keys = [f["key"] for f in json_io_manager.load_factories() if f.get("key")]
        return keys or ["GPT4ALL"]
    except Exception:
        return ["GPT4ALL"]


# Member NAME can't contain '.', so sanitise the name but keep the real key as
# the VALUE (what is shown in the dropdown and passed to langbite's llm_factory).
AIModelProvider = Enum(
    "AIModelProvider",
    {key.replace(".", "_"): key for key in _registered_model_keys()},
    type=str,
)

_DEFAULT_MODEL = (
    AIModelProvider.GPT4ALL
    if "GPT4ALL" in AIModelProvider.__members__
    else next(iter(AIModelProvider))
)


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
    nRetries: int = Field(default=1, title="Number of templates")
    temperature: float = Field(default=1.0, ge=0, le=2, title="Temperature")
    tokens: int = Field(default=60, title="Number of tokens")
    useLLMEval: bool = Field(default=True, title="Use LLMEval")
    aiModels: AIModelProvider = Field(
        default=_DEFAULT_MODEL,
        title="Select AI Model",
        description="Choose the model for this evaluation (from langbite's factories.json registry)"
    )
    model_credential: str = Field(
        default="",
        title="API Key",
        description="API key for the selected model's provider (OpenAI / HuggingFace / Replicate). "
                    "For Ollama, put the base URL here. Leave blank for GPT4ALL (runs locally)."
    )
    requirements: list[RequirementsSchema] = Field(default_factory=list, title="Requirements")
    language: LanguageEnum = Field(
        default=LanguageEnum.en_us,
        title="Language to run"
    )
    