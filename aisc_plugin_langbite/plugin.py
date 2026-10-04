import sys
import time
from typing import Any

import pandas as pd
from pandas import DataFrame

from aisc_plugin_interface import (
    BaseEvaluationPlugin,
    PluginFeatureFlags,
    InputType,
    TaskProgress,
    evaluation_input,
    metric,
)
from aisc_plugin_interface.models.measure import Measure
from aisc_plugin_interface.system_under_test import system_under_test

from .artifact_csv import global_eval_to_csv_bytes
from .custom_dataset_input_provider import CustomDatasetInputProvider
from .models import ConfigFormSchema
from .ui_schema import ui_schema

DATASET_INPUT = "dataset"


def _error_result(message: str) -> dict:
    print(f"[LangBiTe] ERROR: {message}", file=sys.stderr)
    return {"global_evaluation": [], "status": "error", "error": message}


# The model under test is the evaluation's target, reached through its endpoint in Manage: the platform
# sets AISC_TARGET_* for the run, which langbite's AISCTarget model reads. The judge keeps its own
# fields in the form (judge_model, judge_api_key), so it is never pointed at the target.
@system_under_test(protocols=("openai",),
                   env={"AISC_TARGET_BASE_URL": "base_url", "AISC_TARGET_API_KEY": "api_key",
                        "AISC_TARGET_MODEL": "model"})
@evaluation_input(
    name=DATASET_INPUT,
    label="Prompt Template (TSV)",
    input_provider_class=CustomDatasetInputProvider,
    input_type=InputType.DATASET,
    required=True,
)
class LangBiteEvaluationPlugin(BaseEvaluationPlugin[ConfigFormSchema]):
    plugin_name = "LangBiTe"
    ui_icon = "science"
    form_ui_schema = ui_schema

    @property
    def feature_flags(self) -> PluginFeatureFlags:
        # the settings page can fill the form from a LangBiTe settings file (parse_config_from_dataset)
        return PluginFeatureFlags(can_parse_config_from_dataset=True)

    def parse_config_from_dataset(self, file_content: bytes) -> dict | None:
        """The form from a LangBiTe settings file (LangBiTe's own config JSON: nTemplates, requirements with
        communities per language, ...), so a workshop's requirements are not typed in. aiModels and the
        timestamp are not taken: what is tested is the evaluation's target. Anything else (a prompt file)
        is no settings file: None."""
        import json
        try:
            raw = json.loads(file_content)
        except (ValueError, UnicodeDecodeError):
            return None
        if not isinstance(raw, dict) or not isinstance(raw.get("requirements"), list):
            return None
        form = {k: raw[k] for k in ("nTemplates", "nRetries", "temperature", "tokens", "history", "language",
                                    "judge_model") if k in raw}
        form["useLLMEval"] = bool(raw.get("useLLMEval", False))
        requirements = []
        for req in raw["requirements"]:
            communities = req.get("communities") or {}
            if isinstance(communities, dict):
                communities = [{"language": lang, "entries": list(entries)} for lang, entries in communities.items()]
            requirements.append({**{k: v for k, v in req.items() if k != "languages"}, "communities": communities})
        form["requirements"] = requirements
        return ConfigFormSchema(**form).model_dump(mode="json", exclude={"judge_api_key"})

    def form_schema_to_internal(self, config_form_data: ConfigFormSchema) -> dict:
        # mode="json": plain values for langbite (enum members would label results "LanguageEnum.en_us")
        config_data = config_form_data.model_dump(mode="json")
        config_data["aiModels"] = ["AISCTarget"]
        config_data["judge"] = {"model": config_data.pop("judge_model"), "api_key": config_data.pop("judge_api_key")}
        history = config_data.pop("history")
        config_data["history"] = None if history in (None, "none") else history
        for requirement in config_data["requirements"]:
            communities = {}
            languages = []
            for community in requirement["communities"]:
                communities[community["language"]] = community["entries"]
                languages.append(community["language"])
            requirement["languages"] = languages
            requirement["communities"] = communities
        config_data["timestamp"] = int(time.time())
        return config_data

    def evaluate(self, config_data) -> Any:
        try:
            return self._run_evaluation(config_data)
        except Exception as exc:
            import traceback
            print(f"[LangBiTe] Unhandled exception:\n{traceback.format_exc()}", file=sys.stderr)
            return _error_result(str(exc))

    def _run_evaluation(self, config_data) -> dict:
        from langbite.langbite import LangBiTeForAPI

        config: ConfigFormSchema = self.validate_config_form_data(config_data)
        langbite_config = self.form_schema_to_internal(config)
        input_language = langbite_config["language"]

        self.report_progress(TaskProgress(progress=0.05, extra={"stage": "setup"}))

        prompts = self.get_input_data(DATASET_INPUT)
        if not prompts:
            return _error_result("No prompt template provided. Upload a TSV prompt template.")

        self.report_progress(TaskProgress(progress=0.40, extra={"stage": "executing"}))

        langbite = LangBiTeForAPI({
            "prompts": prompts,
            "config": langbite_config,
            "input_language": input_language,
        })
        langbite.generate()
        langbite.execute()
        try:
            report = langbite.report()
        except KeyError as exc:
            # report() groups the results DataFrame by 'Provider','Concern',… ;
            # if NO prompts matched the run it is empty and raises KeyError. That
            # happens when a requirement's concern/inputs/reflections don't match
            # any prompt in the uploaded dataset. Give an actionable message
            # instead of the cryptic KeyError.
            return _error_result(
                "No prompts matched the configuration, so no results were produced. "
                "Check that each requirement's 'concern', 'inputs' and 'reflections' "
                "match the prompts in your uploaded dataset (e.g. the dataset's "
                f"input_type/reflection_type columns). [langbite: empty results, {exc}]"
            )

        # Normalise LangBiTe's report (global_eval DataFrame) into JSON records so
        # the metrics below can consume a plain dict.
        global_eval = report.get("global_eval")
        if isinstance(global_eval, DataFrame):
            global_records = global_eval.to_dict(orient="records")
            self.upload_artifact(
                "global_evaluation.csv",
                global_eval_to_csv_bytes(global_eval),
            )
        else:
            global_records = list(global_eval) if global_eval else []

        self.report_progress(TaskProgress(progress=0.95, extra={"stage": "done"}))
        return {"global_evaluation": global_records, "status": "success"}

    # ── Metrics (ported from the MLA-BiTe plugin, rebranded LangBiTe) ──────────
    @metric("LangBiTe Run Success")
    def export_run_success(self, evaluation_output: dict) -> list[Measure]:
        score = 1.0 if evaluation_output.get("status") == "success" else 0.0
        description = evaluation_output.get("error") or "LangBiTe execution finished"
        return [Measure(name="LangBiTe Run Success", score=score, description=description)]

    @staticmethod
    def _row_name(row: dict) -> str:
        return (f"{row.get('Concern', '')} | {row.get('Model', '')} | {row.get('Language', '')} | "
                f"{row.get('Input Type', '')} | {row.get('Reflection Type', '')}")

    @staticmethod
    def _evaluated(rows: list[dict]) -> list[dict]:
        """Rows with answers to judge: a row where the target refused everything has none, and scoring it
        0% would read as a failure."""
        return [r for r in rows if (r.get("Total") or 0) > 0]

    @metric("Bias Evaluation Results")
    def export_bias_results(self, evaluation_output: dict) -> list[Measure]:
        measures = []
        for row in self._evaluated(evaluation_output.get("global_evaluation", [])):
            total = row.get("Total", 0)
            description = (
                f"Tolerance Evaluation: {row.get('Tolerance Evaluation', 'Unknown')} | "
                f"Tolerance: {row.get('Tolerance', '')} | "
                f"Passed: {row.get('Passed Nr', 0)}/{total} | "
                f"Failed: {row.get('Failed Nr', 0)}/{total} | "
                f"Errors: {row.get('Error Nr', 0)} | Refused: {row.get('Refused Nr', 0)}"
            )
            measures.append(Measure(name=self._row_name(row), score=float(row.get("Passed Pct", 0.0)),
                                    description=description))
        return measures

    @metric("Refusals")
    def export_refusals(self, evaluation_output: dict) -> list[Measure]:
        """How often the target declined: its own outcome, neither passed nor failed."""
        measures = []
        for row in evaluation_output.get("global_evaluation", []):
            refused = row.get("Refused Nr", 0) or 0
            answered = (row.get("Passed Nr", 0) or 0) + (row.get("Failed Nr", 0) or 0) + refused
            if answered:
                measures.append(Measure(name=self._row_name(row), score=refused / answered,
                                        description=f"refused {refused}/{answered}"))
        return measures

    @metric("Overall Pass Rate")
    def export_overall_pass_rate(self, evaluation_output: dict) -> list[Measure]:
        rows = evaluation_output.get("global_evaluation", [])
        if not rows:
            error = evaluation_output.get("error", "")
            return [Measure(name="Overall Pass Rate", score=0.0, description=error or "No evaluations produced")]
        evaluated = self._evaluated(rows)
        if not evaluated:
            return []
        avg = sum(float(r.get("Passed Pct", 0.0)) for r in evaluated) / len(evaluated)
        return [Measure(name="Overall Pass Rate", score=avg)]

    @metric("All Tolerances Passed")
    def export_all_tolerances_passed(self, evaluation_output: dict) -> list[Measure]:
        rows = evaluation_output.get("global_evaluation", [])
        evaluated = self._evaluated(rows)
        if rows and not evaluated:
            return []
        n_passed = sum(1 for r in evaluated if r.get("Tolerance Evaluation") == "Passed")
        all_passed = bool(evaluated) and n_passed == len(evaluated)
        skipped = len(rows) - len(evaluated)
        return [
            Measure(
                name="All Tolerances Passed",
                score=1.0 if all_passed else 0.0,
                description=f"{n_passed}/{len(evaluated)} evaluated tolerance checks passed"
                            + (f" ({skipped} not evaluated)" if skipped else ""),
            )
        ]
