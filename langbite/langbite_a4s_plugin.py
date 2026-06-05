import time

from pandas import DataFrame

from aisc_plugin_interface import BaseEvaluationPlugin, metric
from aisc_plugin_interface.input_providers.base_input_provider import BaseInputProvider
from aisc_plugin_interface.models.measure import Measure
from langbite.a4s_plugin.custom_dataset_input_provider import CustomDatasetInputProvider
from langbite.a4s_plugin.models import ConfigFormSchema, LanguageEnum
from langbite.a4s_plugin.ui_schema import ui_schema


class LangBiteEvaluationPlugin(BaseEvaluationPlugin[ConfigFormSchema]):
    form_ui_schema = ui_schema


    def set_dataset_input_provider(self, file_content: bytes | None) -> BaseInputProvider:
        self.dataset_input_provider = CustomDatasetInputProvider(file_content)
        return self.dataset_input_provider


    def form_schema_to_internal(self, config_form_data: ConfigFormSchema) -> dict:
        config_data = config_form_data.model_dump()
        config_data["aiModels"] = [config_data["aiModels"]]
        for requirement in config_data["requirements"]:
            communities = {}
            languages = []
            for community in requirement["communities"]:
                communities[community["language"]] = community["entries"]
                language = LanguageEnum(community["language"])
                languages.append(language)

            requirement["languages"] = languages
            requirement["communities"] = communities

        config_data["timestamp"] = int(time.time())
        return config_data


    def evaluate(self, config_data) ->  dict[str, DataFrame | None]:
        from langbite.langbite import LangBiTeForAPI

        config: ConfigFormSchema = self.validate_config_form_data(config_data)

        langbite_config = self.form_schema_to_internal(config)
        input_language = langbite_config["language"]

        prompts = self.get_dataset()

        langbite_input = {
            "prompts": prompts,
            "config": langbite_config,
            "input_language": input_language
        }

        langbite = LangBiTeForAPI(langbite_input)

        langbite.generate()
        langbite.execute()
        langbite_report = langbite.report()

        return langbite_report

    @metric("LangBite Passed Percent")
    def export_accuracy(self, langbite_report: dict[str, DataFrame | None]) -> list[Measure]:
        passed_percent = langbite_report['global_eval']['Passed Pct'].values.take(0)
        measure = Measure(name="LangBite Passed Percent", score=passed_percent)
        return [measure]
