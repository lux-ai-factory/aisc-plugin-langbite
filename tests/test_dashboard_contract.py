"""What the results dashboard reads from LangBiTe (aisc docs/superpowers/plugin-dashboards-2026-10-04, T1.7 to
T1.9): each row's measures are named by their metric and carry the row (concern, model, language, input type,
reflection type) as dimensions; the default charts group by those.

Before 0.2.5 a row's pass rate and its refusal rate shared one name (the row), so any average of a concern mixed
the two, and the engine's results page, which asks for the metric names, found neither."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aisc_plugin_interface import ChartType  # noqa: E402
from aisc_plugin_langbite.plugin import LangBiteEvaluationPlugin  # noqa: E402

ROWS = [
    {"Concern": "ageism", "Model": "AISCTarget", "Language": "en_us", "Input Type": "constrained",
     "Reflection Type": "observational", "Passed Nr": 4, "Failed Nr": 0, "Total": 4, "Passed Pct": 1.0,
     "Refused Nr": 0, "Tolerance": 0.8, "Tolerance Evaluation": "Passed", "Error Nr": 0},
    {"Concern": "sexism", "Model": "AISCTarget", "Language": "en_us", "Input Type": "verbose",
     "Reflection Type": "observational", "Passed Nr": 0, "Failed Nr": 2, "Total": 2, "Passed Pct": 0.0,
     "Refused Nr": 1, "Tolerance": 0.8, "Tolerance Evaluation": "Failed", "Error Nr": 0},
]
ROW_DIMS = {"concern", "model", "language", "input_type", "reflection_type"}


def exported():
    return LangBiteEvaluationPlugin().export_metrics({"status": "success", "global_evaluation": ROWS})


def test_t1_7_row_measures_are_named_by_their_metric_and_carry_the_row():
    bias = [m for m in exported() if m.name == "Bias Evaluation Results"]
    refusals = [m for m in exported() if m.name == "Refusals"]
    assert [m.dimensions["concern"] for m in bias] == ["ageism", "sexism"]
    assert [m.score for m in bias] == [1.0, 0.0]
    assert [m.dimensions["concern"] for m in refusals] == ["ageism", "sexism"]
    assert [round(m.score, 3) for m in refusals] == [0.0, round(1 / 3, 3)]
    for m in bias + refusals:
        assert set(m.dimensions) == ROW_DIMS
        assert all(isinstance(v, str) for v in m.dimensions.values())
    assert bias[1].dimensions == {"concern": "sexism", "model": "AISCTarget", "language": "en_us",
                                  "input_type": "verbose", "reflection_type": "observational"}
    assert bias[0].description.startswith("Tolerance Evaluation: Passed")      # the text stays as it was


def test_t1_7_no_two_metrics_share_a_name():
    names = [m.name for m in exported()]
    assert set(names) <= set(LangBiteEvaluationPlugin().get_metrics())


def test_t1_8_default_charts():
    charts = LangBiteEvaluationPlugin().get_metric_visualizations({})
    assert [c.title for c in charts] == ["Overall pass rate", "Pass rate per concern", "Refusals per concern"]
    assert charts[0].chart_type == ChartType.TABLE and charts[0].metrics == ["Overall Pass Rate", "All Tolerances Passed"]
    assert charts[1].chart_type == ChartType.BARS and charts[1].metrics == ["Bias Evaluation Results"]
    assert charts[1].group_by_dimensions == ["concern"]
    assert charts[2].metrics == ["Refusals"] and charts[2].group_by_dimensions == ["concern"]


def test_t1_9_every_metric_is_exported_and_every_grouping_is_carried():
    plugin = LangBiteEvaluationPlugin()
    carried = {}
    for m in exported():
        carried.setdefault(m.name, set()).update((m.dimensions or {}).keys())
    for chart in plugin.get_metric_visualizations({}):
        assert set(chart.metrics) <= set(plugin.get_metrics()), chart.title
        for dim in chart.group_by_dimensions or []:
            for metric in chart.metrics:
                assert dim in carried.get(metric, set()), (chart.title, metric, dim)
