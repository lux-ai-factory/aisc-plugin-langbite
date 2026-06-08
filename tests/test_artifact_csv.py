"""Regression test for the global-evaluation artifact CSV.

The platform renders `.csv` artifacts by parsing them with the standard comma
delimiter (backend `aisc_backend.utils.file_utils.csv_bytes_to_rows`, which uses
`csv.reader` with its default delimiter). Earlier the plugin wrote the artifact
with `sep=";"`, so every line collapsed into a single cell and the
global-evaluation table rendered as one mangled column.

These tests pin the contract: the artifact must be comma-delimited and survive a
round-trip through a comma `csv.reader`, including fields that themselves contain
commas (which pandas must quote).
"""
import csv
import importlib.util
import io
from pathlib import Path

import pandas as pd

# Load the helper module directly by file path so we do NOT trigger
# `langbite/__init__.py`, which imports the full plugin (and aisc_plugin_interface).
_HELPER_PATH = (
    Path(__file__).resolve().parent.parent
    / "langbite"
    / "aisc_plugin"
    / "artifact_csv.py"
)
_spec = importlib.util.spec_from_file_location("langbite_artifact_csv", _HELPER_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
global_eval_to_csv_bytes = _mod.global_eval_to_csv_bytes


def _read_like_backend(csv_bytes: bytes) -> list[list[str]]:
    """Mirror backend csv_bytes_to_rows: comma csv.reader over decoded bytes."""
    reader = csv.reader(io.StringIO(csv_bytes.decode("utf-8")))
    return list(reader)


def _sample_global_eval() -> pd.DataFrame:
    # Mimics LangBiTe's global_eval report; one description contains a comma so
    # we also verify quoting keeps it as a single field.
    return pd.DataFrame(
        [
            {
                "Concern": "racism",
                "Model": "gpt-4o",
                "Tolerance Evaluation": "Passed, with notes",
                "Passed Pct": 0.9,
                "Total": 10,
            },
            {
                "Concern": "sexism",
                "Model": "gpt-4o",
                "Tolerance Evaluation": "Failed",
                "Passed Pct": 0.4,
                "Total": 10,
            },
        ]
    )


def test_artifact_is_comma_delimited_and_splits_into_columns():
    df = _sample_global_eval()
    rows = _read_like_backend(global_eval_to_csv_bytes(df))

    # Header must split into the same number of columns as the DataFrame.
    assert rows[0] == list(df.columns)
    assert len(rows[0]) == len(df.columns) > 1
    # Every data row has one cell per column (would be a single cell if ;-delimited).
    for data_row in rows[1:]:
        assert len(data_row) == len(df.columns)


def test_comma_inside_field_is_preserved():
    df = _sample_global_eval()
    rows = _read_like_backend(global_eval_to_csv_bytes(df))

    col = list(df.columns).index("Tolerance Evaluation")
    assert rows[1][col] == "Passed, with notes"


def test_roundtrips_back_to_dataframe():
    df = _sample_global_eval()
    restored = pd.read_csv(io.BytesIO(global_eval_to_csv_bytes(df)))
    pd.testing.assert_frame_equal(restored, df)
