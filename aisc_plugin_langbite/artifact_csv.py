"""Serialization of the LangBiTe global-evaluation report to a CSV artifact.

The platform's artifact preview parses `.csv` artifacts with the standard comma
delimiter (backend `aisc_backend.utils.file_utils.csv_bytes_to_rows`, which uses
`csv.reader` with its default delimiter). Writing the artifact with any other
delimiter (e.g. ``;``) collapses each line into a single cell, so the
global-evaluation table renders as one mangled column.

We therefore emit standard comma-delimited CSV. pandas quotes any field that
itself contains a comma, so values stay intact through the comma parser.
"""
from pandas import DataFrame


def global_eval_to_csv_bytes(global_eval: DataFrame) -> bytes:
    """Return the global-evaluation report as comma-delimited CSV bytes."""
    return global_eval.to_csv(index=False).encode()
