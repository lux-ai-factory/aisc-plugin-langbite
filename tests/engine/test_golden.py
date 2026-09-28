"""T14: a run without a history reproduces the output captured on 8e8afde."""
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from fakes import RecordingService, register_fake, run_api

GOLDEN = Path(__file__).resolve().parent / "golden"


def test_t14_no_history_run_matches_golden(fake_openai):
    register_fake("FakeYes", RecordingService("FakeYes", "Yes"))
    register_fake("FakeNo", RecordingService("FakeNo", "No"))
    report = run_api(["FakeYes", "FakeNo"])
    for name, df in report.items():
        expected = pd.read_csv(GOLDEN / f"{name}.csv")
        actual = pd.read_csv(pd.io.common.StringIO(df.to_csv(index=False)))
        assert_frame_equal(actual, expected, check_like=False), name
