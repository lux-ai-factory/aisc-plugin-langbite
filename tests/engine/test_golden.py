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
        if "Refused Nr" in actual and "Refused Nr" not in expected:
            # the column refusals added after the golden run (8e8afde): no refusals here, so all zero
            assert (actual.pop("Refused Nr") == 0).all(), name
        assert_frame_equal(actual, expected, check_like=False), name
