from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from openstata import StataFrame, ci_mean, summarize


@pytest.mark.parametrize("dtype", ["float16", "float32", "Float32", "float64"])
def test_summary_reductions_do_not_overflow_low_precision_storage(dtype: str) -> None:
    series = pd.Series([-30000.0, 30000.0, -20000.0, 20000.0], dtype=dtype)
    frame = pd.DataFrame({"signal": series})
    original = frame.copy(deep=True)
    values = series.to_numpy(dtype=np.float64)
    result = summarize(frame, detail=True).loc["signal"]
    assert result["Std. dev."] == pytest.approx(np.std(values, ddof=1), rel=1e-12)
    assert result["Variance"] == pytest.approx(np.var(values, ddof=1), rel=1e-12)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("dtype", ["float32", "Float32", "float64"])
def test_large_finite_means_and_intervals_remain_finite(dtype: str) -> None:
    series = pd.Series([1e38, 1e38, 1e38, 1e38, None], dtype=dtype)
    frame = pd.DataFrame({"signal": series})
    expected = series.dropna().to_numpy(dtype=np.float64).mean()
    result = StataFrame(frame).run("ci means signal").loc["signal"]
    assert np.isfinite(result["Mean"])
    assert result["Mean"] == pytest.approx(expected, rel=1e-12)
    assert result["CI lower"] == pytest.approx(expected, rel=1e-12)
    assert result["CI upper"] == pytest.approx(expected, rel=1e-12)
    assert result["Obs"] == 4


@pytest.mark.parametrize("dtype", ["float32", "Float32"])
def test_large_variation_has_a_finite_student_t_interval(dtype: str) -> None:
    series = pd.Series([-1e20, 1e20, 0.0, None], dtype=dtype)
    values = series.dropna().to_numpy(dtype=np.float64)
    standard_error = np.std(values, ddof=1) / np.sqrt(len(values))
    margin = stats.t.ppf(0.975, df=len(values) - 1) * standard_error
    result = ci_mean(pd.DataFrame({"signal": series})).loc["signal"]
    assert result["Std. err."] == pytest.approx(standard_error, rel=1e-12)
    assert result["CI lower"] == pytest.approx(-margin, rel=1e-12)
    assert result["CI upper"] == pytest.approx(margin, rel=1e-12)
