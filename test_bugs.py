"""
Tests for 4 confirmed bugs in pyGAM.

Each test is expected to FAIL on the unfixed code and PASS after the fix.
Exception: test_confidence_intervals_bad_quantile_still_raises always passes
(it verifies the exception is raised, not the message content).
"""

import pytest
import numpy as np
from pygam import LinearGAM, s
from pygam.utils import check_X


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def fitted_linear_gam():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 1))
    y = rng.standard_normal(50)
    return LinearGAM(s(0)).fit(X, y), X, y


# ---------------------------------------------------------------------------
# Bug 1 - gridsearch() error message: typo "mut be" and stray quote characters
# File: pygam/pygam.py lines 1945-1948
# ---------------------------------------------------------------------------

def test_gridsearch_invalid_objective_message_no_typo(fitted_linear_gam):
    """Error message should say 'must be', not 'mut be'."""
    gam, X, y = fitted_linear_gam
    with pytest.raises(ValueError, match="must be"):
        gam.gridsearch(X, y, objective="invalid")


def test_gridsearch_invalid_objective_message_no_stray_quotes(fitted_linear_gam):
    """Error message should not contain stray quote characters or excess whitespace."""
    gam, X, y = fitted_linear_gam
    with pytest.raises(ValueError) as exc_info:
        gam.gridsearch(X, y, objective="bad_obj")
    msg = str(exc_info.value)
    # The malformed f-string produces literal "'   '" fragments
    assert "'\\" not in msg, f"Stray backslash-quote in message: {msg!r}"
    assert "                             " not in msg, f"Excess whitespace in message: {msg!r}"


# ---------------------------------------------------------------------------
# Bug 2 - gridsearch() error message: "withunknown" missing space
# File: pygam/pygam.py line 1954
# ---------------------------------------------------------------------------

def test_gridsearch_gcv_known_scale_message_has_space():
    """Error for using GCV with a known-scale model should say 'with unknown', not 'withunknown'."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 1))
    y = rng.standard_normal(50)
    # scale=1.0 makes _known_scale=True; GCV is then invalid and should raise
    gam = LinearGAM(s(0), scale=1.0).fit(X, y)
    with pytest.raises(ValueError) as exc_info:
        gam.gridsearch(X, y, objective="GCV")
    msg = str(exc_info.value)
    assert "withunknown" not in msg, f"Missing space in error message: {msg!r}"
    assert "with unknown" in msg, f"Expected 'with unknown' in message: {msg!r}"


# ---------------------------------------------------------------------------
# Bug 3 - _get_quantiles() error message shows full list, not the bad value
# File: pygam/pygam.py line 1392
# ---------------------------------------------------------------------------

def test_confidence_intervals_bad_quantile_still_raises(fitted_linear_gam):
    """Passing an out-of-range quantile must still raise ValueError (baseline, always passes)."""
    gam, X, _ = fitted_linear_gam
    with pytest.raises(ValueError):
        gam.confidence_intervals(X, quantiles=[0.025, 1.5])


def test_confidence_intervals_bad_quantile_message_shows_value(fitted_linear_gam):
    """Error message should contain the specific bad value (1.5), not the full array.

    After conversion to numpy, quantiles become np.array([0.025, 1.5]).
    The buggy code formats the whole array as '[0.025 1.5]'; the fix should
    show just '1.5' (the invalid element).
    """
    gam, X, _ = fitted_linear_gam
    with pytest.raises(ValueError) as exc_info:
        gam.confidence_intervals(X, quantiles=[0.025, 1.5])
    msg = str(exc_info.value)
    assert "1.5" in msg, f"Bad quantile value not in message: {msg!r}"
    # numpy formats the array without a comma: '[0.025 1.5]'
    assert "[0.025 1.5]" not in msg, f"Full array should not appear in message: {msg!r}"


# ---------------------------------------------------------------------------
# Bug 4 - check_X() uses assert instead of ValueError for edge_knots validation
# File: pygam/utils.py line 312
# ---------------------------------------------------------------------------

def test_check_X_odd_edge_knots_raises_value_error():
    """Odd-length edge_knots should raise ValueError, not AssertionError.

    With Python's -O flag, assert statements are stripped entirely, so the
    check silently disappears. Replacing it with an explicit raise ensures
    the error is always surfaced regardless of optimisation level.
    """
    X = np.array([[1.0, 2.0]])
    # Provide 3 edge-knot values for a single feature -- an odd-length list
    # that violates the "must come in (min, max) pairs" contract.
    with pytest.raises(ValueError):
        check_X(
            X,
            n_feats=2,
            edge_knots=[[0.0, 1.0], [2.0]],
            dtypes=["numerical", "numerical"],
            features=[0, 1],
        )
