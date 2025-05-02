"""
mondrian_cp_utils.py

Utility functions for Mondrian (conditional) conformal prediction, including:
- Non-conformity score computation from calibrated probabilities
- Fitting Mondrian/class-conditional conformal predictors
- Evaluation helpers for coverage and set size
- Flexible bin creation strategies for Mondrian partitioning (class, predicted label, difficulty, etc.)

Requires: numpy, pandas, crepes
"""


from __future__ import annotations

import logging
from typing import Tuple, Dict, Optional, Literal

import numpy as np
import pandas as pd
from crepes import ConformalClassifier
from crepes.extras import MondrianCategorizer

ArrayLike = np.ndarray


# ---------------------------------------------------------------------------
# Non‑conformity helper
# ---------------------------------------------------------------------------

def probs_to_alphas(
    p: ArrayLike,
    y: Optional[ArrayLike] = None,
) -> ArrayLike:
    """Convert calibrated class‑probabilities into non‑conformity scores α.

    Parameters
    ----------
    p
        2‑D array, shape (n_samples, n_classes).
    y
        **Optional.** 1‑D integer array, true labels.  If provided we return
        the one‑column vector α_i = 1 − p_{i, y_i}.  If *not* provided, the full
        (n_samples, n_classes) matrix 1 − p is returned – this is exactly what
        you must pass to ``ConformalClassifier.predict_set`` at test time.
    """
    if not isinstance(p, np.ndarray) or p.ndim != 2:
        raise ValueError("`p` must be a 2‑D numpy array of probabilities.")

    if not np.issubdtype(p.dtype, np.floating):
        p = p.astype(float)

    if y is None:
        return 1.0 - p

    y = np.asarray(y)
    if y.ndim != 1 or len(y) != len(p):
        raise ValueError("`y` must be 1‑D and match the first dimension of `p`.")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("`y` must contain integer class indices.")

    return 1.0 - p[np.arange(len(y)), y]


# ---------------------------------------------------------------------------
# Fitting wrapper
# ---------------------------------------------------------------------------

def fit_mondrian_classifier(
    probs_cal: ArrayLike,
    *,
    y_cal: Optional[ArrayLike] = None,
    bins_cal: Optional[ArrayLike] = None,
) -> ConformalClassifier:
    """Fit a (Mondrian) conformal predictor.

    Exactly *one* of ``y_cal`` or ``bins_cal`` must be supplied.
    If ``y_cal`` is given, the result enjoys **class‑conditional** coverage.
    Otherwise, the caller provides an explicit ``bins_cal`` (e.g. from
    :func:`create_bins` below).
    """
    logging.info("--- Fitting Mondrian Conformal Classifier ---")

    if (y_cal is None) == (bins_cal is None):
        raise ValueError("Pass *either* `y_cal` (for class‑conditional CP) *or* "
                         "`bins_cal` (custom Mondrian bins), but not both.")

    if y_cal is not None:
        alphas_cal = probs_to_alphas(probs_cal, y_cal)  # 1‑D
        bins = np.asarray(y_cal, int)
    else:
        alphas_cal = probs_to_alphas(probs_cal)         # 1‑D check below
        bins = np.asarray(bins_cal, int)

    if alphas_cal.ndim != 1:
        raise ValueError("`alphas_cal` must be 1‑D after conversion (got matrix).")

    cc = ConformalClassifier()
    cc.fit(alphas_cal, bins=bins)
    logging.info("--- Mondrian Conformal Classifier Fitted ---")
    return cc


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_mondrian_prediction(
    cc: ConformalClassifier,
    probs_test: ArrayLike,
    y_test_true: ArrayLike,
    *,
    alpha: float = 0.1,
    bins_test: Optional[ArrayLike] = None,
) -> Tuple[float, float, ArrayLike, Dict[int, float]]:
    """Overall & per‑class coverage plus average set size."""
    logging.info(f"--- Evaluating Mondrian Prediction (α={alpha}) ---")

    y_true = np.asarray(y_test_true)
    if len(probs_test) == 0 or len(y_true) == 0:
        raise ValueError("Empty test probabilities or labels.")

    alphas_test = probs_to_alphas(probs_test)  # full matrix
    pred_sets = cc.predict_set(alphas_test, bins=bins_test, confidence=1 - alpha)

    contains = pred_sets[np.arange(len(y_true)), y_true]
    coverage = contains.mean()
    avg_size = pred_sets.sum(1).mean()

    class_cov = {int(c): contains[y_true == c].mean() for c in np.unique(y_true)}

    logging.info(f"coverage={coverage:.3f}, avg_set_size={avg_size:.3f}")
    return coverage, avg_size, pred_sets, class_cov


# ---------------------------------------------------------------------------
# Lightweight bin generator
# ---------------------------------------------------------------------------

def create_bins(
    *,
    strategy: Literal["none", "class_conditional", "predicted_label", "difficulty"],
    X_cal: ArrayLike,
    X_test: ArrayLike,
    calibrator,
    y_cal: Optional[ArrayLike] = None,
    difficulty_metric: Literal["uncertainty", "margin", "entropy"] = "uncertainty",
    num_bins: int = 10,
    min_bin_size: int = 50,
) -> Tuple[Optional[ArrayLike], Optional[ArrayLike]]:
    """Safe and succinct Mondrian bin creation."""

    if strategy == "none":
        return None, None

    if strategy == "class_conditional":
        if y_cal is None:
            raise ValueError("`y_cal` required for class‑conditional strategy.")
        return np.asarray(y_cal, int), None  # test bins unused

    # Both predicted‑label and difficulty need predicted probabilities
    proba_cal = calibrator.predict_proba(X_cal)
    proba_test = calibrator.predict_proba(X_test)

    if strategy == "predicted_label":
        return proba_cal.argmax(1), proba_test.argmax(1)

    if strategy == "difficulty":
        if difficulty_metric == "uncertainty":
            diff_cal = 1 - proba_cal.max(1)
            diff_test = 1 - proba_test.max(1)
        elif difficulty_metric == "margin":
            part = np.partition(-proba_cal, 1, axis=1)
            diff_cal = -(part[:, 0] + part[:, 1])
            part = np.partition(-proba_test, 1, axis=1)
            diff_test = -(part[:, 0] + part[:, 1])
        elif difficulty_metric == "entropy":
            safe_log = lambda p: np.log(np.clip(p, 1e-15, 1))
            diff_cal = -np.sum(proba_cal * safe_log(proba_cal), 1)
            diff_test = -np.sum(proba_test * safe_log(proba_test), 1)
        else:
            raise ValueError(f"Unknown difficulty metric: {difficulty_metric}")

        mc = MondrianCategorizer().fit(diff_cal.reshape(-1, 1), f=None, no_bins=num_bins)
        bins_cal = mc.apply(diff_cal.reshape(-1, 1))
        bins_test = mc.apply(diff_test.reshape(-1, 1))

        # Optional post‑processing: merge bins below `min_bin_size`
        while True:
            uniq, counts = np.unique(bins_cal, return_counts=True)
            if len(counts) == 1:
                raise ValueError("Only one bin remains after merging. Mondrian CP requires at least two bins.")
            if counts.min() >= min_bin_size:
                break
            small = uniq[np.argmin(counts)]
            # merge with nearest numerical label for simplicity
            target = uniq[0] if small != uniq[0] else uniq[1]
            bins_cal[bins_cal == small] = target
            bins_test[bins_test == small] = target
            # re‑index to compact range 0..k‑1
            mapping = {old: new for new, old in enumerate(np.unique(bins_cal))}
            bins_cal = np.vectorize(mapping.get)(bins_cal)
            bins_test = np.vectorize(mapping.get)(bins_test)

        return bins_cal, bins_test

    raise ValueError(f"Unsupported strategy: {strategy}")
