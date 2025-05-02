"""
Mondrian Inductive Conformal Classifier (binary / multiclass)

Works with *any* calibrated‑probability source – including
cross Venn–Abers predictors – and offers four binning strategies:
    • "none"              – ordinary ICP
    • "class_conditional" – one bin per true class (exact Mondrian CP)
    • "predicted_label"   – one bin per arg‑max label
    • "difficulty"        – equal‑sized bins of a chosen difficulty
                              estimate using crepes.extras.*

Requires: crepes ≥ 0.8.0
"""
from __future__ import annotations
import logging
from typing import Literal, Optional, Tuple, Dict

import numpy as np
from crepes import ConformalClassifier
from crepes.extras import (
    hinge,                          # 1‑p_{i,y_i}   (same as your probs_to_alphas)
    margin,                         # optional alternative α
    DifficultyEstimator,
    MondrianCategorizer,
)

Array = np.ndarray
_logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 1.  Non‑conformity
# ------------------------------------------------------------------
def probs_to_alphas(
    probs: Array,
    y: Optional[Array] = None,
    method: Literal["hinge", "margin"] = "hinge",
) -> Array:
    """
    Convert calibrated probabilities → α.

    Parameters
    ----------
    probs : (n, k) float
    y     : (n,) int or None      – required at *fit* time, None at *predict* time
    method: "hinge" | "margin"    – hinge = 1 - p_y ; margin = max_other - p_y
    """
    fn = hinge if method == "hinge" else margin
    return fn(probs, y=y)          # crepes handles both 1‑D and 2‑D return shapes


# ------------------------------------------------------------------
# 2.  Mondrian bin generator
# ------------------------------------------------------------------
def _difficulty_function(
    metric: Literal["uncertainty", "margin", "entropy"],
    calibrator,
):
    """Return a vectorised f(X) usable by MondrianCategorizer."""
    import numpy as np
    from scipy.special import entr     # numerically stable entropy

    if metric == "uncertainty":
        return lambda X: 1.0 - calibrator.predict_proba(X).max(axis=1)

    if metric == "margin":
        def _margin(X):
            p = calibrator.predict_proba(X)
            part = np.partition(-p, 1, axis=1)
            return -(part[:, 0] + part[:, 1])
        return _margin

    if metric == "entropy":
        return lambda X: entr(calibrator.predict_proba(X)).sum(axis=1)

    raise ValueError(f"Unknown difficulty metric: {metric!r}")


def create_bins(
    *,
    strategy: Literal[
        "none", "class_conditional", "predicted_label", "difficulty"
    ],
    X_cal: Array,
    X_test: Array,
    calibrator,                      # must expose .predict_proba(X)
    y_cal: Optional[Array] = None,
    difficulty_metric: Literal["uncertainty", "margin", "entropy"] = "uncertainty",
    num_bins: int = 10,
    min_bin_size: int = 50,
) -> Tuple[Optional[Array], Optional[Array]]:
    """
    Produce (bins_cal, bins_test) following Henrik Boström’s
    Mondrian taxonomy idea :contentReference[oaicite:0]{index=0}
    """

    if strategy == "none":
        return None, None

    if strategy == "class_conditional":
        if y_cal is None:
            raise ValueError("`y_cal` required for class‑conditional bins.")
        return np.asarray(y_cal, int), None           # bins_test unused

    # --------------------------------------------------------------
    # All remaining strategies need predicted probabilities
    # --------------------------------------------------------------
    P_cal = calibrator.predict_proba(X_cal)
    P_test = calibrator.predict_proba(X_test)

    if strategy == "predicted_label":
        return P_cal.argmax(1), P_test.argmax(1)

    # --------------------------------------------------------------
    # Difficulty bins  ––  real “Mondrian ICP” with DifficultyEstimator
    # --------------------------------------------------------------
    if strategy == "difficulty":
        # 1) Build f(X) := difficulty score
        f = _difficulty_function(difficulty_metric, calibrator)

        # 2) Optional: put f inside a DifficultyEstimator if you prefer the
        #    KNN / variance‑based difficulty from crepes.
        #    Here we stick to the simple “external function” route:
        de = DifficultyEstimator()        # dummy – not used when f is given

        # 3) Fit categorizer on *probability* space – that’s enough
        mc = MondrianCategorizer()
        mc.fit(P_cal, f=f, de=de, no_bins=num_bins)   # bins ≈ equal‑sized
        bins_cal = mc.apply(P_cal)
        bins_test = mc.apply(P_test)

        # 4) Merge undersized bins (crepes bins are equal‑sized *by count*,
        #    so usually no merge is needed – but keep a safety net)
        while True:
            labels, counts = np.unique(bins_cal, return_counts=True)
            if counts.min() >= min_bin_size or len(labels) == 1:
                break
            # merge smallest into nearest id
            small = labels[np.argmin(counts)]
            tgt = labels[0] if small != labels[0] else labels[1]
            bins_cal[bins_cal == small] = tgt
            bins_test[bins_test == small] = tgt
            relabel = {old: new for new, old in enumerate(np.unique(bins_cal))}
            bins_cal = np.vectorize(relabel.get)(bins_cal)
            bins_test = np.vectorize(relabel.get)(bins_test)

        return bins_cal.astype(int), bins_test.astype(int)

    raise ValueError(f"Unknown binning strategy: {strategy!r}")


# ------------------------------------------------------------------
# 3.  Fit Mondrian ICP
# ------------------------------------------------------------------
def fit_mondrian_classifier(
    probs_cal: Array,
    *,
    y_cal: Optional[Array] = None,
    bins_cal: Optional[Array] = None,
    alpha_method: Literal["hinge", "margin"] = "hinge",
) -> ConformalClassifier:
    """
    Fit crepes.ConformalClassifier.

    Either supply y_cal (→ class‑conditional ICP) **or**
    pre‑computed bins_cal (from `create_bins`).
    """
    if (y_cal is None) == (bins_cal is None):
        raise ValueError("Provide *exactly one* of `y_cal` or `bins_cal`.")

    alphas_cal = probs_to_alphas(probs_cal, y_cal, method=alpha_method)
    if alphas_cal.ndim != 1:
        raise ValueError("Calibration α must be a 1‑D vector.")

    cc = ConformalClassifier()
    cc.fit(alphas_cal, bins=bins_cal if bins_cal is not None else y_cal)
    return cc


# ------------------------------------------------------------------
# 4.  Evaluate
# ------------------------------------------------------------------
def evaluate_mondrian_prediction(
    cc: ConformalClassifier,
    probs_test: Array,
    y_true: Array,
    *,
    bins_test: Optional[Array] = None,
    alpha: float = 0.1,
    alpha_method: Literal["hinge", "margin"] = "hinge",
) -> Tuple[float, float, Array, Dict[int, float]]:
    """
    Compute coverage, average set size and per‑class coverage.
    """
    alphas_test = probs_to_alphas(probs_test, None, method=alpha_method)
    pred_sets = cc.predict_set(alphas_test, bins=bins_test, confidence=1 - alpha)

    contains = pred_sets[np.arange(len(y_true)), y_true]
    coverage = float(contains.mean())
    avg_set_size = float(pred_sets.sum(1).mean())
    class_cov = {
        int(c): float(contains[y_true == c].mean()) for c in np.unique(y_true)
    }
    return coverage, avg_set_size, pred_sets, class_cov
