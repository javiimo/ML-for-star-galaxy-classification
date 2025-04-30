# raw_cvap.py
"""
Implementation of IVAP **Approach 1** (efficient pre‑calculated lookup) that
works directly with *arbitrary real‑valued scores* and a lightweight CVAP
wrapper that aggregates these single‑fold calibrators.

Key differences from the reference `venn_abers.py` code
-------------------------------------------------------
* **No assumption that inputs are probabilities** – the calibrator accepts any
  real numbers produced by the underlying model (margins, logits, SVM decision
  values, …).
* Removed all clipping/normalisation to `[0,1]` from the *inputs*; only the
  **outputs** are clipped to stay inside the probability simplex.
* Uses the greatest‑convex‑minorant / least‑concave‑majorant construction from
  Vovk et al. (2015, Algorithms 2 & 3) to pre‑compute the tables `p0`, `p1` in
  \(O(k)\) time and answers each query in \(O(\log k)\).
* Provides a small `CVAPPredictorRaw` class that stores the per‑fold
  calibrators and implements the geometric‑mean aggregation (loss="log" by
  default).

The public surface consists of three items:

* `RawVennAbers`   – single‑fold calibrator (`fit`, `predict_proba`).
* `calc_p0p1`, `calc_probs` – standalone helper functions.
* `CVAPPredictorRaw` – minimal cross‑validated predictor with `predict_proba`.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Sequence, Tuple
from sklearn.exceptions import NotFittedError

# ---------------------------------------------------------------------------
#  Internal helpers – cumulative sums & envelopes
# ---------------------------------------------------------------------------

def _cum_sums(scores: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return unique score grid and cumulative counts/positives."""
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = y[order]

    c = np.unique(s_sorted)
    idx = np.searchsorted(s_sorted, c, side="right")

    w = np.diff(np.concatenate(([0], idx)))  # frequency of each unique score
    n_cum = np.cumsum(w)                     # running total n_i
    t_cum = np.cumsum(y_sorted)              # running positives t_i
    return c, idx, n_cum, t_cum


def _gcm_lcm(c: np.ndarray, idx: np.ndarray, n_cum: np.ndarray, t_cum: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Greatest convex minorant (p0) and least concave majorant (p1) slopes."""
    k = len(c)
    n_s = np.concatenate(([0], n_cum))
    t_s = np.concatenate(([0], t_cum))

    # ---- least concave majorant (p1) ----
    g = np.empty(k + 1)
    stack: List[int] = []
    for j in range(1, k + 1):
        stack.append(j)
        while len(stack) >= 3:
            i, h, j2 = stack[-3:]
            if (t_s[j2] - t_s[i]) * (n_s[h] - n_s[i]) <= (t_s[h] - t_s[i]) * (n_s[j2] - n_s[i]):
                stack.pop(-2)
            else:
                break
        if len(stack) >= 2:
            i, j2 = stack[-2:]
            g[j] = (t_s[j2] - t_s[i]) / (n_s[j2] - n_s[i])
    g[0] = g[1]  # extend left to -inf
    p1 = np.column_stack((np.concatenate(([-np.inf], c)), g))

    # ---- greatest convex minorant (p0) ----
    h = np.empty(k + 1)
    stack.clear()
    for j in range(k - 1, -1, -1):
        stack.append(j)
        while len(stack) >= 3:
            i, h2, j2 = stack[-3:]
            if (t_s[h2] - t_s[j2]) * (n_s[j2] - n_s[i]) <= (t_s[j2] - t_s[i]) * (n_s[h2] - n_s[j2]):
                stack.pop(-2)
            else:
                break
        if len(stack) >= 2:
            j2, i = stack[-2:]
            h[j] = (t_s[j2] - t_s[i]) / (n_s[j2] - n_s[i])
    h[-1] = h[-2]  # extend right to +inf
    p0 = np.column_stack((np.concatenate((c, [np.inf])), h))

    # numeric safety
    p0[:, 1] = np.clip(p0[:, 1], 0.0, 1.0)
    p1[:, 1] = np.clip(p1[:, 1], 0.0, 1.0)
    return p0, p1

# ---------------------------------------------------------------------------
#  Public helpers – pre‑compute & query
# ---------------------------------------------------------------------------

def calc_p0p1(scores: Sequence[float], y: Sequence[int], *, precision: int | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre‑compute IVAP tables for an arbitrary array of real‑valued *scores*."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.int8).ravel()
    if precision is not None:
        scores = np.round(scores, precision)
    if scores.shape[0] != y.shape[0]:
        raise ValueError("scores and y must have the same length")

    c, idx, n_cum, t_cum = _cum_sums(scores, y)
    if len(c) == 1:
        mean = y.mean()
        p0 = np.array([[c[0], mean], [np.inf, mean]], dtype=np.float64)
        p1 = np.array([[-np.inf, mean], [c[0], mean]], dtype=np.float64)
        return p0, p1, c

    p0, p1 = _gcm_lcm(c, idx, n_cum, t_cum)
    return p0, p1, c


def calc_probs(p0: np.ndarray, p1: np.ndarray, c: np.ndarray, scores: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate IVAP prediction for a batch of real‑valued *scores*."""
    s = np.asarray(scores, dtype=np.float64).ravel()

    idx_r = np.searchsorted(c, s, side="right")  # maps to p0 rows
    idx_l = np.searchsorted(c, s, side="left")   # maps to p1 rows

    upper = p0[idx_r, 1]
    lower = p1[idx_l, 1]
    bounds = np.column_stack((upper, lower))

    denom = 1.0 - upper + lower
    prob1 = np.where(denom > 0, lower / denom, 0.5)
    prob1 = np.clip(prob1, 0.0, 1.0)
    calibrated = np.column_stack((1.0 - prob1, prob1))
    return calibrated, bounds

# ---------------------------------------------------------------------------
#  Single‑fold calibrator
# ---------------------------------------------------------------------------

class RawVennAbers:
    """Efficient IVAP calibrator that operates on raw, totally ordered scores."""

    def __init__(self, *, precision: int | None = None):
        self.precision = precision
        self._fitted = False

    def fit(self, scores: Sequence[float], y: Sequence[int]):
        self.p0_, self.p1_, self.c_ = calc_p0p1(scores, y, precision=self.precision)
        self._fitted = True
        return self

    def _check(self):
        if not self._fitted:
            raise NotFittedError("RawVennAbers instance is not fitted yet.")

    def predict_proba(self, scores: Sequence[float], *, return_bounds: bool = False):
        self._check()
        proba, bounds = calc_probs(self.p0_, self.p1_, self.c_, scores)
        return (proba, bounds) if return_bounds else proba

# ---------------------------------------------------------------------------
#  K‑fold CVAP wrapper
# ---------------------------------------------------------------------------

@dataclass
class CVAPPredictorRaw:
    """Cross‑Venn‑Abers predictor for models that output raw scores."""

    final_estimator_: object           # Trained base model
    calibrators_: List[RawVennAbers]   # One per CV fold (length = K)
    loss_: str = "log"                # Aggregation loss: "log" or "brier"
    score_method_: str = "decision_function"  # How to obtain raw scores from the model

    def _aggregate(self, p0_stack: np.ndarray, p1_stack: np.ndarray) -> np.ndarray:
        if self.loss_ == "log":
            g1 = np.exp(np.mean(np.log(np.clip(p1_stack, 1e-9, 1.0)), axis=1))
            g0 = np.exp(np.mean(np.log(np.clip(1.0 - p0_stack, 1e-9, 1.0)), axis=1))
            denom = g0 + g1
            return np.where(denom > 0, g1 / denom, 0.5)
        elif self.loss_ == "brier":
            return p1_stack.mean(axis=1) + 0.5 * (p0_stack ** 2).mean(axis=1) - 0.5 * (p1_stack ** 2).mean(axis=1)
        else:
            raise ValueError("loss_ must be 'log' or 'brier'")

    # ---------------------------------------------------------------------
    #  Public API
    # ---------------------------------------------------------------------

    def predict_proba(self, X):
        # obtain raw scores
        score_fn = getattr(self.final_estimator_, self.score_method_)
        raw = score_fn(X)
        if raw.ndim > 1:
            raw = raw.ravel()

        # evaluate bounds for each fold
        p0_cols, p1_cols = [], []
        for va in self.calibrators_:
            _, bounds = va.predict_proba(raw, return_bounds=True)
            p0_cols.append(bounds[:, 0])
            p1_cols.append(bounds[:, 1])

        p0_stack = np.column_stack(p0_cols)
        p1_stack = np.column_stack(p1_cols)
        p1_final = np.clip(self._aggregate(p0_stack, p1_stack), 0.0, 1.0)
        return np.column_stack((1.0 - p1_final, p1_final))

    # sklearn compatibility helpers
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
