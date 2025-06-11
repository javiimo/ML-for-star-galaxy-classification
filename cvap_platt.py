"""
cvap_platt.py  ────────────────────────────────────────────────────────────────

Parametric **CVAP** (Cross‑Venn‑Abers Predictor) that uses **Platt scaling**
instead of the non‑parametric isotonic/Venn‑Abers mapping.

Goal
────
*   Keep the *cross‑validated* training protocol that avoids information leak
    (retain theoretical low‑bias probability estimates).
*   Replace each per‑fold calibrator with a *parametric*, smooth mapping – a
    one‑dimensional logistic regression (a.k.a. **Platt scaling**).
*   Aggregate the per‑fold probabilities with the same geometric‑mean rule
    used for CVAP‑isotonic.  This preserves proper scoring‑rule optimisation
    (log‑loss) while eliminating the step‑function artefacts that appear when
    isotonic has too few positive samples.

Practical usage
───────────────
```python
from cvap_platt import fit_cvap_platt

cvap = fit_cvap_platt(base_estimator, X_train, y_train,
                      cv=5,               # K‑folds
                      score_method='decision_function',
                      C_platt=1.0)        # optional LR strength

probabilities = cvap.predict_proba(X_test)
labels        = cvap.predict(X_test, threshold=0.18)  # any post‑hoc cut‑off
```
The returned object mimics scikit‑learn’s `CalibratedClassifierCV` API: it has
`predict_proba` and `predict` (with a configurable threshold) and can be used
as a drop‑in replacement for your existing `CVAPPredictorRaw`.
"""
from __future__ import annotations

import numpy as np
from typing import Sequence, List, Optional
from dataclasses import dataclass

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import StratifiedKFold

# ──────────────────────────────────────────────────────────────────────────────
#  Single‑fold Platt calibrator
# ──────────────────────────────────────────────────────────────────────────────
class PlattCalibrator:
    """One‑dimensional logistic regression a.k.a. Platt scaling."""

    def __init__(self, *, C: float = 1.0, max_iter: int = 1000):
        self.C = C
        self.max_iter = max_iter
        self._fitted = False

    # ---------------------------------------------------------------------
    def fit(self, scores: Sequence[float], y: Sequence[int], *, sample_weight=None):
        scores = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(y, dtype=np.int8).ravel()

        self._lr = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            max_iter=self.max_iter,
            penalty="l2",
            fit_intercept=True,
        )
        self._lr.fit(scores, y, sample_weight=sample_weight)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def _check(self):
        if not self._fitted:
            raise NotFittedError("PlattCalibrator instance is not fitted yet.")

    # ------------------------------------------------------------------
    def predict_proba(self, scores: Sequence[float]):
        """Return calibrated probabilities (shape = (n_samples, 2))."""
        self._check()
        scores = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        return self._lr.predict_proba(scores)

    # Convenient accessors ------------------------------------------------
    @property
    def A_(self):  # slope
        self._check()
        return self._lr.coef_[0, 0]

    @property
    def B_(self):  # intercept
        self._check()
        return self._lr.intercept_[0]


# ──────────────────────────────────────────────────────────────────────────────
#  Cross‑fold aggregator
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class CVAPPredictorPlatt:
    """CVAP predictor that aggregates *parametric* Platt calibrators."""

    final_estimator_: object            # Fully‑trained base model on *all* data
    calibrators_: List[PlattCalibrator] # One calibrator per CV fold (length = K)
    score_method_: str = "decision_function"  # How to obtain raw scores
    loss_: str = "log"                         # Aggregation loss ("log" or "mean")
    default_threshold_: float = 0.5            # Threshold for `predict`

    # ------------------------------------------------------------------
    def _get_raw_scores(self, X):
        """Extract raw, *totally‑ordered* scores from the underlying model."""
        estimator = self.final_estimator_
        method = self.score_method_

        if method == "decision_function":
            if not hasattr(estimator, "decision_function"):
                raise AttributeError(
                    f"{estimator.__class__.__name__} has no decision_function()"
                )
            scores = estimator.decision_function(X)
        elif method == "predict_proba":
            if not hasattr(estimator, "predict_proba"):
                raise AttributeError(
                    f"{estimator.__class__.__name__} has no predict_proba()"
                )
            scores = estimator.predict_proba(X)[:, 1]
        elif method == "raw_margin_xgb":
            # XGBoost: predict(output_margin=True)
            scores = estimator.predict(X, output_margin=True)
        elif method == "raw_score_lgbm":
            # LightGBM: predict(raw_score=True)
            scores = estimator.predict(X, raw_score=True)
        else:
            # Fallback to a custom method name
            if not hasattr(estimator, method):
                raise ValueError(f"Unsupported score_method: {method}")
            scores = getattr(estimator, method)(X)

        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim > 1:
            scores = scores.ravel()
        return scores

    # ------------------------------------------------------------------
    def _aggregate(self, prob_stack: np.ndarray) -> np.ndarray:
        """Geometric‑mean (log‑loss optimal) or arithmetic aggregation."""
        if self.loss_ == "log":
            return np.exp(np.mean(np.log(np.clip(prob_stack, 1e-9, 1.0)), axis=1))
        elif self.loss_ == "mean":
            return prob_stack.mean(axis=1)
        else:
            raise ValueError("loss_ must be 'log' or 'mean'")

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        raw = self._get_raw_scores(X)
        columns = [cal.predict_proba(raw)[:, 1] for cal in self.calibrators_]
        prob_stack = np.column_stack(columns)  # shape (n_samples, K)

        p1_final = self._aggregate(prob_stack)
        p1_final = np.clip(p1_final, 0.0, 1.0)
        return np.column_stack((1.0 - p1_final, p1_final))

    # ------------------------------------------------------------------
    def predict(self, X, *, threshold: Optional[float] = None):
        """Class labels using an arbitrary cut‑off (default = 0.5)."""
        if threshold is None:
            threshold = self.default_threshold_
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)


# ──────────────────────────────────────────────────────────────────────────────
#  Convenience factory
# ──────────────────────────────────────────────────────────────────────────────

def fit_cvap_platt(
    estimator,
    X,
    y,
    *,
    cv: int = 5,
    score_method: str = "decision_function",
    C_platt: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
):
    """Fit *estimator* and a parametric CVAP‑Platt calibrator in one call.

    Returns
    -------
    CVAPPredictorPlatt
        Drop‑in object with `predict_proba` and `predict` methods.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    calibrators: List[PlattCalibrator] = []

    for train_idx, cal_idx in skf.split(X, y):
        # 1) train base model on K‑1 folds
        est_fold = clone(estimator)
        est_fold.fit(X[train_idx], y[train_idx])

        # 2) obtain raw scores on the *held‑out* fold
        if score_method == "decision_function":
            scores_cal = est_fold.decision_function(X[cal_idx])
        elif score_method == "predict_proba":
            scores_cal = est_fold.predict_proba(X[cal_idx])[:, 1]
        elif score_method == "raw_margin_xgb":
            scores_cal = est_fold.predict(X[cal_idx], output_margin=True)
        elif score_method == "raw_score_lgbm":
            scores_cal = est_fold.predict(X[cal_idx], raw_score=True)
        else:
            scores_cal = getattr(est_fold, score_method)(X[cal_idx])

        # 3) fit Platt on that fold
        platt = PlattCalibrator(C=C_platt, max_iter=max_iter)
        platt.fit(scores_cal, y[cal_idx])
        calibrators.append(platt)

    # 4) retrain final estimator on the *full* data
    final_estimator = clone(estimator).fit(X, y)

    return CVAPPredictorPlatt(
        final_estimator_=final_estimator,
        calibrators_=calibrators,
        score_method_=score_method,
        loss_="log",
    )
