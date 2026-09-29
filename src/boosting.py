"""LightGBM Booster wrapper used by final.py (kept in its own module so saved models load anywhere)."""
import numpy as np


class BoosterModel:
    """Wraps a LightGBM Booster with the predict_proba interface the pipeline expects."""

    def __init__(self, booster):
        self.booster_ = booster
        self.best_iteration_ = booster.best_iteration

    def predict_proba(self, X):
        p = self.booster_.predict(X, num_iteration=self.best_iteration_)
        return np.column_stack([1 - p, p])
