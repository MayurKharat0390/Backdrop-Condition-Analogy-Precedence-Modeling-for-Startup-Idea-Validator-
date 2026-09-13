import pytest
import numpy as np
from src.evaluation import run_paired_bootstrap


def test_paired_bootstrap_identical_models():
    # If two models have identical predictions, delta should be 0.0
    y_test = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1])
    probs = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.1, 0.6, 0.2, 0.85])
    
    res = run_paired_bootstrap(y_test, probs, probs, capacity_k=3, n_resamples=50)
    assert np.isclose(res["mean_delta_precision_at_k"], 0.0)
    assert np.isclose(res["mean_delta_roi"], 0.0)


def test_paired_bootstrap_superior_model():
    y_test = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    p_good = np.array([0.95, 0.90, 0.85, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    p_bad = np.array([0.1, 0.1, 0.1, 0.95, 0.90, 0.85, 0.1, 0.1, 0.1, 0.1])
    
    res = run_paired_bootstrap(y_test, p_good, p_bad, capacity_k=3, n_resamples=50)
    assert res["mean_delta_precision_at_k"] > 0.0
    assert res["mean_delta_roi"] > 0.0
