"""
asymmetric_loss.py: Asymmetric Classification Loss & Focal Objectives for Model Training.

Formulation:
- Standard BCE treats false positives and false negatives symmetrically.
- Asymmetric Loss introduces explicit loss gradient weights w1 and w0:
    L_asymmetric = - (1/N) * sum( w1 * y * log(p) + w0 * (1 - y) * log(1 - p) )
- Asymmetric Focal Loss adds focusing parameter gamma (default gamma=2):
    L_focal = - (1/N) * sum( w1 * y * (1 - p)^gamma * log(p) + w0 * (1 - y) * p^gamma * log(1 - p) )

Research Integrity Note:
Weights w1 and w0 represent training loss penalties, NOT literal dollar amounts.
Monetary valuations are strictly handled downstream in the Economic Decision Layer.
"""

from typing import Union, Tuple
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


def asymmetric_cross_entropy_loss(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    w1: float = 1.0,
    w0: float = 1.0,
    eps: float = 1e-15
) -> float:
    """
    Compute Asymmetric Binary Cross-Entropy Loss.
    
    Parameters:
        y_true: Ground truth binary targets (0 or 1).
        y_pred_proba: Predicted probabilities P(Y=1|X).
        w1: Loss multiplier for positive class (successes).
        w0: Loss multiplier for negative class (failures).
        eps: Small numerical stabilizer to prevent log(0).
        
    Returns:
        Scalar loss value.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_pred_proba, dtype=float), eps, 1.0 - eps)
    
    loss = - (w1 * y * np.log(p) + w0 * (1.0 - y) * np.log(1.0 - p))
    return float(np.mean(loss))


def asymmetric_focal_loss(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    w1: float = 1.0,
    w0: float = 1.0,
    gamma: float = 2.0,
    eps: float = 1e-15
) -> float:
    """
    Compute Asymmetric Focal Loss (Lin et al., with asymmetric class weights).
    
    Focuses gradient updates on hard, borderline instances by modulating factor (1 - p_t)^gamma.
    
    Parameters:
        y_true: Ground truth binary targets (0 or 1).
        y_pred_proba: Predicted probabilities P(Y=1|X).
        w1: Loss weight for successes.
        w0: Loss weight for failures.
        gamma: Focusing parameter (default 2.0).
        eps: Small numerical stabilizer.
        
    Returns:
        Scalar focal loss value.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_pred_proba, dtype=float), eps, 1.0 - eps)
    
    focal_pos = w1 * y * ((1.0 - p) ** gamma) * np.log(p)
    focal_neg = w0 * (1.0 - y) * (p ** gamma) * np.log(1.0 - p)
    
    loss = - (focal_pos + focal_neg)
    return float(np.mean(loss))


def compute_sample_weights_for_asymmetric_loss(
    y: np.ndarray,
    w1: float = 1.0,
    w0: float = 1.0
) -> np.ndarray:
    """
    Compute per-sample weights mapping to asymmetric loss objectives.
    
    In convex cross-entropy minimization:
        sum_i sample_weight_i * BCE(y_i, p_i) 
    is mathematically equivalent to the asymmetric cross-entropy objective.
    """
    y_arr = np.asarray(y, dtype=int)
    weights = np.where(y_arr == 1, float(w1), float(w0))
    return weights


class AsymmetricGradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    """
    Gradient Boosting Classifier trained under asymmetric loss weights.
    """
    _estimator_type = "classifier"
    def __init__(
        self,
        max_iter: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        w1: float = 1.0,
        w0: float = 3.0,
        random_state: int = 42
    ):
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.w1 = w1
        self.w0 = w0
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        sample_weights = compute_sample_weights_for_asymmetric_loss(y, self.w1, self.w0)
        self.model = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state
        )
        self.model.fit(X, y, sample_weight=sample_weights)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
