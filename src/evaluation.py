"""
evaluation.py: Unified Machine Learning, Decision, and Economic Evaluation Framework.

Metrics Taxonomy:
1. Standard ML Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Brier Score.
2. Decision Metrics: Operating Threshold, Selection Rate, Investments Count, Precision@K, Recall@K.
3. Economic Metrics: Expected Monetary Value, Portfolio Net Return, Portfolio ROI, Capital Loss, Opportunity Loss.
4. Portfolio Metrics: Capital Preservation Ratio (CPR), Winner Capture Rate, Failure Avoidance Rate.
"""

from typing import Dict, Union, List, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss
)
from src.portfolio_simulation import simulate_portfolio
from src.economic_utility import calculate_emv


def compute_precision_recall_at_k(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k: int = 100
) -> Dict[str, float]:
    """
    Compute Precision@K and Recall@K for top-K ranked deals by predicted probability.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    
    k = min(k, len(y))
    top_indices = np.argsort(p)[::-1][:k]
    
    top_actuals = y[top_indices]
    prec_k = float(np.sum(top_actuals == 1) / k)
    total_positives = np.sum(y == 1)
    rec_k = float(np.sum(top_actuals == 1) / total_positives) if total_positives > 0 else 0.0
    
    return {f"Precision@{k}": prec_k, f"Recall@{k}": rec_k}


def evaluate_vc_model(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.05,
    investment_amount: float = 1.0,
    success_payoff: float = 20.0
) -> Dict[str, Union[str, float, int]]:
    """
    Compute full multi-layer evaluation metrics for a startup prediction model.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    decisions = (p >= threshold).astype(int)
    
    # 1. Standard ML Metrics
    acc = float(accuracy_score(y, decisions))
    prec = float(precision_score(y, decisions, zero_division=0))
    rec = float(recall_score(y, decisions, zero_division=0))
    f1 = float(f1_score(y, decisions, zero_division=0))
    
    try:
        roc_auc = float(roc_auc_score(y, p))
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = float(average_precision_score(y, p))
    except Exception:
        pr_auc = 0.0
        
    brier = float(brier_score_loss(y, p))
    
    # 2. Portfolio Simulation
    sim = simulate_portfolio(y, decisions, investment_amount, success_payoff)
    
    # 3. Precision@K & Recall@K for K=100 and K=500
    at_100 = compute_precision_recall_at_k(y, p, k=100)
    at_500 = compute_precision_recall_at_k(y, p, k=500)
    
    # 4. Total EMV for the selected portfolio
    if sim["Total_Investments"] > 0:
        tot_emv = float(np.sum(calculate_emv(p[decisions == 1], investment_amount, success_payoff)))
    else:
        tot_emv = 0.0
        
    return {
        "Model": model_name,
        "Threshold": float(threshold),
        # ML Metrics
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Brier_Score": brier,
        # Decision Metrics
        "Total_Investments": sim["Total_Investments"],
        "Selection_Rate": sim["Selection_Rate"],
        "Precision@100": at_100["Precision@100"],
        "Recall@100": at_100["Recall@100"],
        "Precision@500": at_500["Precision@500"],
        "Recall@500": at_500["Recall@500"],
        # Economic Metrics
        "Capital_Deployed_M": sim["Total_Capital_Invested_M"],
        "Portfolio_Net_Return_M": sim["Portfolio_Net_Return_M"],
        "Portfolio_ROI": sim["Portfolio_ROI"],
        "Capital_Loss_M": sim["Capital_Loss_M"],
        "Opportunity_Loss_M": sim["Opportunity_Loss_M"],
        "Total_EMV_M": tot_emv,
        # Portfolio Metrics
        "Capital_Preservation_Ratio": sim["Capital_Preservation_Ratio"],
        "Winner_Capture_Rate": sim["Winner_Capture_Rate"],
        "Portfolio_Success_Rate": sim["Portfolio_Success_Rate"]
    }


def run_paired_bootstrap(
    y_test: np.ndarray,
    probs_bcapm: np.ndarray,
    probs_gbdt: np.ndarray,
    capacity_k: int = 10,
    investment_amount_m: float = 1.0,
    expected_payoff_m: float = 20.0,
    n_resamples: int = 1000,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Execute paired bootstrap resampling on the exact same test opportunities.
    """
    y = np.asarray(y_test, dtype=int)
    p_bcapm = np.asarray(probs_bcapm, dtype=float)
    p_gbdt = np.asarray(probs_gbdt, dtype=float)
    n_samples = len(y)
    
    rng = np.random.default_rng(random_seed)
    
    delta_precisions = []
    delta_rois = []
    
    for _ in range(n_resamples):
        b_idx = rng.choice(n_samples, size=n_samples, replace=True)
        y_b = y[b_idx]
        pb_b = p_bcapm[b_idx]
        pg_b = p_gbdt[b_idx]
        
        top_b_idx = np.argsort(pb_b)[::-1][:capacity_k]
        prec_b = float(np.mean(y_b[top_b_idx] == 1))
        roi_b = float((np.sum(y_b[top_b_idx] == 1) * expected_payoff_m - capacity_k * investment_amount_m) / (capacity_k * investment_amount_m))
        
        top_g_idx = np.argsort(pg_b)[::-1][:capacity_k]
        prec_g = float(np.mean(y_b[top_g_idx] == 1))
        roi_g = float((np.sum(y_b[top_g_idx] == 1) * expected_payoff_m - capacity_k * investment_amount_m) / (capacity_k * investment_amount_m))
        
        delta_precisions.append(prec_b - prec_g)
        delta_rois.append(roi_b - roi_g)
        
    delta_prec_arr = np.array(delta_precisions)
    delta_roi_arr = np.array(delta_rois)
    
    prec_ci_low, prec_ci_high = np.percentile(delta_prec_arr, 2.5), np.percentile(delta_prec_arr, 97.5)
    roi_ci_low, roi_ci_high = np.percentile(delta_roi_arr, 2.5), np.percentile(delta_roi_arr, 97.5)
    
    p_val_prec = float(np.mean(delta_prec_arr <= 0.0))
    p_val_roi = float(np.mean(delta_roi_arr <= 0.0))
    
    return {
        "capacity_k": capacity_k,
        "n_resamples": n_resamples,
        "mean_delta_precision_at_k": float(np.mean(delta_prec_arr)),
        "delta_precision_ci_95": (float(prec_ci_low), float(prec_ci_high)),
        "p_value_precision": p_val_prec,
        "mean_delta_roi": float(np.mean(delta_roi_arr)),
        "delta_roi_ci_95": (float(roi_ci_low), float(roi_ci_high)),
        "p_value_roi": p_val_roi,
        "statistically_significant": bool(p_val_prec < 0.05 or prec_ci_low > 0)
    }

