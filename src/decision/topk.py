"""
topk.py: Capacity-Constrained Top-K Fund Selection & Capital Budget Allocator.

Models realistic venture capital selection where fund managers cannot deploy capital
to all positive-EMV deals, but must select their highest-conviction K opportunities
subject to deal capacity or finite fund capital budgets.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def rank_and_select_top_k(
    probabilities: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    capacity_k: int = 10,
    investment_amount_m: float = 1.0,
    expected_payoff_m: float = 20.0
) -> Dict[str, Any]:
    """
    Rank deals by Expected Monetary Value (EMV) and select top-K candidates.
    """
    p = np.asarray(probabilities, dtype=float)
    emvs = (p * expected_payoff_m) - investment_amount_m
    
    n_samples = len(p)
    k = min(capacity_k, n_samples)
    
    # Sort indices by EMV descending
    ranked_indices = np.argsort(emvs)[::-1]
    top_k_indices = ranked_indices[:k]
    
    decisions = np.zeros(n_samples, dtype=int)
    decisions[top_k_indices] = 1
    
    selected_probs = p[top_k_indices]
    selected_emvs = emvs[top_k_indices]
    
    result = {
        "capacity_k": k,
        "total_evaluated": n_samples,
        "selected_indices": top_k_indices.tolist(),
        "mean_selected_probability": float(np.mean(selected_probs)),
        "min_selected_probability": float(np.min(selected_probs)),
        "total_emv_selected_m": float(np.sum(selected_emvs)),
        "total_capital_deployed_m": float(k * investment_amount_m),
        "decisions": decisions
    }
    
    # If ground truth labels are provided, evaluate selection accuracy
    if ground_truth is not None:
        y = np.asarray(ground_truth, dtype=int)
        top_y = y[top_k_indices]
        winners_captured = int(np.sum(top_y == 1))
        total_winners = int(np.sum(y == 1))
        failures_avoided = int(np.sum((decisions == 0) & (y == 0)))
        total_failures = int(np.sum(y == 0))
        
        prec_k = float(winners_captured / k) if k > 0 else 0.0
        rec_k = float(winners_captured / total_winners) if total_winners > 0 else 0.0
        cpr = float(failures_avoided / total_failures) if total_failures > 0 else 1.0
        
        result.update({
            "precision_at_k": round(prec_k, 4),
            "precision_percentage": f"{prec_k * 100:.1f}%",
            "recall_at_k": round(rec_k, 4),
            "recall_percentage": f"{rec_k * 100:.1f}%",
            "winner_capture_rate": round(rec_k, 4),
            "capital_preservation_ratio": round(cpr, 4),
            "winners_captured": winners_captured,
            "total_winners_in_pool": total_winners
        })
        
    return result


def allocate_by_capital_budget(
    probabilities: np.ndarray,
    fund_budget_m: float = 25.0,
    check_size_m: float = 1.0,
    expected_payoff_m: float = 20.0
) -> Dict[str, Any]:
    """
    Select opportunities subject to a finite fund capital ceiling.
    """
    max_k = int(fund_budget_m // check_size_m)
    return rank_and_select_top_k(
        probabilities=probabilities,
        capacity_k=max_k,
        investment_amount_m=check_size_m,
        expected_payoff_m=expected_payoff_m
    )
