"""
threshold_optimization.py: Multi-Threshold Sweeps & Economic Sensitivity Analysis.

Purpose:
- Sweep across decision thresholds [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50].
- Compute ML metrics + Venture Economic metrics at each threshold.
- Identify optimal operational threshold under different VC mandates (Max ROI, Max Return, Max Preservation).
- Conduct 2D Sensitivity Analysis across (Check Size, Exit Payoff) scenarios.
"""

from typing import List, Dict, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from src.portfolio_simulation import simulate_portfolio
from src.economic_utility import calculate_emv, calculate_optimal_threshold


DEFAULT_THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.80, 0.85, 0.90, 0.95]

DEFAULT_SCENARIOS = [
    {"investment": 0.5, "payoff": 10.0},
    {"investment": 0.5, "payoff": 20.0},
    {"investment": 1.0, "payoff": 10.0},
    {"investment": 1.0, "payoff": 20.0},
    {"investment": 1.0, "payoff": 50.0},
    {"investment": 2.0, "payoff": 20.0},
    {"investment": 2.0, "payoff": 50.0},
]


def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float] = None,
    investment_amount: float = 1.0,
    success_payoff: float = 20.0
) -> pd.DataFrame:
    """
    Sweep across probability thresholds and compute ML, decision, and financial metrics.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
        
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    
    records = []
    for thresh in thresholds:
        decisions = (p >= thresh).astype(int)
        sim = simulate_portfolio(y, decisions, investment_amount, success_payoff)
        
        prec = float(precision_score(y, decisions, zero_division=0))
        rec = float(recall_score(y, decisions, zero_division=0))
        f1 = float(f1_score(y, decisions, zero_division=0))
        
        # Mean EMV for the selected portfolio
        if sim["Total_Investments"] > 0:
            selected_emvs = calculate_emv(p[decisions == 1], investment_amount, success_payoff)
            avg_emv = float(np.mean(selected_emvs))
            total_emv = float(np.sum(selected_emvs))
        else:
            avg_emv = 0.0
            total_emv = 0.0
            
        records.append({
            "Threshold": thresh,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "Investments_Selected": sim["Total_Investments"],
            "Selection_Rate": sim["Selection_Rate"],
            "Capital_Deployed_M": sim["Total_Capital_Invested_M"],
            "Opportunity_Loss_M": sim["Opportunity_Loss_M"],
            "Capital_Loss_M": sim["Capital_Loss_M"],
            "Portfolio_Net_Return_M": sim["Portfolio_Net_Return_M"],
            "Portfolio_ROI": sim["Portfolio_ROI"],
            "Mean_EMV_Per_Deal_M": avg_emv,
            "Total_EMV_M": total_emv,
            "Capital_Preservation_Ratio": sim["Capital_Preservation_Ratio"],
            "Winner_Capture_Rate": sim["Winner_Capture_Rate"]
        })
        
    df_sweep = pd.DataFrame(records)
    return df_sweep


def run_sensitivity_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    scenarios: List[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Run 2D economic sensitivity matrix across check sizes and success payoffs.
    """
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
        
    records = []
    for sc in scenarios:
        inv = sc["investment"]
        pay = sc["payoff"]
        p_star = calculate_optimal_threshold(inv, pay)
        
        # Decision at economic threshold
        decisions = (y_prob >= p_star).astype(int)
        sim = simulate_portfolio(y_true, decisions, inv, pay)
        
        records.append({
            "Investment_M": inv,
            "Success_Payoff_M": pay,
            "Theoretical_Threshold_p_star": p_star,
            "Total_Investments": sim["Total_Investments"],
            "Selection_Rate": sim["Selection_Rate"],
            "Capital_Deployed_M": sim["Total_Capital_Invested_M"],
            "Portfolio_Net_Return_M": sim["Portfolio_Net_Return_M"],
            "Portfolio_ROI": sim["Portfolio_ROI"],
            "Capital_Loss_M": sim["Capital_Loss_M"],
            "Opportunity_Loss_M": sim["Opportunity_Loss_M"],
            "Capital_Preservation_Ratio": sim["Capital_Preservation_Ratio"],
            "Winner_Capture_Rate": sim["Winner_Capture_Rate"]
        })
        
    return pd.DataFrame(records)
