"""
portfolio_simulation.py: Venture Capital Portfolio Simulator & Economic Accounting Engine.

Accounting Definitions:
- Invest in Success (TP): Realized Investment Upside. Net Return = V - I.
- Invest in Failure (FP): Actual Capital Loss. Net Return = -I.
- Reject Failure (TN): Capital Preserved. Net Return = 0.
- Reject Success (FN): Simulated Opportunity Cost. Net Return = 0. Opportunity Loss = V.

Important Research Clarification:
- False Positives represent literal capital deployed that was written off.
- False Negatives represent simulated opportunity cost (foregone upside), NOT fund cash draw.
"""

from typing import Dict, Union, List
import numpy as np
import pandas as pd


def simulate_portfolio(
    y_true: np.ndarray,
    decisions: np.ndarray,
    investment_amount: float = 1.0,
    success_payoff: float = 20.0
) -> Dict[str, Union[float, int, np.ndarray]]:
    """
    Simulate full economic performance of an investment portfolio under given decisions.
    
    Parameters:
        y_true: Ground truth binary outcomes (1 = Success / Exit, 0 = Closed / Failure).
        decisions: Binary decisions (1 = INVEST, 0 = REJECT).
        investment_amount (I): Capital check size ($ millions).
        success_payoff (V): Payoff on successful exit ($ millions).
        
    Returns:
        Dictionary containing portfolio-level financial statement and per-deal return distributions.
    """
    y = np.asarray(y_true, dtype=int)
    d = np.asarray(decisions, dtype=int)
    
    total_startups = len(y)
    n_investments = int(np.sum(d == 1))
    n_rejections = int(np.sum(d == 0))
    
    # Confusion components
    tp = int(np.sum((y == 1) & (d == 1)))
    fp = int(np.sum((y == 0) & (d == 1)))
    fn = int(np.sum((y == 1) & (d == 0)))
    tn = int(np.sum((y == 0) & (d == 0)))
    
    # Financial calculations
    total_capital_invested = float(n_investments * investment_amount)
    
    # Per-deal net returns for invested deals
    deal_returns = np.zeros(n_investments, dtype=float)
    if n_investments > 0:
        invested_actuals = y[d == 1]
        deal_returns = np.where(invested_actuals == 1, success_payoff - investment_amount, -investment_amount)
        portfolio_net_return = float(np.sum(deal_returns))
        portfolio_roi = float(portfolio_net_return / total_capital_invested)
        mean_return_per_deal = float(np.mean(deal_returns))
        median_return_per_deal = float(np.median(deal_returns))
        std_return_per_deal = float(np.std(deal_returns))
    else:
        portfolio_net_return = 0.0
        portfolio_roi = 0.0
        mean_return_per_deal = 0.0
        median_return_per_deal = 0.0
        std_return_per_deal = 0.0
        
    # Literal Capital Loss from failed investments (FP)
    capital_loss = float(fp * investment_amount)
    
    # Simulated Opportunity Loss from missed winners (FN)
    opportunity_loss = float(fn * success_payoff)
    
    # Ratios
    selection_rate = float(n_investments / total_startups) if total_startups > 0 else 0.0
    portfolio_success_rate = float(tp / n_investments) if n_investments > 0 else 0.0
    winner_capture_rate = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    capital_preservation_ratio = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    return {
        "Total_Evaluated": total_startups,
        "Total_Investments": n_investments,
        "Total_Rejections": n_rejections,
        "True_Positives": tp,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Negatives": tn,
        "Total_Capital_Invested_M": total_capital_invested,
        "Portfolio_Net_Return_M": portfolio_net_return,
        "Portfolio_ROI": portfolio_roi,
        "Capital_Loss_M": capital_loss,
        "Opportunity_Loss_M": opportunity_loss,
        "Selection_Rate": selection_rate,
        "Portfolio_Success_Rate": portfolio_success_rate,
        "Winner_Capture_Rate": winner_capture_rate,
        "Capital_Preservation_Ratio": capital_preservation_ratio,
        "Mean_Return_Per_Deal_M": mean_return_per_deal,
        "Median_Return_Per_Deal_M": median_return_per_deal,
        "Std_Return_Per_Deal_M": std_return_per_deal,
        "Deal_Returns_Array": deal_returns
    }
