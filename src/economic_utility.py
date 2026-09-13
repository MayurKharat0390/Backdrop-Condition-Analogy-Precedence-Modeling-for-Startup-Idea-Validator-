"""
economic_utility.py: Venture Capital Economic Decision & Expected Monetary Value (EMV) Layer.

Architecture:
- Layer 1: Predicted Probability Model outputs P(Success | X_i).
- Layer 2: Economic Utility Layer calculates EMV(Invest | X_i) = P(Success | X_i) * V_i - I_i.
- Layer 3: Investment Decision assigns 'INVEST' if EMV > 0 <=> P(Success | X_i) > (I_i / V_i), else 'REJECT'.
- Layer 4: Portfolio Simulation & Evaluation.

Research Principle:
Investment decisions should be governed by expected economic utility under asymmetric payoffs,
rather than arbitrary mathematical thresholds (like 0.50).
"""

from typing import Union, Dict, Tuple
import numpy as np
import pandas as pd


def calculate_optimal_threshold(
    investment_amount: Union[float, np.ndarray] = 1.0,
    success_payoff: Union[float, np.ndarray] = 20.0
) -> Union[float, np.ndarray]:
    """
    Calculate the Bayesian optimal economic investment threshold (p*).
    
    Rule:
        Invest if EMV(Invest) > EMV(Reject)
        <=> P(Success) * V - I > 0
        <=> P(Success) > (I / V)
        
    Parameters:
        investment_amount (I): Check size / capital committed (e.g. $1.0M).
        success_payoff (V): Estimated total gross monetary payoff if successful (e.g. $20.0M).
        
    Returns:
        p_star: Breakeven probability threshold above which investment has positive expected return.
    """
    if np.any(success_payoff <= 0):
        raise ValueError("success_payoff (V) must be strictly greater than 0.")
    return investment_amount / success_payoff


def calculate_emv(
    p_success: Union[float, np.ndarray, pd.Series],
    investment_amount: Union[float, np.ndarray] = 1.0,
    success_payoff: Union[float, np.ndarray] = 20.0
) -> Union[float, np.ndarray]:
    """
    Calculate Expected Monetary Value (EMV) for investing in startup(s).
    
    Formula:
        EMV(Invest | X_i) = P(Success | X_i) * V_i - I_i
        EMV(Reject | X_i) = 0.0
        
    Parameters:
        p_success: Predicted probability of startup success P(Y=1|X).
        investment_amount: Capital check size (I).
        success_payoff: Realized monetary payout on successful exit (V).
        
    Returns:
        emv: Expected monetary value in the same units (e.g. $ millions).
    """
    p = np.asarray(p_success, dtype=float)
    return (p * success_payoff) - investment_amount


def make_investment_decision(
    p_success: Union[float, np.ndarray, pd.Series],
    threshold: Union[float, None] = None,
    investment_amount: float = 1.0,
    success_payoff: float = 20.0
) -> np.ndarray:
    """
    Apply economic decision rule to determine whether to invest or reject.
    
    Decision Rule:
        If threshold is not provided, calculates optimal threshold p* = I / V.
        Decision = 1 ('INVEST') if p_success > threshold, else 0 ('REJECT').
        
    Parameters:
        p_success: Array of predicted probabilities.
        threshold: Explicit threshold override. If None, derived from (investment_amount / success_payoff).
        investment_amount: Investment check size ($M).
        success_payoff: Success payoff ($M).
        
    Returns:
        binary_decisions: Integer array where 1 = INVEST, 0 = REJECT.
    """
    p = np.asarray(p_success, dtype=float)
    if threshold is None:
        threshold = calculate_optimal_threshold(investment_amount, success_payoff)
        
    return (p > threshold).astype(int)


def explain_economic_decision(
    company_name: str,
    p_success: float,
    investment_amount: float = 1.0,
    success_payoff: float = 20.0
) -> Dict[str, Union[str, float]]:
    """
    Generate an interpretable human-readable explanation of the economic decision for a startup.
    """
    p_star = calculate_optimal_threshold(investment_amount, success_payoff)
    emv = calculate_emv(p_success, investment_amount, success_payoff)
    decision = "INVEST" if p_success > p_star else "REJECT"
    
    rationale = (
        f"Under an investment check of ${investment_amount:.2f}M and assumed success payoff of ${success_payoff:.2f}M, "
        f"the breakeven hurdle rate is {p_star * 100:.2f}%. "
        f"Target startup '{company_name}' has an estimated success probability of {p_success * 100:.2f}%, "
        f"yielding an Expected Monetary Value (EMV) of ${emv:+.2f}M. "
        f"Recommendation: {decision}."
    )
    
    return {
        "Company_Name": company_name,
        "P_Success": float(p_success),
        "Economic_Threshold": float(p_star),
        "Expected_Monetary_Value_M": float(emv),
        "Decision": decision,
        "Rationale": rationale
    }
