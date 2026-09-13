"""
monte_carlo_engine.py: Power-Law Heavy-Tailed Return Simulator & Risk-Adjusted Portfolio Metrics.

Venture Capital Return Distribution:
- Failure (y=0): Multiple = 0.0x (Net Return = -I).
- Success (y=1):
    - 60% probability: Modest Exit (1.5x - 3.0x, Median 2.0x)
    - 30% probability: Strong Venture Exit (5.0x - 15.0x, Median 10.0x)
    - 9% probability: Exceptional Outlier (25.0x - 60.0x, Median 40.0x)
    - 1% probability: Power-Law Unicorn (100.0x - 500.0x, Median 200.0x)

Risk-Adjusted Metrics:
- Sharpe Ratio: (Mean ROI - Rf) / Std(ROI)
- Sortino Ratio: (Mean ROI - Rf) / Downside_Std
- CVaR (95%): Expected shortfall in the worst 5% of portfolio draws
- Probability of Loss: P(Net Return < 0)
"""

from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import pandas as pd


def sample_venture_multiples(
    n_successes: int,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Sample exit return multiples for n successful startups from a heavy-tailed power-law mixture.
    """
    if n_successes <= 0:
        return np.array([], dtype=float)
        
    rng = np.random.default_rng(random_state)
    
    # Tier probabilities: [Modest (60%), Strong (30%), Exceptional (9%), Unicorn (1%)]
    tier_choices = rng.choice(
        [0, 1, 2, 3],
        size=n_successes,
        p=[0.60, 0.30, 0.09, 0.01]
    )
    
    multiples = np.zeros(n_successes, dtype=float)
    
    # 0: Modest
    idx0 = (tier_choices == 0)
    multiples[idx0] = rng.uniform(1.5, 3.0, size=np.sum(idx0))
    
    # 1: Strong
    idx1 = (tier_choices == 1)
    multiples[idx1] = rng.uniform(5.0, 15.0, size=np.sum(idx1))
    
    # 2: Exceptional
    idx2 = (tier_choices == 2)
    multiples[idx2] = rng.uniform(25.0, 60.0, size=np.sum(idx2))
    
    # 3: Unicorn Outlier
    idx3 = (tier_choices == 3)
    multiples[idx3] = rng.uniform(100.0, 500.0, size=np.sum(idx3))
    
    return multiples


def run_monte_carlo_simulation(
    y_true: np.ndarray,
    decisions: np.ndarray,
    n_iterations: int = 1000,
    investment_amount: float = 1.0,
    risk_free_rate: float = 0.03,
    random_seed: int = 42
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Run N Monte Carlo iterations simulating portfolio return distributions under power-law venture payoffs.
    """
    y = np.asarray(y_true, dtype=int)
    d = np.asarray(decisions, dtype=int)
    
    invested_mask = (d == 1)
    n_investments = int(np.sum(invested_mask))
    
    if n_investments == 0:
        return {
            "Mean_ROI": 0.0, "Median_ROI": 0.0,
            "ROI_5th": 0.0, "ROI_25th": 0.0, "ROI_75th": 0.0, "ROI_95th": 0.0,
            "Mean_Net_Return_M": 0.0, "Prob_Loss": 1.0,
            "Sharpe_Ratio": 0.0, "Sortino_Ratio": 0.0, "CVaR_95": 0.0,
            "ROI_Distribution": np.zeros(n_iterations)
        }
        
    invested_y = y[invested_mask]
    n_success = int(np.sum(invested_y == 1))
    n_fail = int(np.sum(invested_y == 0))
    total_capital = float(n_investments * investment_amount)
    
    rng = np.random.default_rng(random_seed)
    roi_draws = np.zeros(n_iterations, dtype=float)
    net_return_draws = np.zeros(n_iterations, dtype=float)
    
    for i in range(n_iterations):
        seed_i = rng.integers(0, 1_000_000_000)
        multiples = sample_venture_multiples(n_success, random_state=seed_i)
        
        gross_return = np.sum(multiples * investment_amount)
        net_return = gross_return - total_capital
        roi = net_return / total_capital
        
        roi_draws[i] = roi
        net_return_draws[i] = net_return
        
    mean_roi = float(np.mean(roi_draws))
    median_roi = float(np.median(roi_draws))
    std_roi = float(np.std(roi_draws))
    
    # Downside deviation for Sortino
    downside = roi_draws[roi_draws < risk_free_rate] - risk_free_rate
    downside_std = float(np.sqrt(np.mean(downside**2))) if len(downside) > 0 else 1e-6
    
    sharpe = float((mean_roi - risk_free_rate) / (std_roi + 1e-9))
    sortino = float((mean_roi - risk_free_rate) / (downside_std + 1e-9))
    
    prob_loss = float(np.mean(net_return_draws < 0.0))
    
    # CVaR at 95% (expected shortfall below 5th percentile)
    var_95 = float(np.percentile(roi_draws, 5))
    cvar_95 = float(np.mean(roi_draws[roi_draws <= var_95])) if np.sum(roi_draws <= var_95) > 0 else var_95
    
    return {
        "Mean_ROI": mean_roi,
        "Median_ROI": median_roi,
        "Std_ROI": std_roi,
        "ROI_5th": float(np.percentile(roi_draws, 5)),
        "ROI_25th": float(np.percentile(roi_draws, 25)),
        "ROI_75th": float(np.percentile(roi_draws, 75)),
        "ROI_95th": float(np.percentile(roi_draws, 95)),
        "Mean_Net_Return_M": float(np.mean(net_return_draws)),
        "Prob_Loss": prob_loss,
        "Sharpe_Ratio": sharpe,
        "Sortino_Ratio": sortino,
        "CVaR_95": cvar_95,
        "ROI_Distribution": roi_draws
    }
