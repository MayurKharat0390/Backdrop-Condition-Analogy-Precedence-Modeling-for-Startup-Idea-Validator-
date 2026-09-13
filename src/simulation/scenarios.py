"""
scenarios.py: Configurable Venture Return Scenario Generator for BCAPM.

Replaces hardcoded payoff assumptions with three explicit, auditable scenario profiles:
1. Conservative: High downside protection, low multiple ceiling (modest 75%, strong 22%, outlier 3%).
2. Base: Standard empirical venture capital return distribution (modest 60%, strong 30%, exceptional 9%, unicorn 1%).
3. Aggressive: Heavy-tailed power-law distribution (modest 50%, strong 32%, exceptional 15%, unicorn 3%).

Explicitly models sensitivity: "Does this venture remain economically viable across all three regimes?"
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class ScenarioProfile:
    name: str
    description: str
    tier_probs: np.ndarray        # [Modest, Strong, Exceptional, Unicorn]
    modest_range: tuple           # (min, max)
    strong_range: tuple           # (min, max)
    exceptional_range: tuple      # (min, max)
    unicorn_range: tuple          # (min, max)


SCENARIO_PROFILES = {
    "conservative": ScenarioProfile(
        name="Conservative",
        description="Downside-protection regime with lower multiple ceiling (2x-5x median).",
        tier_probs=np.array([0.75, 0.22, 0.029, 0.001]),
        modest_range=(1.2, 2.5),
        strong_range=(3.0, 8.0),
        exceptional_range=(10.0, 25.0),
        unicorn_range=(30.0, 50.0)
    ),
    "base": ScenarioProfile(
        name="Base",
        description="Standard historical venture capital power-law distribution (5x-15x median).",
        tier_probs=np.array([0.60, 0.30, 0.09, 0.01]),
        modest_range=(1.5, 3.0),
        strong_range=(5.0, 15.0),
        exceptional_range=(25.0, 60.0),
        unicorn_range=(100.0, 250.0)
    ),
    "aggressive": ScenarioProfile(
        name="Aggressive",
        description="Heavy-tailed outlier regime with power-law upside (25x-100x+).",
        tier_probs=np.array([0.50, 0.32, 0.15, 0.03]),
        modest_range=(1.5, 4.0),
        strong_range=(8.0, 20.0),
        exceptional_range=(30.0, 80.0),
        unicorn_range=(100.0, 500.0)
    )
}


def sample_scenario_multiples(
    n_successes: int,
    scenario_name: str = "base",
    random_state: Optional[int] = None
) -> np.ndarray:
    """Sample realized return multiples according to a specific scenario profile."""
    if n_successes <= 0:
        return np.array([], dtype=float)
        
    scenario = SCENARIO_PROFILES.get(scenario_name.lower(), SCENARIO_PROFILES["base"])
    rng = np.random.default_rng(random_state)
    
    tier_choices = rng.choice([0, 1, 2, 3], size=n_successes, p=scenario.tier_probs)
    multiples = np.zeros(n_successes, dtype=float)
    
    # 0: Modest
    idx0 = (tier_choices == 0)
    multiples[idx0] = rng.uniform(scenario.modest_range[0], scenario.modest_range[1], size=np.sum(idx0))
    
    # 1: Strong
    idx1 = (tier_choices == 1)
    multiples[idx1] = rng.uniform(scenario.strong_range[0], scenario.strong_range[1], size=np.sum(idx1))
    
    # 2: Exceptional
    idx2 = (tier_choices == 2)
    multiples[idx2] = rng.uniform(scenario.exceptional_range[0], scenario.exceptional_range[1], size=np.sum(idx2))
    
    # 3: Unicorn
    idx3 = (tier_choices == 3)
    multiples[idx3] = rng.uniform(scenario.unicorn_range[0], scenario.unicorn_range[1], size=np.sum(idx3))
    
    return multiples


def evaluate_scenario_sensitivity(
    p_success: float,
    investment_amount_m: float = 1.0,
    n_simulations: int = 1000
) -> Dict[str, Dict[str, float]]:
    """
    Simulate expected outcome of a single investment across Conservative, Base, and Aggressive scenarios.
    """
    results = {}
    
    for sc_key, sc_obj in SCENARIO_PROFILES.items():
        rng = np.random.default_rng(42)
        net_returns = []
        rois = []
        
        for _ in range(n_simulations):
            # Bernouilli success trial
            success = (rng.random() < p_success)
            if success:
                mult = sample_scenario_multiples(1, scenario_name=sc_key, random_state=rng.integers(0, 1_000_000_000))[0]
                gross = mult * investment_amount_m
                net = gross - investment_amount_m
                roi = net / investment_amount_m
            else:
                net = -investment_amount_m
                roi = -1.0
                
            net_returns.append(net)
            rois.append(roi)
            
        mean_roi = float(np.mean(rois))
        median_roi = float(np.median(rois))
        mean_net = float(np.mean(net_returns))
        prob_loss = float(np.mean(np.array(net_returns) < 0.0))
        
        results[sc_key] = {
            "scenario_name": sc_obj.name,
            "mean_roi": round(mean_roi, 2),
            "median_roi": round(median_roi, 2),
            "mean_net_return_m": round(mean_net, 2),
            "prob_loss": round(prob_loss, 3),
            "description": sc_obj.description
        }
        
    return results
