"""
friction.py: Condition Friction & Macro Alignment Diagnostics for BCAPM.

Identifies:
- Supporting conditions (macro/micro factors historically correlated with liquidity exits)
- Adverse risk conditions (factors historically associated with startup failure/write-offs)
- Condition mismatches between precedent era and current deployment climate
"""

from typing import Dict, List, Any
from src.precedent.schema import StartupIdea
from src.backdrop.condition_vector import BackdropConditionVector


class ConditionFrictionAnalyzer:
    """Diagnoses structural and environmental conditions supporting or working against a venture idea."""

    def analyze_friction(
        self,
        idea: StartupIdea,
        backdrop: BackdropConditionVector,
        retrieval_result: Dict[str, Any],
        backdrop_eval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze supporting, adverse, and mismatching conditions.
        """
        supporting = []
        adverse = []
        mismatches = []
        
        # 1. Micro-Firmographic Rules
        if idea.business_model.upper() in ["B2B", "ENTERPRISE"]:
            supporting.append("B2B SaaS / Enterprise model historically exhibits higher liquidity exit resilience than B2C.")
        else:
            adverse.append("Consumer-facing (B2C) business models in historical cohorts experienced higher write-off rates due to customer acquisition cost (CAC) inflation.")
            
        if idea.founder_count >= 2:
            supporting.append(f"Balanced founding team size ({idea.founder_count} founders) aligns with successful venture precedent structures.")
        else:
            adverse.append("Solo-founder venture structure historically correlates with lower follow-on financing velocity.")
            
        if idea.worked_in_top_companies:
            supporting.append("Founders with pedigree in leading tier-1 tech companies historically demonstrate higher Series A conversion rates.")
            
        # 2. Capital Scale Rules
        if idea.initial_funding_usd >= 1_500_000.0:
            supporting.append(f"Substantial seed check (${idea.initial_funding_usd/1e6:.1f}M) provides 18-24 months of operational runway.")
        elif idea.initial_funding_usd < 500_000.0:
            adverse.append(f"Limited initial capital (${idea.initial_funding_usd/1e3:.0f}K) poses acute near-term financing risk under tightened market conditions.")
            
        # 3. Macro Backdrop & Era Rules
        if backdrop.internet_penetration_pct >= 85.0:
            supporting.append(f"High digital penetration in {backdrop.country_code} ({backdrop.internet_penetration_pct:.0f}%) provides strong addressable TAM connectivity.")
            
        if backdrop.entrepreneurial_financing_index >= 6.5:
            supporting.append(f"Robust domestic venture ecosystem score ({backdrop.entrepreneurial_financing_index:.1f}/10) supports syndication and follow-on rounds.")
            
        era_diff = backdrop_eval.get("mean_era_difference_years", 10.0)
        if era_diff >= 8.0:
            mismatches.append(
                f"Macroeconomic Era Mismatch: Retrieved historical precedents were founded ~{era_diff:.0f} years ago. "
                f"Historical outcomes occurred during an era with differing capital costs, valuation multiples, and tech infrastructure."
            )
            
        if backdrop.macro_failure_rate >= 0.22:
            adverse.append(f"Elevated baseline business mortality ({backdrop.macro_failure_rate*100:.1f}%) in target jurisdiction.")
            
        # 4. Precedent Outcome Summary
        exit_rate = retrieval_result.get("empirical_precedent_exit_rate", 0.5)
        if exit_rate >= 0.60:
            supporting.append(f"Precedent validation: {exit_rate*100:.1f}% of the most analogous historical startups successfully achieved liquidity exits.")
        elif exit_rate <= 0.40:
            adverse.append(f"Precedent caution: Only {exit_rate*100:.1f}% of historical analogues achieved a liquidity exit (60%+ closed).")

        return {
            "supporting_conditions": supporting,
            "adverse_conditions": adverse,
            "condition_mismatches": mismatches,
            "net_condition_signal": "Favorable" if len(supporting) > len(adverse) else ("Neutral" if len(supporting) == len(adverse) else "Challenging")
        }
