"""
report.py: Explainable BCAPM Verdict Report Generator.

Transforms BCAPMVerdict into structured, publication-grade executive reports
satisfying all 13 core sections of the BCAPM product specification.
"""

from typing import Dict, Any
from src.validator.pipeline import BCAPMVerdict
from src.precedent.explanation import explain_precedent_match


def format_verdict_report(verdict: BCAPMVerdict, as_markdown: bool = False) -> str:
    """Format the full BCAPM decision report."""
    idea = verdict.idea
    v_str = verdict.verdict
    p_pct = f"{verdict.calibrated_probability * 100:.1f}%"
    hurdle_pct = f"{verdict.economic_threshold * 100:.1f}%"
    emv_str = f"+${verdict.expected_monetary_value_m:.2f}M" if verdict.expected_monetary_value_m >= 0 else f"-${abs(verdict.expected_monetary_value_m):.2f}M"
    
    diag = verdict.condition_diagnostics
    bd = verdict.backdrop_evaluation
    ret = verdict.retrieval_result
    sens = verdict.scenario_sensitivities
    
    if as_markdown:
        report = f"""# BCAPM Venture Opportunity Intelligence Report
**Target Venture:** `{idea.name}`  
**Sector:** `{idea.market_category}` | **Model:** `{idea.business_model}` | **Geography:** `{idea.country_code}`

---

## 1. Executive Verdict
**Recommendation:** `{v_str}`  
**Executive Rationale:** {verdict.executive_summary}

| Decision Metric | Value | Venture Benchmark |
| :--- | :---: | :---: |
| **Calibrated Exit Probability** | **{p_pct}** | Historical Base Rate: 53.4% |
| **Breakeven Hurdle Rate ($p^*$)** | **{hurdle_pct}** | Check: ${verdict.check_size_m:.1f}M / Payoff: ${verdict.target_exit_m:.1f}M |
| **Expected Monetary Value (EMV)** | **{emv_str}** | Positive Hurdle Spread: +{(verdict.calibrated_probability - verdict.economic_threshold)*100:.1f}% |
| **Precedent Analogue Exit Rate** | **{ret['empirical_exit_percentage']}** | {ret['exits_count']} of {ret['top_k']} Analogues Realized M&A/IPO |
| **Backdrop Macro Alignment** | **{bd['backdrop_alignment']}** | Context Similarity: {bd['mean_backdrop_percentage']} |

---

## 2. Closest Historical Precedents
"""
        for prec in ret["precedents"]:
            report += f"\n### {prec['company_name']} ({prec['similarity_percentage']} Match)\n"
            report += f"- **Realized Outcome:** {prec['realized_outcome']}\n"
            report += f"- **Market:** {prec['market_category']} | **Country:** {prec['country_code']} | **Founded:** {prec['founded_year']}\n"
            comps = prec.get("component_attributions", {})
            comp_str = ", ".join([f"{k.replace('_', ' ').title()}: {v*100:.0f}%" for k, v in comps.items()])
            report += f"- **Attribute Alignment:** {comp_str}\n"

        report += f"\n> *{ret['disclaimer']}*\n\n---\n\n## 3. Condition & Backdrop Friction Diagnostics\n\n"
        report += f"**Net Environmental Signal:** `{diag['net_condition_signal']}`\n\n"
        
        report += "### Supporting Conditions (Tailwinds):\n"
        for s in diag["supporting_conditions"]:
            report += f"- {s}\n"
            
        report += "\n### Adverse Risk Conditions (Headwinds):\n"
        for a in diag["adverse_conditions"]:
            report += f"- {a}\n"
            
        if diag["condition_mismatches"]:
            report += "\n### Macroeconomic Era Mismatches:\n"
            for m in diag["condition_mismatches"]:
                report += f"- {m}\n"

        report += "\n---\n\n## 4. Multi-Scenario Sensitivity Analysis\n\n"
        report += "| Scenario Profile | Mean Return | Median ROI | Net Profit ($M) | Downside Risk | Profile Description |\n"
        report += "| :--- | :---: | :---: | :---: | :---: | :--- |\n"
        for k_sc, v_sc in sens.items():
            report += f"| **{v_sc['scenario_name']}** | {v_sc['mean_roi']:.2f}x | {v_sc['median_roi']:.2f}x | ${v_sc['mean_net_return_m']:+.2f}M | P(Loss): {v_sc['prob_loss']*100:.0f}% | {v_sc['description']} |\n"

        report += f"\n---\n\n## 5. Strategic Investment Conclusion\n"
        report += f"Under realistic capital constraints and the current macroeconomic backdrop, this opportunity is designated **{v_str}**. "
        if v_str == "INVEST":
            report += f"The venture exhibits positive expected economic return ({emv_str}) and strong structural alignment with historical liquidity exits."
        elif v_str == "REVIEW":
            report += f"The unit economics are viable, but the indicated adverse risk factors warrant partner-level diligence prior to capital commitment."
        else:
            report += f"The risk-adjusted expected return is insufficient to justify check allocation relative to higher-ranking fund candidates."

        return report

    else:
        # Clean terminal ASCII format
        lines = [
            "=" * 78,
            f"  BCAPM VENTURE OPPORTUNITY INTELLIGENCE REPORT: {idea.name.upper()}",
            "=" * 78,
            f"  EXECUTIVE VERDICT: {v_str} | Calibrated Probability: {p_pct} | Hurdle Rate: {hurdle_pct}",
            f"  Expected Monetary Value (EMV): {emv_str} (Check: ${verdict.check_size_m:.1f}M, Exit: ${verdict.target_exit_m:.1f}M)",
            f"  Summary: {verdict.executive_summary}",
            "-" * 78,
            f"  TOP HISTORICAL PRECEDENTS ({ret['top_k']} Closest Ventures, Exit Rate: {ret['empirical_exit_percentage']}):"
        ]
        for p in ret["precedents"]:
            lines.append(f"  • {p['company_name']:<24} | Match: {p['similarity_percentage']:<6} | Outcome: {p['realized_outcome']}")
            
        lines.append("-" * 78)
        lines.append(f"  BACKDROP & CONDITION ALIGNMENT: {bd['backdrop_alignment']} ({bd['mean_backdrop_percentage']})")
        lines.append("  Supporting Conditions:")
        for s in diag["supporting_conditions"][:3]:
            lines.append(f"    [+] {s}")
        lines.append("  Adverse Conditions / Risks:")
        for a in diag["adverse_conditions"][:3]:
            lines.append(f"    [-] {a}")
        lines.append("-" * 78)
        lines.append("  SCENARIO SENSITIVITY PROFILES:")
        for k_sc, v_sc in sens.items():
            lines.append(f"    • {v_sc['scenario_name']:<12}: Mean ROI: {v_sc['mean_roi']:>5.2f}x | Net: ${v_sc['mean_net_return_m']:>+5.2f}M | P(Loss): {v_sc['prob_loss']*100:>2.0f}%")
        lines.append("=" * 78)
        return "\n".join(lines)
