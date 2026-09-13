"""
explanation.py: Precedent Analogy Explainer for BCAPM.

Generates human-readable, auditable explanations of why specific historical
precedents were retrieved and which firmographic dimensions match or differ.
"""

from typing import Dict, List, Any


def explain_precedent_match(precedent: Dict[str, Any]) -> str:
    """Generate a concise explanation of why a historical company is analogous."""
    name = precedent["company_name"]
    score = precedent["similarity_percentage"]
    outcome = precedent["realized_outcome"]
    components = precedent.get("component_attributions", {})
    
    # Identify top matching dimensions
    sorted_comps = sorted(components.items(), key=lambda x: x[1], reverse=True)
    top_matches = [k.replace('_', ' ').title() for k, v in sorted_comps if v >= 0.70]
    low_matches = [k.replace('_', ' ').title() for k, v in sorted_comps if v < 0.50]
    
    match_str = ", ".join(top_matches) if top_matches else "moderate overall alignment"
    diff_str = f" Differences observed in: {', '.join(low_matches)}." if low_matches else ""
    
    return (
        f"• {name} (Similarity: {score} | Outcome: {outcome})\n"
        f"  Strongest alignment in: {match_str}.{diff_str}"
    )


def generate_precedent_summary(retrieval_result: Dict[str, Any]) -> str:
    """Format the full precedent retrieval section for the verdict report."""
    query = retrieval_result["query_idea"]
    k = retrieval_result["top_k"]
    rate_pct = retrieval_result["empirical_exit_percentage"]
    exits = retrieval_result["exits_count"]
    
    lines = [
        f"HISTORICAL PRECEDENT ANALOGY ANALYSIS ({k} Closest Historical Ventures)",
        f"Query Idea: '{query}'",
        f"Observed Liquidity Exit Rate: {rate_pct} ({exits} of {k} analogous ventures exited via Acquisition or IPO)\n",
        "Top Historical Precedents:"
    ]
    
    for prec in retrieval_result["precedents"]:
        lines.append(explain_precedent_match(prec))
        
    lines.append(f"\n{retrieval_result['disclaimer']}")
    return "\n".join(lines)
