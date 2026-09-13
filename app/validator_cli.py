"""
validator_cli.py: Interactive Terminal CLI for BCAPM Idea Validation.

Usage Example:
    python -m app.validator_cli --idea "Acuity Health AI" --category "health" --model "B2B" --capital 2.0 --team 3
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from src.precedent.schema import StartupIdea
from src.validator.pipeline import BCAPMValidator
from src.validator.report import format_verdict_report


def main():
    parser = argparse.ArgumentParser(description="BCAPM Precedent-Driven Venture Opportunity Validator")
    parser.add_argument("--idea", type=str, default="Acuity Health AI", help="Name or title of startup idea")
    parser.add_argument("--category", type=str, default="software", help="Market category (e.g. software, health, biotech)")
    parser.add_argument("--model", type=str, default="B2B", choices=["B2B", "B2C", "Hybrid"], help="Business model")
    parser.add_argument("--country", type=str, default="USA", help="3-letter country code (e.g. USA, GBR, IND)")
    parser.add_argument("--capital", type=float, default=1.5, help="Seed check size / initial funding in $ Millions")
    parser.add_argument("--team", type=int, default=2, help="Number of co-founders")
    parser.add_argument("--top_company", action="store_true", help="Founders worked in tier-1 tech companies")
    parser.add_argument("--ml_based", action="store_true", help="Startup core technology is AI/ML driven")
    parser.add_argument("--target_exit", type=float, default=20.0, help="Expected successful exit payoff in $ Millions")
    parser.add_argument("--top_k", type=int, default=5, help="Number of historical precedents to retrieve")
    parser.add_argument("--markdown", action="store_true", help="Output report in GitHub Flavored Markdown")
    args = parser.parse_args()

    idea = StartupIdea(
        name=args.idea,
        market_category=args.category,
        business_model=args.model,
        country_code=args.country,
        initial_funding_usd=args.capital * 1_000_000.0,
        founder_count=args.team,
        worked_in_top_companies=args.top_company,
        is_ml_based=args.ml_based
    )

    validator = BCAPMValidator(top_k=args.top_k)
    verdict = validator.validate_idea(
        idea=idea,
        check_size_m=args.capital,
        target_exit_m=args.target_exit,
        capacity_k=args.top_k
    )

    report_str = format_verdict_report(verdict, as_markdown=args.markdown)
    print("\n" + report_str + "\n")


if __name__ == "__main__":
    main()
