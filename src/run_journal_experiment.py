"""
run_journal_experiment.py: Master Experimental Pipeline for Q1/Q2 Venture Capital Economic Validation.

Full Protocol Implementation:
1. 3-Tier Outcome Benchmarking (Survival, Naturally Balanced Liquidity Exit 53:47, VC Outlier 26:74).
2. Zero-Leakage Temporal Splitting (Train <= 2010, Val 2011-2012, Locked Test >= 2013).
3. 7-Model Baseline Hierarchy (Naive, Logistic, Random Forest, GBDT, SMOTE, Balanced, BCAPM).
4. Power-Law Heavy-Tailed Monte Carlo Simulations (1,000 runs, Sharpe, Sortino, CVaR).
5. Capacity-Constrained Top-K Fund Performance (K in {10, 25, 50, 100, 250}).
6. Stepwise Component Ablation Matrix (Steps 1 to 5).
7. Empirical Pareto Frontier (Capital Preservation vs. Winner Capture).
8. Market Regime Robustness Analysis across 5 economic epochs.
9. 1,000 Bootstrap Confidence Intervals for Statistical Significance.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix
)
from imblearn.over_sampling import SMOTE

from src.config import REPORTS_DIR, RANDOM_SEED
from src.utils import get_logger, save_csv_file
from src.dataset_hierarchy import load_hierarchy_dataframe, get_outcome_dataset, get_temporal_split
from src.calibration import fit_calibrator, evaluate_calibration
from src.asymmetric_loss import AsymmetricGradientBoostingClassifier, compute_sample_weights_for_asymmetric_loss
from src.economic_utility import calculate_emv, calculate_optimal_threshold
from src.portfolio_simulation import simulate_portfolio
from src.monte_carlo_engine import run_monte_carlo_simulation
from src.evaluation import evaluate_vc_model

logger = get_logger("Journal_Experiment")


def run_full_journal_suite(
    investment_amount: float = 1.0,
    success_payoff: float = 20.0
):
    logger.info("=" * 80)
    logger.info("STARTING BCAPM Q1/Q2 JOURNAL EXPERIMENTAL SUITE")
    logger.info(f"Economic Parameters: Check Size = ${investment_amount:.2f}M | Base Success Payoff = ${success_payoff:.2f}M")
    logger.info("=" * 80)

    # 1. Load Master Dataset Hierarchy
    df_master = load_hierarchy_dataframe()

    # =========================================================================
    # EXPERIMENT 1: PRIMARY BENCHMARK ON OUTCOME 2 (LIQUIDITY EXIT: 53% vs 47%)
    # TEMPORAL SPLIT: Train <= 2010 | Val 2011-2012 | Locked Test >= 2013
    # =========================================================================
    logger.info("\n>>> EXECUTING EXPERIMENT 1: TEMPORAL LOCKED BENCHMARK ON OUTCOME 2 (LIQUIDITY EXIT)")
    X_liq, y_liq, sub_liq = get_outcome_dataset(df_master, outcome_tier="outcome_2_liquidity")
    temp_data = get_temporal_split(X_liq, y_liq, sub_liq, train_end_year=2010, val_end_year=2012)

    X_train, y_train = temp_data["X_train"], temp_data["y_train"]
    X_val, y_val = temp_data["X_val"], temp_data["y_val"]
    X_test, y_test = temp_data["X_test"], temp_data["y_test"]

    p_star = calculate_optimal_threshold(investment_amount, success_payoff) # 0.05

    # Define the 7-Model Hierarchy
    models = {
        "Baseline 0: Naive Majority": DummyClassifier(strategy="most_frequent"),
        "Baseline 1: Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "Baseline 2: Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1),
        "Baseline 3: GBDT (Standard BCE)": HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=RANDOM_SEED),
        "Baseline 4: SMOTE + GBDT": None, # Handled separately with resampled X_train
        "Baseline 5: Class-Weighted GBDT": HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=RANDOM_SEED),
        "Proposed: BCAPM Economic Model": None # Calibrated on Validation fold
    }

    # Train Baselines
    logger.info("Training 7-Model Baseline Portfolio...")
    trained_models = {}

    for name, clf in models.items():
        if name == "Baseline 4: SMOTE + GBDT":
            smote = SMOTE(random_state=RANDOM_SEED)
            X_sm, y_sm = smote.fit_resample(X_train, y_train)
            model_sm = HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=RANDOM_SEED)
            model_sm.fit(X_sm, y_sm)
            trained_models[name] = model_sm
        elif name == "Baseline 5: Class-Weighted GBDT":
            w_pos = len(y_train) / (2.0 * np.sum(y_train == 1))
            w_neg = len(y_train) / (2.0 * np.sum(y_train == 0))
            sw = np.where(y_train == 1, w_pos, w_neg)
            clf.fit(X_train, y_train, sample_weight=sw)
            trained_models[name] = clf
        elif name == "Proposed: BCAPM Economic Model":
            # Asymmetric Loss GBDT + Calibrated on Val Fold
            clf_asym = HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=RANDOM_SEED)
            sw_asym = compute_sample_weights_for_asymmetric_loss(y_train, w1=1.0, w0=2.0)
            clf_asym.fit(X_train, y_train, sample_weight=sw_asym)
            cal_bcapm = fit_calibrator(clf_asym, X_val, y_val, method="sigmoid")
            trained_models[name] = cal_bcapm
        else:
            clf.fit(X_train, y_train)
            trained_models[name] = clf

    # Predict Probabilities & Verify Divergence
    probs_dict = {}
    for name, model in trained_models.items():
        if hasattr(model, "predict_proba"):
            probs_dict[name] = model.predict_proba(X_test)[:, 1]
        else:
            probs_dict[name] = model.predict(X_test).astype(float)

    # Compute Divergence
    p_bce = probs_dict["Baseline 3: GBDT (Standard BCE)"]
    p_bcapm = probs_dict["Proposed: BCAPM Economic Model"]
    corr_div = float(np.corrcoef(p_bce, p_bcapm)[0, 1])
    max_div = float(np.max(np.abs(p_bce - p_bcapm)))
    logger.info(f"Model Divergence -> Corr(Standard, BCAPM): {corr_div:.4f} | Max |p_Standard - p_BCAPM|: {max_div:.4f}")

    # Evaluate Table 1: 7-Model Baseline Hierarchy on Locked Test Set
    logger.info("Evaluating 7-Model Hierarchy with Decoupled Performance Metrics...")
    table1_records = []

    for name, probs in probs_dict.items():
        # Operating threshold: Baseline 0-3 use conventional 0.50; BCAPM & weighted use Bayesian hurdle p*
        thresh = 0.50 if "Baseline" in name and "Weighted" not in name and "SMOTE" not in name else p_star
        decisions = (probs >= thresh).astype(int)

        # 1. Predictive Performance
        acc = float(accuracy_score(y_test, decisions))
        prec = float(precision_score(y_test, decisions, zero_division=0))
        rec = float(recall_score(y_test, decisions, zero_division=0))
        f1 = float(f1_score(y_test, decisions, zero_division=0))
        try:
            roc_auc = float(roc_auc_score(y_test, probs))
            pr_auc = float(average_precision_score(y_test, probs))
        except Exception:
            roc_auc, pr_auc = 0.5, 0.0

        # 2. Probability Quality
        brier = float(brier_score_loss(y_test, probs))

        # 3. Venture Economics (Monte Carlo 1,000 runs)
        mc_result = run_monte_carlo_simulation(y_test, decisions, n_iterations=1000, investment_amount=investment_amount)
        sim = simulate_portfolio(y_test, decisions, investment_amount, success_payoff)

        table1_records.append({
            "Model": name,
            "Operating_Threshold": thresh,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc,
            "Brier_Score": brier,
            "Total_Investments": sim["Total_Investments"],
            "Winner_Capture_Rate": sim["Winner_Capture_Rate"],
            "Capital_Preservation_Ratio": sim["Capital_Preservation_Ratio"],
            "Mean_Monte_Carlo_ROI": mc_result["Mean_ROI"],
            "Median_ROI": mc_result["Median_ROI"],
            "Sharpe_Ratio": mc_result["Sharpe_Ratio"],
            "Sortino_Ratio": mc_result["Sortino_Ratio"],
            "CVaR_95": mc_result["CVaR_95"],
            "Prob_Capital_Loss": mc_result["Prob_Loss"]
        })

    df_table1 = pd.DataFrame(table1_records)
    t1_path = REPORTS_DIR / "journal_table1_baselines.csv"
    save_csv_file(df_table1, t1_path, index=False)
    logger.info(f"Exported Table 1 (7-Model Benchmark) to: {t1_path}")

    # =========================================================================
    # EXPERIMENT 2: 3-TIER OUTCOME COMPARISON (SURVIVAL vs LIQUIDITY vs OUTLIER)
    # =========================================================================
    logger.info("\n>>> EXECUTING EXPERIMENT 2: 3-TIER OUTCOME ROBUSTNESS COMPARISON")
    table2_records = []

    for tier_name in ["outcome_1_survival", "outcome_2_liquidity", "outcome_3_vc_outlier"]:
        X_t, y_t, sub_t = get_outcome_dataset(df_master, outcome_tier=tier_name)
        split_t = get_temporal_split(X_t, y_t, sub_t)
        
        # Train GBDT Baseline and BCAPM
        clf_base = HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=RANDOM_SEED)
        clf_base.fit(split_t["X_train"], split_t["y_train"])
        p_base = clf_base.predict_proba(split_t["X_test"])[:, 1]
        
        clf_asym = HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=RANDOM_SEED)
        sw_t = compute_sample_weights_for_asymmetric_loss(split_t["y_train"], w1=1.0, w0=2.0)
        clf_asym.fit(split_t["X_train"], split_t["y_train"], sample_weight=sw_t)
        cal_model = fit_calibrator(clf_asym, split_t["X_val"], split_t["y_val"], method="sigmoid")
        p_bcapm_t = cal_model.predict_proba(split_t["X_test"])[:, 1]
        
        mc_base = run_monte_carlo_simulation(split_t["y_test"], (p_base >= 0.50).astype(int), n_iterations=1000)
        mc_bcapm = run_monte_carlo_simulation(split_t["y_test"], (p_bcapm_t >= p_star).astype(int), n_iterations=1000)
        
        table2_records.append({
            "Outcome_Tier": tier_name,
            "Total_Test_Samples": len(split_t["y_test"]),
            "Success_Rate_Test": float(np.mean(split_t["y_test"])),
            "GBDT_PR_AUC": float(average_precision_score(split_t["y_test"], p_base)),
            "BCAPM_PR_AUC": float(average_precision_score(split_t["y_test"], p_bcapm_t)),
            "GBDT_Sharpe": mc_base["Sharpe_Ratio"],
            "BCAPM_Sharpe": mc_bcapm["Sharpe_Ratio"],
            "GBDT_Mean_ROI": mc_base["Mean_ROI"],
            "BCAPM_Mean_ROI": mc_bcapm["Mean_ROI"],
            "BCAPM_Sortino": mc_bcapm["Sortino_Ratio"]
        })

    df_table2 = pd.DataFrame(table2_records)
    t2_path = REPORTS_DIR / "journal_table2_outcomes.csv"
    save_csv_file(df_table2, t2_path, index=False)
    logger.info(f"Exported Table 2 (3-Tier Outcome Benchmark) to: {t2_path}")

    # =========================================================================
    # EXPERIMENT 3: CAPACITY-CONSTRAINED TOP-K FUND PERFORMANCE (K in 10 to 250)
    # =========================================================================
    logger.info("\n>>> EXECUTING EXPERIMENT 3: CAPACITY-CONSTRAINED TOP-K ALLOCATION")
    k_values = [10, 25, 50, 100, 250]
    table3_records = []

    # Sort test deals by BCAPM predicted score vs GBDT predicted score
    top_models = {
        "GBDT (Standard)": probs_dict["Baseline 3: GBDT (Standard BCE)"],
        "Random Forest": probs_dict["Baseline 2: Random Forest"],
        "Proposed: BCAPM": probs_dict["Proposed: BCAPM Economic Model"]
    }

    for k in k_values:
        for m_name, probs_m in top_models.items():
            top_k_indices = np.argsort(probs_m)[::-1][:k]
            top_k_decisions = np.zeros(len(y_test), dtype=int)
            top_k_decisions[top_k_indices] = 1
            
            top_y = y_test[top_k_indices]
            prec_k = float(np.mean(top_y == 1))
            total_winners = np.sum(y_test == 1)
            rec_k = float(np.sum(top_y == 1) / total_winners) if total_winners > 0 else 0.0
            
            mc_k = run_monte_carlo_simulation(y_test, top_k_decisions, n_iterations=1000, investment_amount=investment_amount)
            
            table3_records.append({
                "Strategy": m_name,
                "Fund_Capacity_K": k,
                "Precision@K": prec_k,
                "Recall@K": rec_k,
                "Mean_ROI@K": mc_k["Mean_ROI"],
                "Median_ROI@K": mc_k["Median_ROI"],
                "Sharpe_Ratio@K": mc_k["Sharpe_Ratio"],
                "Prob_Loss@K": mc_k["Prob_Loss"]
            })

    df_table3 = pd.DataFrame(table3_records)
    t3_path = REPORTS_DIR / "journal_table3_top_k.csv"
    save_csv_file(df_table3, t3_path, index=False)
    logger.info(f"Exported Table 3 (Top-K Fund Allocation) to: {t3_path}")

    # =========================================================================
    # EXPERIMENT 4: STEPWISE ABLATION MATRIX (STEPS 1 TO 5)
    # =========================================================================
    logger.info("\n>>> EXECUTING EXPERIMENT 4: STEPWISE ARCHITECTURAL ABLATION")
    
    # Step 1: Raw Conventional Classifier (BCE, threshold=0.50)
    p_step1 = probs_dict["Baseline 3: GBDT (Standard BCE)"]
    d_step1 = (p_step1 >= 0.50).astype(int)
    mc_s1 = run_monte_carlo_simulation(y_test, d_step1, n_iterations=1000)
    
    # Step 2: Step 1 + Calibration (Platt Sigmoid, threshold=0.50)
    cal_s2 = fit_calibrator(trained_models["Baseline 3: GBDT (Standard BCE)"], X_val, y_val, method="sigmoid")
    p_step2 = cal_s2.predict_proba(X_test)[:, 1]
    d_step2 = (p_step2 >= 0.50).astype(int)
    mc_s2 = run_monte_carlo_simulation(y_test, d_step2, n_iterations=1000)
    
    # Step 3: Step 2 + Bayesian Economic Hurdle (p* = 0.05)
    d_step3 = (p_step2 >= p_star).astype(int)
    mc_s3 = run_monte_carlo_simulation(y_test, d_step3, n_iterations=1000)
    
    # Step 4: Step 3 + Asymmetric Loss Training (w1=1, w0=2)
    clf_asym_s4 = HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=RANDOM_SEED)
    sw_s4 = compute_sample_weights_for_asymmetric_loss(y_train, w1=1.0, w0=2.0)
    clf_asym_s4.fit(X_train, y_train, sample_weight=sw_s4)
    p_step4 = clf_asym_s4.predict_proba(X_test)[:, 1]
    d_step4 = (p_step4 >= p_star).astype(int)
    mc_s4 = run_monte_carlo_simulation(y_test, d_step4, n_iterations=1000)
    
    # Step 5: Full BCAPM (Asymmetric + Calibrated + Economic Hurdle)
    p_step5 = p_bcapm
    d_step5 = (p_step5 >= p_star).astype(int)
    mc_s5 = run_monte_carlo_simulation(y_test, d_step5, n_iterations=1000)

    ablation_records = [
        {"Step": "Step 1: Raw Conventional GBDT (BCE, p=0.50)", "PR_AUC": float(average_precision_score(y_test, p_step1)), "Brier_Score": float(brier_score_loss(y_test, p_step1)), "Mean_ROI": mc_s1["Mean_ROI"], "Sharpe_Ratio": mc_s1["Sharpe_Ratio"]},
        {"Step": "Step 2: Step 1 + Probability Calibration", "PR_AUC": float(average_precision_score(y_test, p_step2)), "Brier_Score": float(brier_score_loss(y_test, p_step2)), "Mean_ROI": mc_s2["Mean_ROI"], "Sharpe_Ratio": mc_s2["Sharpe_Ratio"]},
        {"Step": "Step 3: Step 2 + Bayesian Hurdle (p*=0.05)", "PR_AUC": float(average_precision_score(y_test, p_step2)), "Brier_Score": float(brier_score_loss(y_test, p_step2)), "Mean_ROI": mc_s3["Mean_ROI"], "Sharpe_Ratio": mc_s3["Sharpe_Ratio"]},
        {"Step": "Step 4: Step 3 + Asymmetric Loss Objective", "PR_AUC": float(average_precision_score(y_test, p_step4)), "Brier_Score": float(brier_score_loss(y_test, p_step4)), "Mean_ROI": mc_s4["Mean_ROI"], "Sharpe_Ratio": mc_s4["Sharpe_Ratio"]},
        {"Step": "Step 5: Full BCAPM (Calibrated + Asymmetric + EMV)", "PR_AUC": float(average_precision_score(y_test, p_step5)), "Brier_Score": float(brier_score_loss(y_test, p_step5)), "Mean_ROI": mc_s5["Mean_ROI"], "Sharpe_Ratio": mc_s5["Sharpe_Ratio"]}
    ]
    df_ablation = pd.DataFrame(ablation_records)
    t4_path = REPORTS_DIR / "journal_table4_ablations.csv"
    save_csv_file(df_ablation, t4_path, index=False)
    logger.info(f"Exported Table 4 (Ablation Matrix) to: {t4_path}")

    # =========================================================================
    # EXPERIMENT 5: MARKET REGIME RESILIENCE ANALYSIS
    # =========================================================================
    logger.info("\n>>> EXECUTING EXPERIMENT 5: MARKET REGIME ANALYSIS ACROSS 5 EPOCHS")
    all_years = sub_liq["founded_year"].values
    all_y = y_liq
    all_p_bce = trained_models["Baseline 3: GBDT (Standard BCE)"].predict_proba(X_liq)[:, 1]
    all_p_bcapm = trained_models["Proposed: BCAPM Economic Model"].predict_proba(X_liq)[:, 1]

    regimes = [
        {"name": "Pre-2005 (Early Web Era)", "mask": (all_years <= 2004)},
        {"name": "2005-2007 (Web 2.0 Boom)", "mask": (all_years >= 2005) & (all_years <= 2007)},
        {"name": "2008-2009 (Financial Crisis)", "mask": (all_years >= 2008) & (all_years <= 2009)},
        {"name": "2010-2012 (Cloud & Mobile)", "mask": (all_years >= 2010) & (all_years <= 2012)},
        {"name": "2013-2015 (Unicorn Era)", "mask": (all_years >= 2013)}
    ]

    regime_records = []
    for reg in regimes:
        m = reg["mask"]
        if np.sum(m) > 10:
            y_sub = all_y[m]
            p_sub_bce = all_p_bce[m]
            p_sub_bcapm = all_p_bcapm[m]
            
            sim_bce = simulate_portfolio(y_sub, (p_sub_bce >= 0.50).astype(int), investment_amount, success_payoff)
            sim_bcapm = simulate_portfolio(y_sub, (p_sub_bcapm >= p_star).astype(int), investment_amount, success_payoff)
            
            regime_records.append({
                "Regime": reg["name"],
                "Sample_Count": int(np.sum(m)),
                "Empirical_Success_Rate": float(np.mean(y_sub)),
                "GBDT_PR_AUC": float(average_precision_score(y_sub, p_sub_bce)) if len(np.unique(y_sub)) > 1 else 0.0,
                "BCAPM_PR_AUC": float(average_precision_score(y_sub, p_sub_bcapm)) if len(np.unique(y_sub)) > 1 else 0.0,
                "GBDT_ROI": sim_bce["Portfolio_ROI"],
                "BCAPM_ROI": sim_bcapm["Portfolio_ROI"],
                "BCAPM_Capital_Preservation": sim_bcapm["Capital_Preservation_Ratio"]
            })

    df_regimes = pd.DataFrame(regime_records)
    t5_path = REPORTS_DIR / "journal_table5_regimes.csv"
    save_csv_file(df_regimes, t5_path, index=False)
    logger.info(f"Exported Table 5 (Market Regime Breakdown) to: {t5_path}")

    # =========================================================================
    # EXPERIMENT 6: 1,000 BOOTSTRAP CONFIDENCE INTERVALS (STATISTICAL SIGNIFICANCE)
    # =========================================================================
    logger.info("\n>>> EXECUTING EXPERIMENT 6: 1,000 BOOTSTRAP RESAMPLES FOR STATISTICAL SIGNIFICANCE")
    n_boot = 1000
    rng_b = np.random.default_rng(RANDOM_SEED)
    
    bcapm_boot_rois = []
    gbdt_boot_rois = []
    
    d_bcapm = (p_bcapm >= p_star).astype(int)
    d_gbdt = (p_bce >= 0.50).astype(int)
    
    for _ in range(n_boot):
        b_idx = rng_b.choice(len(y_test), size=len(y_test), replace=True)
        sim_b = simulate_portfolio(y_test[b_idx], d_bcapm[b_idx], investment_amount, success_payoff)
        sim_g = simulate_portfolio(y_test[b_idx], d_gbdt[b_idx], investment_amount, success_payoff)
        bcapm_boot_rois.append(sim_b["Portfolio_ROI"])
        gbdt_boot_rois.append(sim_g["Portfolio_ROI"])
        
    bcapm_ci_low, bcapm_ci_high = np.percentile(bcapm_boot_rois, 2.5), np.percentile(bcapm_boot_rois, 97.5)
    gbdt_ci_low, gbdt_ci_high = np.percentile(gbdt_boot_rois, 2.5), np.percentile(gbdt_boot_rois, 97.5)
    
    logger.info(f"BCAPM Portfolio ROI 95% CI: [{bcapm_ci_low:.2f}, {bcapm_ci_high:.2f}] (Mean: {np.mean(bcapm_boot_rois):.2f})")
    logger.info(f"GBDT Portfolio ROI 95% CI:  [{gbdt_ci_low:.2f}, {gbdt_ci_high:.2f}] (Mean: {np.mean(gbdt_boot_rois):.2f})")

    # =========================================================================
    # 7. GENERATE 8 JOURNAL-READY MANUSCRIPT FIGURES
    # =========================================================================
    logger.info("\n>>> GENERATING 8 JOURNAL-READY MANUSCRIPT FIGURES...")
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # Figure 1: Calibration Curve (Raw GBDT vs Platt vs Isotonic)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    cal_iso = fit_calibrator(trained_models["Baseline 3: GBDT (Standard BCE)"], X_val, y_val, method="isotonic")
    p_iso = cal_iso.predict_proba(X_test)[:, 1]
    e_raw = evaluate_calibration(y_test, p_bce)
    e_platt = evaluate_calibration(y_test, p_step2)
    e_iso = evaluate_calibration(y_test, p_iso)
    ax.plot(e_raw["prob_pred"], e_raw["prob_true"], "s-", label=f"Raw GBDT (Brier={e_raw['brier_score']:.4f})", lw=2)
    ax.plot(e_platt["prob_pred"], e_platt["prob_true"], "o-", label=f"Platt Calibrated (Brier={e_platt['brier_score']:.4f})", lw=2.5, color="darkgreen")
    ax.plot(e_iso["prob_pred"], e_iso["prob_true"], "^--", label=f"Isotonic Calibrated (Brier={e_iso['brier_score']:.4f})", lw=2, color="purple")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.set_title("Probability Calibration Diagram (Reliability Curves)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Empirical Fraction of Positives")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "journal_fig1_calibration.png", dpi=300)
    plt.close()

    # Figure 2: Precision-Recall Benchmark on Locked Test Set
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for m_name in ["Baseline 1: Logistic Regression", "Baseline 2: Random Forest", "Baseline 3: GBDT (Standard BCE)", "Proposed: BCAPM Economic Model"]:
        pr_vals, rc_vals, _ = precision_recall_curve(y_test, probs_dict[m_name])
        score = average_precision_score(y_test, probs_dict[m_name])
        ax.plot(rc_vals, pr_vals, label=f"{m_name.split(':')[1]} (PR-AUC = {score:.3f})", lw=2.5)
    ax.set_title("Precision-Recall Benchmark (Temporal Out-of-Time Test)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "journal_fig2_pr_curve.png", dpi=300)
    plt.close()

    # Figure 3: Empirical Pareto Frontier (Capital Preservation vs Winner Capture)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    thresholds_pareto = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    wcr_vals, cpr_vals = [], []
    for t in thresholds_pareto:
        sim_p = simulate_portfolio(y_test, (p_bcapm >= t).astype(int), investment_amount, success_payoff)
        wcr_vals.append(sim_p["Winner_Capture_Rate"])
        cpr_vals.append(sim_p["Capital_Preservation_Ratio"])
    ax.plot(wcr_vals, cpr_vals, "o-", color="crimson", lw=2.5, markersize=8)
    for i, t in enumerate(thresholds_pareto):
        ax.annotate(f"p={t:.2f}", (wcr_vals[i], cpr_vals[i]), textcoords="offset points", xytext=(-15, 8), fontsize=9)
    ax.set_title("Empirical Pareto Frontier: Capital Preservation vs. Winner Capture", fontsize=13, fontweight='bold')
    ax.set_xlabel("Winner Capture Rate (Recall on Successes)")
    ax.set_ylabel("Capital Preservation Ratio (Failures Avoided)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "journal_fig3_pareto_frontier.png", dpi=300)
    plt.close()

    # Figure 4: Top-K Fund Performance (Precision@K and ROI@K)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    df_t3 = pd.DataFrame(table3_records)
    sns.lineplot(data=df_t3, x="Fund_Capacity_K", y="Precision@K", hue="Strategy", marker="o", ax=ax1, lw=2.5)
    ax1.set_title("Fund Capacity (K) vs. Precision@K", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Fund Capacity (Number of Investments)")
    ax1.set_ylabel("Precision@K")
    
    sns.lineplot(data=df_t3, x="Fund_Capacity_K", y="Mean_ROI@K", hue="Strategy", marker="s", ax=ax2, lw=2.5)
    ax2.set_title("Fund Capacity (K) vs. Mean Monte Carlo ROI", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Fund Capacity (Number of Investments)")
    ax2.set_ylabel("Portfolio ROI Multiple")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "journal_fig4_top_k_efficiency.png", dpi=300)
    plt.close()

    # Figure 5: Monte Carlo Return Distributions (1,000 Draws KDE)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    mc_gbdt_draws = run_monte_carlo_simulation(y_test, d_gbdt, n_iterations=1000)["ROI_Distribution"]
    mc_bcapm_draws = run_monte_carlo_simulation(y_test, d_bcapm, n_iterations=1000)["ROI_Distribution"]
    sns.kdeplot(mc_gbdt_draws, fill=True, label="Standard GBDT (Conventional)", color="#e65100", ax=ax)
    sns.kdeplot(mc_bcapm_draws, fill=True, label="Proposed BCAPM (Economic Layer)", color="#1b5e20", ax=ax)
    ax.axvline(0, color="black", linestyle="--", lw=1.2)
    ax.set_title("Monte Carlo Portfolio Return Distribution (1,000 Iterations)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Simulated Portfolio ROI Multiple")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "journal_fig5_monte_carlo_distribution.png", dpi=300)
    plt.close()

    # Figure 6: Stepwise Architectural Ablation (ROI & Brier Score)
    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    steps = [f"S{i+1}" for i in range(len(df_ablation))]
    ax2 = ax1.twinx()
    ax1.bar(steps, df_ablation["Mean_ROI"], color="#2b5c8f", alpha=0.8, width=0.4, label="Mean Portfolio ROI")
    ax2.plot(steps, df_ablation["Brier_Score"], color="crimson", marker="o", lw=2.5, label="Brier Score (Lower is Better)")
    ax1.set_ylabel("Mean Portfolio ROI Multiple", color="#2b5c8f")
    ax2.set_ylabel("Brier Score", color="crimson")
    ax1.set_title("Stepwise Architectural Ablation Impact", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Architectural Step (S1: Raw -> S5: Full BCAPM)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "journal_fig6_ablation_progression.png", dpi=300)
    plt.close()

    # Figure 7: Bootstrap ROI Distributions with 95% Confidence Intervals
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.boxplot(data=pd.DataFrame({"GBDT Baseline": gbdt_boot_rois, "BCAPM Framework": bcapm_boot_rois}), palette=["#ffb74d", "#81c784"], ax=ax)
    ax.set_title("1,000 Bootstrap Resamples: Statistical Confidence Comparison", fontsize=13, fontweight='bold')
    ax.set_ylabel("Portfolio ROI Multiple")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "journal_fig7_bootstrap_ci.png", dpi=300)
    plt.close()

    # Figure 8: Market Regime Performance Trajectory
    fig, ax = plt.subplots(figsize=(8.5, 5))
    if len(df_regimes) > 0:
        df_reg_plot = df_regimes.set_index("Regime")[["GBDT_ROI", "BCAPM_ROI"]]
        df_reg_plot.plot(kind="bar", ax=ax, color=["#f57c00", "#388e3c"], width=0.6)
        ax.set_title("Macroeconomic Regime Resilience across Historical Epochs", fontsize=13, fontweight='bold')
        ax.set_ylabel("Portfolio ROI Multiple")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "journal_fig8_regime_resilience.png", dpi=300)
        plt.close()

    logger.info("=" * 80)
    logger.info("JOURNAL EXPERIMENTAL SUITE COMPLETE! ALL 5 TABLES & 8 FIGURES EXPORTED TO reports/")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete BCAPM Q1/Q2 journal experimental suite.")
    parser.add_argument("--investment", type=float, default=1.0, help="Check size in $M (default: 1.0)")
    parser.add_argument("--payoff", type=float, default=20.0, help="Base exit payoff in $M (default: 20.0)")
    args = parser.parse_args()
    
    run_full_journal_suite(investment_amount=args.investment, success_payoff=args.payoff)
