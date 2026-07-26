# Backdrop-Conditioned Analogy Precedent Modeling (BCAPM) for Startup Success Prediction

An end-to-end production Machine Learning pipeline for predicting startup success through **Backdrop-Conditioned Analogy Precedent Modeling (BCAPM)**. This research project integrates multi-source micro-level startup firmographics, founder human capital metrics, Hacker News community sentiment, and World Bank macroeconomic backdrop indicators.

---

## Project Structure

```
BCAMP/
├── DATASET_ANALYSIS.md          # Comprehensive catalog & audit of all 37 raw CSV datasets
├── PIPELINE_DESIGN.md           # In-depth 4-stage data pipeline architecture
├── Final.zip                    # Source raw dataset zip archive
├── run_pipeline.py              # Top-level CLI orchestrator running stages 1-4
├── requirements.txt             # Project dependencies
├── .gitignore                   # Git ignore settings
├── README.md                    # Project documentation
│
├── notebooks/                   # Jupyter notebooks for interactive analysis
├── reports/                     # Output figures, heatmaps, and evaluation tables
│   ├── dataset_collection_summary.csv
│   ├── merge_v2_report.txt
│   ├── SelectedFeatures.csv
│   ├── model_comparison.csv
│   └── *.png                    # EDA charts (distribution, heatmap, histograms)
│
├── models/                      # Serialized trained ML model artifacts (.joblib)
├── data/                        # Tiered data storage directory
│   ├── raw/                     # Extracted raw CSV files from Final.zip
│   ├── interim/                 # Staged master outputs (BCAPM_Master_V1, BCAPM_Master_V2)
│   └── processed/               # Cleaned & ML-ready tensors (BCAPM_Clean, BCAPM_Preprocessed)
│
└── src/                         # Modular production-grade Python package
    ├── config.py                # System paths, constants, and hyperparameters
    ├── utils.py                 # String canonicalization, currency parsing & validation
    ├── 01_collect_data.py       # Ingestion audit and schema report generation
    ├── 02_merge_v1.py           # Entity resolution & micro-level startup merge
    ├── 03_merge_v2.py           # Macroeconomic backdrop & World Bank indicator merge
    ├── 04_clean_data.py         # Missing value imputation & type normalization
    ├── 05_feature_engineering.py # Financial ratios, founder scores & climate indices
    ├── 06_preprocessing.py      # One-hot encoding & continuous feature scaling
    ├── 07_feature_selection.py  # Pearson, Mutual Info & Random Forest feature ranking
    ├── 08_eda.py                # Exploratory data visualization suite
    ├── 09_train_models.py       # Training Logistic Regression, DT, RF, XGBoost & ANN
    └── 10_similarity_engine.py  # BCAPM Cosine Similarity Analogy Precedent Engine
```

---

## Data Engineering Pipeline Architecture

The pipeline processes raw data through four distinct evolutionary tiers:

1. **Raw Datasets (`data/raw/`)**: Ingests Crunchbase, Y Combinator, Hacker News, CAX, and World Bank datasets.
2. **Stage 1: `BCAPM_Master_V1.csv`**: Resolves entity names and consolidates micro-level startup signals (firmographics, founders, funding rounds, sentiment).
3. **Stage 2: `BCAPM_Master_V2.csv`**: Embeds macroeconomic environmental backdrop features (World Bank internet penetration by founding year, country HDI, policy support, macro failure rates).
4. **Stage 3: `BCAPM_Clean.csv`**: Deduplicates records, enforces missing value imputation (sector median + global fallback), and cleans data types.
5. **Stage 4: `BCAPM_Preprocessed.csv`**: Generates engineered features (`CapitalEfficiency`, `BackdropScore`, `MacroStartupClimateScore`, etc.), applies one-hot encoding, and normalizes continuous features via `StandardScaler`.

---

## Quick Start & Execution

### 1. Installation
Ensure Python 3.10+ is installed. Install required packages:
```bash
pip install -r requirements.txt
```

### 2. Run Complete Data Pipeline
Execute the full 4-stage pipeline:
```bash
python run_pipeline.py
```

### 3. Run Individual Pipeline Stages (`src/`)

```bash
# 1. Audit Raw Datasets
python src/01_collect_data.py

# 2. Merge Micro Startup Datasets -> BCAPM_Master_V1.csv
python src/02_merge_v1.py

# 3. Merge Macro Backdrop Indicators -> BCAPM_Master_V2.csv
python src/03_merge_v2.py

# 4. Clean & Impute Data -> BCAPM_Clean.csv
python src/04_clean_data.py

# 5. Domain Feature Engineering -> BCAPM_Engineered.csv
python src/05_feature_engineering.py

# 6. ML Preprocessing & Scaling -> BCAPM_Preprocessed.csv
python src/06_preprocessing.py

# 7. Feature Selection & Importance Ranking
python src/07_feature_selection.py

# 8. Exploratory Data Analysis & Chart Generation
python src/08_eda.py

# 9. Train & Evaluate ML Models (LR, DT, RF, XGBoost, ANN)
python src/09_train_models.py

# 10. Query the BCAPM Analogy Precedent Engine
python src/10_similarity_engine.py --company "Dropbox" --top_k 5
```

---

## Precedent Analogy Engine Usage

To query historical precedent analogs for any target startup:

```python
from src.similarity_engine import BCAPMSimilarityEngine

engine = BCAPMSimilarityEngine()
results = engine.find_analogs_by_name("Airbnb", top_k=5)

print("Predicted Success Probability:", results["Precedent_Success_Probability"])
for analog in results["Top_K_Analogs"]:
    print(f"- {analog['Company_Name']} (Similarity: {analog['Similarity_Score']:.4f})")
```

---

## Verification & Outputs

- **Dataset Audit Report**: [DATASET_ANALYSIS.md](file:///d:/PROJECTS/BCAPM/BCAMP/DATASET_ANALYSIS.md)
- **Pipeline Architecture Specs**: [PIPELINE_DESIGN.md](file:///d:/PROJECTS/BCAPM/BCAMP/PIPELINE_DESIGN.md)
- **Model Evaluation Metrics**: `reports/model_comparison.csv`
- **EDA Visualizations**: `reports/*.png`
- **Trained Model Artifacts**: `models/*.joblib`
