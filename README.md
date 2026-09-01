# Detecting Bid-Rigging Cartels in Public Procurement

Machine learning models that flag likely bid-rigging cartels in Russian public
procurement auctions (44-FZ / 223-FZ) using indirect signals.

## Problem

Antitrust authorities review procurement cases for collusion by hand, which is slow
and expensive. Cartels are rare, so reviewers spend most of their time on clean
cases. The goal is not to convict anyone automatically — it is to **reorder the
review queue** so investigators see far more real cartels per case opened.

## Data

- **15,138 procurement auctions**, combining a commercial procurement database with
  published antitrust enforcement decisions used as labels
- **Severe class imbalance:** cartels are 1.7% of the sample (254 of 15,138)
- Split into 12,110 training and 3,028 test auctions (51 cartels in the test set)
- Engineered three families of features:
  - **price-based** — bid levels, price drop behaviour, contract value
  - **structural** — auction and participant characteristics
  - **graph-based** — repeated co-participation links between bidders

> Raw data is not included in this repository. Notebook outputs containing
> row-level auction records were cleared before publishing.

## Methods

- **Models:** Logistic Regression, Random Forest, XGBoost — each trained on base
  features and again with graph features added
- **Imbalance handling:** SMOTE oversampling and class weighting (`scale_pos_weight`)
- **Tuning:** RandomizedSearchCV for tree models
- **Validation:** cross-validation plus bootstrap 95% confidence intervals
- **Metrics:** rare-class performance (precision, recall, F1, lift), never accuracy —
  a model predicting "no cartel" everywhere scores 98.3% accuracy and is worthless

## Results

Test-set performance with bootstrap confidence intervals and cross-validation means:

| Model | Test F1 | 95% CI | CV mean F1 |
|---|---|---|---|
| Logistic Regression | 0.059 | [0.049, 0.070] | 0.066 |
| Random Forest — base | 0.232 | [0.170, 0.295] | 0.152 |
| **Random Forest — + graph features** | **0.270** | **[0.203, 0.339]** | 0.174 |
| XGBoost — base | 0.178 | [0.130, 0.227] | 0.212 |
| XGBoost — + graph features | 0.206 | [0.164, 0.249] | 0.229 |

**Best model: Random Forest with graph features.** It reaches precision 0.181 and
recall 0.529 — catching just over half of all cartels while flagging 149 of 3,028
auctions for review.

Against a 1.7% base rate, that precision means the model **concentrates cartels in
the review queue by 10.6×**. In practice: to find the same number of violations, an
investigator reviews roughly an order of magnitude fewer cases.

Two findings worth noting:

- **Graph features help.** Adding co-participation structure improved F1 for both
  tree models (RF 0.232 → 0.270, XGBoost 0.178 → 0.206).
- **XGBoost led on cross-validation but not on the test set**, and its CV variance
  was far lower than Random Forest's (std 0.024 vs 0.082) — a sign that the RF test
  result sits at the optimistic end of its range, worth stating honestly rather than
  reporting the single best number.

## Tech stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `imbalanced-learn`

## Repository structure

```
├── notebooks/
│   └── cartel_detection.ipynb   # analysis, feature engineering, model training
├── docs/
│   └── coursework_report.pdf    # full written report (in Russian)
├── .gitignore
└── README.md

data/   — raw datasets (not published)
```

## About

Coursework at HSE University, "Data-Driven Company Management" programme.
Team project by Korenkov, Semchugov and Makarov.

My contribution: feature engineering, model training and comparison, and the
evaluation methodology for the rare-class problem.
