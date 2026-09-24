# ML Classification & SHAP Biomarker Screening on WGCNA Modules (METABRIC)

Machine learning + SHAP interpretability pipeline in Python, applied to gene modules identified by a prior WGCNA analysis on the METABRIC breast cancer dataset.

## Related Analyses

- WGCNA (gene module identification): [https://github.com/nuzlanrasjid/wgcna-analysis-metabric]

## Dataset

- **Source:** [Breast Cancer Gene Expression Profiles (METABRIC)](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric) — Kaggle
- **Samples:** 1,904 breast cancer patients (1,240 after filtering to the 5 PAM50 classes and dropping missing values)
- **Features used:** 137 total — 132 genes from 4 significant WGCNA modules (`blue`, `turquoise`, `brown`, `yellow`) + 5 clinical traits (age at diagnosis, tumor size, tumor stage, lymph nodes examined positive, Nottingham Prognostic Index)
- **Target classes:** 5 PAM50 molecular subtypes — Basal, LumA, LumB, Her2, Normal

> The raw CSV and the WGCNA module-assignment file are not included in this repository due to size/licensing — download `METABRIC_RNA_Mutation.csv` from the Kaggle link above and place it, along with `metabric_gene_modules_R.csv` (output of the WGCNA analysis linked above), as `data/`.

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib
```

Tested in Python on Google Colab.

## Pipeline

All steps below live in a single script, [`wgcna_after_ml.py`](./wgcna_after_ml.py):

1. **Load dataset & identify column types** (clinical / mutation / gene) — base pandas
2. **Select genes from significant WGCNA modules**, with name-matching validation — `isin()`, `df.columns` matching
3. **Add clinical traits**, with name-matching validation — base pandas
4. **Filter to 5 PAM50 classes, drop missing values, define X/Y** — `dropna()`, `LabelEncoder`
5. **Train-test split & feature scaling** — `train_test_split()`, `StandardScaler`
6. **Train classifiers** — `XGBClassifier`, `RandomForestClassifier`, `SVC`
7. **Compute SHAP values per model**, with array-shape validation — `shap.TreeExplainer`, `shap.KernelExplainer`
8. **Per-class + cross-model biomarker ranking** — `mean(|SHAP value|)`, rank averaging
9. **Directional check & stability check** on top candidates — `shap.summary_plot()`, repeated train-test splits

## Results

- **Model accuracy:** XGBoost ≈ 0.77, Random Forest ≈ 0.80, SVM ≈ 0.80 on PAM50 subtype classification (see `perbandingan_akurasi_model.png`).
- **SHAP importance:** per-model summary and bar plots generated for all three classifiers (`shap_summary_*.png`, `shap_bar_*.png`).
- **Cross-model biomarker screening for the Basal subtype:** `egfr` and `gata3` emerged as the strongest candidates — consistently ranked important across XGBoost, Random Forest, and SVM (by rank agreement, not raw SHAP value), with a clean, biologically consistent direction (low GATA3 / high EGFR pushing predictions toward Basal — both patterns match known Basal-like breast cancer biology) and stable ranking across 5 resampled train-test splits.
- `map2` and `e2f3` were also stable across splits, but showed a weaker or less consistent directional signal and are treated as secondary candidates.

See `output/tables/shap_rank_agreement_<class>.csv` for the full cross-model ranking, `output/figures/shap_directional_xgb_<class>.png` for the directional check, and `output/tables/shap_stability_<class>.csv` for the stability check across resampled splits.

*(Fill in final figures/tables paths once results are exported from Colab.)*

## Repository structure

```
.
├── data/                              # place METABRIC_RNA_Mutation.csv and metabric_gene_modules_R.csv here (not tracked)
├── wgcna_after_ml.py                  # full annotated ML + SHAP + biomarker screening pipeline
├── output/
│   ├── figures/
│   │   ├── perbandingan_akurasi_model.png
│   │   ├── shap_summary_xgboost.png / shap_bar_xgboost.png
│   │   ├── shap_summary_rf.png / shap_bar_rf.png
│   │   ├── shap_summary_svm.png / shap_bar_svm.png
│   │   └── shap_directional_xgb_<class>.png
│   └── tables/
│       ├── shap_rank_agreement_<class>.csv
│       └── shap_stability_<class>.csv
└── README.md
```

## Notes / lessons learned

- Features that directly define the target subtype (e.g. ER/HER2/PR status) were excluded from model inputs to avoid data leakage.
- SHAP values for a multiclass model come back as a 3D array `(n_samples, n_features, n_classes)`; passing this directly into `summary_plot()` without slicing a class can cause some `shap` versions to misread it as interaction values instead of per-class importance.
- SVM's SHAP values were computed via `KernelExplainer` on a 20-sample subset (versus the full test set used for the tree models), so its importance ranking is noisier and less directly comparable — cross-model agreement was assessed by rank, not raw SHAP magnitude, for this reason.
- A single train-test split can give an unstable "top gene" ranking, especially when candidate genes come from the same co-expression module and are correlated with each other; the stability check (5 resampled splits) was added specifically to catch this before treating any gene as a credible biomarker candidate.
