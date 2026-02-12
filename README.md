# Breast Cancer Biomarker Discovery Using Machine Learning

This repository contains an end-to-end bioinformatics and machine learning workflow for discovering breast cancer biomarker candidates from GEO microarray datasets and testing their predictive performance on external cohorts.

## What This Project Does

This project starts from raw GEO series matrix files, converts probe-level microarray data into gene-level expression using GPL annotation files, harmonizes multiple studies, builds training and external test matrices, and then applies feature selection plus model evaluation. The goal is to identify robust gene signatures that separate tumor and normal samples, then explain those signatures biologically with pathway enrichment analysis.

In practical terms, you are running a full translational pipeline:
- data curation and preprocessing from raw GEO exports
- cross-dataset harmonization and label assignment
- machine learning model training with leakage-aware validation
- external test evaluation and threshold analysis
- biomarker shortlist generation and pathway interpretation

## Datasets Used

- Training cohorts (5): `GSE24124`, `GSE32641`, `GSE36295`, `GSE42568`, `GSE53752`
- External test cohorts (2): `GSE70947`, `GSE109169`
- Platform annotation files: `data/annotation/GPL*.txt`

## Quick Start

```bash
cd "C:\Users\mmsid\Downloads\Breast-Cancer-Biomarkers-Discover-Using-ML-main"
pip install -r scripts/requirements.txt
```

### Option A (Notebook-first, recommended)
1. Run `notebooks/preprocessing.ipynb` (generates processed train/test matrices)
2. Run `notebooks/ML_Analysis.ipynb` (modeling, results, enrichment)

### Option B (Script entry)
```bash
python scripts/main_analysis.py
```
This executes `notebooks/ML_Analysis.ipynb` via `nbconvert`.

## Why Preprocessing Is Necessary

Raw GEO microarray files are probe-level and platform-specific. ML models require a consistent gene-level feature matrix. Preprocessing is therefore essential to:
- map probes to genes correctly for each platform
- collapse duplicate probes mapped to the same gene
- align features across studies by common genes
- create consistent binary labels (tumor/normal)
- produce train/test files with compatible feature spaces

Without these steps, model features become inconsistent, performance estimates become unreliable, and biological interpretation is weak.

## Main Outputs You Will Get

### Processed data
- `data/processed/Training_Dataset_Preprocessed.csv`
- `data/processed/Test_Dataset_Preprocessed.csv`
- `data/processed/common_genes.txt`

### Core ML outputs
- `results/model_results_ML_Analysis.csv`
- `results/biomarkers_ML_Analysis.csv`
- `results/ROC_ML_Analysis.png`
- `results/external_validation_results.csv`

### Extended evaluation outputs
- `results/panel_comparison_external.csv`
- `results/precision_recall_external.png`
- `results/threshold_tuning_summary.csv`
- `results/external_full_threshold_0_5_confusion_matrix.csv`
- `results/external_full_threshold_0_5_confusion_matrix.png`
- `results/external_full_threshold_0_5_classification_report.csv`
- `results/external_full_threshold_tuned_confusion_matrix.csv`
- `results/external_full_threshold_tuned_confusion_matrix.png`
- `results/external_full_threshold_tuned_classification_report.csv`
- `results/leave_one_dataset_out_validation.csv`

### Pathway enrichment outputs
- `results/pathway_enrichment/enrichment_results.csv`
- `results/pathway_enrichment/enrichment_top25.csv`
- `results/pathway_enrichment/GO_Biological_Process_2023.Human.enrichr.reports.txt`
- `results/pathway_enrichment/KEGG_2021_Human.Human.enrichr.reports.txt`
- Optional PNG plots from notebook:
  - `results/pathway_enrichment/GO_enrichment_top_terms.png`
  - `results/pathway_enrichment/KEGG_enrichment_top_terms.png`

## Methods Used

- Probe-to-gene mapping using GPL annotations
- Gene intersection across cohorts
- Missing value handling and imputed model-ready matrix usage
- L1-regularized logistic feature selection (LASSO-style)
- Nested cross-validation style model assessment in notebook workflow
- Models: Logistic Regression, SVM, Random Forest, Gradient Boosting, ANN
- Metrics: Accuracy, AUROC, F1, Sensitivity, Specificity, AUPRC
- Threshold tuning for sensitivity/specificity trade-off
- Biological pathway enrichment (GO/KEGG via Enrichr/GSEApy)

## Documentation

- Detailed pipeline guide: `docs/COMPLETE_PIPELINE_GUIDE.md`
- Full results interpretation report: `docs/EXCLUSIVE_RESULTS_REPORT.md`

