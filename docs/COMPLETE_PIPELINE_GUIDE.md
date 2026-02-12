# Complete Pipeline Guide (Raw GEO -> Biomarkers -> Validation -> Pathway Enrichment)

This document explains the full project workflow in practical and scientific terms: what each step does, why it is required, what files are used, what statistical methods are applied, and what outputs to expect.

## 1) Pipeline Objective

The objective of this project is to discover breast cancer biomarker candidates from gene expression microarray data and evaluate their predictive performance in a robust way. The pipeline combines bioinformatics preprocessing (probe-to-gene mapping, cohort harmonization) with machine learning evaluation (feature selection, model comparison, external testing, threshold optimization) and biological interpretation (GO/KEGG pathway enrichment).

The pipeline is designed to answer four core questions:
1. Can raw GEO data from multiple studies be integrated into a usable gene-level matrix?
2. Can we identify a stable, interpretable biomarker panel that predicts tumor vs normal?
3. Does model performance hold up on external datasets?
4. Are the selected genes biologically coherent in known pathways?

## 2) Input Data and Cohort Design

### 2.1 Raw expression and annotation files
- Raw expression files are in `data/raw/` as `GSE*_series_matrix.txt`
- Annotation files are in `data/annotation/` as `GPL*.txt`

### 2.2 Train/test cohort split
Training datasets (5):
- `GSE24124`
- `GSE32641`
- `GSE36295`
- `GSE42568`
- `GSE53752`

External test datasets (2):
- `GSE70947`
- `GSE109169`

This split is important because external cohorts are used to test generalization rather than in-sample fitting.

## 3) Preprocessing: What You Did and Why It Was Necessary

Preprocessing is not optional in this project. GEO microarray data are probe-level and platform-specific, while machine learning requires a consistent gene-level feature matrix.

### 3.1 Probe-to-gene mapping
Implemented in `notebooks/preprocessing.py` / `notebooks/preprocessing.ipynb`.

Why it matters:
- Different platforms use different probe IDs.
- Multiple probes can map to the same gene.
- Without mapping, features are not biologically consistent across datasets.

How it works:
- Parse each GPL annotation file.
- Normalize probe IDs and gene symbols.
- Join expression probes with annotation mapping.
- Aggregate duplicated probe mappings by gene.

### 3.2 Label assignment
Sample titles/metadata are parsed to assign:
- `label = 1` for tumor
- `label = 0` for normal

Why it matters:
- Classification quality depends directly on clean labels.
- Mislabeling creates false signal and unstable biomarkers.

### 3.3 Feature harmonization across studies
After mapping each dataset to genes, the pipeline intersects genes across cohorts and builds merged matrices.

Outputs:
- `data/processed/Training_Dataset_Preprocessed.csv`
- `data/processed/Test_Dataset_Preprocessed.csv`
- `data/processed/common_genes.txt`

Observed dimensions in this run:
- Train preprocessed: `(468, 12508)`
- Test preprocessed: `(346, 11982)`
- Common genes file: `12505` genes

Class counts:
- Training: 394 tumor, 74 normal (imbalanced)
- Test: 173 tumor, 173 normal (balanced)

### 3.4 Model-ready training matrix
`notebooks/ML_Analysis.ipynb` currently trains from:
- `data/processed/imputed_gene_expression_Training Dataset.csv`

This file is a curated/imputed training matrix with no missing values and reduced feature dimensionality, helping model stability and speed.

## 4) ML Analysis: Methods and Statistical Design

The main notebook is `notebooks/ML_Analysis.ipynb`.

### 4.1 LASSO-style feature selection
The notebook uses L1-regularized logistic regression (`penalty='l1'`) via `LogisticRegressionCV` for sparse feature selection.

Why this is important:
- Omics data are high-dimensional.
- L1 penalty drives irrelevant coefficients to zero.
- Result is a compact gene set with better interpretability.

### 4.2 Model families evaluated
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting
- Artificial Neural Network (ANN)

### 4.3 Evaluation metrics
Used across model comparison and external validation:
- Accuracy
- AUROC
- F1 score
- Sensitivity (recall for tumor)
- Specificity (recall for normal)
- AUPRC (added in extended analysis)

### 4.4 Data leakage awareness
The notebook uses fold-based feature handling and selection logic to reduce leakage risk. External testing is performed separately on independent cohorts.

## 5) Core Results: What They Mean

### 5.1 Internal model comparison
From `results/model_results_ML_Analysis.csv`:
- Logistic Regression: Accuracy ~0.987, AUROC ~0.994
- SVM: Accuracy ~0.991, AUROC ~0.993
- Random Forest: Accuracy ~0.991, AUROC ~0.995
- Gradient Boosting: Accuracy ~0.985, AUROC ~0.992
- ANN: Accuracy ~0.961, AUROC ~0.979

Interpretation:
- Internal discrimination is very strong across models.
- The signal in selected features is clearly predictive.

### 5.2 External validation
From `results/external_validation_results.csv`:
- AUROC: ~0.889
- Accuracy: ~0.743
- Sensitivity: ~0.486
- Specificity: 1.000

Interpretation:
- AUROC remains strong on external cohorts.
- Model is conservative: very high specificity but lower sensitivity.
- This motivates threshold optimization and panel simplification.

### 5.3 Biomarker stability file
From `results/biomarkers_ML_Analysis.csv`:
- Contains genes and fold-selection counts.
- Genes with higher `folds_selected` are more stable candidates.

## 6) Extended Analysis Added

A full extension script/cell was integrated to produce deeper diagnostics.

### 6.1 Confusion matrix and classification reports
Files:
- `results/external_full_threshold_0_5_confusion_matrix.csv`
- `results/external_full_threshold_0_5_confusion_matrix.png`
- `results/external_full_threshold_0_5_classification_report.csv`
- `results/external_full_threshold_tuned_confusion_matrix.csv`
- `results/external_full_threshold_tuned_confusion_matrix.png`
- `results/external_full_threshold_tuned_classification_report.csv`

Why this is important:
- Provides direct false negative and false positive counts.
- Class-wise precision/recall highlights practical operating behavior.

### 6.2 Precision-Recall and AUPRC
File:
- `results/precision_recall_external.png`

Why this is important:
- For class-imbalanced settings and clinical screening contexts, PR behavior is often more informative than ROC alone.

### 6.3 Threshold tuning
File:
- `results/threshold_tuning_summary.csv`

Current tuned threshold:
- ~0.475 for full panel

Method:
- CV-based rule maximizing sensitivity while enforcing specificity >= 0.85

### 6.4 Compact panel comparison
File:
- `results/panel_comparison_external.csv`

Comparison includes:
- Top-5 genes
- Top-10 genes
- Full panel (23 genes)

Observed outcome:
- Top-10 panel gives best overall external tradeoff in this run.

### 6.5 Leave-one-dataset-out validation
File:
- `results/leave_one_dataset_out_validation.csv`

Purpose:
- Stress-test robustness by holding out one training cohort at a time.

### 6.6 Pathway enrichment
Files:
- `results/pathway_enrichment/enrichment_results.csv`
- `results/pathway_enrichment/enrichment_top25.csv`
- Enrichr raw reports in the same folder
- PNG pathway bar plots from notebook:
  - `results/pathway_enrichment/GO_enrichment_top_terms.png`
  - `results/pathway_enrichment/KEGG_enrichment_top_terms.png`

Why this is important:
- Converts a predictive gene list into biologically interpretable themes.
- Supports translational relevance beyond pure classification metrics.

## 7) End-to-End Run Order

### Notebook mode (recommended)
1. `notebooks/preprocessing.ipynb`
2. `notebooks/ML_Analysis.ipynb`
3. Extended evaluation cells at notebook end
4. Pathway PNG plotting cell

### Script mode
- `python scripts/main_analysis.py`
- `python scripts/extended_evaluation.py`

## 8) Files Created and Their Purpose

### Processed data files
- `data/processed/Training_Dataset_Preprocessed.csv`: merged training matrix with metadata and genes
- `data/processed/Test_Dataset_Preprocessed.csv`: merged external test matrix
- `data/processed/common_genes.txt`: feature-space intersecti
