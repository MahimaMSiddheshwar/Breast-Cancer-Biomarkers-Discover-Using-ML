# Breast Cancer Biomarker Discovery Using Machine Learning

**Team Members:** Mahima Mahabaleshwar Siddheshwar, Muni Manasa Vema 

 **Course:** Machine Learning for Bioinformatics
 
 **Date:** May 8, 2025
 
 **Languages & Tools:** Python (pandas, numpy, scikit-learn), matplotlib, seaborn, joblib, 
  imbalanced-learn
  
 **Dataset Source:** Gene Expression Omnibus (GEO)

## Project Overview

This project applies machine learning to identify robust gene expression biomarkers for breast cancer. Using five GEO datasets for training and two for external validation, models such as LASSO logistic regression, SVM, Random Forest, Gradient Boosting, and ANN were trained to distinguish tumor from normal tissue. The goal is to discover diagnostic genes for future clinical validation.

## Dataset Summary

### Training and Validation Datasets

* GSE24124: 99 tumor, 20 normal
* GSE32641: 95 tumor, 7 normal
* GSE36295: 45 tumor, 5 normal
* GSE42568: 104 tumor, 17 normal
* GSE53752: 51 tumor, 25 normal
* **Total:** 468 samples (394 tumor, 74 normal)

### External Test Datasets

* GSE70947: 148 tumor, 148 normal
* GSE109169: 25 tumor, 25 normal
* **Total:** 346 samples

## Workflow

1. **Data Preprocessing**

   * Load and clean GEO datasets.
   * Map probes to gene symbols.
   * Impute missing values and apply batch correction.
   * Assign labels: 0 = normal, 1 = tumor.

2. **Feature Selection**

   * Use LASSO regression to select top 28 predictive genes.

3. **Model Training**

   * Models: LASSO, SVM (RBF), Random Forest, GBM, ANN.
   * Stratified 80/20 split for validation.
   * Performance metrics: Accuracy, AUROC, F1-score, Sensitivity, Specificity.
   * Two settings:

     * Baseline (`class_weight='balanced'`)
     * SMOTE-enhanced (oversampling)

4. **External Testing**

   * Evaluate trained models on GSE70947 and GSE109169.

5. **Biomarker Analysis**

   * Visualize gene performance and statistical significance.
   * Generate violin plots, ROC curves, heatmaps.

## Performance Summary

### Internal Validation (Standard Training)

| Model               | Accuracy | AUROC | F1-Score | Sensitivity | Specificity |
| ------------------- | -------- | ----- | -------- | ----------- | ----------- |
| Logistic Regression | 0.968    | 0.997 | 0.981    | 0.962       | 1.000       |
| SVM                 | 0.989    | 0.997 | 0.994    | 1.000       | 0.933       |
| Random Forest       | 0.979    | 0.999 | 0.988    | 1.000       | 0.867       |
| GBM                 | 0.979    | 0.970 | 0.988    | 1.000       | 0.867       |
| ANN                 | 0.989    | 0.998 | 0.994    | 1.000       | 0.933       |

### Internal Validation (SMOTE-Enhanced Training)

| Model               | Accuracy | AUROC | F1-Score | Sensitivity | Specificity |
| ------------------- | -------- | ----- | -------- | ----------- | ----------- |
| Logistic Regression | 0.979    | 0.998 | 0.987    | 0.975       | 1.000       |
| SVM                 | 0.989    | 0.998 | 0.994    | 1.000       | 0.933       |
| Random Forest       | 0.989    | 0.999 | 0.994    | 1.000       | 0.933       |
| GBM                 | 0.979    | 0.998 | 0.988    | 1.000       | 0.867       |
| ANN                 | 0.989    | 0.998 | 0.994    | 1.000       | 0.933       |

### External Test (Standard Models)

| Model               | Accuracy | AUROC | F1-Score | Sensitivity | Specificity |
| ------------------- | -------- | ----- | -------- | ----------- | ----------- |
| Logistic Regression | 0.775    | 0.885 | 0.713    | 0.561       | 0.988       |
| SVM                 | 0.572    | 0.479 | 0.700    | 1.000       | 0.145       |
| Random Forest       | 0.659    | 0.861 | 0.726    | 0.902       | 0.416       |
| GBM                 | 0.558    | 0.699 | 0.496    | 0.361       | 0.958       |
| ANN                 | 0.730    | 0.876 | 0.796    | 0.665       | 0.994       |

### External Test (SMOTE-Trained Models)

| Model               | Accuracy | AUROC | F1-Score | Sensitivity | Specificity |
| ------------------- | -------- | ----- | -------- | ----------- | ----------- |
| Logistic Regression | 0.757    | 0.882 | 0.684    | 0.526       | 0.988       |
| SVM                 | 0.572    | 0.529 | 0.700    | 1.000       | 0.145       |
| Random Forest       | 0.630    | 0.866 | 0.418    | 0.266       | 0.994       |
| GBM                 | 0.558    | 0.699 | 0.496    | 0.361       | 0.958       |
| ANN                 | 0.730    | 0.876 | 0.757    | 0.613       | 0.994       |

## Confusion Matrix Analysis

Confusion matrices were generated for standard-trained models to provide deeper insight into model predictions.

### Confusion Matrix - LASSO (Standard)

```
              Predicted
              0      1
Actual 0    171      2
       1     76     97
```

**File:** `conf_matrix_lasso.png`

### Confusion Matrix - ANN (Standard)

```
              Predicted
              0      1
Actual 0    172      1
       1     58    115
```

**File:** `conf_matrix_ann.png`

## Top Biomarkers Identified

| Gene  | AUROC | Role                                                                 |
| ----- | ----- | -------------------------------------------------------------------- |
| POSTN | 0.847 | Involved in extracellular matrix remodeling and tumor invasion.      |
| LSR   | 0.791 | Regulates lipid metabolism and associated with metastasis.           |
| COMP  | 0.686 | Participates in TGF-β signaling and tumor extracellular matrix.      |
| CRTAM | 0.709 | Related to immune cell adhesion and cytotoxic activity.              |
| MYOC  | 0.179 | Limited individual utility, associated with eye pressure regulation. |
| OGN   | 0.265 | Implicated in collagen binding; low discriminative power.            |
| PENK  | 0.338 | Opioid precursor gene with minimal differential expression.          |
| EFHB  | 0.419 | Calcium-binding protein with unclear cancer-specific roles.          |
| EHF   | 0.613 | Transcription factor involved in epithelial differentiation.         |
| GZMB  | 0.559 | Immune cytotoxic effector gene; moderate classifier strength.        |

## Visualizations

* ROC curves for individual genes
* Violin plots of gene expression
* Box plots for tumor vs. normal comparison
* Heatmaps of top 10 gene expression

## Output Files

| Filename                         | Description                                   |
| -------------------------------- | --------------------------------------------- |
| `Imputed_gene_exprs_dataset.csv` | Final dataset with 28 selected genes          |
| `boxplot.png`                    | Box plots of top genes                        |
| `violinplot_test_top_genes.png`  | Violin plots comparing expression levels      |
| `gene_roc_curves.png`            | ROC curves for each top gene                  |
| `heatmap_top_genes.png`          | Heatmap of top gene expression                |
| `roc_comparison.png`             | Standard vs. SMOTE model ROC comparison       |
| `conf_matrix_lasso.png`          | Confusion matrix: Logistic Regression (LASSO) |
| `conf_matrix_ann.png`            | Confusion matrix: Artificial Neural Network   |

## Key Takeaways

* LASSO and ANN (standard models) had best generalization to external data.
* SMOTE improved class balance internally but reduced sensitivity externally.
* POSTN and LSR emerged as top diagnostic markers.
* Multi-gene models outperformed single gene predictors.

## Future Directions

* Incorporate clinical metadata (e.g., subtype, survival).
* Apply explainability techniques (e.g., SHAP values).
* Experimentally validate top genes in wet-lab studies.
* Compare models with commercial panels like Oncotype DX.


