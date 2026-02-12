# Breast Cancer Biomarker Discovery Report
## End-to-End Technical and Biological Analysis (Author Narrative)

I wrote this report to document my project exactly the way I would explain it to a reviewer, supervisor, collaborator, or future version of myself. My objective is to provide a complete account of what I did, why each step was necessary, what I observed in the outputs, how I interpret the findings, and what still needs to be done before clinical-level claims can be made.

This project is not just model training on a prepared table. I started from heterogeneous public GEO microarray data, converted probe-level measurements to gene-level features using platform annotation files, built harmonized training and external test matrices, trained multiple machine learning models, extracted stable biomarkers using L1-based selection, and then added deeper diagnostic and biological interpretation layers (confusion matrices, threshold tuning, panel-size comparison, precision-recall analysis, leave-one-dataset-out validation, and pathway enrichment).

My conclusion is that this project is successful as a **candidate biomarker discovery pipeline** with strong computational evidence and meaningful external performance. At the same time, it is not yet final clinical proof, and I explain that distinction clearly in this report.

---

## Quick Definitions (At a Glance)

I added this short section so any reader can immediately understand the terms before going into deep analysis.

### Performance metrics
- **Accuracy:** overall percent of correct predictions.
- **AUROC:** how well the model ranks tumor above normal across all thresholds.
- **F1 score:** balance between precision and recall.
- **Sensitivity (Recall for tumor):** how many true tumor samples are correctly detected.
- **Specificity:** how many true normal samples are correctly identified.
- **AUPRC:** precision-recall quality across thresholds (very useful in imbalanced detection tasks).

### Models compared
- **Logistic Regression:** linear, interpretable baseline and strong for biomarker panels.
- **SVM:** margin-based classifier, strong in high-dimensional feature spaces.
- **Random Forest:** ensemble of decision trees, robust for nonlinear tabular patterns.
- **Gradient Boosting:** sequential tree boosting, often high performance on structured data.
- **ANN:** neural network model for complex nonlinear relationships.

For full theory + practical interpretation, see Sections 26 and 27.

---


## 1. Project Purpose and Study Logic

The core question I addressed is: **Can I identify a robust breast cancer gene signature that generalizes beyond the datasets used for model training?**

To answer that, I designed the workflow around four principles:
1. biological correctness in preprocessing,
2. statistical discipline in feature/model evaluation,
3. external validation to test generalization,
4. biological interpretation to move beyond black-box prediction.

This matters because in omics ML it is easy to obtain inflated internal metrics if preprocessing and validation design are weak. I specifically aimed to avoid that by structuring preprocessing and external evaluation as separate, explicit stages.

---

## 2. Data Foundation: What I Used and Why

I used seven GEO datasets in total, split by role:

### Training cohorts (for model development)
- GSE24124
- GSE32641
- GSE36295
- GSE42568
- GSE53752

### External test cohorts (for independent evaluation)
- GSE70947
- GSE109169

I also used platform annotation files (`GPL*.txt`) because raw GEO expression is probe-level and platform-specific.

The reason this split is important is simple: if I evaluate only inside development data, I can overestimate how good the model really is. The external cohorts are the first real stress test of transferability.

---

## 3. Why Preprocessing Was Essential (Not Optional)

Preprocessing in this project is not cosmetic cleanup. It is the scientific foundation.

### 3.1 Probe-to-gene mapping
Raw microarray files report probe intensities, not final gene-level features. Different platforms define probes differently, and one gene can map to multiple probes. If I skip mapping, the model learns technical IDs that do not transfer cleanly across cohorts.

So I parsed each dataset with its corresponding GPL annotation file and mapped probe IDs to gene symbols. I normalized IDs and symbols to avoid hidden mismatch errors and aggregated duplicated probes per gene.

### 3.2 Label assignment
I extracted tumor/normal labels from sample metadata. This step is critical because any labeling inconsistency directly contaminates classification targets.

### 3.3 Cross-cohort harmonization
After each dataset was converted to gene-level format, I aligned cohorts using common genes and generated merged outputs:
- `data/processed/Training_Dataset_Preprocessed.csv`
- `data/processed/Test_Dataset_Preprocessed.csv`
- `data/processed/common_genes.txt`

In my current run, the processed dimensions were:
- Training preprocessed: `(468, 12508)`
- Test preprocessed: `(346, 11982)`
- Common genes list: `12505`

Label distribution:
- Training: 394 tumor, 74 normal
- Test: 173 tumor, 173 normal

This tells me preprocessing succeeded in producing valid multi-cohort matrices. It also highlights a class imbalance in training that I need to account for in model interpretation.

### 3.4 Model-ready training matrix usage
In `ML_Analysis.ipynb`, the training source is currently:
- `data/processed/imputed_gene_expression_Training Dataset.csv`

I confirmed it has shape `(468, 1855)` with zero missing values. This table is a curated/imputed training representation and is useful for stable modeling.

---

## 4. Modeling Strategy and Statistical Methods

### 4.1 Feature selection (LASSO-style)
I used L1-regularized logistic regression CV (`penalty='l1'`) to perform sparse feature selection. In this classification context, this is LASSO-style selection. The practical benefit is that many coefficients are shrunk to zero, leaving a compact panel of informative genes.

### 4.2 Models benchmarked
I compared:
- Logistic Regression
- SVM
- Random Forest
- Gradient Boosting
- ANN

This combination gives a useful spread from linear to nonlinear and ensemble behavior.

### 4.3 Metrics tracked
I interpreted performance with:
- Accuracy
- AUROC
- F1
- Sensitivity
- Specificity
- AUPRC (extended analysis)

I did not rely on a single metric because each reflects different behavior. In biomarker screening contexts, sensitivity, specificity, and PR behavior are especially important.

---

## 5. Internal Results: What I Observed

From `results/model_results_ML_Analysis.csv`, my internal performance was very high:
- Logistic Regression: Accuracy `0.987 ± 0.011`, AUROC `0.994 ± 0.011`
- SVM: Accuracy `0.991 ± 0.013`, AUROC `0.993 ± 0.013`
- Random Forest: Accuracy `0.991 ± 0.008`, AUROC `0.995 ± 0.009`
- Gradient Boosting: Accuracy `0.985 ± 0.009`, AUROC `0.992 ± 0.012`
- ANN: Accuracy `0.961 ± 0.023`, AUROC `0.979 ± 0.034`

My interpretation: there is clearly a strong predictive signal in the feature space. The consistency across model families suggests this is not only a single-model artifact.

However, I do not treat internal metrics as final evidence. External behavior is what determines whether this signature has practical value beyond its training domain.

---

## 6. External Validation: The Most Important Reality Check

From `results/external_validation_results.csv`, I obtained:
- AUROC: `0.889`
- Accuracy: `0.743`
- Sensitivity: `0.486`
- Specificity: `1.000`

This profile is very informative.

### What is strong
- AUROC near 0.89 indicates strong discrimination on independent cohorts.
- Specificity of 1.0 means no false positives at that operating point.

### What is limiting
- Sensitivity around 0.49 means the model misses a meaningful fraction of tumor cases.

So external performance is good overall, but the operating point is conservative. In other words, when the model says "tumor" it is usually right, but it does not catch every tumor case.

---

## 7. Biomarker Candidates and Stability

From `results/biomarkers_ML_Analysis.csv`, I obtained a 23-gene candidate set with fold-selection frequencies.

Genes with highest stability (`folds_selected = 5`) include:
- GZMK
- SELE
- OXTR
- HBB
- EFHB
- GZMB
- MYOC
- OGN
- INHBA
- PENK
- COMP

I interpret fold frequency as computational stability under resampling. It is not final biological proof by itself, but it is valuable evidence for prioritization.

---

## 8. Extended Diagnostic Layer I Added and Why

I expanded the pipeline because core metrics alone were not enough for a complete interpretation.

### 8.1 Confusion matrix and classification reports
I generated baseline and tuned-threshold confusion outputs.

Baseline (`threshold = 0.5`) confusion matrix:
- TN = 173
- FP = 0
- FN = 88
- TP = 85

Tuned (`threshold = 0.475`) confusion matrix:
- TN = 173
- FP = 0
- FN = 87
- TP = 86

This confirms the model currently produces almost no false positives, and false negatives are the dominant error class.

Classification report files quantify per-class behavior and reinforce that tumor recall is the bottleneck.

### 8.2 Precision-Recall analysis
I added PR curve output (`results/preci

---

## 17. Extended Technical Notes on Model Behavior and Generalization

In this section I explain in more depth how I interpret the gap between internal and external performance, because this is one of the most important aspects of the whole project.

When I look at internal results (`AUROC` close to 0.99 for several models), I read that as confirmation that the transformed feature space contains strong class signal. But when I test externally and get `AUROC` around 0.89 with lower sensitivity, I do not treat that as a failure. I treat it as expected behavior in cross-cohort omics settings. Different cohorts can differ in lab protocols, patient characteristics, tissue handling, platform chemistry, and hidden batch structure. Even with robust preprocessing, these differences can shift score distributions.

The critical point is that external discrimination stayed high enough to remain useful, which means the learned signature still carries transferable information. If the external AUROC had collapsed to near random (around 0.5), that would suggest overfitting or severe domain mismatch. That did not happen here.

At the same time, the sensitivity-specificity profile tells me the model currently sits in a conservative operating region. I get very few false positives, but many false negatives. If this were used in a screening context, I would need to shift operating policy toward higher recall. If this were used as a confirmatory adjunct where false positives are costly, the present profile might be acceptable.

This is why I added threshold tuning and panel-size comparison: not just to improve one score, but to understand how operating policy changes model behavior.

### Why top-10 can outperform full panel
A frequent assumption in biomarker work is that more genes should produce better performance. My panel comparison shows that this is not always true. In my run, the top-10 panel outperformed the 23-gene full panel on external data. I interpret this as a likely reduction of noise and cohort-specific overfitting. Compact panels can generalize better because they force the classifier to rely on the most stable signal components.

This is also helpful for downstream validation and potential translational work. Smaller panels are easier to test in wet-lab and easier to standardize.

### Why I care about AUPRC in addition to AUROC
AUROC measures ranking quality globally, but AUPRC focuses more directly on positive-class retrieval under thresholding pressure. In many clinical-like contexts, precision-recall behavior is operationally more meaningful than ROC alone. By adding PR analysis, I can evaluate whether high AUROC is supported by practical precision/recall trade-offs.

---

## 18. Statistical and Experimental Caveats I Explicitly Acknowledge

A rigorous report should state caveats clearly. These are the key caveats I acknowledge for my current project state.

### 18.1 Retrospective public data bias
All datasets are retrospective public cohorts. This is excellent for discovery and benchmarking, but clinical deployment requires prospective evaluation.

### 18.2 Cohort-level confounding risks
Even with harmonization, cohorts can encode hidden structure (site effects, sampling differences, subtype prevalence). Very high internal or cohort-holdout performance can be partly influenced by such structure. That is why truly independent external cohorts remain important.

### 18.3 Label quality dependence
My labels are metadata-derived. This is standard in GEO workflows, but any metadata inconsistency can introduce noise.

### 18.4 Biomarker versus mechanism distinction
A feature can be highly predictive without being mechanistically causal. I therefore treat selected genes as candidate markers and pathway clues, not final mechanistic proof.

### 18.5 Threshold dependence
Sensitivity and specificity are threshold-dependent. Reporting a single threshold without rationale can mislead interpretation. I addressed this by documenting threshold tuning and confusion-matrix shifts.

---

## 19. Practical Interpretation for Different Audiences

I find that different readers care about different parts of the result. I summarize how I would communicate the same project to each group.

### For ML reviewers
I would emphasize:
- robust preprocessing from raw probe-level data,
- sparse feature selection,
- multi-model comparison,
- external validation,
- threshold and PR diagnostics.

### For biological collaborators
I would emphasize:
- stable candidate genes,
- enrichment themes (immune, adhesion, ECM, differentiation),
- coherence between selected markers and pathway-level interpretation,
- prioritized candidates for qPCR/IHC follow-up.

### For clinical audiences
I would emphasize:
- external AUROC is good,
- specificity is high,
- sensitivity still needs improvement for screening use,
- top-10 panel appears more practical and potentially translatable.

---

## 20. Detailed Rationale for Candidate Biomarker Framing

I want to be very explicit about language because scientific wording matters.

When I say "candidate biomarker discovery," I mean:
- my genes are reproducibly selected in computational workflows,
- they help classification on external cohorts,
- they map to biologically plausible processes,
- they are suitable for next-stage validation.

When I avoid saying "clinically proven biomarkers," I mean:
- I have not yet run prospective, protocol-controlled, real-time validation,
- I have not completed wet-lab confirmation for top markers,
- I have not demonstrated decision impact versus clinical standard care pathways.

This distinction keeps the project scientifically honest and strengthens credibility.

---

## 21. Expanded Candidate Gene Prioritization for Follow-Up

If I need to prioritize a shortlist for immediate follow-up, I would organize it into tiers.

### Tier 1: high-priority validation genes
- INHBA
- COMP
- OGN
- SELE
- GZMB
- GZMK

Reason: strong computational stability and pathway coherence with tumor microenvironment, immune activity, and matrix/adhesion processes.

### Tier 2: structural and differentiation context genes
- DSP
- LSR
- CSTA
- MYOC

Reason: consistent with junction/differentiation themes and adds biological context depth.

### Tier 3: exploratory candidates needing careful confirmation
- OXTR
- PENK
- EFHB
- HBB
- CRTAM
- CIDEA

Reason: informative in current signatures, but they should be validated with extra caution for cohort/context effects.

For a qPCR pilot, I would start with Tier 1 + selected Tier 2 genes and compare tumor vs normal across an independent sample set.

---

## 22. Quality-Control Checklist I Would Include Before Submission

Before I finalize a thesis chapter or manuscript submission, I would confirm:
1. All notebook cells execute from clean kernel in order.
2. Output files regenerate consistently with fixed random seeds.
3. Figure captions explicitly state cohort and threshold context.
4. Methods section clearly distinguishes training, internal validation, and external validation.
5. Biomarker claims are framed as candidate-level.
6. Limitations section includes sensitivity caveat and validation roadmap.

I would also add a compact reproducibility appendix listing Python version, package versions, and run commands.

---

## 23. Final Integrated Interpretation

Putting all results together, this is how I interpret my project as a complete story:

I successfully built a multi-cohort, platform-aware breast cancer biomarker discovery pipeline that starts from raw GEO microarray files and ends with externally tested candidate panels plus biological pathway interpretation. The workflow is technically coherent, reproducible, and transparent. Internal metrics indicate strong class signal; external testing confirms meaningful transferability but reveals an operating-point tradeoff where sensitivity remains lower than desired for broad screening. Extended analyses show that a compact top-10 gene panel currently provides the best external balance and that the selected gene set is biologically coherent in immune, adhesion, and ECM/differentiation-rela

---

## 26. Metrics Explained (Theory + Practical Meaning)

In this section, I explain each metric the way I would write it in a thesis or viva discussion: first the theory, then what it practically means in my project.

### 26.1 Accuracy
**Theory:**
Accuracy is the fraction of total predictions that are correct.

Formula:
Accuracy = (TP + TN) / (TP + TN + FP + FN)

**Practical meaning in this project:**
Accuracy gives me a quick overall view, but by itself it is not enough. In cancer detection, two models can have similar accuracy but very different clinical behavior. A model with high accuracy can still miss many tumor cases if classes are imbalanced or if threshold is conservative. That is why I never interpret accuracy alone.

### 26.2 AUROC
**Theory:**
AUROC (Area Under Receiver Operating Characteristic Curve) measures how well the model ranks positives above negatives across all thresholds. It is threshold-independent.

Interpretation scale (rough guideline):
- 0.5: random ranking
- 0.7-0.8: acceptable
- 0.8-0.9: good
- >0.9: excellent

**Practical meaning in this project:**
AUROC tells me whether tumor samples generally receive higher probability scores than normal samples. My external AUROC around 0.89 means the ranking quality is strong even on independent cohorts. However, AUROC does not tell me how many tumors are missed at my chosen threshold.

### 26.3 F1 Score
**Theory:**
F1 is the harmonic mean of precision and recall.

Formula:
F1 = 2 * (Precision * Recall) / (Precision + Recall)

**Practical meaning in this project:**
F1 is useful because it balances two competing needs:
- precision (how many predicted tumors are truly tumors),
- recall/sensitivity (how many true tumors I correctly detect).

In my outputs, F1 helps compare models/panels when one has very high specificity but lower sensitivity.

### 26.4 Sensitivity (Recall for Tumor)
**Theory:**
Sensitivity = TP / (TP + FN)
It measures how many actual tumor cases are detected.

**Practical meaning in this project:**
Sensitivity is extremely important for screening-like tasks. Lower sensitivity means I am missing true tumor cases (false negatives). In my external results, sensitivity was the main limiting metric at baseline threshold, which is why I added threshold tuning and panel comparison.

### 26.5 Specificity
**Theory:**
Specificity = TN / (TN + FP)
It measures how many actual normal cases are correctly identified as normal.

**Practical meaning in this project:**
Specificity indicates false alarm control. My model has very high specificity externally, meaning almost no false positives. This is strong for confirmatory contexts but can come at the cost of lower sensitivity.

### 26.6 AUPRC
**Theory:**
AUPRC (Area Under Precision-Recall Curve) summarizes precision-recall behavior across thresholds.

**Practical meaning in this project:**
AUPRC is particularly valuable when positive detection quality matters and class distributions are uneven or operating decisions are threshold-sensitive. I added AUPRC in extended analysis to complement AUROC. It helped show that the top-10 panel not only ranked well but also had stronger precision-recall balance than alternatives.

### 26.7 Why I use all metrics together
If I use only one metric, I risk over-claiming. I therefore read metrics as a profile:
- AUROC: ranking quality
- Accuracy: broad correctness snapshot
- Sensitivity: tumor detection strength
- Specificity: false alarm control
- F1: precision-recall balance at threshold
- AUPRC: threshold-robust positive detection quality

This gives me a more clinically meaningful interpretation than any single number.

---

## 27. Model Families Compared (Theory + Practicality)

I compared five model types. I explain what each model is, why it is useful, and how to interpret it in my pipeline.

### 27.1 Logistic Regression
**Theory:**
A linear probabilistic classifier. It models log-odds of class membership as a weighted sum of features.

**Why useful in biomarker projects:**
- highly interpretable coefficients,
- stable baseline,
- works well with regularization (L1/L2),
- easy to calibrate and deploy.

**Practical interpretation in my project:**
It performs strongly and is often the most transparent option for candidate biomarker panels. I also use logistic models for LASSO-style feature selection.

### 27.2 Support Vector Machine (SVM)
**Theory:**
SVM finds a separating margin between classes; with kernels it can model nonlinear boundaries.

**Why useful:**
- strong performance in high-dimensional spaces,
- robust to many irrelevant features with proper tuning.

**Practical interpretation in my project:**
SVM performs very well internally. It is a strong comparator to confirm that signal is not specific to one model family.

### 27.3 Random Forest
**Theory:**
An ensemble of decision trees trained on bootstrapped samples and random feature subsets. Final prediction is aggregated across trees.

**Why useful:**
- captures nonlinear interactions,
- robust to noisy features,
- often strong default for tabular biology data.

**Practical interpretation in my project:**
Random Forest gives top internal metrics, indicating nonlinearity and interaction signal exist. However, I still prioritize external behavior and interpretability when choosing practical panels.

### 27.4 Gradient Boosting
**Theory:**
Builds trees sequentially, where each new tree focuses on correcting previous errors.

**Why useful:**
- often very high predictive power,
- models subtle nonlinear patterns.

**Practical interpretation in my project:**
It remains strong, but as with all boosting methods, careful validation is essential to ensure gains are real and not overfitting artifacts.

### 27.5 ANN (Artificial Neural Network)
**Theory:**
A layered nonlinear function approximator optimized by gradient descent.

**Why useful:**
- can capture complex nonlinear relationships,
- flexible architecture.

**Practical interpretation in my project:**
ANN performs well but not clearly better than simpler models in my current data regime. Given sample size and interpretability needs, simpler models may be preferable at this stage.

---

## 28. How I Compare Models Fairly in Practice

When I compare these models, I avoid comparing only one score. I evaluate:
1. internal consistency,
2. external transferability,
3. class-wise behavior (sensitivity/specificity),
4. panel simplicity and interpretability,
5. robustness under threshold variation.

This is why my final recommendation is not automatically the model with highest internal AUROC. I value practical external behavior and interpretability for biomarker development.

---

## 29. Theory-to-Practice Summary Table (Narrative)

If I summarize my interpretation in one practical framework:
- **Need ranking quality?** Check AUROC.
- **Need positive detection quality?** Check AUPRC + sensitivity.
- **Need low false alarms?** Check specificity.
- **Need balanced threshold behavior?** Check F1 + confusion matrix.
- **Need deployable panel?** Compare top-5/top-10/full externally.
- **Need biological plausibility?** Check enrichment terms and gene context.

This combined reading is what turns model scores into a scientifically defendable biomarker story.

