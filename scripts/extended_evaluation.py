import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs():
    train_df = pd.read_csv(DATA_DIR / "imputed_gene_expression_Training Dataset.csv", index_col=0)
    test_df = pd.read_csv(DATA_DIR / "Test_Dataset_Preprocessed.csv", index_col=0)
    biomarkers = pd.read_csv(RESULTS_DIR / "biomarkers_ML_Analysis.csv")

    x_train = train_df.drop(columns=["label"])
    y_train = train_df["label"].astype(int)

    drop_cols = [c for c in ["label", "DiseaseType", "Dataset"] if c in test_df.columns]
    x_test = test_df.drop(columns=drop_cols)
    y_test = test_df["label"].astype(int)

    return x_train, y_train, x_test, y_test, biomarkers


def choose_threshold(y_true, y_prob, min_specificity=0.85):
    thresholds = np.linspace(0.05, 0.95, 181)
    best = None

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if spec >= min_specificity:
            score = (sens, f1, spec)
            if best is None or score > best[0]:
                best = (score, thr)

    if best is not None:
        return float(best[1])

    best_j, best_thr = -1.0, 0.5
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_thr = j, thr
    return float(best_thr)


def tune_threshold_with_cv(x_train, y_train, genes):
    x = x_train[genes].copy()
    med = x.median()
    x = x.fillna(med)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_prob = np.zeros(len(y_train), dtype=float)

    for tr_idx, val_idx in cv.split(x, y_train):
        x_tr, x_val = x.iloc[tr_idx], x.iloc[val_idx]
        y_tr = y_train.iloc[tr_idx]

        scaler = StandardScaler()
        x_tr_s = scaler.fit_transform(x_tr)
        x_val_s = scaler.transform(x_val)

        model = LogisticRegression(C=1.0, max_iter=3000, random_state=42, class_weight="balanced")
        model.fit(x_tr_s, y_tr)
        oof_prob[val_idx] = model.predict_proba(x_val_s)[:, 1]

    return choose_threshold(y_train.values, oof_prob, min_specificity=0.85)


def fit_predict_prob(x_train, y_train, x_test, genes):
    x_tr = x_train[genes].copy()
    x_te = x_test[genes].copy()

    med = x_tr.median()
    x_tr = x_tr.fillna(med)
    x_te = x_te.fillna(med)

    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_te_s = scaler.transform(x_te)

    model = LogisticRegression(C=1.0, max_iter=3000, random_state=42, class_weight="balanced")
    model.fit(x_tr_s, y_train)
    return model.predict_proba(x_te_s)[:, 1]


def metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Sensitivity": sens,
        "Specificity": spec,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def save_confusion_and_report(y_true, y_prob, threshold, prefix):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    cm_df = pd.DataFrame(cm, index=["Actual_Normal", "Actual_Tumor"], columns=["Pred_Normal", "Pred_Tumor"])
    cm_df.to_csv(RESULTS_DIR / f"{prefix}_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Normal", "Pred Tumor"])
    ax.set_yticklabels(["Actual Normal", "Actual Tumor"])
    ax.set_title(f"Confusion Matrix ({prefix})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_confusion_matrix.png", dpi=300)
    plt.close(fig)

    report = classification_report(y_true, y_pred, target_names=["Normal", "Tumor"], output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(RESULTS_DIR / f"{prefix}_classification_report.csv")


def run_pathway_enrichment(genes):
    out_dir = RESULTS_DIR / "pathway_enrichment"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import gseapy as gp

        enr = gp.enrichr(
            gene_list=genes,
            gene_sets=["GO_Biological_Process_2023", "KEGG_2021_Human"],
            organism="Human",
            outdir=str(out_dir),
            no_plot=True,
            cutoff=1.0,
        )
        if hasattr(enr, "results") and enr.results is not None and len(enr.results) > 0:
            enr.results.to_csv(out_dir / "enrichment_results.csv", index=False)
            enr.results.head(25).to_csv(out_dir / "enrichment_top25.csv", index=False)
        else:
            (out_dir / "enrichment_status.txt").write_text("Enrichment ran but returned no terms.", encoding="utf-8")
    except Exception as exc:
        (out_dir / "enrichment_status.txt").write_text(
            f"Pathway enrichment not completed. Reason: {exc}", encoding="utf-8"
        )


def validate_leave_one_dataset_out(full_genes):
    train_pre = pd.read_csv(DATA_DIR / "Training_Dataset_Preprocessed.csv", index_col=0)
    if "Dataset" not in train_pre.columns:
        return
    datasets = sorted(train_pre["Dataset"].dropna().unique().tolist())
    genes = [g for g in full_genes if g in train_pre.columns]
    if not genes:
        return

    rows = []
    for holdout in datasets:
        tr = train_pre[train_pre["Dataset"] != holdout].copy()
        te = train_pre[train_pre["Dataset"] == holdout].copy()

        y_tr = tr["label"].astype(int)
        y_te = te["label"].astype(int)
        x_tr = tr[genes].copy()
        x_te = te[genes].copy()

        med = x_tr.median()
        x_tr = x_tr.fillna(med)
        x_te = x_te.fillna(med)

        scaler = StandardScaler()
        x_tr_s = scaler.fit_transform(x_tr)
        x_te_s = scaler.transform(x_te)

        model = LogisticRegression(C=1.0, max_iter=3000, random_state=42, class_weight="balanced")
        model.fit(x_tr_s, y_tr)

        prob = model.predict_proba(x_te_s)[:, 1]
        pred = (prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan

        rows.append(
            {
                "Holdout_Dataset": holdout,
                "Samples": len(te),
                "Tumor": int((y_te == 1).sum()),
                "Normal": int((y_te == 0).sum()),
                "AUROC": roc_auc_score(y_te, prob),
                "Accuracy": accuracy_score(y_te, pred),
                "Sensitivity": sens,
                "Specificity": spec,
            }
        )

    pd.DataFrame(rows).sort_values("Holdout_Dataset").to_csv(RESULTS_DIR / "leave_one_dataset_out_validation.csv", index=False)


def main():
    x_train, y_train, x_test, y_test, biomarkers = load_inputs()

    ranked = biomarkers.sort_values(["folds_selected", "gene"], ascending=[False, True])["gene"].tolist()
    consistent = biomarkers[biomarkers["folds_selected"] >= 3].sort_values(
        ["folds_selected", "gene"], ascending=[False, True]
    )["gene"].tolist()
    if not consistent:
        consistent = ranked

    def panel(genes, n):
        common = [g for g in genes if g in x_train.columns and g in x_test.columns]
        return common if n is None else common[:n]

    full_panel = panel(consistent, None)
    top10_panel = panel(ranked, 10)
    top5_panel = panel(ranked, 5)

    if not full_panel:
        raise RuntimeError("No overlapping genes found between train and test for evaluation panels.")

    tuned_threshold = tune_threshold_with_cv(x_train, y_train, full_panel)

    panels = {"top5": top5_panel, "top10": top10_panel, "full": full_panel}
    rows = []
    pr_curves = {}

    for name, genes in panels.items():
        if not genes:
            continue
        prob = fit_predict_prob(x_train, y_train, x_test, genes)
        thr = tuned_threshold if name == "full" else 0.5
        m = metrics(y_test.values, prob, thr)
        rows.append({"Panel": name, "Genes": len(genes), "Threshold": thr, **m})

        precision, recall, _ = precision_recall_curve(y_test.values, prob)
        pr_curves[name] = (recall, precision, m["AUPRC"])

        if name == "full":
            save_confusion_and_report(y_test.values, prob, 0.5, "external_full_threshold_0_5")
            save_confusion_and_report(y_test.values, prob, tuned_threshold, "external_full_threshold_tuned")

    pd.DataFrame(rows).sort_values("Panel").to_csv(RESULTS_DIR / "panel_comparison_external.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (recall, precision, auprc) in pr_curves.items():
        ax.plot(recall, precision, linewidth=2, label=f"{name} (AUPRC={auprc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("External Precision-Recall Curves")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "precision_recall_external.png", dpi=300)
    plt.close(fig)

    pd.DataFrame(
        {
            "Panel": ["full"],
            "Tuned_Threshold": [tuned_threshold],
            "Threshold_Method": ["maximize sensitivity with specificity >= 0.85 on 5-fold CV"],
        }
    ).to_csv(RESULTS_DIR / "threshold_tuning_summary.csv", index=False)

    run_pathway_enrichment(full_panel)
    validate_leave_one_dataset_out(full_panel)

    print("Saved extended evaluation outputs to results/.")


if __name__ == "__main__":
    main()
