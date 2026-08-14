#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baseline_classifiers_preproc.py
===============================
Non-graph baselines on the VALID preprocessed (ABIDE I func_preproc) FC, with
CV folds identical to the preprocessed GNN (gnn_multisite_framework).

Why this is the clean comparison:
  - func_preproc is MNI-normalized -> every subject has the full atlas, so the
    flattened-FC design matrix is rectangular and NO subject is dropped.
  - We rebuild the sample list in the SAME os.listdir order FCDataset uses and
    apply StratifiedKFold(shuffle=True, random_state=SEED) -> the val folds are
    the same subjects the GNN validated on, fold-for-fold.

The GNN here uses the full FC matrix as node features + message passing, so the
right non-graph control is a classifier on the flattened upper-triangle FC
(same connectivity information, no message passing). If the GNN beats these,
message passing earns its place; if it ties, the contribution is the multi-atlas
benchmark, not a graph-specific accuracy gain. Either way it answers Reviewer C.

Outputs -> <BASE_DATA_DIR>/baselines_preproc/
  <atlas>_<version>_per_fold.csv / _summary.csv / _preds.npz
  baseline_preproc_master_summary.csv

Requirements: numpy, pandas, scikit-learn
"""

import os
import re
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG  -- mirror the preprocessed framework's config.py
# ---------------------------------------------------------------------------
BASE_DATA_DIR = os.environ.get("ABIDE_DATA_DIR", "data")
BASE_FC_DIR   = os.path.join(BASE_DATA_DIR, "fc_matrices_from_pre_proc")
CSV_FILE      = os.path.join(BASE_DATA_DIR, "Phenotypic_V1_0b_preprocessed1.csv")
RESULTS_DIR   = os.path.join(BASE_DATA_DIR, "baselines_preproc")

ATLASES  = ["AAL", "Schaefer100", "Schaefer200", "Schaefer400"]
VERSIONS = ["v1", "v2"]

N_SPLITS = 5
SEED     = 42                              # MUST equal the framework's SEED

PCA_COMPONENTS_IF_HIGHDIM = 200
HIGHDIM_THRESHOLD         = 2000

os.makedirs(RESULTS_DIR, exist_ok=True)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Replicate FCDataset's subject/label construction AND ordering
# ---------------------------------------------------------------------------
def sid7(name):
    m = re.search(r"\d{7}", os.path.basename(name))
    return int(m.group(0)) if m else None


def load_phenotype():
    df = pd.read_csv(CSV_FILE)
    subj_col = "subject" if "subject" in df.columns else "SUB_ID"
    label_map = dict(zip(df[subj_col].astype(int),
                         (df["DX_GROUP"].astype(int) - 1)))   # 0=ASD, 1=Control
    return label_map


def build_dataset(atlas, version, label_map):
    """
    Build the sample matrix in a DETERMINISTIC order (sorted by subject id).

    Both this baseline and the GNN use StratifiedKFold(shuffle=True,
    random_state=SEED); folds are identical iff the label vector is in the same
    order. Relying on os.listdir order is fragile (filesystem/machine dependent),
    so we sort by subject id for reproducibility. Report as "same CV protocol,
    same seed" -- fold membership is stable rather than tied to listing order.
    """
    atlas_dir = os.path.join(BASE_FC_DIR, atlas)

    entries = []
    for fname in os.listdir(atlas_dir):
        if not fname.endswith("_{}.npz".format(version)):
            continue
        sid = sid7(fname)
        if sid is None or sid not in label_map:
            continue
        entries.append((sid, os.path.join(atlas_dir, fname)))
    entries.sort(key=lambda e: e[0])          # deterministic: by subject id

    X, y, ids = [], [], []
    for sid, path in entries:
        fc = np.load(path)["fc"].astype(np.float32)
        fc = np.nan_to_num(fc)
        np.fill_diagonal(fc, 0.0)
        iu = np.triu_indices(fc.shape[0], k=1)
        X.append(fc[iu])
        y.append(int(label_map[sid]))
        ids.append(sid)
    return np.asarray(X, dtype=np.float32), np.asarray(y), ids


def make_models(n_features):
    pre = [("scale", StandardScaler())]
    if PCA_COMPONENTS_IF_HIGHDIM and n_features > HIGHDIM_THRESHOLD:
        pre.append(("pca", PCA(n_components=min(PCA_COMPONENTS_IF_HIGHDIM, n_features),
                               random_state=SEED)))

    def pipe(clf):
        return Pipeline(pre + [("clf", clf)])

    return {
        "LogReg":       pipe(LogisticRegression(max_iter=2000, C=1.0)),
        "LinearSVM":    pipe(SVC(kernel="linear", C=1.0, probability=True, random_state=SEED)),
        "RBF_SVM":      pipe(SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=SEED)),
        "RandomForest": pipe(RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
        "MLP_128":      pipe(MLPClassifier(hidden_layer_sizes=(128,), max_iter=500,
                                           early_stopping=True, random_state=SEED)),
    }


def scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    s = model.decision_function(X)
    return (s - s.min()) / (s.max() - s.min() + 1e-12)


def run(atlas, version, label_map):
    print("\n===== {} | {} =====".format(atlas, version))
    X, y, _ = build_dataset(atlas, version, label_map)
    if len(X) == 0:
        print("  [skip] no data")
        return None
    print("  X: {}  pos_rate(control)={:.3f}".format(X.shape, y.mean()))

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    models = make_models(X.shape[1])
    rows = []
    preds = {m: {"y_true": [], "y_prob": [], "fold": []} for m in models}

    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        Xtr, Xva, ytr, yva = X[tr], X[va], y[tr], y[va]
        for name, model in models.items():
            t0 = time.time(); model.fit(Xtr, ytr); fit_t = time.time() - t0
            t1 = time.time(); yp = model.predict(Xva); ps = scores(model, Xva)
            inf_ms = 1000.0 * (time.time() - t1) / len(yva)
            try:
                auc = roc_auc_score(yva, ps)
            except ValueError:
                auc = float("nan")
            rows.append({
                "atlas": atlas, "version": version, "model": name, "fold": fold,
                "accuracy": accuracy_score(yva, yp),
                "precision": precision_score(yva, yp, zero_division=0),
                "recall": recall_score(yva, yp, zero_division=0),
                "f1": f1_score(yva, yp, zero_division=0),
                "auc": auc, "fit_time_sec": fit_t, "infer_ms_per_sample": inf_ms,
                "confusion_matrix": str(confusion_matrix(yva, yp)),
            })
            preds[name]["y_true"].extend(yva.tolist())
            preds[name]["y_prob"].extend(ps.tolist())
            preds[name]["fold"].extend([fold] * len(yva))

    fdf = pd.DataFrame(rows)
    fdf.to_csv(os.path.join(RESULTS_DIR, "{}_{}_per_fold.csv".format(atlas, version)), index=False)
    num = ["accuracy", "precision", "recall", "f1", "auc", "infer_ms_per_sample", "fit_time_sec"]
    summ = fdf.groupby("model")[num].mean().reset_index()
    summ.insert(0, "version", version); summ.insert(0, "atlas", atlas)
    summ.to_csv(os.path.join(RESULTS_DIR, "{}_{}_summary.csv".format(atlas, version)), index=False)
    np.savez(os.path.join(RESULTS_DIR, "{}_{}_preds.npz".format(atlas, version)),
             **{m: {k: np.array(v) for k, v in d.items()} for m, d in preds.items()})
    print(summ.to_string(index=False))
    return summ


def main():
    label_map = load_phenotype()
    all_s = []
    for atlas in ATLASES:
        if not os.path.isdir(os.path.join(BASE_FC_DIR, atlas)):
            print("[skip] no folder for", atlas); continue
        for v in VERSIONS:
            s = run(atlas, v, label_map)
            if s is not None:
                all_s.append(s)
    if all_s:
        full = pd.concat(all_s, ignore_index=True)
        full.to_csv(os.path.join(RESULTS_DIR, "baseline_preproc_master_summary.csv"), index=False)
        print("\n===== MASTER SUMMARY =====")
        print(full.to_string(index=False))
        print("\nWrote results to:", RESULTS_DIR)
        print("\nNOTE: compare GNN vs baselines primarily on AUC (threshold-free).")
        print("      Baseline accuracy/F1/etc. use the default 0.5 threshold,")
        print("      while the GNN uses a train-tuned (Youden-J) threshold, so")
        print("      those thresholded columns are NOT directly comparable.")


if __name__ == "__main__":
    main()
