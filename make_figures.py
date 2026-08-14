#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py
===============
Generate the figures for the revision from the saved run outputs. No retraining;
reads *_arrays.npz (y_true / y_prob / fold / embeddings) and the *.json files.

Figures produced (PNG + PDF, 300 dpi) into an output folder:

  1. roc_<atlas>_<ver>.*        ROC curve with per-fold + mean AUC  (Reviewer C
                                "visualize model output"; honest AUC-based figure)
  2. tsne_<atlas>_<ver>.*       t-SNE of graph embeddings colored by ASD/Control
                                (class separability -> where classes overlap =
                                 where the model struggles)
  3. confusion_<atlas>_<ver>.*  mean confusion matrix (shows the false-negative
                                bias: missed ASD)
  4. training_curve_<atlas>_<ver>.*  val AUC / train loss vs epoch (from JSON
                                histories; pooled_cv only)
  5. topk_ablation.*            AUC & inference-time vs k curve (the ablation)

Usage:
    # point at your merged/main results folder for ROC/t-SNE/CM/curves:
    python3 make_figures.py --results results/run_20260620_073443 \
                            --atlas Schaefer200 --ver v2

    # or all cells found in a folder:
    python3 make_figures.py --results results/run_20260620_073443 --all

    # top-k ablation figure (edit the K_RUNS dict below to your folders):
    python3 make_figures.py --topk

Requires: numpy, matplotlib, scikit-learn (for t-SNE and roc_curve).
    pip install matplotlib scikit-learn
"""

import os
import glob
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc as sk_auc

OUT_DIR = "figures"

# ---- top-k ablation: map k -> results folder (EDIT to your verified folders) ----
K_RUNS = {
    5:  "results/run_20260705_114258",
    10: "results/run_20260620_073443",   # from main grid (Schaefer200/v2)
    20: "results/run_20260705_225127",
    30: "results/run_20260705_192300",
}
TOPK_ATLAS, TOPK_VER = "Schaefer200", "v2"


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", os.path.join(OUT_DIR, name) + ".png/.pdf")


def _load_arrays(results, atlas, ver, mode="pooled_cv"):
    path = os.path.join(results, f"{mode}_{atlas}_{ver}_arrays.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _load_json(results, atlas, ver, mode="pooled_cv"):
    path = os.path.join(results, f"{mode}_{atlas}_{ver}.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))


# ============================================================
# 1. ROC curve (per-fold + mean)
# ============================================================
def fig_roc(results, atlas, ver):
    arr = _load_arrays(results, atlas, ver)
    if arr is None:
        print(f"  [skip ROC] no arrays for {atlas} {ver}")
        return
    y_true, y_prob, fold = arr["y_true"], arr["y_prob"], arr["fold"]

    fig, ax = plt.subplots(figsize=(5, 5))
    mean_fpr = np.linspace(0, 1, 200)
    tprs, aucs = [], []
    for f in sorted(np.unique(fold)):
        m = fold == f
        fpr, tpr, _ = roc_curve(y_true[m], y_prob[m])
        a = sk_auc(fpr, tpr)
        aucs.append(a)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        ax.plot(fpr, tpr, lw=1, alpha=0.35, label=f"Fold {int(f)} (AUC {a:.2f})")

    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs); std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color="k", lw=2.5,
            label=f"Mean (AUC {mean_auc:.3f} ± {std_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {atlas} ({ver})")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_aspect("equal")
    _save(fig, f"roc_{atlas}_{ver}")


# ============================================================
# 2. t-SNE of embeddings
# ============================================================
def fig_tsne(results, atlas, ver):
    arr = _load_arrays(results, atlas, ver)
    if arr is None or "embeddings" not in arr or arr["embeddings"] is None:
        print(f"  [skip t-SNE] no embeddings for {atlas} {ver}")
        return
    emb = np.asarray(arr["embeddings"])
    if emb.ndim != 2 or emb.shape[0] != len(arr["y_true"]):
        print(f"  [skip t-SNE] embedding shape mismatch for {atlas} {ver}")
        return
    y = arr["y_true"]

    from sklearn.manifold import TSNE
    perp = max(5, min(30, (len(y) - 1) // 3))
    z = TSNE(n_components=2, perplexity=perp, init="pca",
             random_state=42).fit_transform(emb)

    fig, ax = plt.subplots(figsize=(5.2, 5))
    for lab, name, c in [(0, "ASD", "#d1495b"), (1, "Control", "#30638e")]:
        m = y == lab
        ax.scatter(z[m, 0], z[m, 1], s=12, alpha=0.6, label=name, color=c)
    ax.set_title(f"Graph embeddings (t-SNE) — {atlas} ({ver})")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=9)
    _save(fig, f"tsne_{atlas}_{ver}")


# ============================================================
# 3. Confusion matrix (mean over folds)
# ============================================================
def fig_confusion(results, atlas, ver):
    d = _load_json(results, atlas, ver)
    if d is None or d.get("mean_confusion_matrix") is None:
        print(f"  [skip CM] no confusion matrix for {atlas} {ver}")
        return
    cm = np.array(d["mean_confusion_matrix"], dtype=float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4.2, 4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    labels = ["ASD", "Control"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion (row-normalized) — {atlas} ({ver})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}\n(n={cm[i,j]:.0f})",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    _save(fig, f"confusion_{atlas}_{ver}")


# ============================================================
# 4. Training curves (pooled_cv histories)
# ============================================================
def fig_training(results, atlas, ver):
    d = _load_json(results, atlas, ver)
    if d is None or "histories" not in d or not d["histories"]:
        print(f"  [skip curves] no histories for {atlas} {ver}")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    for i, h in enumerate(d["histories"]):
        ax1.plot(h.get("train_loss", []), lw=1, alpha=0.6, label=f"Fold {i+1}")
        ax2.plot(h.get("val_auc", []), lw=1, alpha=0.6)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss"); ax1.set_title("Training loss")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val AUC"); ax2.set_title("Validation AUC")
    ax1.legend(fontsize=7)
    fig.suptitle(f"Training curves — {atlas} ({ver})")
    _save(fig, f"training_curve_{atlas}_{ver}")


# ============================================================
# 5. Top-k ablation curve
# ============================================================
def fig_topk():
    ks, aucs, stds, infers = [], [], [], []
    for k in sorted(K_RUNS):
        d = _load_json(K_RUNS[k], TOPK_ATLAS, TOPK_VER)
        if d is None:
            print(f"  [topk] missing json for k={k} in {K_RUNS[k]}")
            continue
        ks.append(k)
        aucs.append(d["mean"]["auc"])
        stds.append(d["std"]["auc"])
        infers.append(d["mean"].get("infer_ms_per_sample", np.nan))

    if not ks:
        print("  [skip topk] no data")
        return

    fig, ax1 = plt.subplots(figsize=(5.5, 4))
    ax1.errorbar(ks, aucs, yerr=stds, marker="o", color="#30638e",
                 lw=2, capsize=4, label="AUC")
    ax1.set_xlabel("Top-k edges per node")
    ax1.set_ylabel("Pooled CV AUC", color="#30638e")
    ax1.tick_params(axis="y", labelcolor="#30638e")
    ax1.set_xticks(ks)

    ax2 = ax1.twinx()
    ax2.plot(ks, infers, marker="s", color="#d1495b", lw=2,
             linestyle="--", label="Inference ms/sample")
    ax2.set_ylabel("Inference ms / sample", color="#d1495b")
    ax2.tick_params(axis="y", labelcolor="#d1495b")

    # mark the peak
    best = ks[int(np.argmax(aucs))]
    ax1.axvline(best, color="gray", lw=1, ls=":")
    ax1.set_title(f"Top-k ablation — {TOPK_ATLAS} ({TOPK_VER})")
    _save(fig, "topk_ablation")


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None, help="a run_* folder")
    ap.add_argument("--atlas", default="Schaefer200")
    ap.add_argument("--ver", default="v2")
    ap.add_argument("--all", action="store_true",
                    help="all atlas/version cells found in --results")
    ap.add_argument("--topk", action="store_true", help="make top-k ablation figure")
    args = ap.parse_args()

    if args.topk:
        print("Top-k ablation figure:")
        fig_topk()

    if args.results:
        cells = []
        if args.all:
            for jf in glob.glob(os.path.join(args.results, "pooled_cv_*.json")):
                base = os.path.basename(jf)[len("pooled_cv_"):-len(".json")]
                # split "<atlas>_<ver>"
                atlas, ver = base.rsplit("_", 1)
                cells.append((atlas, ver))
        else:
            cells = [(args.atlas, args.ver)]

        for atlas, ver in cells:
            print(f"\nFigures for {atlas} {ver}:")
            fig_roc(args.results, atlas, ver)
            fig_tsne(args.results, atlas, ver)
            fig_confusion(args.results, atlas, ver)
            fig_training(args.results, atlas, ver)

    if not args.topk and not args.results:
        ap.print_help()


if __name__ == "__main__":
    main()
