import os
import csv
import json
import torch
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader

from config import (
    BATCH_SIZE,
    LR,
    EPOCHS,
    N_SPLITS,
    SEED,
    WEIGHT_DECAY,
    HIDDEN_DIM,
    DROPOUT,
    get_topk_mode,
)
from models.gcn import GCNClassifier
from training.trainer import train_model, set_seed
from data.dataset import FCDataset


# ============================================
# Utilities
# ============================================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_results(results, save_path):
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)


# metric keys we treat as numeric scalars when aggregating
_SCALAR_KEYS = [
    "auc", "accuracy", "balanced_accuracy", "f1", "precision", "recall",
    "specificity", "threshold", "infer_time_sec", "infer_ms_per_sample",
    "n_eval", "train_time_sec", "epochs_run", "n_params",
]


def _aggregate(list_of_metric_dicts):
    """Mean/std over a list of best_metrics dicts (ignores confusion_matrix)."""
    keys = [k for k in list_of_metric_dicts[0]
            if k in _SCALAR_KEYS]
    mean = {k: float(np.mean([r[k] for r in list_of_metric_dicts])) for k in keys}
    std = {k: float(np.std([r[k] for r in list_of_metric_dicts])) for k in keys}
    cms = [np.array(r["confusion_matrix"]) for r in list_of_metric_dicts
           if "confusion_matrix" in r]
    mean_cm = np.mean(cms, axis=0).tolist() if cms else None
    return mean, std, mean_cm


def _save_arrays(output_dir, tag, fold_arrays):
    """
    Persist per-fold y_true / y_prob / embeddings for ROC curves and t-SNE.
    fold_arrays: list of dicts {y_true, y_prob, embeddings} (any may be None).
    Saved as <tag>_arrays.npz with concatenated arrays + a fold index vector.
    """
    y_true, y_prob, embs, fold_id = [], [], [], []
    have_emb = True
    for i, a in enumerate(fold_arrays):
        if a is None:
            continue
        yt = np.asarray(a.get("y_true"))
        yp = np.asarray(a.get("y_prob"))
        y_true.append(yt)
        y_prob.append(yp)
        fold_id.append(np.full(len(yt), i + 1))
        e = a.get("embeddings")
        if e is None:
            have_emb = False
        else:
            embs.append(e)

    if not y_true:
        return

    out = {
        "y_true": np.concatenate(y_true),
        "y_prob": np.concatenate(y_prob),
        "fold": np.concatenate(fold_id),
    }
    if have_emb and embs:
        out["embeddings"] = np.concatenate(embs, axis=0)

    np.savez(os.path.join(output_dir, f"{tag}_arrays.npz"), **out)


def _append_csv(output_dir, rows):
    """Append per-fold rows to a single flat results CSV for easy plotting."""
    csv_path = os.path.join(output_dir, "all_results.csv")
    fieldnames = [
        "mode", "atlas", "fc_version", "fold_or_site",
        "auc", "accuracy", "balanced_accuracy", "f1", "precision",
        "recall", "specificity", "threshold",
        "train_time_sec", "infer_ms_per_sample", "epochs_run", "n_params",
        "n_eval",
    ]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def _row(mode, atlas, fc_version, fold_or_site, m):
    """Build a flat CSV row from a metrics dict."""
    row = {"mode": mode, "atlas": atlas, "fc_version": fc_version,
           "fold_or_site": fold_or_site}
    for k in ("auc", "accuracy", "balanced_accuracy", "f1", "precision",
              "recall", "specificity", "threshold", "train_time_sec",
              "infer_ms_per_sample", "epochs_run", "n_params", "n_eval"):
        row[k] = m.get(k)
    return row


def _build_dataset(atlas_dir, csv_file, fc_version):
    return FCDataset(
        atlas_dir,
        csv_file,
        topk_mode=get_topk_mode(os.path.basename(atlas_dir)),
        fc_version=fc_version,
    )


def _new_model(input_dim, device):
    return GCNClassifier(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_classes=2,
        dropout=DROPOUT,
    ).to(device)


# ============================================
# POOLED CROSS VALIDATION
# ============================================

def run_pooled_cv(atlas_dir, csv_file, fc_version, output_dir):
    set_seed(SEED)
    atlas_name = os.path.basename(atlas_dir)
    print(f"\nRunning Pooled CV | {atlas_name} | {fc_version}")

    dataset = _build_dataset(atlas_dir, csv_file, fc_version)
    input_dim = dataset[0].x.shape[1]
    labels = [s["label"] for s in dataset.samples]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    device = get_device()

    fold_metrics = []
    fold_arrays = []
    histories = []
    csv_rows = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\nFold {fold+1}/{N_SPLITS}")

        train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx),
                                  batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx),
                                batch_size=BATCH_SIZE)

        model = _new_model(input_dim, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        history, best_metrics = train_model(model, train_loader, val_loader,
                                            optimizer, device, epochs=EPOCHS)

        fold_metrics.append(best_metrics)
        fold_arrays.append(history.pop("best_arrays", None))
        histories.append(history)
        csv_rows.append(_row("pooled_cv", atlas_name, fc_version, fold + 1, best_metrics))

    mean_metrics, std_metrics, mean_cm = _aggregate(fold_metrics)

    print("\n===== Pooled CV Results =====")
    for k in mean_metrics:
        print(f"{k}: {mean_metrics[k]:.4f} +/- {std_metrics[k]:.4f}")

    tag = f"pooled_cv_{atlas_name}_{fc_version}"
    save_results(
        {
            "mode": "pooled_cv",
            "atlas": atlas_name,
            "fc_version": fc_version,
            "folds": fold_metrics,
            "mean": mean_metrics,
            "std": std_metrics,
            "mean_confusion_matrix": mean_cm,
            "histories": histories,
        },
        os.path.join(output_dir, f"{tag}.json"),
    )
    _save_arrays(output_dir, tag, fold_arrays)
    _append_csv(output_dir, csv_rows)


# ============================================
# LEAVE ONE SITE OUT
# ============================================

def run_leave_one_site_out(atlas_dir, csv_file, fc_version, output_dir):
    set_seed(SEED)
    atlas_name = os.path.basename(atlas_dir)
    print(f"\nRunning Leave-One-Site-Out | {atlas_name} | {fc_version}")

    dataset = _build_dataset(atlas_dir, csv_file, fc_version)
    input_dim = dataset[0].x.shape[1]
    device = get_device()

    sites = sorted(set(s["site"] for s in dataset.samples))
    print(f"Sites: {sites}")

    per_site_metrics = []
    fold_arrays = []
    csv_rows = []

    for test_site in sites:
        train_idx = [i for i, s in enumerate(dataset.samples) if s["site"] != test_site]
        test_idx = [i for i, s in enumerate(dataset.samples) if s["site"] == test_site]

        test_labels = [dataset.samples[i]["label"] for i in test_idx]
        if len(set(test_labels)) < 2:
            print(f"  Skipping {test_site}: only one class in test set")
            continue

        print(f"\nTest site: {test_site}  (n={len(test_idx)})")

        train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx),
                                  batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx),
                                 batch_size=BATCH_SIZE)

        model = _new_model(input_dim, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        history, best_metrics = train_model(model, train_loader, test_loader,
                                            optimizer, device, epochs=EPOCHS)
        best_metrics["test_site"] = test_site
        per_site_metrics.append(best_metrics)
        fold_arrays.append(history.pop("best_arrays", None))
        csv_rows.append(_row("loso", atlas_name, fc_version, test_site, best_metrics))

    if not per_site_metrics:
        print("  No valid sites for LOSO; skipping save.")
        return

    mean_metrics, std_metrics, mean_cm = _aggregate(per_site_metrics)

    print("\n===== Leave-One-Site-Out Results (mean over sites) =====")
    for k in mean_metrics:
        print(f"{k}: {mean_metrics[k]:.4f} +/- {std_metrics[k]:.4f}")

    tag = f"loso_{atlas_name}_{fc_version}"
    save_results(
        {
            "mode": "leave_one_site_out",
            "atlas": atlas_name,
            "fc_version": fc_version,
            "per_site": per_site_metrics,
            "mean": mean_metrics,
            "std": std_metrics,
            "mean_confusion_matrix": mean_cm,
        },
        os.path.join(output_dir, f"{tag}.json"),
    )
    _save_arrays(output_dir, tag, fold_arrays)
    _append_csv(output_dir, csv_rows)


# ============================================
# WITHIN SITE CV
# ============================================

def run_within_site_cv(atlas_dir, csv_file, fc_version, output_dir):
    set_seed(SEED)
    atlas_name = os.path.basename(atlas_dir)
    print(f"\nRunning Within-Site CV | {atlas_name} | {fc_version}")

    dataset = _build_dataset(atlas_dir, csv_file, fc_version)
    input_dim = dataset[0].x.shape[1]
    device = get_device()

    sites = sorted(set(s["site"] for s in dataset.samples))
    per_site_mean = {}        # site -> mean metric dict across its folds
    csv_rows = []

    for site in sites:
        site_idx = [i for i, s in enumerate(dataset.samples) if s["site"] == site]
        labels = [dataset.samples[i]["label"] for i in site_idx]
        class_counts = Counter(labels)

        if len(class_counts) < 2 or min(class_counts.values()) < N_SPLITS:
            print(f"Skipping {site}: too few samples/class for {N_SPLITS}-fold")
            continue

        print(f"\nSite: {site}  (n={len(site_idx)})")
        site_subset = torch.utils.data.Subset(dataset, site_idx)
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

        site_fold_metrics = []
        for tr, va in skf.split(np.zeros(len(labels)), labels):
            train_loader = DataLoader(torch.utils.data.Subset(site_subset, tr),
                                      batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(torch.utils.data.Subset(site_subset, va),
                                    batch_size=BATCH_SIZE)

            model = _new_model(input_dim, device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

            _, best_metrics = train_model(model, train_loader, val_loader,
                                          optimizer, device, epochs=EPOCHS)
            site_fold_metrics.append(best_metrics)

        site_mean, _, _ = _aggregate(site_fold_metrics)
        per_site_mean[site] = site_mean
        csv_rows.append(_row("within_site", atlas_name, fc_version, site, site_mean))
        print(f"  {site}: AUC={site_mean['auc']:.3f}  Acc={site_mean['accuracy']:.3f}")

    if not per_site_mean:
        print("  No sites met the within-site CV criteria; skipping save.")
        return

    # average the per-site means into an overall number
    keys = list(next(iter(per_site_mean.values())).keys())
    mean_metrics = {k: float(np.mean([m[k] for m in per_site_mean.values()])) for k in keys}
    std_metrics = {k: float(np.std([m[k] for m in per_site_mean.values()])) for k in keys}

    print("\n===== Within-Site CV (mean over sites) =====")
    for k in mean_metrics:
        print(f"{k}: {mean_metrics[k]:.4f} +/- {std_metrics[k]:.4f}")

    tag = f"within_site_{atlas_name}_{fc_version}"
    save_results(
        {
            "mode": "within_site_cv",
            "atlas": atlas_name,
            "fc_version": fc_version,
            "per_site": per_site_mean,
            "mean": mean_metrics,
            "std": std_metrics,
        },
        os.path.join(output_dir, f"{tag}.json"),
    )
    _append_csv(output_dir, csv_rows)

