import time
import torch
import torch.nn.functional as F
import numpy as np
import random

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
)


# ============================================
# Reproducibility
# ============================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================
# Train One Epoch
# ============================================

def train_one_epoch(model, loader, optimizer, device, class_weights):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)

        optimizer.zero_grad()
        out = model(data)

        loss = F.cross_entropy(out, data.y, weight=class_weights)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ============================================
# Threshold selection (Youden's J)
# ============================================

def find_best_threshold(labels, probs):
    """Youden's J: threshold maximizing (sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx])


# ============================================
# Collect probabilities/labels (and optional embeddings) from a loader
# ============================================

def _collect(model, loader, device, with_embeddings=False):
    model.eval()
    all_probs, all_labels, all_embs = [], [], []
    n_samples, infer_time = 0, 0.0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            t0 = time.time()
            out = model(data)
            probs = torch.softmax(out, dim=1)[:, 1]
            infer_time += time.time() - t0
            n_samples += int(data.y.size(0))

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

            if with_embeddings:
                emb = _get_embeddings(model, data)
                if emb is not None:
                    all_embs.append(emb)

    probs = np.array(all_probs)
    labels = np.array(all_labels)
    embs = np.concatenate(all_embs, axis=0) if all_embs else None
    return probs, labels, embs, infer_time, n_samples


# ============================================
# Embedding extraction (graph-level, pre-classifier)
# ============================================

def _get_embeddings(model, data):
    """
    Output of global_mean_pool (pre-classifier) for t-SNE/UMAP plots.
    Mirrors GCNClassifier.forward up to pooling; fails soft (returns None)
    if the architecture differs.
    """
    try:
        from torch_geometric.nn import global_mean_pool
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = model.conv1(x, edge_index)
        x = model.bn1(x)
        x = F.relu(x)

        x = model.conv2(x, edge_index)
        x = model.bn2(x)
        x = F.relu(x)

        return global_mean_pool(x, batch).cpu().numpy()
    except Exception:
        return None


# ============================================
# Metrics at a GIVEN threshold (no peeking)
# ============================================

def metrics_at_threshold(labels, probs, threshold, infer_time=0.0, n_samples=0):
    preds = (probs >= threshold).astype(int)

    try:
        auc = roc_auc_score(labels, probs)      # threshold-independent
    except Exception:
        auc = 0.5

    cm = confusion_matrix(labels, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0

    return {
        "auc": float(auc),
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),   # sensitivity
        "specificity": float(specificity),
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
        "infer_time_sec": float(infer_time),
        "infer_ms_per_sample": float(1000.0 * infer_time / max(n_samples, 1)),
        "n_eval": int(n_samples),
    }


# ============================================
# Full Training Loop
# ============================================

def train_model(model,
                train_loader,
                val_loader,
                optimizer,
                device,
                epochs=100,
                early_stopping=20):

    best_val_auc = -1.0
    best_metrics = None
    best_arrays = None
    patience = 0

    history = {
        "train_loss": [],
        "val_auc": [],
        "val_accuracy": [],
        "epoch_time_sec": [],
    }

    # ---- class weights from training loader ----
    train_labels_all = []
    for batch in train_loader:
        train_labels_all.extend(batch.y.cpu().numpy())

    class_counts = np.bincount(train_labels_all)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    train_start = time.time()

    for epoch in range(epochs):
        ep_t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, class_weights)

        # --- threshold chosen on TRAINING predictions (no test-fold peeking) ---
        tr_probs, tr_labels, _, _, _ = _collect(model, train_loader, device,
                                                 with_embeddings=False)
        if len(set(tr_labels.tolist())) > 1:
            thr = find_best_threshold(tr_labels, tr_probs)
        else:
            thr = 0.5

        # --- evaluate validation at that fixed threshold ---
        va_probs, va_labels, va_embs, va_inf_t, va_n = _collect(
            model, val_loader, device, with_embeddings=True)
        metrics = metrics_at_threshold(va_labels, va_probs, thr, va_inf_t, va_n)

        ep_time = time.time() - ep_t0

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"AUC: {metrics['auc']:.4f} | "
            f"BalAcc: {metrics['balanced_accuracy']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"thr(train): {thr:.3f}"
        )

        history["train_loss"].append(train_loss)
        history["val_auc"].append(metrics["auc"])
        history["val_accuracy"].append(metrics["accuracy"])
        history["epoch_time_sec"].append(ep_time)

        # early stop on AUC (threshold-independent -> stable criterion)
        if metrics["auc"] > best_val_auc:
            best_val_auc = metrics["auc"]
            best_metrics = metrics
            best_arrays = {
                "y_true": va_labels,
                "y_prob": va_probs,
                "embeddings": va_embs,
            }
            patience = 0
        else:
            patience += 1

        if patience >= early_stopping:
            print("Early stopping triggered.")
            break

    total_train_time = time.time() - train_start

    n_params = int(sum(p.numel() for p in model.parameters()))
    best_metrics["train_time_sec"] = float(total_train_time)
    best_metrics["epochs_run"] = int(len(history["train_loss"]))
    best_metrics["n_params"] = n_params

    print("\nBest Validation Metrics:")
    for k, v in best_metrics.items():
        if k != "confusion_matrix":
            print(f"{k}: {v}")

    history["best_arrays"] = best_arrays
    return history, best_metrics
