# Multi-Atlas GCN for Autism Classification from Resting-State fMRI

**Paper:** *A Multi Atlas Evaluation of Functional Connectivity Based Graph Neural Networks for Autism Classification*

**Journal:** To be Added

---

## 1. Project Overview

This repository contains the code and results for evaluating **Graph Convolutional Networks (GCN)** for classifying Autism Spectrum Disorder (ASD) from resting-state fMRI functional connectivity (FC) matrices.

**What the project does:**
- Builds brain graphs from functional connectivity matrices extracted under **four brain atlases** (AAL, Schaefer-100, Schaefer-200, Schaefer-400) and **two FC estimation strategies** (V1: Pearson correlation, V2: Ledoit-Wolf shrinkage covariance).
- Trains a 2-layer GCN with global mean pooling for binary classification (ASD vs. Control).
- Evaluates under **three cross-validation protocols**: pooled 5-fold CV, leave-one-site-out (LOSO), and within-site CV.
- Compares GCN against **five non-graph baselines** (Logistic Regression, Linear SVM, RBF SVM, Random Forest, MLP) on the same data splits.
- Includes a **top-k ablation study** (k ∈ {5, 10, 20, 30}) for graph sparsification.

**Key findings:**
- **Schaefer-200** is the computational sweet spot, same AUC (~0.72 pooled, ~0.77 LOSO) as Schaefer-400 at roughly half the training time and inference cost.
- LOSO evaluation (weighted mean AUC up to 0.77) shows the model generalizes across acquisition sites.
- Per-node top-k = 10 edges balances accuracy and speed; denser graphs (k = 20, 30) add cost without improving AUC.

---

## 2. Repository Structure

```
gnn_multisite_framework_v2/
│
├── main.py                           # Entry point: runs all experiment modes across atlases
├── config.py                         # All hyperparameters, paths, atlas/mode configuration
├── requirements.txt                  # Python package dependencies
├── run.sh                            # Shell script: launches main.py with nohup logging
│
├── models/
│   └── gcn.py                        # 2-layer GCNConv + BatchNorm + global_mean_pool classifier
│
├── data/
│   └── dataset.py                    # FCDataset: loads FC .npz, builds PyG graphs, top-k sparsification
│
├── training/
│   └── trainer.py                    # Training loop, AUC early stopping, Youden-J threshold, metrics
│
├── experiments/
│   └── experiment_runner.py          # Pooled CV, LOSO, within-site CV orchestration; CSV/JSON/NPZ output
│
├── baseline_classifiers_preproc.py   # 5 non-graph baselines (LogReg, SVM, RF, MLP) on flattened FC
│
├── generate_fc_all_v1.py             # FC generation: Pearson correlation (V1) for all atlases
├── generate_fc_all_v2.py             # FC generation: Ledoit-Wolf shrinkage (V2) for all atlases
│
├── summarize_results.py              # Pretty-prints summary tables from all_results.csv or JSON files
├── merge_runs.py                     # Merges multiple run folders into one combined CSV
├── loso_per_site.py                  # Per-site LOSO breakdown: size-weighted AUC, site reliability
├── make_figures.py                   # Generates ROC, t-SNE, confusion matrix, training curves, top-k ablation
├── plot_loso_per_site.py             # Per-site LOSO AUC horizontal bar chart
├── abide1_demographics.py            # Cohort demographics summary and distribution figures
│
├── Phenotypic_V1_0b_preprocessed1.csv  # ABIDE I phenotype file (labels, demographics, site IDs)
│
├── figures/                          # Generated publication figures (PNG + PDF)
│   ├── roc_<atlas>_<ver>.*           # ROC curves per atlas/FC version
│   ├── tsne_<atlas>_<ver>.*          # t-SNE of graph embeddings
│   ├── confusion_<atlas>_<ver>.*     # Mean confusion matrices
│   ├── training_curve_<atlas>_<ver>.*# Training loss and validation AUC curves
│   ├── topk_ablation.*              # Top-k ablation: AUC vs inference time
│   ├── loso_per_site_Schaefer200_v2.*# Per-site LOSO AUC bar chart
│   ├── fig_age_distribution.*       # Age distribution by group
│   ├── fig_gender_distribution.*    # Gender distribution by group
│   └── fig_site_distribution.*      # Subject count per site
│
└── results/                          # Experiment outputs (JSON metrics, NPZ arrays, CSV)
    ├── run_20260620_073443/          # Main grid: pooled+LOSO+within-site, AAL/S100/S200 (v1/v2)
    ├── run_20260629_163108/          # Schaefer-400 gap-fill: pooled+LOSO+within-site (v1/v2)
    ├── merged/                       # Merged all_results.csv - full 24-cell grid
    ├── run_20260705_114258/          # Top-k ablation k=5 (Schaefer200/v2/pooled)
    ├── run_20260705_225127/          # Top-k ablation k=20 (Schaefer200/v2/pooled)
    └── run_20260705_192300/          # Top-k ablation k=30 (Schaefer200/v2/pooled)
```

---

## 3. Dataset

### ABIDE I Preprocessed

This study uses the **Autism Brain Imaging Data Exchange I (ABIDE I) Preprocessed** dataset.

| Property | Value |
|---|---|
| **Source** | [ABIDE Preprocessed Initiative](http://preprocessed-connectomes-project.org/abide/) |
| **Pipeline** | CPAC (Configurable Pipeline for the Analysis of Connectomes) |
| **Strategy** | `filt_global` (band-pass filtered, global signal regression) |
| **Derivative** | `func_preproc` (preprocessed functional MRI, MNI152-normalized) |
| **Subjects** | 884 (408 ASD / 476 Control) |
| **Sites** | 17 acquisition sites |

### Download

Download the preprocessed data from the [ABIDE Preprocessed](http://preprocessed-connectomes-project.org/abide/) website. You need the `func_preproc` derivative under the CPAC pipeline with `filt_global` strategy.

### Expected folder structure

```
data/
├── Phenotypic_V1_0b_preprocessed1.csv          # Phenotype file (included in this repo)
└── fc_matrices_from_pre_proc/
    ├── AAL/
    │   ├── CMU_b_0050669_func_preproc_v1.npz   # Pearson FC
    │   ├── CMU_b_0050669_func_preproc_v2.npz   # Ledoit-Wolf FC
    │   └── ...
    ├── Schaefer100/
    ├── Schaefer200/
    └── Schaefer400/
```

Each `.npz` file contains a key `"fc"` holding an N×N functional connectivity matrix.

### Note on ABIDE II

ABIDE II raw data was not used because some atlas-derived ROI time-series showed spatial misregistration artifacts for subjects not in MNI152 space. ABIDE I Preprocessed (`func_preproc`) guarantees MNI normalization, so every subject has the full atlas coverage and no subject is dropped.

---

## 4. Installation

### Requirements

- **Python** ≥ 3.9 (developed on 3.14)
- **PyTorch** ≥ 2.0
- **PyTorch Geometric** ≥ 2.4

### Setup

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch (adjust for your CUDA version - see https://pytorch.org)
pip install torch torchvision torchaudio

# 3. Install PyTorch Geometric (see https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
pip install torch-geometric

# 4. Install remaining dependencies
pip install -r requirements.txt
```

> **Note:** PyTorch Geometric installation depends on your PyTorch and CUDA versions. Follow the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for your specific setup.

---

## 5. How to Run

### Step 1: Configure paths

Edit `config.py` and set `BASE_DATA_DIR` to point to your local ABIDE data directory, or set the `ABIDE_DATA_DIR` environment variable:

```bash
export ABIDE_DATA_DIR="/path/to/your/abide/data"
```

The directory must contain `Phenotypic_V1_0b_preprocessed1.csv` and the `fc_matrices_from_pre_proc/` folder structure described above.

### Step 2: Generate FC matrices

Two scripts are provided to compute functional connectivity matrices from the `func_preproc` NIfTI files:

```bash
# V1: Pearson correlation + Fisher z-transform
python3 generate_fc_all_v1.py

# V2: Ledoit-Wolf shrinkage covariance + Fisher z-transform
python3 generate_fc_all_v2.py
```

Both scripts read `.nii.gz` files from `$ABIDE_DATA_DIR` (or `data/`) and write `.npz` files to `fc_matrices_from_pre_proc/<Atlas>/` with one file per subject per atlas. Each `.npz` contains a `"fc"` key holding the N×N connectivity matrix.

**Additional dependencies for FC generation:** `pip install nibabel nilearn tqdm`

Output naming convention: `{SITE}_{SUBJECT_ID}_func_preproc_{v1|v2}.npz`

### Step 3: Run GNN experiments

```bash
python3 main.py
```

This runs all atlas × FC version × experiment mode combinations defined in `config.py`. Results are saved to a timestamped `results/run_YYYYMMDD_HHMMSS/` folder. For long runs, use the provided shell script:

```bash
bash run.sh
```

### Step 4: Run baselines

```bash
python3 baseline_classifiers_preproc.py
```

Runs 5 non-graph classifiers (Logistic Regression, Linear SVM, RBF SVM, Random Forest, MLP) on flattened upper-triangle FC features using the same CV protocol and seed.

### Step 5: Summarize results

```bash
python3 summarize_results.py results/merged
```

Prints compact summary tables from the merged results CSV.

### Step 6: Generate figures

```bash
# All figures for a specific results folder:
python3 make_figures.py --results results/run_20260620_073443 --all

# Top-k ablation figure:
python3 make_figures.py --topk

# Demographics figures:
python3 abide1_demographics.py --pheno Phenotypic_V1_0b_preprocessed1.csv
```

### Step 7: LOSO per-site breakdown

```bash
python3 loso_per_site.py results/merged

# Per-site bar chart:
python3 plot_loso_per_site.py
```

---

## 6. Experiment Results

### 6.1 Pooled 5-Fold Cross-Validation (Main Results)

| Atlas | FC | AUC | Bal.Acc | Sensitivity | Specificity | Train (s) | Infer (ms) |
|---|---|---|---|---|---|---|---|
| AAL | V1 | 0.697±0.029 | 0.642 | 0.647 | 0.637 | 345 | 3.1 |
| AAL | V2 | 0.675±0.042 | 0.614 | 0.623 | 0.605 | 269 | 3.2 |
| Schaefer-100 | V1 | 0.670±0.027 | 0.630 | 0.601 | 0.660 | 312 | 2.4 |
| Schaefer-100 | V2 | 0.672±0.008 | 0.627 | 0.623 | 0.630 | 211 | 2.7 |
| Schaefer-200 | V1 | **0.727±0.019** | 0.668 | 0.671 | 0.664 | 666 | 6.3 |
| Schaefer-200 | V2 | 0.722±0.018 | 0.657 | 0.654 | 0.660 | 548 | 6.6 |
| Schaefer-400 | V1 | 0.725±0.020 | 0.673 | 0.671 | 0.674 | 1297 | 14.1 |
| Schaefer-400 | V2 | 0.723±0.027 | 0.649 | 0.720 | 0.577 | 1376 | 13.6 |

> **Note:** Sensitivity = ASD detection rate (recall); Specificity = Control detection rate. Threshold selected via Youden's J on training data only.

### 6.2 Leave-One-Site-Out (Size-Weighted Mean over 17 Sites)

| Atlas | FC | AUC (weighted) | Bal.Acc |
|---|---|---|---|
| AAL | V1 | 0.725 | 0.654 |
| AAL | V2 | 0.714 | 0.643 |
| Schaefer-100 | V1 | 0.704 | 0.614 |
| Schaefer-100 | V2 | 0.708 | 0.618 |
| Schaefer-200 | V1 | 0.761 | 0.692 |
| Schaefer-200 | V2 | **0.770** | 0.684 |
| Schaefer-400 | V1 | 0.762 | 0.660 |
| Schaefer-400 | V2 | 0.761 | 0.662 |

### 6.3 Top-k Ablation (Schaefer-200, V2, Pooled CV)

| k | AUC | Infer (ms/sample) |
|---|---|---|
| 5 | 0.714±0.016 | 3.27 |
| **10** | **0.722±0.018** | 6.6 |
| 20 | 0.705±0.006 | 11.78 |
| 30 | 0.687±0.020 | 17.62 |

Top-k = 10 achieves the best AUC at moderate inference cost.

---

## 7. Result Logs and Run Folders

Each run folder in `results/` contains:
- `all_results.csv` - one row per fold/site with all metrics
- `<mode>_<atlas>_<ver>.json` - aggregated results (mean, std, per-fold detail, training histories)
- `<mode>_<atlas>_<ver>_arrays.npz` - per-fold predictions and embeddings (for ROC/t-SNE figures)

| Folder | Contents |
|---|---|
| `results/run_20260620_073443/` | Main grid: pooled + LOSO + within-site for AAL, Schaefer-100, Schaefer-200 (v1/v2) |
| `results/run_20260629_163108/` | Schaefer-400 gap-fill: pooled + LOSO + within-site (v1/v2) |
| `results/merged/` | Merged `all_results.csv` - full 24-cell grid (4 atlases × 2 FC × 3 modes) |
| `results/run_20260705_114258/` | Top-k ablation k = 5 (Schaefer-200 / v2 / pooled) |
| `results/run_20260705_225127/` | Top-k ablation k = 20 (Schaefer-200 / v2 / pooled) |
| `results/run_20260705_192300/` | Top-k ablation k = 30 (Schaefer-200 / v2 / pooled) |

---

## 8. Figures

All figures are saved in `figures/` in both PNG (300 dpi) and PDF formats.

| Figure | Description | Paper Usage |
|---|---|---|
| `roc_<atlas>_<ver>` | ROC curves with per-fold and mean AUC | Model performance comparison |
| `confusion_<atlas>_<ver>` | Row-normalized confusion matrix (mean over folds) | Classification error analysis |
| `tsne_<atlas>_<ver>` | t-SNE of graph-level embeddings colored by ASD/Control | Embedding separability visualization |
| `training_curve_<atlas>_<ver>` | Training loss and validation AUC vs. epoch | Convergence analysis |
| `topk_ablation` | AUC and inference time vs. top-k value | Graph sparsification ablation |
| `loso_per_site_Schaefer200_v2` | Per-site LOSO AUC bar chart (color-coded by site size) | Cross-site generalization |
| `fig_age_distribution` | Age distribution histogram by group | Dataset demographics |
| `fig_gender_distribution` | Gender distribution bar chart by group | Dataset demographics |
| `fig_site_distribution` | Per-site subject counts by group | Dataset demographics |

---

## 9. Citation

To be added

---

## 10. License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2026 Tushar Dhole

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
