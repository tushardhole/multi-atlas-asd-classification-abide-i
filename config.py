import os
from datetime import datetime

# =============================
# PATHS  (edit these to match your local data location)
# =============================
# Valid arm = ABIDE I func_preproc (CPAC / filt_global), already in MNI space.
BASE_DATA_DIR = os.environ.get("ABIDE_DATA_DIR", "data")
BASE_FC_DIR   = os.path.join(BASE_DATA_DIR, "fc_matrices_from_pre_proc")
CSV_FILE      = os.path.join(BASE_DATA_DIR, "Phenotypic_V1_0b_preprocessed1.csv")

# =============================
# ATLASES  (all four for the multi-atlas comparison)
# =============================
ATLASES = [
    "AAL",
    "Schaefer100",
    "Schaefer200",
    "Schaefer400",
]

# =============================
# FC VERSIONS (v1 = Pearson, v2 = Ledoit-Wolf robust)
# =============================
FC_VERSIONS = ["v1", "v2"]

# =============================
# EXPERIMENT MODES
# =============================
# pooled_cv          -> core 5-fold result (the main table)
# leave_one_site_out -> cross-site generalization (train on all sites but one)
# within_site_cv     -> per-site 5-fold (sites with >= N_SPLITS per class only)
#
# CPU NOTE: running all three over 4 atlases x 2 versions is a lot of training.
# If you are CPU-bound, do a first pass with only "pooled_cv", then enable the
# cross-site modes for a second run (the JSON/CSV outputs accumulate).
EXPERIMENT_MODES = [
    "pooled_cv",
    "leave_one_site_out",
    "within_site_cv",
]

# =============================
# TOP-K GRAPH SPARSIFICATION  (now a real, ablatable knob)
# =============================
# scheme:
#   "topk"    -> per-node top-k strongest edges (uniform degree ~k, no isolated
#                nodes). DEFAULT / primary method.
#   "density" -> global threshold keeping the strongest TOPK_DENSITY fraction of
#                all edges (kept only as a comparison point).
#   "full"    -> fully connected (reproduces the old, unintended behaviour).
TOPK_SCHEME  = "topk"
TOPK_K = 10
TOPK_DENSITY = 0.10        # used when TOPK_SCHEME == "density"

# --- ABLATION (top-k edges, Reviewer C) ---
# To sweep, restrict ATLASES/FC_VERSIONS above to one each (e.g. Schaefer200,
# v2) and EXPERIMENT_MODES to ["pooled_cv"], then run main.py once per value:
#   TOPK_K = 5, then 10, then 20, then 30.
# Each run appends to all_results.csv with its own tagged JSON.

# =============================
# TRAINING CONFIG
# =============================
BATCH_SIZE   = 16
LR           = 1e-3
EPOCHS       = 100        # ceiling; AUC early-stopping (patience 20 in trainer)
                          # will usually stop well before this. Lower to ~40-50
                          # if CPU runtime is too long.
N_SPLITS     = 5
SEED         = 42
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.2
HIDDEN_DIM   = 128

# =============================
# TOP-K MODE PER ATLAS (legacy hook)
# =============================
# Kept so the runner's import/signature stays valid. Sparsification is now
# controlled by TOPK_SCHEME/TOPK_K/TOPK_DENSITY above (read by FCDataset);
# this just returns the scheme name for the (ignored) topk_mode kwarg.
def get_topk_mode(atlas):
    return TOPK_SCHEME

# =============================
# OUTPUT DIRECTORY (timestamped) -- legacy helper, unused by main.py
# =============================
def create_output_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

