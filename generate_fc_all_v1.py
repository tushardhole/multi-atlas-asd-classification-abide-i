import os
import time
import numpy as np
import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker
from nilearn import datasets


# -----------------------------
# CONFIG  (edit these to match your local data location)
# -----------------------------
INPUT_DIR = os.environ.get("ABIDE_DATA_DIR", "data")       # folder with .nii.gz files
OUTPUT_DIR = os.path.join(INPUT_DIR, "fc_matrices_from_pre_proc")  # output: <atlas>/<subject>_v1.npz


# -----------------------------
# Atlas Loader
# -----------------------------
def load_atlases():

    atlases = {}

    print("Loading AAL atlas...")
    aal = datasets.fetch_atlas_aal()
    atlases["AAL"] = (aal['maps'], len(aal['labels']))

    print("Loading Schaefer atlas (100 ROIs)...")
    sch100 = datasets.fetch_atlas_schaefer_2018(n_rois=100)
    atlases["Schaefer100"] = (sch100['maps'], 100)

    print("Loading Schaefer atlas (200 ROIs)...")
    sch200 = datasets.fetch_atlas_schaefer_2018(n_rois=200)
    atlases["Schaefer200"] = (sch200['maps'], 200)

    print("Loading Schaefer atlas (400 ROIs)...")
    sch400 = datasets.fetch_atlas_schaefer_2018(n_rois=400)
    atlases["Schaefer400"] = (sch400['maps'], 400)

    return atlases


# -----------------------------
# FC Computation
# -----------------------------
def compute_fc(time_series):

    corr = np.corrcoef(time_series.T)

    corr = np.nan_to_num(corr)
    corr = np.clip(corr, -0.999999, 0.999999)

    fisher = np.arctanh(corr)

    np.fill_diagonal(fisher, 0.0)

    return fisher


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def generate_fc():

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    atlases = load_atlases()

    fmri_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.endswith(".nii.gz")
    ]

    print("Found {} fMRI scans".format(len(fmri_files)))

    for atlas_name in atlases:

        atlas_file, n_rois = atlases[atlas_name]

        print("\n==============================")
        print("Processing Atlas:", atlas_name)
        print("ROIs:", n_rois)
        print("==============================")

        atlas_out = os.path.join(OUTPUT_DIR, atlas_name)
        if not os.path.exists(atlas_out):
            os.makedirs(atlas_out)

        masker = NiftiLabelsMasker(
            labels_img=atlas_file,
            standardize=True,
            detrend=True
        )

        for file in sorted(fmri_files):

            start_time = time.time()

            base = file.replace(".nii.gz", "")
            out_path = os.path.join(atlas_out, base + ".npz")

            if os.path.exists(out_path):
                continue

            filepath = os.path.join(INPUT_DIR, file)

            try:
                img = nib.load(filepath)
                ts = masker.fit_transform(img)

                fc = compute_fc(ts)

                np.savez(
                    out_path,
                    fc=fc,
                    atlas=atlas_name,
                    n_rois=n_rois
                )

                elapsed = time.time() - start_time

                print("Saved {} | {} ROIs | {:.2f}s".format(
                    base, n_rois, elapsed
                ))

            except Exception as e:
                print("ERROR processing {} : {}".format(file, str(e)))

    print("\nAll FC matrices generated.")


# -----------------------------
if __name__ == "__main__":
    generate_fc()

