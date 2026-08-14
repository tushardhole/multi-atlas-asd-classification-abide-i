#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_fc_all_v2.py
=====================
Compute robust FC matrices for multiple atlases with Ledoit-Wolf shrinkage.
Saves .npz files with fc, atlas, and ROI count.

Requirements:
    pip install numpy nibabel nilearn scikit-learn tqdm
"""

import os
import time
import numpy as np
import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import LedoitWolf
from tqdm import tqdm

# -----------------------------
# CONFIG  (edit these to match your local data location)
# -----------------------------
INPUT_DIR = os.environ.get("ABIDE_DATA_DIR", "data")       # folder with .nii.gz files
OUTPUT_DIR = os.path.join(INPUT_DIR, "fc_matrices_from_pre_proc")  # output: <atlas>/<subject>_v2.npz

SMOOTHING_FWHM = 6
LOW_PASS = 0.1
HIGH_PASS = 0.01
TR = 2.0

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
# Compute FC with shrinkage
# -----------------------------
def compute_fc(time_series, kind='correlation'):
    """
    Compute Fisher-z-transformed FC using Ledoit-Wolf shrinkage.
    kind: 'correlation' or 'partial'
    """
    try:
        conn = ConnectivityMeasure(
            kind=kind,
            cov_estimator=LedoitWolf()
        )
        fc = conn.fit_transform([time_series])[0]
        # Fisher z-transform
        fc = np.clip(fc, -0.999999, 0.999999)
        fc = np.arctanh(fc)
        np.fill_diagonal(fc, 0.0)
        return fc.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"FC computation failed: {e}")

# -----------------------------
# Main Pipeline
# -----------------------------
def generate_fc():

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    atlases = load_atlases()

    fmri_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.endswith(".nii.gz")
    ]

    print(f"\nFound {len(fmri_files)} fMRI scans in {INPUT_DIR}")

    for atlas_name, (atlas_file, n_rois) in atlases.items():

        print("\n==============================")
        print(f"Processing Atlas: {atlas_name} | ROIs: {n_rois}")
        print("==============================")

        atlas_out = os.path.join(OUTPUT_DIR, atlas_name)
        if not os.path.exists(atlas_out):
            os.makedirs(atlas_out)

        masker = NiftiLabelsMasker(
            labels_img=atlas_file,
            standardize=True,
            detrend=True,
            smoothing_fwhm=SMOOTHING_FWHM,
            low_pass=LOW_PASS,
            high_pass=HIGH_PASS,
            t_r=TR
        )

        for file in tqdm(sorted(fmri_files), desc=f"{atlas_name} FC"):

            start_time = time.time()
            base = file.replace(".nii.gz", "")
            out_path = os.path.join(atlas_out, base + ".npz")

            # Skip if already exists
            if os.path.exists(out_path):
                continue

            filepath = os.path.join(INPUT_DIR, file)
            try:
                img = nib.load(filepath)
                ts = masker.fit_transform(img)

                fc = compute_fc(ts, kind='correlation')

                np.savez(
                    out_path,
                    fc=fc,
                    atlas=atlas_name,
                    n_rois=n_rois
                )

                elapsed = time.time() - start_time
                print(f"Saved {base} | {n_rois} ROIs | {elapsed:.2f}s")

            except Exception as e:
                print(f"ERROR {file} -> {e}")

    print("\nAll FC matrices generated successfully.")

# -----------------------------
if __name__ == "__main__":
    start_total = time.time()
    generate_fc()
    print(f"\nTotal elapsed time: {time.time() - start_total:.2f}s")

