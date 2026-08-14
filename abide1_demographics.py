#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
abide1_demographics.py
======================
Generate demographics summary and distribution figures for the ABIDE I
Preprocessed cohort (replacing old Figures 4 and 5 which showed ABIDE II).

Reads Phenotypic_V1_0b_preprocessed1.csv and filters to the 884 subjects
that have func_preproc FC files (by matching subject IDs from the FC folder).

Produces:
  - Console summary: total N, ASD/Control counts, age stats, gender breakdown,
    site counts
  - figures/fig_age_distribution.png/.pdf   (replaces old Figure 4)
  - figures/fig_gender_distribution.png/.pdf (replaces old Figure 5)
  - figures/fig_site_distribution.png/.pdf   (new — shows per-site sample sizes)

Usage:
    python3 abide1_demographics.py

    # or with explicit paths:
    python3 abide1_demographics.py --pheno path/to/Phenotypic_V1_0b_preprocessed1.csv \
                                   --fc_dir path/to/fc_matrices_from_pre_proc/AAL
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "figures"


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT_DIR}/{name}.png/.pdf")


def get_fc_subjects(fc_dir):
    """Get the set of subject IDs that have FC files (= our actual cohort)."""
    if fc_dir is None or not os.path.isdir(fc_dir):
        return None
    sids = set()
    for f in os.listdir(fc_dir):
        if f.endswith(".npz"):
            m = re.search(r"\d{7}", f)
            if m:
                sids.add(int(m.group(0)))
    return sids if sids else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pheno", default=None,
                    help="path to Phenotypic_V1_0b_preprocessed1.csv")
    ap.add_argument("--fc_dir", default=None,
                    help="path to one atlas FC folder (e.g. fc_matrices_from_pre_proc/AAL) "
                         "to filter to actual cohort; if omitted, uses all subjects in phenotype")
    args = ap.parse_args()

    # auto-find phenotype CSV
    pheno_path = args.pheno
    if pheno_path is None:
        for cand in [
            "Phenotypic_V1_0b_preprocessed1.csv",
            "../testdata/Phenotypic_V1_0b_preprocessed1.csv",
            os.path.expanduser("~/asd_project_local/testdata/Phenotypic_V1_0b_preprocessed1.csv"),
            os.path.expanduser("~/Downloads/testdata/Phenotypic_V1_0b_preprocessed1.csv"),
        ]:
            if os.path.exists(cand):
                pheno_path = cand
                break
    if pheno_path is None or not os.path.exists(pheno_path):
        raise SystemExit("Cannot find phenotype CSV. Use --pheno <path>.")

    df = pd.read_csv(pheno_path)
    subj_col = "subject" if "subject" in df.columns else "SUB_ID"

    # filter to actual cohort if FC dir provided
    fc_sids = get_fc_subjects(args.fc_dir)
    if fc_sids is not None:
        df = df[df[subj_col].astype(int).isin(fc_sids)].copy()
        print(f"Filtered to {len(df)} subjects with FC files in {args.fc_dir}\n")
    else:
        # fallback: filter to subjects that would have func_preproc
        # (those with a FILE_ID, which the PCP release uses)
        if "FILE_ID" in df.columns:
            df = df[df["FILE_ID"].notna() & (df["FILE_ID"] != "no_filename")].copy()
        print(f"Using {len(df)} subjects from phenotype CSV\n")

    # map labels
    df["group"] = df["DX_GROUP"].map({1: "ASD", 2: "Control"})
    df["sex_label"] = df["SEX"].map({1: "Male", 2: "Female"})

    # ============ CONSOLE SUMMARY ============
    n = len(df)
    n_asd = (df["group"] == "ASD").sum()
    n_ctrl = (df["group"] == "Control").sum()

    print("=" * 60)
    print("ABIDE I Preprocessed — Cohort Demographics")
    print("=" * 60)
    print(f"Total subjects:  {n}")
    print(f"  ASD:           {n_asd} ({100*n_asd/n:.1f}%)")
    print(f"  Control:       {n_ctrl} ({100*n_ctrl/n:.1f}%)")

    if "AGE_AT_SCAN" in df.columns:
        age = pd.to_numeric(df["AGE_AT_SCAN"], errors="coerce").dropna()
        print(f"\nAge at scan:")
        print(f"  range:         {age.min():.1f} – {age.max():.1f} years")
        print(f"  mean ± std:    {age.mean():.1f} ± {age.std():.1f}")
        print(f"  median:        {age.median():.1f}")
        for grp in ["ASD", "Control"]:
            a = pd.to_numeric(df.loc[df["group"] == grp, "AGE_AT_SCAN"], errors="coerce").dropna()
            print(f"  {grp:<8}:      {a.mean():.1f} ± {a.std():.1f}  (range {a.min():.1f}–{a.max():.1f})")

    if "SEX" in df.columns:
        print(f"\nGender:")
        for grp in ["ASD", "Control"]:
            g = df[df["group"] == grp]
            nm = (g["SEX"] == 1).sum()
            nf = (g["SEX"] == 2).sum()
            print(f"  {grp:<8}:      Male {nm} ({100*nm/len(g):.1f}%)  Female {nf} ({100*nf/len(g):.1f}%)")

    if "SITE_ID" in df.columns:
        print(f"\nSites: {df['SITE_ID'].nunique()}")
        sc = df.groupby("SITE_ID").agg(
            total=("group", "size"),
            ASD=("group", lambda x: (x == "ASD").sum()),
            Control=("group", lambda x: (x == "Control").sum()),
        ).sort_values("total", ascending=False)
        print(sc.to_string())

    # ============ FIGURES ============

    # --- Figure: Age Distribution by Group ---
    if "AGE_AT_SCAN" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bins = np.arange(0, df["AGE_AT_SCAN"].max() + 5, 5)
        for grp, color, alpha in [("ASD", "#d1495b", 0.6), ("Control", "#30638e", 0.5)]:
            ages = pd.to_numeric(df.loc[df["group"] == grp, "AGE_AT_SCAN"], errors="coerce").dropna()
            ax.hist(ages, bins=bins, alpha=alpha, color=color, label=grp, edgecolor="white")
        ax.set_xlabel("Age at Scan (years)")
        ax.set_ylabel("Number of Subjects")
        ax.set_title("Age Distribution by Group")
        ax.legend()
        _save(fig, "fig_age_distribution")

    # --- Figure: Gender Distribution by Group ---
    if "SEX" in df.columns:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        groups = ["ASD", "Control"]
        males = [(df[(df["group"] == g) & (df["SEX"] == 1)].shape[0]) for g in groups]
        females = [(df[(df["group"] == g) & (df["SEX"] == 2)].shape[0]) for g in groups]
        x = np.arange(len(groups))
        w = 0.35
        ax.bar(x - w/2, males, w, label="Male", color="#30638e", alpha=0.8)
        ax.bar(x + w/2, females, w, label="Female", color="#d1495b", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylabel("Count")
        ax.set_title("Gender Distribution by Group")
        ax.legend()
        for i, (m, f) in enumerate(zip(males, females)):
            ax.text(i - w/2, m + 3, str(m), ha="center", fontsize=9)
            ax.text(i + w/2, f + 3, str(f), ha="center", fontsize=9)
        _save(fig, "fig_gender_distribution")

    # --- Figure: Site Distribution ---
    if "SITE_ID" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sc = df.groupby(["SITE_ID", "group"]).size().unstack(fill_value=0)
        sc = sc.reindex(columns=["ASD", "Control"], fill_value=0)
        sc = sc.sort_values("ASD", ascending=False)
        sc.plot(kind="bar", ax=ax, color=["#d1495b", "#30638e"], alpha=0.8,
                edgecolor="white")
        ax.set_xlabel("Site")
        ax.set_ylabel("Number of Subjects")
        ax.set_title("Subject Distribution by Site and Group")
        ax.legend(title="Group")
        plt.xticks(rotation=45, ha="right")
        _save(fig, "fig_site_distribution")

    print(f"\nDone. Figures in {OUT_DIR}/")
    print("\nFor the paper dataset section, use:")
    print(f"  Total: {n} subjects ({n_asd} ASD, {n_ctrl} Control)")
    if "AGE_AT_SCAN" in df.columns:
        print(f"  Age: {age.min():.0f}–{age.max():.0f} years (mean {age.mean():.1f} ± {age.std():.1f})")
    print(f"  Sites: {df['SITE_ID'].nunique()}")


if __name__ == "__main__":
    main()
