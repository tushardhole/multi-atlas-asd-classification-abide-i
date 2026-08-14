#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_results.py
====================
Read the runner's outputs and print compact, paste-ready summary tables.

Primary source: all_results.csv (one row per fold/site, written by the runner).
Fallback: the per-run *.json files (pooled_cv_*, loso_*, within_site_*).

Prints, per (mode, atlas, fc_version): mean +/- std for the key metrics, plus
train/inference time and model size. Small enough to paste into chat.

Usage:
    python3 summarize_results.py                      # auto-find results dir
    python3 summarize_results.py /path/to/results_dir
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd

METRICS = ["auc", "balanced_accuracy", "accuracy", "f1",
           "precision", "recall", "specificity"]
EXTRAS  = ["train_time_sec", "infer_ms_per_sample", "epochs_run", "n_params", "n_eval"]


def find_results_dir(arg):
    if arg and os.path.isdir(arg):
        return arg
    # common locations: ./results/*, ./outputs/*, cwd
    for pat in ["results/*", "outputs/*", "."]:
        for d in sorted(glob.glob(pat), reverse=True):
            if os.path.isdir(d) and (
                glob.glob(os.path.join(d, "all_results.csv")) or
                glob.glob(os.path.join(d, "*.json"))):
                return d
    return "."


def fmt(mean, std):
    return "{:.3f}+/-{:.3f}".format(mean, std)


def from_csv(path):
    df = pd.read_csv(path)
    group_cols = ["mode", "atlas", "fc_version"]
    rows = []
    for (mode, atlas, ver), g in df.groupby(group_cols):
        row = {"mode": mode, "atlas": atlas, "ver": ver, "n_folds": len(g)}
        for m in METRICS:
            if m in g:
                row[m] = fmt(g[m].mean(), g[m].std())
        for e in EXTRAS:
            if e in g:
                row[e] = round(float(g[e].mean()), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def from_json(results_dir):
    rows = []
    for jf in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            d = json.load(open(jf))
        except Exception:
            continue
        if "mean" not in d:
            continue
        mean, std = d["mean"], d.get("std", {})
        row = {"mode": d.get("mode", os.path.basename(jf)),
               "atlas": d.get("atlas", "?"),
               "ver": d.get("fc_version", "?")}
        for m in METRICS:
            if m in mean:
                row[m] = fmt(mean[m], std.get(m, 0.0))
        for e in EXTRAS:
            if e in mean:
                row[e] = round(float(mean[e]), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def order(df):
    atlas_rank = {"AAL": 0, "Schaefer100": 1, "Schaefer200": 2, "Schaefer400": 3}
    mode_rank = {"pooled_cv": 0, "leave_one_site_out": 1, "loso": 1, "within_site_cv": 2}
    df = df.copy()
    df["_a"] = df["atlas"].map(lambda x: atlas_rank.get(x, 9))
    df["_m"] = df["mode"].map(lambda x: mode_rank.get(x, 9))
    return df.sort_values(["_m", "_a", "ver"]).drop(columns=["_a", "_m"])


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    rdir = find_results_dir(arg)
    print("Results dir:", os.path.abspath(rdir))

    csv_path = os.path.join(rdir, "all_results.csv")
    if os.path.exists(csv_path):
        print("Source: all_results.csv\n")
        df = from_csv(csv_path)
    else:
        print("Source: per-run JSON files\n")
        df = from_json(rdir)

    if df.empty:
        print("No results found.")
        return

    df = order(df)

    for mode in ["pooled_cv", "leave_one_site_out", "loso", "within_site_cv"]:
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        print("\n========== {} ==========".format(mode.upper()))
        cols = ["atlas", "ver"] + [m for m in METRICS if m in sub.columns] \
               + [e for e in EXTRAS if e in sub.columns]
        print(sub[cols].to_string(index=False))

    # compact headline: best AUC per atlas in pooled_cv
    pc = df[df["mode"] == "pooled_cv"]
    if not pc.empty and "auc" in pc.columns:
        print("\n---------- HEADLINE (pooled_cv AUC) ----------")
        for _, r in pc.iterrows():
            print("  {:<12} {:<3} AUC {}".format(r["atlas"], r["ver"], r["auc"]))


if __name__ == "__main__":
    main()
