#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_runs.py
=============
Combine per-fold results from multiple run folders into one CSV so the full grid
can be summarized together (e.g. the 22-cell main run + the Schaefer400 gap-fill).

Concatenates each folder's all_results.csv, de-duplicates on
(mode, atlas, fc_version, fold_or_site), and on conflict keeps the row from the
folder listed LATER on the command line (put the newer/gap-fill folder last).

Usage:
    python3 merge_runs.py results/run_20260620_073443 results/run_20260629_163108 \
            -o results/merged/all_results.csv
    python3 summarize_results.py results/merged
"""

import os
import sys
import argparse
import pandas as pd

KEY = ["mode", "atlas", "fc_version", "fold_or_site"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+",
                    help="run folders (each with all_results.csv); later wins on conflict")
    ap.add_argument("-o", "--out", default="results/merged/all_results.csv")
    args = ap.parse_args()

    frames = []
    for i, folder in enumerate(args.folders):
        path = os.path.join(folder, "all_results.csv")
        if not os.path.exists(path):
            print(f"[warn] no all_results.csv in {folder} -- skipping")
            continue
        df = pd.read_csv(path)
        df["_src_order"] = i
        df["_src"] = os.path.basename(folder.rstrip("/"))
        frames.append(df)
        print(f"  loaded {len(df):4d} rows from {folder}")

    if not frames:
        sys.exit("No all_results.csv found in any folder.")

    allrows = pd.concat(frames, ignore_index=True)
    for k in KEY:
        if k not in allrows.columns:
            sys.exit(f"Expected column '{k}' missing -- are these new-runner CSVs?")

    # normalize key dtypes so 1 (int) and '1' (str) fold ids match
    for k in KEY:
        allrows[k] = allrows[k].astype(str).str.strip()

    before = len(allrows)
    # stable sort by preference, keep the LAST (later folder) per key group
    allrows = allrows.sort_values("_src_order", kind="mergesort")
    merged = allrows.drop_duplicates(subset=KEY, keep="last").copy()
    after = len(merged)

    merged = merged.drop(columns=["_src_order"])
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    merged.to_csv(args.out, index=False)

    print(f"\nMerged {before} -> {after} rows (removed {before - after} duplicates).")
    print("Wrote:", args.out)

    print("\nCells present (mode | atlas | version | n_rows | source):")
    grp = (merged.groupby(["mode", "atlas", "fc_version"])
                 .agg(n=("fold_or_site", "size"), src=("_src", "first"))
                 .reset_index())
    atlas_rank = {"AAL": 0, "Schaefer100": 1, "Schaefer200": 2, "Schaefer400": 3}
    mode_rank = {"pooled_cv": 0, "loso": 1, "leave_one_site_out": 1,
                 "within_site": 2, "within_site_cv": 2}
    grp["_a"] = grp["atlas"].map(lambda x: atlas_rank.get(x, 9))
    grp["_m"] = grp["mode"].map(lambda x: mode_rank.get(x, 9))
    grp = grp.sort_values(["_m", "_a", "fc_version"])
    for _, r in grp.iterrows():
        print(f"  {r['mode']:<18} {r['atlas']:<12} {r['fc_version']:<3} "
              f"n={r['n']:<3} [{r['src']}]")


if __name__ == "__main__":
    main()
