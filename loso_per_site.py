#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loso_per_site.py
================
Read the per-site detail from loso_*.json files and report, per atlas/version:
  - each held-out site's AUC and n (sorted worst -> best)
  - which sites are too small to trust (n < MIN_TRUST)
  - the plain mean AUC vs the SIZE-WEIGHTED mean AUC (so a 5-subject site does
    not count as much as NYU ~180)

The per-fold all_results.csv only stores the LOSO average; the site-level
numbers live in the JSON ("per_site": [ {test_site, n_eval, auc, ...}, ... ]).

Usage:
    python3 loso_per_site.py results/merged
    python3 loso_per_site.py results/run_20260620_073443 results/run_20260629_163108
"""

import os
import sys
import glob
import json
import numpy as np

MIN_TRUST = 20     # sites with fewer test subjects than this are flagged unreliable


def load_loso_files(folders):
    files = {}
    for folder in folders:
        for f in glob.glob(os.path.join(folder, "loso_*.json")):
            files[os.path.basename(f)] = f      # later folder overrides same name
    return [files[k] for k in sorted(files)]


def main():
    folders = sys.argv[1:] or ["results/merged"]
    files = load_loso_files(folders)
    if not files:
        sys.exit("No loso_*.json found in: {}".format(folders))

    for jf in files:
        try:
            d = json.load(open(jf))
        except Exception as e:
            print("[skip] {}: {}".format(jf, e))
            continue

        atlas = d.get("atlas", "?")
        ver = d.get("fc_version", "?")
        per_site = d.get("per_site", [])
        if not per_site:
            continue

        rows = []
        for s in per_site:
            site = s.get("test_site", "?")
            n = int(s.get("n_eval", 0))
            auc = float(s.get("auc", float("nan")))
            rec = float(s.get("recall", float("nan")))
            spec = float(s.get("specificity", float("nan")))
            rows.append((site, n, auc, rec, spec))

        rows.sort(key=lambda r: r[2])          # by AUC, worst first

        aucs = np.array([r[2] for r in rows], dtype=float)
        ns = np.array([r[1] for r in rows], dtype=float)
        plain = float(np.nanmean(aucs))
        weighted = float(np.nansum(aucs * ns) / np.nansum(ns))

        # weighted over trustworthy sites only
        big = ns >= MIN_TRUST
        if big.sum() > 0:
            w_big = float(np.nansum(aucs[big] * ns[big]) / np.nansum(ns[big]))
        else:
            w_big = float("nan")

        print("\n===== LOSO  {} | {} =====".format(atlas, ver))
        print("  {:<10} {:>4}  {:>6} {:>6} {:>6}".format("site", "n", "AUC", "rec", "spec"))
        for site, n, auc, rec, spec in rows:
            flag = "  <-- small (n<{})".format(MIN_TRUST) if n < MIN_TRUST else ""
            print("  {:<10} {:>4}  {:>6.3f} {:>6.3f} {:>6.3f}{}".format(
                site, n, auc, rec, spec, flag))
        print("  ------")
        print("  plain mean AUC        : {:.3f}".format(plain))
        print("  size-weighted AUC     : {:.3f}".format(weighted))
        print("  weighted (n>={} only) : {:.3f}".format(MIN_TRUST, w_big))
        print("  sites total={}  small(<{})={}".format(
            len(rows), MIN_TRUST, int((ns < MIN_TRUST).sum())))


if __name__ == "__main__":
    main()
