#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_loso_per_site.py
=====================
Generate a per-site LOSO AUC bar chart for the paper.

Reads loso_*.json (which contains per_site: [{test_site, n_eval, auc, ...}])
and plots a horizontal bar chart sorted by AUC, colored by site size,
with the weighted-mean AUC as a vertical reference line.

Usage:
    python3 plot_loso_per_site.py

    # or with explicit paths:
    python3 plot_loso_per_site.py \
        --json results/run_20260620_073443/loso_Schaefer200_v2.json \
        --atlas "Schaefer-200" --ver "V2"

    # if your LOSO JSONs are split across two run folders (main + gap-fill):
    python3 plot_loso_per_site.py \
        --json results/run_20260620_073443/loso_Schaefer200_v2.json
"""

import os
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "figures"

# Default: Schaefer200 V2 (the headline atlas) from the main grid run
DEFAULT_JSONS = [
    "results/run_20260620_073443/loso_Schaefer200_v2.json",
]


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT_DIR}/{name}.png/.pdf")


def find_json(paths):
    """Try each candidate path and return the first that exists."""
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None,
                    help="path to a loso_*.json file")
    ap.add_argument("--atlas", default="Schaefer-200",
                    help="atlas name for the title")
    ap.add_argument("--ver", default="V2",
                    help="FC version for the title")
    args = ap.parse_args()

    # find the JSON
    if args.json and os.path.exists(args.json):
        jpath = args.json
    else:
        jpath = find_json(DEFAULT_JSONS)
    if jpath is None:
        raise SystemExit(
            "Cannot find LOSO JSON. Use --json <path>.\n"
            "Expected at: " + str(DEFAULT_JSONS))

    print(f"Reading: {jpath}")
    d = json.load(open(jpath))
    atlas = d.get("atlas", args.atlas)
    ver = d.get("fc_version", args.ver)
    per_site = d.get("per_site", [])

    if not per_site:
        raise SystemExit("No per_site data in the JSON.")

    # extract site, n, auc
    sites, ns, aucs = [], [], []
    for s in per_site:
        sites.append(s.get("test_site", "?"))
        ns.append(int(s.get("n_eval", 0)))
        aucs.append(float(s.get("auc", 0)))

    # sort by AUC ascending (worst at top for horizontal bars)
    order = np.argsort(aucs)
    sites = [sites[i] for i in order]
    ns = [ns[i] for i in order]
    aucs = [aucs[i] for i in order]

    # weighted mean AUC
    ns_arr = np.array(ns, dtype=float)
    aucs_arr = np.array(aucs, dtype=float)
    weighted_auc = float(np.sum(aucs_arr * ns_arr) / np.sum(ns_arr))

    # color by site size (larger = darker)
    norm = plt.Normalize(vmin=min(ns), vmax=max(ns))
    cmap = plt.colormaps["YlOrRd"]
    colors = [cmap(norm(n)) for n in ns]

    # --- plot ---
    fig, ax = plt.subplots(figsize=(7, 6))

    y_pos = np.arange(len(sites))
    bars = ax.barh(y_pos, aucs, color=colors, edgecolor="white", height=0.7)

    # site labels with (n=XX)
    labels = [f"{s} (n={n})" for s, n in zip(sites, ns)]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)

    # AUC value on each bar
    for i, (a, n) in enumerate(zip(aucs, ns)):
        xoff = 0.005
        ha = "left"
        color = "black"
        if a > 0.5:
            ax.text(a - xoff, i, f"{a:.3f}", va="center", ha="right",
                    fontsize=8, color="white", fontweight="bold")
        else:
            ax.text(a + xoff, i, f"{a:.3f}", va="center", ha="left",
                    fontsize=8, color="black")

    # weighted mean reference line
    ax.axvline(weighted_auc, color="#1a3c5e", linestyle="--", linewidth=1.5,
               label=f"Weighted mean AUC = {weighted_auc:.3f}")

    # mark small sites
    for i, n in enumerate(ns):
        if n < 20:
            ax.text(0.02, i, "⚠ small", va="center", fontsize=7,
                    color="gray", style="italic")

    ax.set_xlabel("AUC", fontsize=11)
    ax.set_title(f"LOSO Per-Site AUC — {atlas} ({ver})", fontsize=12)
    ax.set_xlim(0.4, 0.95)
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()  # best at top after inversion

    # colorbar for site size
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30, shrink=0.6)
    cbar.set_label("Site size (n)", fontsize=9)

    _save(fig, f"loso_per_site_{atlas}_{ver}")
    print(f"\nWeighted mean AUC: {weighted_auc:.3f}")
    print(f"Sites: {len(sites)} (small n<20: {sum(1 for n in ns if n < 20)})")


if __name__ == "__main__":
    main()
