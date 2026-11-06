"""Scatter plots of SimCSE and judge scores against the gold score for the new dataset.

  python src/make_figures.py        # writes results/figures/scatter_own.pdf and .png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from analysis import load_set


def main():
    os.makedirs("results/figures", exist_ok=True)
    df = load_set("own")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), sharex=True)
    for ax, col, title in [(axes[0], "simcse_sim", "SimCSE (cosine)"), (axes[1], "llm_score", "LLM judge (0 to 4)")]:
        d = df[[col, "gold", "expected_band"]].dropna() if "expected_band" in df else df[[col, "gold"]].dropna()
        rho = spearmanr(d["gold"], d[col])[0]
        ax.scatter(d["gold"], d[col], s=14, alpha=0.7, color="#2b6cb0", edgecolors="none")
        ax.set_title(f"{title}: Spearman {rho:.2f}", fontsize=9)
        ax.set_xlabel("Gold score (mean of 3 raters)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Model score", fontsize=8)
    fig.tight_layout()
    fig.savefig("results/figures/scatter_own.pdf")
    fig.savefig("results/figures/scatter_own.png", dpi=200)
    print("wrote results/figures/scatter_own.pdf")


if __name__ == "__main__":
    main()
