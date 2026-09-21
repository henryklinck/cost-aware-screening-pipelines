"""Figure generation for the Phase-1 simulation (spec section 8, figures 1-5).

Pure plotting: every function takes already-computed DataFrames and writes a PNG. No
data is fabricated here -- figures only visualise what the simulation produced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def fig1_recall_vs_cost(
    constrained: pd.DataFrame,
    frontier: pd.DataFrame,
    out: Path,
    single_model: pd.DataFrame | None = None,
) -> Path:
    """Figure 1 -- recall vs computational cost; Pareto frontier highlighted."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        constrained["total_cost"], constrained["recall"],
        s=10, alpha=0.25, color="#9aa0a6", label="all configurations",
    )
    ax.plot(
        frontier["total_cost"], frontier["recall"],
        "-o", color="#1a73e8", markersize=5, linewidth=2, label="Pareto frontier",
    )
    if single_model is not None and not single_model.empty:
        ax.scatter(
            single_model["total_cost"], single_model["recall"],
            s=60, marker="D", edgecolor="#d93025", facecolor="none",
            linewidth=1.5, label="best single-model (per model)",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Computational cost (compute units, log scale)")
    ax.set_ylabel("Recall of true hits (final shortlist)")
    ax.set_title("Figure 1 — Hit recall vs computational cost")
    ax.legend(loc="lower right", framealpha=0.9)
    return _save(fig, out)


def fig2_cost_per_hit(frontier: pd.DataFrame, out: Path) -> Path:
    """Figure 2 -- cost per recovered hit along the frontier."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        frontier["recall"], frontier["cost_per_hit"],
        "-o", color="#188038", markersize=5, linewidth=2,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Recall of true hits")
    ax.set_ylabel("Cost per recovered hit (compute units / hit, log scale)")
    ax.set_title("Figure 2 — Cost per recovered hit (Pareto frontier)")
    return _save(fig, out)


def fig3_ordering(subset_df: pd.DataFrame, out: Path, model_set_label: str) -> Path:
    """Figure 3 -- pipeline ordering vs performance for a fixed model set.

    Points are coloured by the *first* model in the pipeline.  Cheap-first orderings
    cluster at low cost for comparable recall; expensive-first orderings are pushed
    right -- ordering, not just model choice, drives the trade-off.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    first_models = sorted(subset_df["first_model"].unique())
    cmap = plt.get_cmap("viridis", len(first_models))
    for i, fm in enumerate(first_models):
        d = subset_df[subset_df["first_model"] == fm]
        ax.scatter(
            d["total_cost"], d["recall"], s=18, alpha=0.7,
            color=cmap(i), label=f"starts with {fm}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Computational cost (log scale)")
    ax.set_ylabel("Recall of true hits")
    ax.set_title(f"Figure 3 — Ordering vs performance (model set {model_set_label})")
    ax.legend(loc="lower right", framealpha=0.9)
    return _save(fig, out)


def fig4_prevalence(frontiers: Dict[float, pd.DataFrame], out: Path) -> Path:
    """Figure 4 -- effect of hit prevalence on the cost/recall frontier."""
    fig, ax = plt.subplots(figsize=(7, 5))
    keys = sorted(frontiers)
    cmap = plt.get_cmap("plasma", len(keys))
    for i, prev in enumerate(keys):
        f = frontiers[prev]
        ax.plot(f["total_cost"], f["recall"], "-o", markersize=4,
                color=cmap(i), label=f"prevalence {prev:g}")
    ax.set_xscale("log")
    ax.set_xlabel("Computational cost (log scale)")
    ax.set_ylabel("Recall of true hits")
    ax.set_title("Figure 4 — Effect of prevalence on the frontier")
    ax.legend(loc="lower right", framealpha=0.9)
    return _save(fig, out)


def fig5_correlation(frontiers: Dict[float, pd.DataFrame], out: Path) -> Path:
    """Figure 5 -- effect of model error correlation on the cost/recall frontier."""
    fig, ax = plt.subplots(figsize=(7, 5))
    keys = sorted(frontiers)
    cmap = plt.get_cmap("cividis", len(keys))
    for i, rho in enumerate(keys):
        f = frontiers[rho]
        ax.plot(f["total_cost"], f["recall"], "-o", markersize=4,
                color=cmap(i), label=f"error corr ρ={rho:g}")
    ax.set_xscale("log")
    ax.set_xlabel("Computational cost (log scale)")
    ax.set_ylabel("Recall of true hits")
    ax.set_title("Figure 5 — Effect of model error correlation on the frontier")
    ax.legend(loc="lower right", framealpha=0.9)
    return _save(fig, out)
