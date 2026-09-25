"""Publication figures, rendered from result files at 300 dpi.

The reviewer's complaint about the previous submission was specific: a
four-panel composite with unreadable labels. Every figure here is rendered as a
separate full-width file, with axis labels, explicit units, confidence bands
where an interval exists, and no reliance on colour alone to distinguish
series.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Colour-blind safe qualitative palette (Okabe-Ito), paired with distinct
# line styles and markers so the figures survive greyscale printing.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#F0E442", "#000000"]
STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)), "-"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

DPI = 300


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def fig_training_curves(results: list[dict], out: Path, method: str = "fomaml") -> None:
    """Meta-training and meta-validation loss, and validation accuracy.

    Rendered as two separate figures rather than a shared panel, because the
    two quantities have different units and the previous composite made both
    unreadable.
    """
    plt = _setup()
    runs = [r for r in results
            if r["config"]["algorithm"]["name"].lower() == method and r.get("history")]
    if not runs:
        log.warning("no training history for %s; skipping training curves", method)
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, r in enumerate(runs):
        h = r["history"]
        seed = r["config"]["seed"]
        ax.plot(h["steps"], h["train_loss"], color=PALETTE[0],
                linestyle=STYLES[i % len(STYLES)], alpha=0.85,
                label=f"meta-train (seed {seed})")
        ax.plot(h["steps"], h["val_loss"], color=PALETTE[1],
                linestyle=STYLES[i % len(STYLES)], alpha=0.85,
                label=f"meta-validation (seed {seed})")
    ax.set_xlabel("Meta-training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Meta-training and meta-validation loss")
    ax.legend(ncol=2, frameon=False)
    fig.savefig(out / "fig_loss.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, r in enumerate(runs):
        h = r["history"]
        n_way = r["config"]["episodes"]["n_way"]
        ax.plot(h["steps"], h["val_accuracy"], color=PALETTE[i % len(PALETTE)],
                linestyle=STYLES[i % len(STYLES)], marker=MARKERS[i % len(MARKERS)],
                markersize=4, label=f"seed {r['config']['seed']}")
        if h.get("val_accuracy_lo"):
            ax.fill_between(h["steps"], h["val_accuracy_lo"], h["val_accuracy_hi"],
                            color=PALETTE[i % len(PALETTE)], alpha=0.15)
    ax.axhline(1.0 / n_way, color="grey", linestyle=":", linewidth=1.2)
    ax.annotate(f"chance ({100 / n_way:.0f}%)", xy=(0.01, 1.0 / n_way),
                xycoords=("axes fraction", "data"), va="bottom", fontsize=9,
                color="grey")
    ax.set_xlabel("Meta-training step")
    ax.set_ylabel("Meta-validation accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"Meta-validation accuracy ({n_way}-way, shaded: 95% CI)")
    ax.legend(frameon=False)
    fig.savefig(out / "fig_val_accuracy.png")
    plt.close(fig)


def fig_adaptation_curve(summary: dict, out: Path) -> None:
    """Accuracy versus number of inner-loop gradient steps."""
    curves = summary.get("adaptation_curves") or {}
    if not curves:
        return
    plt = _setup()
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (method, curve) in enumerate(sorted(curves.items())):
        ax.plot(curve["steps"], curve["accuracy"],
                color=PALETTE[i % len(PALETTE)], linestyle=STYLES[i % len(STYLES)],
                marker=MARKERS[i % len(MARKERS)], label=method)
    ax.set_xlabel("Inner-loop gradient steps applied at test time")
    ax.set_ylabel("Meta-test accuracy")
    ax.set_title("Adaptation dynamics")
    ax.legend(frameon=False)
    # Step 0 is pre-adaptation: the output head has not yet been told which
    # logit corresponds to which episode-local class, so chance there is
    # structural and not a failure.
    ax.axvspan(-0.2, 0.2, color="grey", alpha=0.12)
    ax.annotate("pre-adaptation", xy=(0, ax.get_ylim()[0]), xytext=(0.3, 0.02),
                textcoords=("data", "axes fraction"), fontsize=9, color="grey")
    fig.savefig(out / "fig_adaptation_curve.png")
    plt.close(fig)


def fig_baselines(summary: dict, out: Path) -> None:
    """Horizontal bar chart of meta-test accuracy with confidence intervals."""
    table = summary.get("baselines") or {}
    if not table:
        return
    from .analyse import METHOD_ORDER

    ordered = [m for m in METHOD_ORDER if m in table]
    ordered += [m for m in table if m not in ordered]
    if not ordered:
        return

    plt = _setup()
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(ordered) + 2))
    labels, means, errs = [], [], []
    for method in ordered:
        row = table[method]["across_seeds"]
        labels.append(table[method]["label"])
        means.append(row["mean"] * 100)
        errs.append([(row["mean"] - row["lo"]) * 100, (row["hi"] - row["mean"]) * 100])
    y = np.arange(len(labels))
    ax.barh(y, means, xerr=np.array(errs).T, color=PALETTE[0], alpha=0.8,
            error_kw={"ecolor": "black", "capsize": 3, "lw": 1})
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Meta-test accuracy (%), 95% CI across seeds")
    ax.set_title("Baseline comparison under an identical episodic protocol")
    for i, m in enumerate(means):
        ax.text(m + 1.5, i, f"{m:.1f}", va="center", fontsize=9)
    fig.savefig(out / "fig_baselines.png")
    plt.close(fig)


def fig_ablations(summary: dict, out: Path) -> None:
    """One panel per ablation factor, each with confidence intervals."""
    ablations = summary.get("ablations") or {}
    if not ablations:
        return
    plt = _setup()
    for factor, levels in ablations.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        names = list(levels)
        means = [levels[n]["mean"] * 100 for n in names]
        errs = [[(levels[n]["mean"] - levels[n]["lo"]) * 100 for n in names],
                [(levels[n]["hi"] - levels[n]["mean"]) * 100 for n in names]]
        x = np.arange(len(names))
        ax.errorbar(x, means, yerr=errs, color=PALETTE[0], marker="o",
                    capsize=4, linestyle="-")
        ax.set_xticks(x, names)
        ax.set_xlabel(factor.replace("_", " "))
        ax.set_ylabel("Meta-test accuracy (%)")
        ax.set_title(f"Ablation: {factor.replace('_', ' ')}")
        fig.savefig(out / f"fig_ablation_{factor}.png")
        plt.close(fig)


def fig_protocol_b(summary: dict, out: Path) -> None:
    """Per-family binary accuracy and F1 with intervals and value labels."""
    table = summary.get("protocol_b") or {}
    if not table:
        return
    plt = _setup()
    families = list(table)
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.38
    x = np.arange(len(families))
    for j, metric in enumerate(("accuracy", "f1")):
        means = [table[f][metric]["mean"] for f in families]
        errs = [[table[f][metric]["mean"] - table[f][metric]["lo"] for f in families],
                [table[f][metric]["hi"] - table[f][metric]["mean"] for f in families]]
        bars = ax.bar(x + (j - 0.5) * width, means, width, yerr=errs,
                      color=PALETTE[j], alpha=0.85, capsize=3,
                      label=metric.replace("_", " ").title())
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, m + 0.02, f"{m:.3f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x, families, rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score (benign vs. family)")
    ax.set_title("Protocol B: per-family binary performance after adaptation")
    ax.legend(frameon=False)
    fig.savefig(out / "fig_protocol_b.png")
    plt.close(fig)


def fig_attention(model, X: np.ndarray, feature_names: list[str], out: Path,
                  top_k: int = 20) -> None:
    """Mean [CLS] attention over features, as an interpretability aid.

    Presented as attention mass, not as feature importance. Attention weights
    are not explanations (Jain and Wallace, 2019), and the figure caption in
    the manuscript should say so; SHAP is the attribution method of record.
    """
    import torch

    plt = _setup()
    with torch.no_grad():
        maps = model.attention_maps(torch.as_tensor(X))
    # Average over samples and blocks, then read the [CLS] row (token 0).
    stacked = torch.stack([m.mean(0) for m in maps]).mean(0)
    cls_attention = stacked[0, 1:].cpu().numpy()

    order = np.argsort(cls_attention)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(7, 0.32 * top_k + 1.5))
    ax.barh(np.arange(len(order)), cls_attention[order], color=PALETTE[0], alpha=0.85)
    ax.set_yticks(np.arange(len(order)),
                  [feature_names[i] for i in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean [CLS] attention mass")
    ax.set_title(f"Features most attended by the [CLS] token (top {top_k})")
    fig.savefig(out / "fig_attention.png")
    plt.close(fig)


def render_all(results: list[dict], summary: dict, out: Path) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig_training_curves(results, out)
    fig_adaptation_curve(summary, out)
    fig_baselines(summary, out)
    fig_ablations(summary, out)
    fig_protocol_b(summary, out)
    log.info("figures written to %s", out)
