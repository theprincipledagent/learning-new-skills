#!/usr/bin/env python3
"""Plot LLM evaluator scores vs actual benchmark accuracy across epochs.

Usage:
    python plot_eval_vs_accuracy.py                # save to eval_vs_accuracy.png
    python plot_eval_vs_accuracy.py --show         # also display interactively
    python plot_eval_vs_accuracy.py --out fig.png  # custom output path
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path("data/epochs")
HISTORY_PATH = Path("data/history.json")

_DPI = 300

_color_palette = {
    "primary": "#0A58CA",    # Strong Blue
    "secondary": "#6A7C92",  # Slate Gray
    "accent": "#6F42C1",     # Digital Violet
    "background": "#F8F9FA", # Alabaster White
    "text": "#212529",       # Dark Charcoal
}


def load_epoch_metrics() -> list[dict]:
    """Load per-epoch metrics from metrics.json files."""
    metrics = []
    for epoch_dir in sorted(DATA_DIR.glob("epoch_*")):
        metrics_path = epoch_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            data = json.loads(metrics_path.read_text())
            metrics.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return sorted(metrics, key=lambda m: m.get("epoch", 0))


def plot_eval_vs_accuracy(metrics: list[dict], output_path: str):
    """Dual-axis plot: accuracy and avg eval score over epochs."""
    plt.style.use('seaborn-v0_8-whitegrid')

    epochs = [m["epoch"] for m in metrics]
    accuracies = [m["accuracy"] * 100 for m in metrics]
    avg_scores = [m["avg_overall_score"] for m in metrics]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.set_facecolor(_color_palette["background"])

    # Accuracy line (left axis)
    ax1.plot(epochs, accuracies, color=_color_palette["accent"], linewidth=2.5,
             marker="o", markersize=6, label="Benchmark Accuracy (%)")
    ax1.set_xlabel("Epoch", color=_color_palette["text"], fontsize=12)
    ax1.set_ylabel("Accuracy (%)", color=_color_palette["accent"], fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.set_facecolor(_color_palette["background"])
    ax1.tick_params(axis="y", colors=_color_palette["accent"])
    ax1.tick_params(axis="x", colors=_color_palette["text"])
    ax1.spines['bottom'].set_color(_color_palette["secondary"])
    ax1.spines['left'].set_color(_color_palette["secondary"])
    ax1.spines['right'].set_color(_color_palette["secondary"])
    ax1.spines['top'].set_color(_color_palette["secondary"])

    # Eval score line (right axis)
    ax2 = ax1.twinx()
    ax2.plot(epochs, avg_scores, color=_color_palette["primary"], linewidth=2.5,
             marker="s", markersize=6, linestyle="--", label="Avg Eval Score (/10)")
    ax2.set_ylabel("Avg Evaluator Score (/10)", color=_color_palette["primary"], fontsize=12)
    ax2.set_ylim(0, 10)
    ax2.tick_params(axis="y", colors=_color_palette["primary"])
    ax2.spines['right'].set_color(_color_palette["secondary"])

    # X-axis ticks
    ax1.set_xticks(epochs)

    # Title
    ax1.set_title("Evaluator Score vs Benchmark Accuracy",
                   color=_color_palette["text"], fontsize=16, weight="bold")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left",
                        fontsize=9, framealpha=0.9,
                        facecolor=_color_palette["background"],
                        edgecolor=_color_palette["secondary"])
    for text in legend.get_texts():
        text.set_color(_color_palette["text"])

    plt.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved to {output_path}")

    return fig


def plot_score_dimensions(metrics: list[dict], output_path: str):
    """Plot individual eval dimensions alongside accuracy."""
    plt.style.use('seaborn-v0_8-whitegrid')

    epochs = [m["epoch"] for m in metrics]
    accuracies = [m["accuracy"] * 100 for m in metrics]
    avg_scores_dict = [m.get("avg_scores", {}) for m in metrics]

    dimensions = ["helpfulness", "accuracy", "reasoning_quality",
                   "tool_selection", "knowledge_application"]
    dim_colors = [
        _color_palette["primary"],
        _color_palette["accent"],
        _color_palette["secondary"],
        "#E67E22",  # Orange
        "#27AE60",  # Green
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.set_facecolor(_color_palette["background"])

    # Top: accuracy
    ax1.plot(epochs, accuracies, color=_color_palette["accent"], linewidth=2.5,
             marker="o", markersize=6)
    ax1.set_ylabel("Accuracy (%)", color=_color_palette["text"], fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.set_title("Benchmark Accuracy Over Epochs",
                   color=_color_palette["text"], fontsize=16, weight="bold")
    ax1.set_facecolor(_color_palette["background"])
    ax1.tick_params(colors=_color_palette["text"])
    ax1.spines['bottom'].set_color(_color_palette["secondary"])
    ax1.spines['left'].set_color(_color_palette["secondary"])

    # Bottom: eval dimensions
    for dim, color in zip(dimensions, dim_colors):
        values = [s.get(dim, 0) for s in avg_scores_dict]
        ax2.plot(epochs, values, color=color, linewidth=2, marker="o",
                 markersize=4, label=dim.replace("_", " ").title())

    ax2.set_xlabel("Epoch", color=_color_palette["text"], fontsize=12)
    ax2.set_ylabel("Avg Score (/10)", color=_color_palette["text"], fontsize=12)
    ax2.set_ylim(0, 10)
    ax2.set_title("Evaluator Score Dimensions Over Epochs",
                   color=_color_palette["text"], fontsize=16, weight="bold")
    ax2.set_facecolor(_color_palette["background"])
    ax2.tick_params(colors=_color_palette["text"])
    ax2.spines['bottom'].set_color(_color_palette["secondary"])
    ax2.spines['left'].set_color(_color_palette["secondary"])
    ax2.set_xticks(epochs)

    legend = ax2.legend(fontsize=8, framealpha=0.9,
                        facecolor=_color_palette["background"],
                        edgecolor=_color_palette["secondary"])
    for text in legend.get_texts():
        text.set_color(_color_palette["text"])

    plt.tight_layout(pad=2.0)

    dim_path = output_path.rsplit(".", 1)
    dim_output = f"{dim_path[0]}_dimensions.{dim_path[1]}" if len(dim_path) == 2 else f"{output_path}_dimensions"
    fig.savefig(dim_output, dpi=_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved to {dim_output}")

    return fig


def load_per_question_data() -> list[dict]:
    """Load eval score, correctness, and task_id for every question across all epochs."""
    records = []

    for epoch_dir in sorted(DATA_DIR.glob("epoch_*")):
        eval_path = epoch_dir / "evaluations.json"
        if not eval_path.exists():
            continue
        try:
            evals = json.loads(eval_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for e in evals:
            rec = {
                "task_id": e.get("task_id", ""),
                "score": e.get("overall_score", 0),
                "is_correct": 1 if e.get("is_correct") else 0,
            }
            for dim, val in e.get("scores", {}).items():
                rec[f"dim_{dim}"] = val
            records.append(rec)

    return records


def _get_volatile_task_ids(records: list[dict]) -> set[str]:
    """Return task IDs that were correct at least once AND wrong at least once."""
    task_correct = {}
    task_wrong = {}
    for r in records:
        tid = r["task_id"]
        if r["is_correct"]:
            task_correct[tid] = True
        else:
            task_wrong[tid] = True
    return {tid for tid in task_correct if tid in task_wrong}


def _plot_correlation_scatter(scores, correct, title, subtitle_extra, output_file):
    """Shared scatter plot logic for correlation charts."""
    plt.style.use('seaborn-v0_8-whitegrid')

    scores = np.array(scores)
    correct = np.array(correct)

    rng = np.random.default_rng(42)
    y_jitter = correct + rng.uniform(-0.08, 0.08, size=len(correct))

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.set_facecolor(_color_palette["background"])

    ax.scatter(scores, y_jitter, c=_color_palette["accent"], s=15, alpha=0.5, zorder=5)

    # Bin scores and plot accuracy per bin
    bins = np.arange(0.5, 11.5, 1)
    bin_indices = np.digitize(scores, bins)
    bin_centers = []
    bin_accuracies = []
    for b in range(1, len(bins)):
        mask = bin_indices == b
        if mask.sum() > 0:
            bin_centers.append((bins[b - 1] + bins[b]) / 2)
            bin_accuracies.append(correct[mask].mean())

    ax2 = ax.twinx()
    ax2.plot(bin_centers, [a * 100 for a in bin_accuracies],
             color=_color_palette["primary"], linewidth=2.5,
             marker="s", markersize=6, zorder=10, label="Accuracy per score bin")
    ax2.set_ylabel("Accuracy in bin (%)", color=_color_palette["primary"], fontsize=12)
    ax2.set_ylim(-5, 105)
    ax2.tick_params(axis="y", colors=_color_palette["primary"])
    ax2.spines['right'].set_color(_color_palette["secondary"])

    # Point-biserial correlation
    corr = np.corrcoef(scores, correct)[0, 1] if len(scores) > 1 else 0
    ax.text(0.05, 0.95, f"r = {corr:.2f}  (n={len(scores)}){subtitle_extra}",
            transform=ax.transAxes, fontsize=10,
            color=_color_palette["text"], va="top", fontweight="bold")

    ax.set_xlabel("Evaluator Score (/10)", color=_color_palette["text"], fontsize=12)
    ax.set_ylabel("Correct (0/1, jittered)", color=_color_palette["text"], fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Wrong", "Correct"])
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.3, 1.3)
    ax.set_title(title, color=_color_palette["text"], fontsize=16, weight="bold")
    ax.set_facecolor(_color_palette["background"])
    ax.tick_params(colors=_color_palette["text"])
    ax.spines['bottom'].set_color(_color_palette["secondary"])
    ax.spines['left'].set_color(_color_palette["secondary"])

    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax.legend(lines2, labels2, loc="lower right", fontsize=9,
                       framealpha=0.9, facecolor=_color_palette["background"],
                       edgecolor=_color_palette["secondary"])
    for text in legend.get_texts():
        text.set_color(_color_palette["text"])

    plt.tight_layout(pad=2.0)
    fig.savefig(output_file, dpi=_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved to {output_file}")

    return fig


def _output_variant(base_path: str, suffix: str) -> str:
    parts = base_path.rsplit(".", 1)
    if len(parts) == 2:
        return f"{parts[0]}_{suffix}.{parts[1]}"
    return f"{base_path}_{suffix}"


def plot_correlation(metrics: list[dict], output_path: str):
    """Generate correlation scatter plots: all questions + volatile-only."""
    records = load_per_question_data()
    if not records:
        print("  No per-question evaluation data found, skipping correlation plots.")
        return None

    # All questions
    all_scores = [r["score"] for r in records]
    all_correct = [r["is_correct"] for r in records]
    fig1 = _plot_correlation_scatter(
        all_scores, all_correct,
        "Per-Question: Eval Score vs Correctness", "",
        _output_variant(output_path, "correlation"),
    )

    # Volatile only (correct at least once AND wrong at least once)
    volatile_ids = _get_volatile_task_ids(records)
    volatile = [r for r in records if r["task_id"] in volatile_ids]
    if volatile:
        vol_scores = [r["score"] for r in volatile]
        vol_correct = [r["is_correct"] for r in volatile]
        n_tasks = len(volatile_ids)
        fig2 = _plot_correlation_scatter(
            vol_scores, vol_correct,
            "Volatile Questions: Eval Score vs Correctness",
            f"\n{n_tasks} questions with mixed results",
            _output_variant(output_path, "correlation_volatile"),
        )
    else:
        print("  No volatile questions found (all always-right or always-wrong).")

    # Volatile: per-dimension breakdown
    if volatile:
        plot_volatile_dimensions(volatile, volatile_ids, output_path)

    return fig1


def plot_volatile_dimensions(volatile: list[dict], volatile_ids: set[str], output_path: str):
    """One subplot per eval dimension for volatile questions."""
    plt.style.use('seaborn-v0_8-whitegrid')

    dimensions = [
        ("dim_helpfulness", "Helpfulness"),
        ("dim_accuracy", "Accuracy"),
        ("dim_reasoning_quality", "Reasoning Quality"),
        ("dim_tool_selection", "Tool Selection"),
        ("dim_knowledge_application", "Knowledge Application"),
    ]

    n_dims = len(dimensions)
    fig, axes = plt.subplots(n_dims, 1, figsize=(8, 3.2 * n_dims), sharex=True)
    fig.set_facecolor(_color_palette["background"])
    fig.suptitle(
        f"Volatile Questions: Eval Dimensions vs Correctness\n"
        f"({len(volatile_ids)} questions with mixed results)",
        color=_color_palette["text"], fontsize=16, weight="bold", y=0.98,
    )

    rng = np.random.default_rng(42)
    bins = np.arange(0.5, 11.5, 1)

    for i, (dim_key, dim_label) in enumerate(dimensions):
        ax = axes[i]

        scores = np.array([r.get(dim_key, 0) for r in volatile])
        correct = np.array([r["is_correct"] for r in volatile])
        y_jitter = correct + rng.uniform(-0.08, 0.08, size=len(correct))

        ax.scatter(scores, y_jitter, c=_color_palette["accent"], s=12, alpha=0.45, zorder=5)

        # Accuracy per bin
        bin_indices = np.digitize(scores, bins)
        bin_centers = []
        bin_accuracies = []
        for b in range(1, len(bins)):
            mask = bin_indices == b
            if mask.sum() > 0:
                bin_centers.append((bins[b - 1] + bins[b]) / 2)
                bin_accuracies.append(correct[mask].mean())

        ax2 = ax.twinx()
        ax2.plot(bin_centers, [a * 100 for a in bin_accuracies],
                 color=_color_palette["primary"], linewidth=2,
                 marker="s", markersize=5, zorder=10)
        ax2.set_ylim(-5, 105)
        ax2.tick_params(axis="y", colors=_color_palette["primary"], labelsize=8)
        ax2.spines['right'].set_color(_color_palette["secondary"])
        if i == n_dims // 2:
            ax2.set_ylabel("Accuracy in bin (%)", color=_color_palette["primary"], fontsize=10)

        # Correlation
        corr = np.corrcoef(scores, correct)[0, 1] if len(scores) > 1 else 0
        ax.text(0.02, 0.92, f"r = {corr:.2f}", transform=ax.transAxes, fontsize=9,
                color=_color_palette["text"], fontweight="bold", va="top")

        ax.set_ylabel(dim_label, color=_color_palette["text"], fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Wrong", "Correct"], fontsize=8)
        ax.set_xlim(0, 11)
        ax.set_ylim(-0.3, 1.3)
        ax.set_facecolor(_color_palette["background"])
        ax.tick_params(colors=_color_palette["text"], labelsize=8)
        ax.spines['bottom'].set_color(_color_palette["secondary"])
        ax.spines['left'].set_color(_color_palette["secondary"])
        ax.spines['top'].set_color(_color_palette["secondary"])

    axes[-1].set_xlabel("Evaluator Score (/10)", color=_color_palette["text"], fontsize=12)

    plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.96])

    out_file = _output_variant(output_path, "correlation_volatile_dimensions")
    fig.savefig(out_file, dpi=_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved to {out_file}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot evaluator scores vs benchmark accuracy")
    parser.add_argument("--show", action="store_true", help="Display interactively")
    parser.add_argument("--out", type=str, default="eval_vs_accuracy.png",
                        help="Output file path")
    args = parser.parse_args()

    metrics = load_epoch_metrics()
    if not metrics:
        print("No metrics data found in data/epochs/")
        return

    print(f"Loaded metrics for {len(metrics)} epochs")

    fig1 = plot_eval_vs_accuracy(metrics, args.out)
    fig2 = plot_score_dimensions(metrics, args.out)
    fig3 = plot_correlation(metrics, args.out)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
