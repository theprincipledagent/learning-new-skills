#!/usr/bin/env python3
"""Plot a correctness grid: epochs (x) vs questions (y), colored by correct/wrong.

Usage:
    python plot_correctness.py                # save to correctness_grid.png
    python plot_correctness.py --show         # also display interactively
    python plot_correctness.py --questions    # label y-axis with question text
    python plot_correctness.py --out fig.png  # custom output path
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

DATA_DIR = Path("data/epochs")

_DPI = 300

_color_palette = {
    "primary": "#0A58CA",    # Strong Blue
    "secondary": "#6A7C92",  # Slate Gray
    "accent": "#6F42C1",     # Digital Violet
    "background": "#F8F9FA", # Alabaster White
    "text": "#212529",       # Dark Charcoal
}

_grid_colors = {
    "correct": _color_palette["accent"],     # Digital Violet
    "wrong": "#FFFFFF",                      # White
    "missing": "#FFFFFF",                    # White
}


def load_all_rollouts() -> dict[int, dict[str, dict]]:
    """Load all rollouts, keyed by epoch -> task_id -> rollout data."""
    all_data: dict[int, dict[str, dict]] = {}

    for epoch_dir in sorted(DATA_DIR.glob("epoch_*")):
        try:
            epoch = int(epoch_dir.name.split("_")[1])
        except (ValueError, IndexError):
            continue

        rollouts_dir = epoch_dir / "rollouts"
        if not rollouts_dir.exists():
            continue

        epoch_data: dict[str, dict] = {}
        for json_file in rollouts_dir.glob("*/*.json"):
            try:
                data = json.loads(json_file.read_text())
                task_id = data.get("task_id", json_file.stem)
                epoch_data[task_id] = data
            except (json.JSONDecodeError, OSError):
                continue

        if epoch_data:
            all_data[epoch] = epoch_data

    return all_data


def build_grid(all_data: dict[int, dict[str, dict]]):
    """Build the correctness matrix and metadata."""
    epochs = sorted(all_data.keys())

    # Collect all task IDs across all epochs
    all_task_ids = set()
    for epoch_data in all_data.values():
        all_task_ids.update(epoch_data.keys())

    # Compute success rate per task for sorting
    task_success: dict[str, float] = {}
    task_questions: dict[str, str] = {}
    for task_id in all_task_ids:
        correct_count = 0
        total_count = 0
        for epoch in epochs:
            rollout = all_data.get(epoch, {}).get(task_id)
            if rollout:
                total_count += 1
                if rollout.get("is_correct"):
                    correct_count += 1
                if task_id not in task_questions:
                    task_questions[task_id] = rollout.get("question", "")
        task_success[task_id] = correct_count / total_count if total_count > 0 else 0

    # Sort: always-wrong at top, always-right at bottom
    sorted_tasks = sorted(all_task_ids, key=lambda t: (task_success[t], t))

    # Build matrix: 0=wrong, 1=correct, -1=missing
    matrix = np.full((len(sorted_tasks), len(epochs)), -1, dtype=float)
    for col, epoch in enumerate(epochs):
        for row, task_id in enumerate(sorted_tasks):
            rollout = all_data.get(epoch, {}).get(task_id)
            if rollout:
                matrix[row, col] = 1 if rollout.get("is_correct") else 0

    return matrix, epochs, sorted_tasks, task_questions, task_success


def plot(matrix, epochs, tasks, task_questions, task_success, show_questions, output_path):
    """Render the grid plot."""
    plt.style.use('seaborn-v0_8-whitegrid')

    n_tasks, n_epochs = matrix.shape

    # Colormap: missing / wrong / correct
    cmap = mcolors.ListedColormap([
        _grid_colors["missing"],
        _grid_colors["wrong"],
        _grid_colors["correct"],
    ])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig_width = max(4, n_epochs * 0.3 + 2)
    fig_height = max(4, n_tasks * 0.12 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.set_facecolor(_color_palette["background"])

    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    # X-axis: epochs at top
    ax.set_xticks(range(n_epochs))
    ax.set_xticklabels(
        [f"Epoch {e}" for e in epochs],
        rotation=45, ha="left",
        color=_color_palette["text"], fontsize=9,
    )
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Y-axis: task labels
    if show_questions:
        labels = []
        for task_id in tasks:
            q = task_questions.get(task_id, "")[:55]
            rate = task_success.get(task_id, 0)
            labels.append(f"{q}...  ({rate:.0%})")
        ax.set_yticks(range(n_tasks))
        ax.set_yticklabels(labels, fontsize=6, color=_color_palette["text"])
    else:
        labels = []
        for task_id in tasks:
            rate = task_success.get(task_id, 0)
            labels.append(f"{task_id[:12]}...  ({rate:.0%})")
        ax.set_yticks(range(n_tasks))
        ax.set_yticklabels(labels, fontsize=7, color=_color_palette["text"],
                           fontfamily="monospace")

    ax.set_facecolor(_color_palette["background"])
    ax.tick_params(colors=_color_palette["text"], which="both")
    ax.spines['top'].set_color(_color_palette["secondary"])
    ax.spines['bottom'].set_color(_color_palette["secondary"])
    ax.spines['left'].set_color(_color_palette["secondary"])
    ax.spines['right'].set_color(_color_palette["secondary"])

    ax.tick_params(which="minor", size=0)

    # Per-epoch accuracy below columns
    for col, epoch in enumerate(epochs):
        col_data = matrix[:, col]
        valid = col_data[col_data >= 0]
        if len(valid) > 0:
            accuracy = np.sum(valid) / len(valid)
            ax.text(
                col, n_tasks + 0.5, f"{accuracy:.0%}",
                ha="center", va="top",
                fontsize=9, fontweight="bold",
                color=_color_palette["text"],
            )

    ax.set_ylabel(
        "Questions (sorted by success rate)",
        color=_color_palette["text"], fontsize=12,
    )
    ax.set_title(
        "Correctness Grid: Epoch vs Question",
        color=_color_palette["text"], fontsize=16, weight="bold",
        pad=40,
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_grid_colors["correct"], edgecolor=_color_palette["secondary"],
              label="Correct"),
        Patch(facecolor=_grid_colors["wrong"], edgecolor=_color_palette["secondary"],
              label="Wrong / Missing"),
    ]
    legend = ax.legend(
        handles=legend_elements, loc="lower right",
        fontsize=9, framealpha=0.9,
        facecolor=_color_palette["background"],
        edgecolor=_color_palette["secondary"],
    )
    for text in legend.get_texts():
        text.set_color(_color_palette["text"])

    plt.tight_layout(pad=2.0)
    fig.savefig(
        output_path,
        dpi=_DPI,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    print(f"Saved to {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot correctness grid across epochs")
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    parser.add_argument("--questions", action="store_true",
                        help="Label y-axis with question text")
    parser.add_argument("--out", type=str, default="correctness_grid.png",
                        help="Output file path")
    args = parser.parse_args()

    all_data = load_all_rollouts()
    if not all_data:
        print("No rollout data found in data/epochs/")
        return

    matrix, epochs, tasks, task_questions, task_success = build_grid(all_data)

    n_tasks = len(tasks)
    n_epochs = len(epochs)
    total_correct = int(np.sum(matrix == 1))
    total_wrong = int(np.sum(matrix == 0))
    total_missing = int(np.sum(matrix == -1))
    print(f"Loaded {n_tasks} tasks across {n_epochs} epochs")
    print(f"  Correct: {total_correct}  Wrong: {total_wrong}  Missing: {total_missing}")

    fig = plot(matrix, epochs, tasks, task_questions, task_success, args.questions, args.out)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
