#!/usr/bin/env python3
"""Compare rollouts for the same question across epochs.

Usage:
    python compare_rollouts.py <task_id>
    python compare_rollouts.py <task_id> --epochs 0 1 2
    python compare_rollouts.py --list              # list all task IDs from latest epoch
    python compare_rollouts.py --list --epoch 0    # list all task IDs from epoch 0
"""

import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path("data/epochs")


def list_task_ids(epoch: int | None = None) -> list[str]:
    """List all task IDs from a given epoch (or the latest)."""
    if epoch is None:
        epoch_dirs = sorted(DATA_DIR.glob("epoch_*"), key=lambda p: int(p.name.split("_")[1]))
        if not epoch_dirs:
            print("No epoch directories found.", file=sys.stderr)
            sys.exit(1)
        epoch_dir = epoch_dirs[-1]
    else:
        epoch_dir = DATA_DIR / f"epoch_{epoch}"

    if not epoch_dir.exists():
        print(f"Epoch directory {epoch_dir} not found.", file=sys.stderr)
        sys.exit(1)

    rollouts_dir = epoch_dir / "rollouts"
    task_ids = sorted(d.name for d in rollouts_dir.iterdir() if d.is_dir())
    return task_ids


def load_rollout(epoch: int, task_id: str) -> dict | None:
    """Load a rollout JSON for a given epoch and task ID."""
    rollout_path = DATA_DIR / f"epoch_{epoch}" / "rollouts" / task_id / f"{task_id}.json"
    if not rollout_path.exists():
        return None
    return json.loads(rollout_path.read_text())


def load_evaluation(epoch: int, task_id: str) -> dict | None:
    """Load evaluation data for a given epoch and task ID."""
    eval_path = DATA_DIR / f"epoch_{epoch}" / "evaluations.json"
    if not eval_path.exists():
        return None
    evals = json.loads(eval_path.read_text())
    for e in evals:
        if e.get("task_id") == task_id:
            return e
    return None


def get_available_epochs() -> list[int]:
    """Get sorted list of available epoch numbers."""
    epochs = []
    for d in DATA_DIR.iterdir():
        if d.is_dir() and d.name.startswith("epoch_"):
            try:
                epochs.append(int(d.name.split("_")[1]))
            except ValueError:
                pass
    return sorted(epochs)


def print_separator(char="=", width=80):
    print(char * width)


def compare(task_id: str, epochs: list[int] | None = None):
    """Compare rollouts for a task across epochs."""
    available = get_available_epochs()
    if not available:
        print("No epoch data found.", file=sys.stderr)
        sys.exit(1)

    if epochs is None:
        epochs = available
    else:
        missing = [e for e in epochs if e not in available]
        if missing:
            print(f"Warning: epochs {missing} not found, skipping.", file=sys.stderr)
        epochs = [e for e in epochs if e in available]

    if not epochs:
        print("No valid epochs to compare.", file=sys.stderr)
        sys.exit(1)

    # Load all rollouts
    rollouts = {}
    for epoch in epochs:
        r = load_rollout(epoch, task_id)
        if r:
            rollouts[epoch] = r

    if not rollouts:
        print(f"No rollouts found for task {task_id} in epochs {epochs}.", file=sys.stderr)
        sys.exit(1)

    # Print header
    first = next(iter(rollouts.values()))
    print_separator("=")
    print(f"TASK: {task_id}")
    print_separator("=")
    print()
    print(f"QUESTION: {first['question']}")
    print()
    print(f"GROUND TRUTH: {first['ground_truth']}")
    print(f"LEVEL: {first.get('level', '?')}")
    print()

    # Summary table
    print_separator("-")
    print(f"{'Epoch':<8} {'Correct':<10} {'Final Answer':<50} {'Eval Score'}")
    print_separator("-")
    for epoch in epochs:
        r = rollouts.get(epoch)
        ev = load_evaluation(epoch, task_id)
        if r:
            correct = "YES" if r["is_correct"] else "NO"
            answer = (r.get("final_answer") or "(no answer)")[:50]
            score = str(ev["overall_score"]) + "/10" if ev else "n/a"
            print(f"{epoch:<8} {correct:<10} {answer:<50} {score}")
        else:
            print(f"{epoch:<8} {'---':<10} {'(no rollout)':<50} {'n/a'}")
    print()

    # Full details per epoch
    for epoch in epochs:
        r = rollouts.get(epoch)
        if not r:
            continue

        ev = load_evaluation(epoch, task_id)

        print_separator("=")
        print(f"EPOCH {epoch}")
        print_separator("=")
        print()

        print(f"  Correct:      {r['is_correct']}")
        print(f"  Final Answer: {r.get('final_answer') or '(no answer)'}")
        print(f"  Exit Code:    {r.get('exit_code', '?')}")

        if ev:
            print()
            print(f"  Evaluation Score: {ev['overall_score']}/10")
            print(f"  Scores: {json.dumps(ev.get('scores', {}), indent=4)}")
            print(f"  Feedback: {ev.get('feedback', 'n/a')}")
            areas = ev.get("improvement_areas", [])
            if areas:
                print(f"  Improvement Areas:")
                for area in areas:
                    print(f"    - {area}")

        print()
        print_separator("-")
        print("TRANSCRIPT")
        print_separator("-")
        print(r.get("transcript_formatted", "(no transcript)"))
        print()


def main():
    parser = argparse.ArgumentParser(description="Compare rollouts across epochs")
    parser.add_argument("task_id", nargs="?", help="Task ID to compare")
    parser.add_argument("--epochs", nargs="*", type=int, help="Specific epochs to compare")
    parser.add_argument("--list", action="store_true", help="List all task IDs")
    parser.add_argument("--epoch", type=int, default=None, help="Epoch for --list")

    args = parser.parse_args()

    if args.list:
        task_ids = list_task_ids(args.epoch)
        epoch_label = f"epoch {args.epoch}" if args.epoch is not None else "latest epoch"
        print(f"Task IDs from {epoch_label} ({len(task_ids)} total):\n")
        for tid in task_ids:
            r = load_rollout(args.epoch or get_available_epochs()[-1], tid)
            if r:
                correct = "CORRECT" if r["is_correct"] else "WRONG"
                answer = (r.get("final_answer") or "(no answer)")[:40]
                print(f"  {tid}  [{correct:>7}]  {answer}")
            else:
                print(f"  {tid}")
        sys.exit(0)

    if not args.task_id:
        parser.print_help()
        sys.exit(1)

    compare(args.task_id, args.epochs)


if __name__ == "__main__":
    main()
