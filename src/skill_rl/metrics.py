"""Metrics computation and logging."""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .evaluator import EvaluationResult


@dataclass
class EpochMetrics:
    epoch: int
    n_questions: int
    n_correct: int
    accuracy: float
    accuracy_by_level: dict[int, float]
    avg_scores: dict[str, float]
    avg_overall_score: float
    n_skills: int
    evolution_summary: str
    test_n_questions: int = 0
    test_n_correct: int = 0
    test_accuracy: float = 0.0


class MetricsTracker:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.history: list[EpochMetrics] = []

    def compute_epoch_metrics(
        self,
        epoch: int,
        evaluations: list[EvaluationResult],
        n_skills: int,
        evolution_summary: str,
        test_results: list | None = None,
    ) -> EpochMetrics:
        """Compute metrics for a single epoch."""
        n_questions = len(evaluations)
        n_correct = sum(1 for e in evaluations if e.is_correct)
        accuracy = n_correct / n_questions if n_questions > 0 else 0.0

        # Accuracy by level
        by_level: dict[int, list[bool]] = defaultdict(list)
        for e in evaluations:
            by_level[e.level].append(e.is_correct)
        accuracy_by_level = {
            level: sum(results) / len(results)
            for level, results in sorted(by_level.items())
        }

        # Average scores
        score_sums: dict[str, float] = defaultdict(float)
        for e in evaluations:
            for dim, score in e.scores.items():
                score_sums[dim] += score
        avg_scores = {
            dim: total / n_questions
            for dim, total in score_sums.items()
        } if n_questions > 0 else {}

        avg_overall = (
            sum(e.overall_score for e in evaluations) / n_questions
            if n_questions > 0
            else 0.0
        )

        # Test set metrics
        test_n = len(test_results) if test_results else 0
        test_correct = sum(1 for r in test_results if r.is_correct) if test_results else 0
        test_acc = test_correct / test_n if test_n > 0 else 0.0

        metrics = EpochMetrics(
            epoch=epoch,
            n_questions=n_questions,
            n_correct=n_correct,
            accuracy=accuracy,
            accuracy_by_level=accuracy_by_level,
            avg_scores=avg_scores,
            avg_overall_score=avg_overall,
            n_skills=n_skills,
            evolution_summary=evolution_summary,
            test_n_questions=test_n,
            test_n_correct=test_correct,
            test_accuracy=test_acc,
        )

        self.history.append(metrics)
        return metrics

    def save_metrics(self, epoch: int, metrics: EpochMetrics) -> None:
        """Save epoch metrics and full history."""
        epoch_dir = self.data_dir / "epochs" / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        # Per-epoch metrics
        epoch_data = {
            "epoch": metrics.epoch,
            "n_questions": metrics.n_questions,
            "n_correct": metrics.n_correct,
            "accuracy": metrics.accuracy,
            "accuracy_by_level": {
                str(k): v for k, v in metrics.accuracy_by_level.items()
            },
            "avg_scores": metrics.avg_scores,
            "avg_overall_score": metrics.avg_overall_score,
            "n_skills": metrics.n_skills,
            "evolution_summary": metrics.evolution_summary,
            "test_n_questions": metrics.test_n_questions,
            "test_n_correct": metrics.test_n_correct,
            "test_accuracy": metrics.test_accuracy,
        }
        (epoch_dir / "metrics.json").write_text(json.dumps(epoch_data, indent=2))

        # Full history
        history_data = []
        for m in self.history:
            history_data.append({
                "epoch": m.epoch,
                "accuracy": m.accuracy,
                "test_accuracy": m.test_accuracy,
                "avg_overall_score": m.avg_overall_score,
                "n_skills": m.n_skills,
            })
        (self.data_dir / "history.json").write_text(
            json.dumps(history_data, indent=2)
        )

    def print_summary(self, metrics: EpochMetrics) -> None:
        """Print epoch summary with progress across epochs."""
        print(f"\n{'=' * 60}")
        print(f"EPOCH {metrics.epoch} SUMMARY")
        print(f"{'=' * 60}")
        print(f"Training set pass@1:")
        print(f"  Questions: {metrics.n_questions}")
        print(f"  Correct:   {metrics.n_correct}/{metrics.n_questions} "
              f"({metrics.accuracy:.1%})")
        if metrics.test_n_questions > 0:
            print(f"\nTest set pass@1 (heldout):")
            print(f"  Questions: {metrics.test_n_questions}")
            print(f"  Correct:   {metrics.test_n_correct}/{metrics.test_n_questions} "
                  f"({metrics.test_accuracy:.1%})")
        print(f"\nSkills:    {metrics.n_skills}")
        print()

        # Accuracy by level
        print("Accuracy by level:")
        for level, acc in sorted(metrics.accuracy_by_level.items()):
            bar = _progress_bar(acc, 20)
            print(f"  Level {level}: {bar} {acc:.1%}")

        # Evaluation scores
        print("\nEvaluation scores (avg):")
        for dim, score in metrics.avg_scores.items():
            bar = _progress_bar(score / 10, 20)
            print(f"  {dim:25s}: {bar} {score:.1f}/10")
        print(f"  {'overall':25s}: "
              f"{_progress_bar(metrics.avg_overall_score / 10, 20)} "
              f"{metrics.avg_overall_score:.1f}/10")

        # History
        if len(self.history) > 1:
            print("\nProgress across epochs:")
            print(f"  {'Epoch':>5s}  {'Train':>8s}  {'Test':>8s}  {'Eval Score':>10s}  {'Skills':>6s}")
            for m in self.history:
                test_str = f"{m.test_accuracy:8.1%}" if m.test_n_questions > 0 else "     N/A"
                print(f"  {m.epoch:5d}  {m.accuracy:8.1%}  {test_str}  "
                      f"{m.avg_overall_score:10.1f}  {m.n_skills:6d}")

        print(f"\nEvolution: {metrics.evolution_summary}")
        print(f"{'=' * 60}\n")


def _progress_bar(fraction: float, width: int = 20) -> str:
    """Create a simple progress bar."""
    filled = int(fraction * width)
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "-" * (width - filled) + "]"
