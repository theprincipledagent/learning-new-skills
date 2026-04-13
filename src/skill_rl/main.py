"""Main entry point for skill_rl."""

import argparse
import difflib
import json
import sys
from pathlib import Path

from .actor import ActorManager
from .config import Config
from .docker_utils import DockerManager
from .evaluator import EvaluatorManager
from .evolver import EvolverManager
from .gaia import GaiaDataset
from .generalization_test import get_held_out_questions
from .metrics import MetricsTracker
from .skills import Skill, SkillManager


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Skill RL: evolve agent skills via GAIA benchmark"
    )

    # LLM
    parser.add_argument("--api-base", type=str, default=None,
                        help="LiteLLM API base URL (default: None, uses provider default)")
    parser.add_argument("--model-id", type=str, default="gemini/gemini-2.5-flash-lite",
                        help="LiteLLM model ID (default: gemini/gemini-2.5-flash-lite)")

    # Training
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--questions-per-epoch", type=int, default=100)
    parser.add_argument("--max-actor-turns", type=int, default=25)
    parser.add_argument("--max-parallel-actors", type=int, default=5)
    parser.add_argument("--trust-region-threshold", type=float, default=0.3)
    parser.add_argument("--bottom-percentile", type=float, default=0.2)
    parser.add_argument("--rollback-threshold", type=float, default=0.05,
                        help="Roll back skills if accuracy drops by this amount (default: 0.05)")
    parser.add_argument("--actor-timeout", type=int, default=1800,
                        help="Actor timeout in seconds")

    # Paths
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--skills-dir", type=Path, default=Path("skills"))
    parser.add_argument("--prompts-dir", type=Path, default=Path("prompts"))

    # Docker
    parser.add_argument("--actor-image", type=str,
                        default="skill-rl-actor:latest")

    # Evaluation mode
    parser.add_argument("--use-benchmark-score", action="store_true",
                        help="Use ground truth to diagnose failures instead of blind evaluation")

    # Test set
    parser.add_argument("--test-questions", type=int, default=50,
                        help="Number of heldout test questions for pass@1 reporting (default: 50)")
    parser.add_argument("--test-seed", type=int, default=123,
                        help="RNG seed for sampling heldout test questions (default: 123)")

    # Resume
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="Epoch to resume from")

    args = parser.parse_args()

    return Config(
        api_base=args.api_base,
        model_id=args.model_id,
        num_epochs=args.num_epochs,
        questions_per_epoch=args.questions_per_epoch,
        max_actor_turns=args.max_actor_turns,
        max_parallel_actors=args.max_parallel_actors,
        trust_region_threshold=args.trust_region_threshold,
        bottom_percentile=args.bottom_percentile,
        rollback_threshold=args.rollback_threshold,
        actor_timeout_seconds=args.actor_timeout,
        data_dir=args.data_dir,
        skills_dir=args.skills_dir,
        prompts_dir=args.prompts_dir,
        actor_image=args.actor_image,
        use_benchmark_score=args.use_benchmark_score,
        test_questions=args.test_questions,
        test_seed=args.test_seed,
        start_epoch=args.start_epoch,
    )


def main():
    config = parse_args()

    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize managers
    docker_mgr = DockerManager()
    skill_mgr = SkillManager(config.skills_dir)
    metrics_tracker = MetricsTracker(config.data_dir)

    actor_mgr = ActorManager(config, docker_mgr)
    evaluator_mgr = EvaluatorManager(config)
    evolver_mgr = EvolverManager(config)

    # Build Docker images
    print("Building Docker images...")
    docker_mgr.ensure_images(
        actor_image=config.actor_image,
        actor_dir="docker/actor",
    )

    # Load GAIA dataset
    print("Loading GAIA dataset...")
    gaia = GaiaDataset()
    gaia.load()
    print(f"  Loaded {len(gaia.questions)} questions")

    # Load existing skills
    skill_mgr.load_all()
    print(f"  Loaded {len(skill_mgr.skills)} existing skills")

    # Sample heldout test set (fixed across all epochs)
    test_questions = get_held_out_questions(
        gaia,
        training_n=config.questions_per_epoch,
        training_seed=42,
        test_n=config.test_questions,
        test_seed=config.test_seed,
    )
    print(f"  Sampled {len(test_questions)} heldout test questions")

    # Rollback state: tracks the last "good" checkpoint
    best_accuracy: float | None = None
    best_skills: dict[str, Skill] | None = None

    # Main training loop
    for epoch in range(config.start_epoch, config.num_epochs):
        print(f"\n{'#' * 60}")
        print(f"# EPOCH {epoch}")
        print(f"{'#' * 60}")

        # 1. Sample questions (fixed seed for reproducibility)
        questions = gaia.sample(config.questions_per_epoch, seed=42)
        print(f"\nSampled {len(questions)} questions")

        # 2. Actor phase
        print(f"\n--- Actor Phase ---")
        rollouts = actor_mgr.run_epoch(questions, gaia, skill_mgr, epoch)
        n_correct = sum(1 for r in rollouts if r.is_correct)
        print(f"  Actor phase complete: {n_correct}/{len(rollouts)} correct")

        # 3. Evaluator phase
        print(f"\n--- Evaluator Phase ---")
        evaluations = evaluator_mgr.evaluate_rollouts(
            rollouts, skill_mgr, epoch
        )
        avg_score = (
            sum(e.overall_score for e in evaluations) / len(evaluations)
            if evaluations
            else 0
        )
        print(f"  Evaluation complete: avg score {avg_score:.1f}/10")

        # Compute current accuracy for rollback check
        current_accuracy = n_correct / len(rollouts) if rollouts else 0.0

        # 4. Test set actor phase (heldout pass@1, before evolution)
        print(f"\n--- Test Set Phase ---")
        test_rollouts = actor_mgr.run_epoch(
            test_questions, gaia, skill_mgr, epoch,
            rollouts_subdir="test_rollouts",
        )
        test_correct = sum(1 for r in test_rollouts if r.is_correct)
        print(f"  Test phase complete: {test_correct}/{len(test_rollouts)} correct")

        # 5. Evolver phase (or rollback)
        degraded = (
            best_accuracy is not None
            and current_accuracy < best_accuracy - config.rollback_threshold
        )

        if degraded:
            print(f"\n--- Rollback ---")
            print(f"  Accuracy dropped {best_accuracy:.1%} -> {current_accuracy:.1%} "
                  f"(threshold: {config.rollback_threshold:.1%})")
            print(f"  Rolling back to previous skills")

            old_skills = dict(skill_mgr.skills)
            skill_mgr.skills = {name: skill for name, skill in best_skills.items()}
            skill_mgr.save_all()

            _save_rollback_record(
                config.epoch_dir(epoch), epoch, best_accuracy,
                current_accuracy, old_skills, skill_mgr.skills,
            )

            evolution_summary = (
                f"ROLLBACK: {len(skill_mgr.skills)} skills "
                f"(accuracy {best_accuracy:.1%} -> {current_accuracy:.1%})"
            )
        else:
            print(f"\n--- Evolver Phase ---")
            best_skills = {name: skill for name, skill in skill_mgr.skills.items()}
            best_accuracy = current_accuracy

            evolver_mgr.evolve_skills(evaluations, skill_mgr, epoch)
            evolution_summary = (
                f"{len(skill_mgr.skills)} skills after evolution"
            )

        # 6. Compute and save metrics
        metrics = metrics_tracker.compute_epoch_metrics(
            epoch=epoch,
            evaluations=evaluations,
            n_skills=len(skill_mgr.skills),
            evolution_summary=evolution_summary,
            test_results=test_rollouts,
        )
        metrics_tracker.save_metrics(epoch, metrics)
        metrics_tracker.print_summary(metrics)

    print("\nTraining complete!")


def _save_rollback_record(
    epoch_dir: Path,
    epoch: int,
    prev_accuracy: float,
    current_accuracy: float,
    old_skills: dict[str, Skill],
    restored_skills: dict[str, Skill],
) -> None:
    """Save an evolution.json record for a rollback event."""
    skill_changes = {}
    all_names = sorted(set(old_skills.keys()) | set(restored_skills.keys()))
    for name in all_names:
        before = old_skills.get(name)
        after = restored_skills.get(name)
        before_content = before.to_file_content() if before else ""
        after_content = after.to_file_content() if after else ""

        if before is None:
            change_type = "added"
        elif after is None:
            change_type = "removed"
        elif before_content == after_content:
            change_type = "unchanged"
        else:
            change_type = "modified"

        diff_text = ""
        if change_type in ("modified", "added", "removed"):
            diff_text = "\n".join(difflib.unified_diff(
                before_content.splitlines(),
                after_content.splitlines(),
                fromfile=f"before/{name}/SKILL.md",
                tofile=f"after/{name}/SKILL.md",
                lineterm="",
            ))

        skill_changes[name] = {
            "change_type": change_type,
            "diff": diff_text,
        }

    record = {
        "epoch": epoch,
        "action": "rollback",
        "prev_accuracy": prev_accuracy,
        "current_accuracy": current_accuracy,
        "n_skills_before": len(old_skills),
        "n_skills_after": len(restored_skills),
        "skill_filenames": list(restored_skills.keys()),
        "skill_changes": skill_changes,
        "rejections": [],
        "reasoning": (
            f"Accuracy dropped from {prev_accuracy:.1%} to {current_accuracy:.1%}. "
            f"Rolled back skills to previous checkpoint."
        ),
    }
    (epoch_dir / "evolution.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
