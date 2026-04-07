"""Generalization test: run the actor with evolved skills on unseen GAIA questions.

Samples 50 questions from the GAIA validation split that were NOT used during
skill-rl training (which uses seed=42 to sample 100 questions).  Runs the
exact same actor pipeline (Docker + Claude Code CLI + skills) and reports
accuracy.

Usage:
    python -m skill_rl.generalization_test --api-key <KEY> [OPTIONS]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from .actor import ActorManager, ActorResult
from .config import Config
from .docker_utils import DockerManager
from .gaia import GaiaDataset, GaiaQuestion
from .skills import SkillManager


def parse_args() -> tuple[Config, int, int, Path, int, int]:
    parser = argparse.ArgumentParser(
        description="Generalization test: evaluate evolved skills on unseen GAIA questions"
    )

    # Auth
    parser.add_argument("--api-key", type=str, default="",
                        help="Anthropic API key")
    parser.add_argument("--oauth-token", type=str, default="",
                        help="Claude Code OAuth token")

    # Model
    parser.add_argument("--model", type=str, default="haiku",
                        help="Actor model (default: haiku)")

    # Test params
    parser.add_argument("--num-questions", type=int, default=50,
                        help="Number of held-out questions to test (default: 50)")
    parser.add_argument("--max-actor-turns", type=int, default=25)
    parser.add_argument("--max-parallel-actors", type=int, default=5)
    parser.add_argument("--actor-timeout", type=int, default=600,
                        help="Actor timeout in seconds")
    parser.add_argument("--test-seed", type=int, default=123,
                        help="RNG seed for sampling held-out questions (default: 123)")

    # Training params used to identify the training set
    parser.add_argument("--training-questions-per-epoch", type=int, default=100,
                        help="questions_per_epoch used during training (to exclude them)")
    parser.add_argument("--training-seed", type=int, default=42,
                        help="Seed used during training sampling (default: 42)")

    # Paths
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/generalization_test"),
                        help="Output directory for results")
    parser.add_argument("--skills-dir", type=Path, default=Path("skills"))
    parser.add_argument("--prompts-dir", type=Path, default=Path("prompts"))

    # Docker
    parser.add_argument("--actor-image", type=str,
                        default="skill-rl-actor:latest")
    parser.add_argument("--llm-image", type=str,
                        default="skill-rl-llm:latest")

    args = parser.parse_args()

    config = Config(
        api_key=args.api_key,
        oauth_token=args.oauth_token,
        model=args.model,
        num_epochs=1,
        questions_per_epoch=args.num_questions,
        max_actor_turns=args.max_actor_turns,
        max_parallel_actors=args.max_parallel_actors,
        actor_timeout_seconds=args.actor_timeout,
        data_dir=args.output_dir,
        skills_dir=args.skills_dir,
        prompts_dir=args.prompts_dir,
        actor_image=args.actor_image,
        llm_image=args.llm_image,
    )

    return (
        config,
        args.num_questions,
        args.test_seed,
        args.output_dir,
        args.training_questions_per_epoch,
        args.training_seed,
    )


def get_held_out_questions(
    gaia: GaiaDataset,
    training_n: int,
    training_seed: int,
    test_n: int,
    test_seed: int,
) -> list[GaiaQuestion]:
    """Return questions NOT used during training."""
    import random

    # Reproduce the exact training sample
    training_questions = gaia.sample(training_n, seed=training_seed)
    training_ids = {q.task_id for q in training_questions}

    # Get all questions not in the training set
    held_out = [q for q in gaia.questions if q.task_id not in training_ids]

    if len(held_out) < test_n:
        print(f"Warning: only {len(held_out)} held-out questions available "
              f"(requested {test_n}). Using all of them.")
        test_n = len(held_out)

    # Stratified sampling from held-out set
    rng = random.Random(test_seed)
    by_level: dict[int, list[GaiaQuestion]] = {}
    for q in held_out:
        by_level.setdefault(q.level, []).append(q)

    for level in by_level:
        by_level[level].sort(key=lambda q: q.task_id)

    total = len(held_out)
    sampled: list[GaiaQuestion] = []

    for level in sorted(by_level.keys()):
        level_qs = by_level[level]
        level_n = max(1, round(test_n * len(level_qs) / total))
        level_n = min(level_n, len(level_qs))
        sampled.extend(rng.sample(level_qs, level_n))

    if len(sampled) > test_n:
        rng.shuffle(sampled)
        sampled = sampled[:test_n]
    elif len(sampled) < test_n:
        remaining = [q for q in held_out if q not in sampled]
        rng.shuffle(remaining)
        sampled.extend(remaining[: test_n - len(sampled)])

    return sampled


def progress_bar(fraction: float, width: int = 20) -> str:
    filled = int(fraction * width)
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def print_results(results: list[ActorResult]) -> dict:
    """Print and return accuracy metrics."""
    n_total = len(results)
    n_correct = sum(1 for r in results if r.is_correct)
    accuracy = n_correct / n_total if n_total else 0.0

    # By level
    by_level: dict[int, list[bool]] = defaultdict(list)
    for r in results:
        by_level[r.level].append(r.is_correct)
    accuracy_by_level = {
        level: sum(correct) / len(correct)
        for level, correct in sorted(by_level.items())
    }

    print(f"\n{'=' * 60}")
    print("GENERALIZATION TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"Questions:  {n_total}")
    print(f"Correct:    {n_correct}/{n_total} ({accuracy:.1%})")
    print()

    print("Accuracy by level:")
    for level, acc in sorted(accuracy_by_level.items()):
        n_level = len(by_level[level])
        n_level_correct = sum(by_level[level])
        bar = progress_bar(acc)
        print(f"  Level {level}: {bar} {acc:.1%}  ({n_level_correct}/{n_level})")

    print(f"{'=' * 60}\n")

    return {
        "n_questions": n_total,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "accuracy_by_level": {str(k): v for k, v in accuracy_by_level.items()},
        "n_by_level": {str(k): len(v) for k, v in sorted(by_level.items())},
    }


def main():
    config, num_questions, test_seed, output_dir, training_n, training_seed = parse_args()

    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize managers
    docker_mgr = DockerManager()
    skill_mgr = SkillManager(config.skills_dir)

    actor_mgr = ActorManager(config, docker_mgr)

    # Build Docker images
    print("Building Docker images...")
    docker_mgr.ensure_images(
        actor_image=config.actor_image,
        llm_image=config.llm_image,
        actor_dir="docker/actor",
        llm_dir="docker/llm",
    )

    # Load GAIA dataset
    print("Loading GAIA dataset...")
    gaia = GaiaDataset()
    gaia.load()
    print(f"  Total questions in validation split: {len(gaia.questions)}")

    # Load current evolved skills
    skill_mgr.load_all()
    print(f"  Loaded {len(skill_mgr.skills)} evolved skills")

    # Get held-out questions
    print(f"\nIdentifying held-out questions...")
    questions = get_held_out_questions(
        gaia, training_n, training_seed, num_questions, test_seed,
    )

    training_questions = gaia.sample(training_n, seed=training_seed)
    training_ids = {q.task_id for q in training_questions}
    overlap = [q for q in questions if q.task_id in training_ids]
    if overlap:
        print(f"  WARNING: {len(overlap)} questions overlap with training set!")
    else:
        print(f"  Confirmed: 0 overlap with training set")

    print(f"  Selected {len(questions)} held-out questions")
    by_level = defaultdict(int)
    for q in questions:
        by_level[q.level] += 1
    for level in sorted(by_level):
        print(f"    Level {level}: {by_level[level]} questions")

    # Set up output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run actor phase (reuses epoch 0 directory structure)
    print(f"\n{'#' * 60}")
    print(f"# RUNNING GENERALIZATION TEST")
    print(f"# {len(questions)} unseen questions, {len(skill_mgr.skills)} evolved skills")
    print(f"{'#' * 60}")

    print(f"\n--- Actor Phase ---")
    results = actor_mgr.run_epoch(questions, gaia, skill_mgr, epoch=0)

    # Print results
    metrics = print_results(results)

    # Save detailed results
    results_data = {
        "test_config": {
            "num_questions": num_questions,
            "test_seed": test_seed,
            "training_seed": training_seed,
            "training_questions_per_epoch": training_n,
            "model": config.model,
            "max_actor_turns": config.max_actor_turns,
            "n_skills": len(skill_mgr.skills),
            "skill_names": list(skill_mgr.skills.keys()),
        },
        "metrics": metrics,
        "per_question": [
            {
                "task_id": r.task_id,
                "question": r.question,
                "level": r.level,
                "final_answer": r.final_answer,
                "ground_truth": r.ground_truth,
                "is_correct": r.is_correct,
                "exit_code": r.exit_code,
            }
            for r in results
        ],
    }
    results_path = output_dir / "generalization_results.json"
    results_path.write_text(json.dumps(results_data, indent=2))
    print(f"Results saved to {results_path}")

    # Also save the question IDs used (for reproducibility)
    question_ids_path = output_dir / "test_question_ids.json"
    question_ids_path.write_text(json.dumps(
        [q.task_id for q in questions], indent=2
    ))
    print(f"Question IDs saved to {question_ids_path}")


if __name__ == "__main__":
    main()
