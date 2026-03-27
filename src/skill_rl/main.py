"""Main entry point for skill_rl."""

import argparse
import sys
from pathlib import Path

from .actor import ActorManager
from .config import Config
from .docker_utils import DockerManager
from .evaluator import EvaluatorManager
from .evolver import EvolverManager
from .gaia import GaiaDataset
from .metrics import MetricsTracker
from .skills import SkillManager


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Skill RL: evolve Claude Code skills via GAIA benchmark"
    )

    # Auth
    parser.add_argument("--api-key", type=str, default="",
                        help="Anthropic API key")
    parser.add_argument("--oauth-token", type=str, default="",
                        help="Claude Code OAuth token")

    # Model
    parser.add_argument("--model", type=str, default="haiku",
                        help="Actor model (default: haiku)")

    # Training
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--questions-per-epoch", type=int, default=100)
    parser.add_argument("--max-actor-turns", type=int, default=25)
    parser.add_argument("--max-parallel-actors", type=int, default=5)
    parser.add_argument("--trust-region-threshold", type=float, default=0.3)
    parser.add_argument("--bottom-percentile", type=float, default=0.2)
    parser.add_argument("--actor-timeout", type=int, default=600,
                        help="Actor timeout in seconds")

    # Paths
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--skills-dir", type=Path, default=Path("skills"))
    parser.add_argument("--prompts-dir", type=Path, default=Path("prompts"))

    # Docker
    parser.add_argument("--actor-image", type=str,
                        default="skill-rl-actor:latest")
    parser.add_argument("--llm-image", type=str,
                        default="skill-rl-llm:latest")

    # Resume
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="Epoch to resume from")

    args = parser.parse_args()

    return Config(
        api_key=args.api_key,
        oauth_token=args.oauth_token,
        model=args.model,
        num_epochs=args.num_epochs,
        questions_per_epoch=args.questions_per_epoch,
        max_actor_turns=args.max_actor_turns,
        max_parallel_actors=args.max_parallel_actors,
        trust_region_threshold=args.trust_region_threshold,
        bottom_percentile=args.bottom_percentile,
        actor_timeout_seconds=args.actor_timeout,
        data_dir=args.data_dir,
        skills_dir=args.skills_dir,
        prompts_dir=args.prompts_dir,
        actor_image=args.actor_image,
        llm_image=args.llm_image,
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
    evaluator_mgr = EvaluatorManager(config, docker_mgr)
    evolver_mgr = EvolverManager(config, docker_mgr)

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
    print(f"  Loaded {len(gaia.questions)} questions")

    # Load existing skills
    skill_mgr.load_all()
    print(f"  Loaded {len(skill_mgr.skills)} existing skills")

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

        # 4. Evolver phase
        print(f"\n--- Evolver Phase ---")
        evolver_mgr.evolve_skills(evaluations, skill_mgr, epoch)
        evolution_summary = (
            f"{len(skill_mgr.skills)} skills after evolution"
        )

        # 5. Compute and save metrics
        metrics = metrics_tracker.compute_epoch_metrics(
            epoch=epoch,
            evaluations=evaluations,
            n_skills=len(skill_mgr.skills),
            evolution_summary=evolution_summary,
        )
        metrics_tracker.save_metrics(epoch, metrics)
        metrics_tracker.print_summary(metrics)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
