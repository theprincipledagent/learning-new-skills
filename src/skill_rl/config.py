"""Configuration dataclass for skill_rl."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # LLM
    api_base: str | None = None
    model_id: str = "gemini/gemini-2.5-flash-lite"

    # Training params
    num_epochs: int = 10
    questions_per_epoch: int = 100
    max_actor_turns: int = 25
    max_parallel_actors: int = 5
    trust_region_threshold: float = 0.3
    bottom_percentile: float = 0.2
    rollback_threshold: float = 0.05
    actor_timeout_seconds: int = 1800

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    skills_dir: Path = field(default_factory=lambda: Path("skills"))
    prompts_dir: Path = field(default_factory=lambda: Path("prompts"))

    # Docker image tag
    actor_image: str = "skill-rl-actor:latest"

    # Evaluation mode
    use_benchmark_score: bool = False

    # Test set
    test_questions: int = 50
    test_seed: int = 123

    # Resume
    start_epoch: int = 0

    def epoch_dir(self, epoch: int) -> Path:
        d = self.data_dir / "epochs" / f"epoch_{epoch}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def rollouts_dir(self, epoch: int) -> Path:
        d = self.epoch_dir(epoch) / "rollouts"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def validate(self) -> None:
        if self.questions_per_epoch < 1:
            raise ValueError("questions_per_epoch must be >= 1")
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
