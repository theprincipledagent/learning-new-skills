"""Configuration dataclass for skill_rl."""

from dataclasses import dataclass, field
from pathlib import Path

# Claude Code CLI accepts short names; the API needs full model IDs
MODEL_API_IDS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-5-20250929",
    "opus": "claude-opus-4-6",
}


@dataclass
class Config:
    # Auth (one required)
    api_key: str = ""
    oauth_token: str = ""

    # Model
    model: str = "haiku"

    # Training params
    num_epochs: int = 10
    questions_per_epoch: int = 100
    max_actor_turns: int = 25
    max_parallel_actors: int = 5
    trust_region_threshold: float = 0.3
    bottom_percentile: float = 0.2
    actor_timeout_seconds: int = 600

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    skills_dir: Path = field(default_factory=lambda: Path("skills"))
    prompts_dir: Path = field(default_factory=lambda: Path("prompts"))

    # Docker image tags
    actor_image: str = "skill-rl-actor:latest"
    llm_image: str = "skill-rl-llm:latest"

    # Evaluation mode
    use_benchmark_score: bool = False

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

    @property
    def api_model(self) -> str:
        """Full model ID for the Anthropic API."""
        return MODEL_API_IDS.get(self.model, self.model)

    def validate(self) -> None:
        if not self.api_key and not self.oauth_token:
            raise ValueError("Either --api-key or --oauth-token must be provided")
        if self.questions_per_epoch < 1:
            raise ValueError("questions_per_epoch must be >= 1")
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
