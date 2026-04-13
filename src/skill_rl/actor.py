"""Actor container orchestration for running smolagents on GAIA questions."""

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .docker_utils import DockerManager
from .gaia import GaiaDataset, GaiaQuestion, check_answer
from .skills import SkillManager


@dataclass
class ActorResult:
    task_id: str
    question: str
    level: int
    final_answer: str | None
    transcript_raw: str
    transcript_formatted: str
    exit_code: int
    ground_truth: str
    is_correct: bool


class ActorManager:
    def __init__(self, config: Config, docker_mgr: DockerManager):
        self.config = config
        self.docker = docker_mgr

    def run_epoch(
        self,
        questions: list[GaiaQuestion],
        gaia: GaiaDataset,
        skill_mgr: SkillManager,
        epoch: int,
        rollouts_subdir: str = "rollouts",
    ) -> list[ActorResult]:
        """Run all questions in parallel, return results."""
        epoch_dir = self.config.epoch_dir(epoch)
        rollouts_dir = epoch_dir / rollouts_subdir
        rollouts_dir.mkdir(parents=True, exist_ok=True)

        # Build the task prompt template with skills
        task_prompt_template = self._build_task_prompt_template(skill_mgr)

        # Save rendered template for debugging
        (epoch_dir / "actor_task_prompt_template.txt").write_text(task_prompt_template)

        results: list[ActorResult] = []
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_actors) as pool:
            futures = {
                pool.submit(
                    self._run_single, q, task_prompt_template, rollouts_dir
                ): q
                for q in questions
            }
            for future in as_completed(futures):
                q = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    status = "CORRECT" if result.is_correct else "WRONG"
                    answer_display = result.final_answer or "(no answer)"
                    print(
                        f"  [{status}] {q.task_id}: "
                        f"{answer_display} (truth: {q.final_answer})"
                    )
                except Exception as e:
                    print(f"  [ERROR] {q.task_id}: {e}")
                    results.append(
                        ActorResult(
                            task_id=q.task_id,
                            question=q.question,
                            level=q.level,
                            final_answer=None,
                            transcript_raw="",
                            transcript_formatted=f"Error: {e}",
                            exit_code=-1,
                            ground_truth=q.final_answer,
                            is_correct=False,
                        )
                    )

        return results

    def _build_task_prompt_template(self, skill_mgr: SkillManager) -> str:
        """Build the full task prompt with skills injected.

        smolagents manages its own system prompt for tool orchestration,
        so our skills and instructions go into the task prompt instead.
        """
        template = (self.config.prompts_dir / "actor_task.txt").read_text()
        skills_block = skill_mgr._build_skills_block()
        return template.replace("{skills_block}", skills_block)

    def _run_single(
        self,
        question: GaiaQuestion,
        task_prompt_template: str,
        rollouts_dir: Path,
    ) -> ActorResult:
        """Run a single question in a Docker container."""
        work_dir = rollouts_dir / question.task_id
        work_dir.mkdir(parents=True, exist_ok=True)

        # Write the raw question (for reference)
        (work_dir / "prompt.txt").write_text(question.question)

        # Build the full task prompt: template with question filled in
        task_prompt = task_prompt_template.replace("{question}", question.question)
        (work_dir / "task_prompt.txt").write_text(task_prompt)

        # Write container config
        container_config = {
            "model_id": self.config.model_id,
            "max_steps": self.config.max_actor_turns,
        }
        if self.config.api_base:
            container_config["api_base"] = self.config.api_base
        (work_dir / "config.json").write_text(json.dumps(container_config, indent=2))

        # Copy attached files if present
        if question.file_path:
            src = Path(question.file_path)
            if src.exists():
                dst = work_dir / src.name
                shutil.copy2(src, dst)

        volumes = {
            str(work_dir.resolve()): {"bind": "/work", "mode": "rw"},
        }

        # Forward API key env vars to the container
        env = {}
        for key in ("EXA_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            val = os.environ.get(key)
            if val:
                env[key] = val

        exit_code, stdout, stderr = self.docker.run_container(
            image=self.config.actor_image,
            volumes=volumes,
            env=env,
            timeout=self.config.actor_timeout_seconds,
        )

        # Read structured output from container
        output_path = work_dir / "output.json"
        if output_path.exists():
            output = json.loads(output_path.read_text())
            final_answer = output.get("final_answer")
            transcript_formatted = output.get("transcript", "")
            error = output.get("error")
            if error:
                print(f"  Agent error for {question.task_id}: {error}")
        else:
            final_answer = None
            transcript_formatted = (
                f"No output.json. stdout: {stdout[:2000]}\nstderr: {stderr[:2000]}"
            )

        transcript_raw = stdout + "\n" + stderr
        is_correct = check_answer(final_answer, question.final_answer)

        # Save rollout
        rollout_data = {
            "task_id": question.task_id,
            "question": question.question,
            "level": question.level,
            "final_answer": final_answer,
            "ground_truth": question.final_answer,
            "is_correct": is_correct,
            "exit_code": exit_code,
            "transcript_formatted": transcript_formatted,
        }
        (work_dir / f"{question.task_id}.json").write_text(
            json.dumps(rollout_data, indent=2)
        )

        return ActorResult(
            task_id=question.task_id,
            question=question.question,
            level=question.level,
            final_answer=final_answer,
            transcript_raw=transcript_raw,
            transcript_formatted=transcript_formatted,
            exit_code=exit_code,
            ground_truth=question.final_answer,
            is_correct=is_correct,
        )
