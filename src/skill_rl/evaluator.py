"""Evaluator orchestration: scores rollouts using LLM judge."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from .actor import ActorResult
from .config import Config
from .docker_utils import DockerManager
from .skills import SkillManager


@dataclass
class EvaluationResult:
    task_id: str
    question: str
    level: int
    scores: dict[str, int]
    overall_score: int
    feedback: str
    improvement_areas: list[str]
    final_answer: str | None
    ground_truth: str
    is_correct: bool


DEFAULT_SCORES = {
    "helpfulness": 1,
    "accuracy": 1,
    "reasoning_quality": 1,
    "tool_selection": 1,
    "knowledge_application": 1,
}

MAX_TRANSCRIPT_CHARS = 100_000
TRUNCATION_HALF = 50_000


class EvaluatorManager:
    def __init__(self, config: Config, docker_mgr: DockerManager):
        self.config = config
        self.docker = docker_mgr

    def evaluate_rollouts(
        self,
        rollouts: list[ActorResult],
        skill_mgr: SkillManager,
        epoch: int,
    ) -> list[EvaluationResult]:
        """Evaluate all rollouts via a single LLM container."""
        epoch_dir = self.config.epoch_dir(epoch)
        work_dir = epoch_dir / "evaluator_work"
        work_dir.mkdir(parents=True, exist_ok=True)

        skills_text = skill_mgr._build_skills_block()

        # Build batch of requests
        requests = []
        for rollout in rollouts:
            transcript = rollout.transcript_formatted
            # Truncate long transcripts
            if len(transcript) > MAX_TRANSCRIPT_CHARS:
                transcript = (
                    transcript[:TRUNCATION_HALF]
                    + "\n\n... [TRUNCATED] ...\n\n"
                    + transcript[-TRUNCATION_HALF:]
                )

            user_content = json.dumps({
                "question": rollout.question,
                "transcript": transcript,
                "skills": skills_text,
            })

            requests.append({
                "id": rollout.task_id,
                "messages": [{"role": "user", "content": user_content}],
                "model": self.config.api_model,
                "max_tokens": 2048,
            })

        input_data = {"requests": requests}

        # Run LLM container
        env = {}
        if self.config.api_key:
            env["ANTHROPIC_API_KEY"] = self.config.api_key

        prompt_path = self.config.prompts_dir / "evaluator_system.txt"

        output = self.docker.run_container_with_volume_io(
            image=self.config.llm_image,
            input_data=input_data,
            work_dir=work_dir,
            env=env,
            timeout=600,
            prompt_mount=prompt_path,
        )

        # Parse responses
        results = []
        response_map: dict[str, dict] = {}
        if output and "responses" in output:
            for resp in output["responses"]:
                response_map[resp["id"]] = resp

        for rollout in rollouts:
            resp = response_map.get(rollout.task_id)
            eval_result = self._parse_evaluation(rollout, resp)
            results.append(eval_result)

        # Save evaluations
        eval_data = [
            {
                "task_id": r.task_id,
                "question": r.question,
                "level": r.level,
                "scores": r.scores,
                "overall_score": r.overall_score,
                "feedback": r.feedback,
                "improvement_areas": r.improvement_areas,
                "final_answer": r.final_answer,
                "ground_truth": r.ground_truth,
                "is_correct": r.is_correct,
            }
            for r in results
        ]
        (epoch_dir / "evaluations.json").write_text(
            json.dumps(eval_data, indent=2)
        )

        return results

    def _parse_evaluation(
        self, rollout: ActorResult, resp: dict | None
    ) -> EvaluationResult:
        """Parse an LLM response into an EvaluationResult."""
        if not resp or resp.get("error") or not resp.get("content"):
            return EvaluationResult(
                task_id=rollout.task_id,
                question=rollout.question,
                level=rollout.level,
                scores=DEFAULT_SCORES.copy(),
                overall_score=1,
                feedback="Evaluation failed",
                improvement_areas=[],
                final_answer=rollout.final_answer,
                ground_truth=rollout.ground_truth,
                is_correct=rollout.is_correct,
            )

        content = resp["content"].strip()
        # Strip markdown fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last fence lines
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return EvaluationResult(
                task_id=rollout.task_id,
                question=rollout.question,
                level=rollout.level,
                scores=DEFAULT_SCORES.copy(),
                overall_score=1,
                feedback=f"Failed to parse evaluation JSON: {content[:200]}",
                improvement_areas=[],
                final_answer=rollout.final_answer,
                ground_truth=rollout.ground_truth,
                is_correct=rollout.is_correct,
            )

        scores = data.get("scores", DEFAULT_SCORES)
        # Ensure all expected keys exist
        for key in DEFAULT_SCORES:
            if key not in scores:
                scores[key] = 1

        return EvaluationResult(
            task_id=rollout.task_id,
            question=rollout.question,
            level=rollout.level,
            scores=scores,
            overall_score=data.get("overall_score", 1),
            feedback=data.get("feedback", ""),
            improvement_areas=data.get("improvement_areas", []),
            final_answer=rollout.final_answer,
            ground_truth=rollout.ground_truth,
            is_correct=rollout.is_correct,
        )
