"""Evaluator orchestration: scores rollouts using Claude Code CLI."""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .actor import ActorResult
from .config import Config
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
    def __init__(self, config: Config):
        self.config = config

    def evaluate_rollouts(
        self,
        rollouts: list[ActorResult],
        skill_mgr: SkillManager,
        epoch: int,
    ) -> list[EvaluationResult]:
        """Evaluate all rollouts sequentially via Claude Code CLI."""
        epoch_dir = self.config.epoch_dir(epoch)
        skills_text = skill_mgr._build_skills_block()

        results = []
        for i, rollout in enumerate(rollouts):
            print(f"  Evaluating {i + 1}/{len(rollouts)}: {rollout.task_id}")
            if self.config.use_benchmark_score:
                eval_result = self._evaluate_single_benchmark(rollout, skills_text)
            else:
                eval_result = self._evaluate_single(rollout, skills_text)
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

    def _evaluate_single(
        self, rollout: ActorResult, skills_text: str
    ) -> EvaluationResult:
        """Evaluate a single rollout via Claude Code CLI."""
        transcript = rollout.transcript_formatted
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            transcript = (
                transcript[:TRUNCATION_HALF]
                + "\n\n... [TRUNCATED] ...\n\n"
                + transcript[-TRUNCATION_HALF:]
            )

        data_json = json.dumps({
            "question": rollout.question,
            "transcript": transcript,
            "skills": skills_text,
        }, indent=2)

        user_content = (
            "Evaluate the following agent rollout.\n\n"
            "IMPORTANT: Respond with ONLY a JSON object, no other text. The JSON must have this structure:\n"
            '{"scores": {"helpfulness": N, "accuracy": N, "reasoning_quality": N, "tool_selection": N, "knowledge_application": N}, '
            '"overall_score": N, "feedback": "...", "improvement_areas": ["..."]}\n\n'
            f"Data:\n{data_json}"
        )

        system_prompt_path = self.config.prompts_dir / "evaluator_system.txt"

        try:
            content = call_claude_cli(
                user_content=user_content,
                system_prompt_file=str(system_prompt_path),
                model=self.config.model,
                config=self.config,
            )
        except Exception as e:
            print(f"    Evaluation failed: {e}")
            return self._default_result(rollout, f"CLI error: {e}")

        return self._parse_evaluation(rollout, content)

    def _evaluate_single_benchmark(
        self, rollout: ActorResult, skills_text: str
    ) -> EvaluationResult:
        """Evaluate a single rollout with ground truth knowledge."""
        transcript = rollout.transcript_formatted
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            transcript = (
                transcript[:TRUNCATION_HALF]
                + "\n\n... [TRUNCATED] ...\n\n"
                + transcript[-TRUNCATION_HALF:]
            )

        status = "CORRECT" if rollout.is_correct else "WRONG"

        data_json = json.dumps({
            "question": rollout.question,
            "transcript": transcript,
            "skills": skills_text,
            "is_correct": rollout.is_correct,
            "ground_truth": rollout.ground_truth,
            "final_answer": rollout.final_answer,
        }, indent=2)

        user_content = (
            f"Diagnose this {status} agent rollout.\n\n"
            "IMPORTANT: Respond with ONLY a JSON object, no other text. The JSON must have this structure:\n"
            '{"scores": {"helpfulness": N, "accuracy": N, "reasoning_quality": N, "tool_selection": N, "knowledge_application": N}, '
            '"overall_score": N, "feedback": "...", "improvement_areas": ["..."]}\n\n'
            f"Data:\n{data_json}"
        )

        system_prompt_path = self.config.prompts_dir / "evaluator_benchmark_system.txt"

        try:
            content = call_claude_cli(
                user_content=user_content,
                system_prompt_file=str(system_prompt_path),
                model=self.config.model,
                config=self.config,
            )
        except Exception as e:
            print(f"    Benchmark evaluation failed: {e}")
            return self._default_result(rollout, f"CLI error: {e}")

        return self._parse_evaluation(rollout, content)

    def _parse_evaluation(
        self, rollout: ActorResult, content: str
    ) -> EvaluationResult:
        """Parse CLI output into an EvaluationResult."""
        content = content.strip()
        # Strip markdown fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return self._default_result(
                rollout, f"Failed to parse JSON: {content[:200]}"
            )

        scores = data.get("scores", DEFAULT_SCORES)
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

    def _default_result(
        self, rollout: ActorResult, feedback: str
    ) -> EvaluationResult:
        return EvaluationResult(
            task_id=rollout.task_id,
            question=rollout.question,
            level=rollout.level,
            scores=DEFAULT_SCORES.copy(),
            overall_score=1,
            feedback=feedback,
            improvement_areas=[],
            final_answer=rollout.final_answer,
            ground_truth=rollout.ground_truth,
            is_correct=rollout.is_correct,
        )


def call_claude_cli(
    user_content: str,
    system_prompt_file: str,
    model: str,
    config: "Config",
) -> str:
    """Call Claude Code CLI with prompt piped via stdin. Returns the result text."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write prompt to file
        prompt_file = Path(tmpdir) / "prompt.txt"
        prompt_file.write_text(user_content)

        # Write a shell script to avoid escaping issues
        script_file = Path(tmpdir) / "run.sh"
        script_file.write_text(
            '#!/bin/bash\n'
            f'claude -p "$(cat {prompt_file})" '
            f'--model {model} '
            f'--max-turns 1 '
            f'--output-format json '
            f'--append-system-prompt-file {system_prompt_file} '
            f'--dangerously-skip-permissions\n'
        )
        script_file.chmod(0o755)

        env = dict(os.environ)
        if config.api_key:
            env["ANTHROPIC_API_KEY"] = config.api_key
        if config.oauth_token:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = config.oauth_token

        result = subprocess.run(
            ["bash", str(script_file)],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (exit {result.returncode}): "
            f"{result.stderr[:500]}"
        )

    # Parse JSON output to get the result text
    try:
        output = json.loads(result.stdout)
        return output.get("result", result.stdout)
    except json.JSONDecodeError:
        return result.stdout
