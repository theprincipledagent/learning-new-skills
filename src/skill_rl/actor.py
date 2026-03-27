"""Actor container orchestration for running Claude Code on GAIA questions."""

import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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
    ) -> list[ActorResult]:
        """Run all questions in parallel, return results."""
        epoch_dir = self.config.epoch_dir(epoch)
        rollouts_dir = self.config.rollouts_dir(epoch)

        # Render actor system prompt with current skills
        system_prompt = skill_mgr.build_system_prompt(
            self.config.prompts_dir / "actor_system.txt"
        )
        system_prompt_path = epoch_dir / "actor_system_prompt.txt"
        system_prompt_path.write_text(system_prompt)

        results: list[ActorResult] = []
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_actors) as pool:
            futures = {
                pool.submit(
                    self._run_single, q, system_prompt_path, rollouts_dir
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

    def _run_single(
        self,
        question: GaiaQuestion,
        system_prompt_path: Path,
        rollouts_dir: Path,
    ) -> ActorResult:
        """Run a single question in a Docker container."""
        work_dir = rollouts_dir / question.task_id
        work_dir.mkdir(parents=True, exist_ok=True)

        # Write the question prompt to a file
        prompt_path = work_dir / "prompt.txt"
        prompt_path.write_text(question.question)

        # Copy attached files if present
        if question.file_path:
            src = Path(question.file_path)
            if src.exists():
                dst = work_dir / src.name
                shutil.copy2(src, dst)

        # Build env vars
        env = {}
        if self.config.api_key:
            env["ANTHROPIC_API_KEY"] = self.config.api_key
        if self.config.oauth_token:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = self.config.oauth_token

        # Build command
        cmd = (
            f'claude -p "$(cat /work/prompt.txt)" '
            f"--model {self.config.model} "
            f"--output-format stream-json "
            f"--append-system-prompt-file /work/system_prompt.txt "
            f'--allowedTools "Bash,Read,Write,Edit,Glob,Grep,WebSearch,WebFetch" '
            f"--max-turns {self.config.max_actor_turns} "
            f"--dangerously-skip-permissions"
        )

        volumes = {
            str(work_dir.resolve()): {"bind": "/work", "mode": "rw"},
            str(system_prompt_path.resolve()): {
                "bind": "/work/system_prompt.txt",
                "mode": "ro",
            },
        }

        exit_code, stdout = self.docker.run_container(
            image=self.config.actor_image,
            command=cmd,
            env=env,
            volumes=volumes,
            timeout=self.config.actor_timeout_seconds,
        )

        # Parse stream-json output
        transcript_raw = stdout
        transcript_formatted = self._format_transcript(stdout)
        final_answer = self._extract_answer(stdout)
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

    def _format_transcript(self, raw_output: str) -> str:
        """Parse stream-json NDJSON output into a formatted transcript."""
        lines = []
        for line in raw_output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                lines.append(f"[raw] {line}")
                continue

            event_type = event.get("type", "")
            if event_type == "assistant":
                msg = event.get("message", {})
                content = msg.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                lines.append(f"[assistant] {block.get('text', '')}")
                            elif block.get("type") == "tool_use":
                                lines.append(
                                    f"[tool_use: {block.get('name', '')}] "
                                    f"{json.dumps(block.get('input', {}))[:500]}"
                                )
                elif isinstance(content, str):
                    lines.append(f"[assistant] {content}")
            elif event_type == "tool_result":
                content = event.get("content", "")
                if isinstance(content, str):
                    lines.append(f"[tool_result] {content[:500]}")
            elif event_type == "result":
                result_text = event.get("result", "")
                if isinstance(result_text, str):
                    lines.append(f"[result] {result_text}")

        return "\n".join(lines) if lines else raw_output[:5000]

    def _extract_answer(self, raw_output: str) -> str | None:
        """Extract FINAL ANSWER from output."""
        # Try from stream-json result events first
        for line in reversed(raw_output.strip().split("\n")):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if event.get("type") == "result":
                    text = event.get("result", "")
                    match = re.search(
                        r"FINAL ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE
                    )
                    if match:
                        return match.group(1).strip()
            except json.JSONDecodeError:
                pass

        # Fallback: search entire output
        match = re.search(
            r"FINAL ANSWER:\s*(.+?)(?:\n|$)", raw_output, re.IGNORECASE
        )
        if match:
            return match.group(1).strip()

        return None
