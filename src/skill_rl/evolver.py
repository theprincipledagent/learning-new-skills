"""Evolver orchestration: rewrites skills based on worst evaluations."""

import json
import math
from pathlib import Path

import yaml

from .config import Config
from .docker_utils import DockerManager
from .evaluator import EvaluationResult
from .skills import Skill, SkillManager, apply_trust_region


class EvolverManager:
    def __init__(self, config: Config, docker_mgr: DockerManager):
        self.config = config
        self.docker = docker_mgr

    def evolve_skills(
        self,
        evaluations: list[EvaluationResult],
        skill_mgr: SkillManager,
        epoch: int,
    ) -> None:
        """Evolve skills based on worst-performing evaluations."""
        epoch_dir = self.config.epoch_dir(epoch)
        work_dir = epoch_dir / "evolver_work"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Select bottom percentile
        sorted_evals = sorted(evaluations, key=lambda e: e.overall_score)
        n_worst = max(1, math.ceil(len(sorted_evals) * self.config.bottom_percentile))
        worst_evals = sorted_evals[:n_worst]

        print(f"  Evolving based on {n_worst} worst evaluations "
              f"(scores: {[e.overall_score for e in worst_evals]})")

        # Build evolver input data
        eval_summaries = []
        for e in worst_evals:
            eval_summaries.append({
                "question": e.question,
                "scores": e.scores,
                "overall_score": e.overall_score,
                "feedback": e.feedback,
                "improvement_areas": e.improvement_areas,
                "final_answer": e.final_answer,
            })

        current_skills = skill_mgr.get_skills_with_notes()

        user_content = json.dumps({
            "epoch": epoch,
            "evaluations": eval_summaries,
            "current_skills": current_skills,
        })

        input_data = {
            "requests": [{
                "id": f"evolve_epoch_{epoch}",
                "messages": [{"role": "user", "content": user_content}],
                "model": self.config.api_model,
                "max_tokens": 8192,
            }],
        }

        # Run LLM container
        env = {}
        if self.config.api_key:
            env["ANTHROPIC_API_KEY"] = self.config.api_key

        prompt_path = self.config.prompts_dir / "evolver_system.txt"

        output = self.docker.run_container_with_volume_io(
            image=self.config.llm_image,
            input_data=input_data,
            work_dir=work_dir,
            env=env,
            timeout=300,
            prompt_mount=prompt_path,
        )

        if not output or "responses" not in output:
            print("  WARNING: Evolution failed - no output from LLM container")
            self._save_evolution_record(epoch_dir, epoch, worst_evals, {}, [], "No output")
            return

        resp = output["responses"][0]
        if resp.get("error"):
            print(f"  WARNING: Evolution failed - {resp['error']}")
            self._save_evolution_record(epoch_dir, epoch, worst_evals, {}, [], resp["error"])
            return

        # Parse response
        content = resp["content"].strip()
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
            print(f"  WARNING: Failed to parse evolution JSON: {content[:500]}")
            self._save_evolution_record(
                epoch_dir, epoch, worst_evals, {}, [],
                f"JSON parse error: {content[:200]}"
            )
            return

        # Parse new skills
        new_skills: dict[str, Skill] = {}
        for filename, yaml_content in data.get("skills", {}).items():
            skill = self._parse_skill_yaml(filename, yaml_content)
            if skill:
                new_skills[filename] = skill

        if not new_skills:
            print("  WARNING: Evolution produced no valid skills")
            self._save_evolution_record(
                epoch_dir, epoch, worst_evals, {}, [],
                data.get("reasoning", "No skills produced")
            )
            return

        # Apply trust region
        accepted, rejections = apply_trust_region(
            skill_mgr.skills, new_skills, self.config.trust_region_threshold
        )

        if rejections:
            print(f"  Trust region rejections:")
            for msg in rejections:
                print(f"    {msg}")

        # Update skill manager
        skill_mgr.skills = accepted
        skill_mgr.save_all()

        reasoning = data.get("reasoning", "")
        print(f"  Evolution complete: {len(accepted)} skills "
              f"({len(rejections)} changes rejected)")
        print(f"  Reasoning: {reasoning[:200]}")

        self._save_evolution_record(
            epoch_dir, epoch, worst_evals, accepted, rejections, reasoning
        )

    def _parse_skill_yaml(self, filename: str, yaml_content: str) -> Skill | None:
        """Parse a skill from YAML content."""
        import re

        # Extract evolution comments
        evolution_notes = re.findall(
            r"#\s*(\[EVOLUTION[^\]]*\][^\n]*)", yaml_content
        )

        try:
            data = yaml.safe_load(yaml_content)
        except yaml.YAMLError:
            return None

        if not isinstance(data, dict):
            return None

        raw_instructions = data.get("instructions", [])
        instructions = [str(i) if not isinstance(i, str) else i for i in raw_instructions]

        return Skill(
            filename=filename,
            name=data.get("name", filename.replace(".yaml", "")),
            description=data.get("description", ""),
            instructions=instructions,
            evolution_notes=evolution_notes,
        )

    def _save_evolution_record(
        self,
        epoch_dir: Path,
        epoch: int,
        worst_evals: list[EvaluationResult],
        accepted: dict[str, Skill],
        rejections: list[str],
        reasoning: str,
    ) -> None:
        """Save evolution metadata."""
        record = {
            "epoch": epoch,
            "n_worst_evaluated": len(worst_evals),
            "worst_scores": [e.overall_score for e in worst_evals],
            "n_skills_after": len(accepted),
            "skill_filenames": list(accepted.keys()),
            "rejections": rejections,
            "reasoning": reasoning,
        }
        (epoch_dir / "evolution.json").write_text(json.dumps(record, indent=2))
