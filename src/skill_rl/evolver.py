"""Evolver orchestration: rewrites skills based on worst evaluations."""

import difflib
import json
import math
import re
from pathlib import Path

import yaml

from .config import Config
from .evaluator import EvaluationResult
from .llm import complete as llm_complete
from .skills import Skill, SkillManager, apply_trust_region


def _lenient_json_parse(text: str) -> dict | None:
    """Try to parse JSON that has unescaped newlines inside string values.

    LLMs often produce JSON like {"key": "line1\nline2"} with literal newlines
    instead of escaped \\n. This function fixes that by escaping newlines
    that appear inside JSON string literals.
    """
    # Strategy: walk character by character, track whether we're inside a
    # JSON string, and escape bare newlines found inside strings.
    result = []
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '\\' and in_string:
            # Escaped character — pass through both chars
            result.append(ch)
            if i + 1 < len(text):
                i += 1
                result.append(text[i])
            i += 1
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            i += 1
            continue
        if in_string and ch == '\n':
            result.append('\\n')
            i += 1
            continue
        if in_string and ch == '\t':
            result.append('\\t')
            i += 1
            continue
        result.append(ch)
        i += 1

    try:
        return json.loads("".join(result))
    except json.JSONDecodeError:
        return None


SKILL_FORMAT_EXAMPLE = """---
name: research-methodology
description: Systematic approach to answering research questions using available tools
---

When researching a question, follow these steps:

1. **Identify the question type**: Determine if it requires file analysis, web research, computation, or a combination
2. **Locate resources first**: Check for any attached files in /work/ and read them using Python code
3. **Verify your sources**: Cross-reference findings from multiple sources when possible

Keep your approach systematic and always show your work.
"""


class EvolverManager:
    def __init__(self, config: Config):
        self.config = config

    def evolve_skills(
        self,
        evaluations: list[EvaluationResult],
        skill_mgr: SkillManager,
        epoch: int,
    ) -> None:
        """Evolve skills based on worst-performing evaluations."""
        epoch_dir = self.config.epoch_dir(epoch)

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

        data_json = json.dumps({
            "epoch": epoch,
            "evaluations": eval_summaries,
            "current_skills": current_skills,
        }, indent=2)

        user_content = (
            "Analyze the following evaluation data and produce evolved skills.\n\n"
            "IMPORTANT: Respond with ONLY a JSON object, no other text. The JSON must have this structure:\n"
            '{"skills": {"skill-name": "SKILL.md file content...", ...}, "reasoning": "explanation..."}\n\n'
            "The key is the skill name (used as folder name). The value is the full SKILL.md content.\n"
            "Each skill must be a markdown file with YAML frontmatter, like this example:\n"
            f"```\n{SKILL_FORMAT_EXAMPLE}```\n\n"
            "The skill body is free-form markdown — use numbered lists, bold text, paragraphs, etc.\n"
            "The name in the frontmatter MUST match the key in the JSON.\n"
            "Include ALL skills (modified and unmodified). If no skills exist yet, create 2-3 foundational ones.\n"
            "Evolution notes should be added as HTML comments: <!-- [EVOLUTION cycle N] note -->\n\n"
            f"Data:\n{data_json}"
        )

        system_prompt = (self.config.prompts_dir / "evolver_system.txt").read_text()

        try:
            content = llm_complete(
                user_content=user_content,
                system_prompt=system_prompt,
                model_id=self.config.model_id,
                api_base=self.config.api_base,
            )
        except Exception as e:
            print(f"  WARNING: Evolution failed - {e}")
            self._save_evolution_record(
                epoch_dir, epoch, worst_evals,
                skill_mgr.skills, skill_mgr.skills, [], str(e)
            )
            return

        # Parse response
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        # Save raw response for debugging
        (epoch_dir / "evolver_raw_response.txt").write_text(content)

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # LLMs often produce JSON with unescaped newlines in string values.
            # Try to fix by escaping newlines inside JSON string literals.
            data = _lenient_json_parse(content)
            if data is None:
                print(f"  WARNING: Failed to parse evolution JSON: {content[:500]}")
                self._save_evolution_record(
                    epoch_dir, epoch, worst_evals,
                    skill_mgr.skills, skill_mgr.skills, [],
                    f"JSON parse error: {content[:200]}"
                )
                return

        # Parse new skills
        raw_skills = data.get("skills", {})
        if not raw_skills:
            print(f"  WARNING: No 'skills' key in response. Keys: {list(data.keys())}")
            print(f"  Response preview: {content[:500]}")
        new_skills: dict[str, Skill] = {}
        for skill_name, file_content in raw_skills.items():
            skill = self._parse_skill_file(skill_name, file_content)
            if skill:
                new_skills[skill.name] = skill
            else:
                print(f"  WARNING: Failed to parse skill {skill_name}")
                print(f"    Content preview: {str(file_content)[:200]}")

        if not new_skills:
            print("  WARNING: Evolution produced no valid skills")
            self._save_evolution_record(
                epoch_dir, epoch, worst_evals,
                skill_mgr.skills, skill_mgr.skills, [],
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

        # Capture old skills before overwrite (for diff in evolution record)
        old_skills = dict(skill_mgr.skills)

        # Update skill manager
        skill_mgr.skills = accepted
        skill_mgr.save_all()

        reasoning = data.get("reasoning", "")
        print(f"  Evolution complete: {len(accepted)} skills "
              f"({len(rejections)} changes rejected)")
        print(f"  Reasoning: {reasoning[:200]}")

        self._save_evolution_record(
            epoch_dir, epoch, worst_evals, old_skills, accepted, rejections, reasoning
        )

    def _parse_skill_file(self, skill_name: str, file_content: str | dict) -> Skill | None:
        """Parse a skill from SKILL.md file content or a dict."""
        # Handle case where LLM returns a dict instead of a string
        if isinstance(file_content, dict):
            return Skill(
                name=file_content.get("name", skill_name),
                description=file_content.get("description", ""),
                body=str(file_content.get("body", file_content.get("instructions", ""))),
                evolution_notes=file_content.get("evolution_notes", []),
            )

        file_content = str(file_content)

        # Extract evolution notes from HTML comments
        evolution_notes = re.findall(r"<!--\s*(.*?)\s*-->", file_content)

        # Parse frontmatter
        match = re.match(r"^---\n(.*?)\n---\n(.*)", file_content, re.DOTALL)
        if not match:
            # Lenient — LLM didn't include frontmatter
            return Skill(
                name=skill_name,
                description="",
                body=file_content.strip(),
                evolution_notes=evolution_notes,
            )

        try:
            frontmatter = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            return None

        if not isinstance(frontmatter, dict):
            return None

        body = match.group(2).strip()
        clean_body = re.sub(r"\n*<!--\s*.*?\s*-->\n*", "", body).strip()

        return Skill(
            name=frontmatter.get("name", skill_name),
            description=frontmatter.get("description", ""),
            body=clean_body,
            evolution_notes=evolution_notes,
        )

    def _save_evolution_record(
        self,
        epoch_dir: Path,
        epoch: int,
        worst_evals: list[EvaluationResult],
        old_skills: dict[str, Skill],
        accepted: dict[str, Skill],
        rejections: list[str],
        reasoning: str,
    ) -> None:
        """Save evolution metadata with per-skill diffs."""
        # Compute per-skill changes
        skill_changes = {}
        all_names = sorted(set(old_skills.keys()) | set(accepted.keys()))
        for name in all_names:
            before = old_skills.get(name)
            after = accepted.get(name)
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
            "n_worst_evaluated": len(worst_evals),
            "worst_scores": [e.overall_score for e in worst_evals],
            "n_skills_before": len(old_skills),
            "n_skills_after": len(accepted),
            "skill_filenames": list(accepted.keys()),
            "skill_changes": skill_changes,
            "rejections": rejections,
            "reasoning": reasoning,
        }
        (epoch_dir / "evolution.json").write_text(json.dumps(record, indent=2))
