"""Skill management: load, save, strip, trust-region.

Skills live in folders: skills/<skill-name>/SKILL.md
Each SKILL.md has YAML frontmatter + markdown body:

---
name: skill-name
description: When to use this skill
---

Markdown instructions the agent follows.
"""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import yaml


@dataclass
class Skill:
    name: str
    description: str
    body: str
    evolution_notes: list[str] = field(default_factory=list)

    def to_file_content(self) -> str:
        """Full SKILL.md content with evolution notes as HTML comments."""
        lines = [
            "---",
            f"name: {self.name}",
            f"description: {self.description}",
            "---",
            "",
            self.body,
        ]
        if self.evolution_notes:
            lines.append("")
            for en in self.evolution_notes:
                lines.append(f"<!-- {en} -->")
        content = "\n".join(lines)
        if not content.endswith("\n"):
            content += "\n"
        return content

    def to_stripped_content(self) -> str:
        """Content without evolution notes (for trust region comparison)."""
        lines = [
            "---",
            f"name: {self.name}",
            f"description: {self.description}",
            "---",
            "",
            self.body,
        ]
        content = "\n".join(lines)
        if not content.endswith("\n"):
            content += "\n"
        return content

    def to_system_prompt_block(self) -> str:
        """Formatted text block for inclusion in actor system prompt."""
        return f"### {self.name}\n{self.description}\n\n{self.body}"


class SkillManager:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills: dict[str, Skill] = {}

    def load_all(self) -> None:
        """Load all skills from skills/<name>/SKILL.md."""
        self.skills.clear()
        if not self.skills_dir.exists():
            self.skills_dir.mkdir(parents=True, exist_ok=True)
            return

        for skill_md in sorted(self.skills_dir.glob("*/SKILL.md")):
            skill = self._load_skill(skill_md)
            if skill:
                self.skills[skill.name] = skill

    def save_all(self) -> None:
        """Save all skills to skills/<name>/SKILL.md."""
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        for skill in self.skills.values():
            skill_dir = self.skills_dir / skill.name
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(skill.to_file_content())

    def _load_skill(self, path: Path) -> Skill | None:
        """Load a skill from a SKILL.md file."""
        content = path.read_text()

        match = re.match(r"^---\n(.*?)\n---\n(.*)", content, re.DOTALL)
        if not match:
            return None

        try:
            frontmatter = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            return None

        if not isinstance(frontmatter, dict):
            return None

        body = match.group(2).strip()

        # Extract evolution notes from HTML comments
        evolution_notes = re.findall(r"<!--\s*(.*?)\s*-->", content)

        # Remove evolution note comments from body
        clean_body = re.sub(r"\n*<!--\s*.*?\s*-->\n*", "", body).strip()

        # Skill name comes from folder name or frontmatter
        folder_name = path.parent.name
        name = frontmatter.get("name", folder_name)

        return Skill(
            name=name,
            description=frontmatter.get("description", ""),
            body=clean_body,
            evolution_notes=evolution_notes,
        )

    def build_system_prompt(self, template_path: Path) -> str:
        """Render the actor system prompt template with skills."""
        template = template_path.read_text()
        skills_block = self._build_skills_block()
        return template.replace("{skills_block}", skills_block)

    def _build_skills_block(self) -> str:
        if not self.skills:
            return "(No skills learned yet. This is the first epoch.)"

        blocks = []
        for skill in self.skills.values():
            blocks.append(skill.to_system_prompt_block())
        return "\n\n".join(blocks)

    def get_skills_with_notes(self) -> dict[str, str]:
        """Get all skills with evolution notes (for evolver)."""
        return {
            name: skill.to_file_content()
            for name, skill in self.skills.items()
        }


def apply_trust_region(
    current_skills: dict[str, Skill],
    new_skills: dict[str, Skill],
    threshold: float,
) -> tuple[dict[str, Skill], list[str]]:
    """Apply trust region constraint to skill evolution.

    Compares stripped content (without evolution notes).
    New skills always accepted. Deletions rejected.
    """
    accepted: dict[str, Skill] = {}
    rejections: list[str] = []

    for name in current_skills:
        if name not in new_skills:
            rejections.append(
                f"Rejected deletion of {name}: skill removal not allowed"
            )
            accepted[name] = current_skills[name]

    for name, new_skill in new_skills.items():
        if name not in current_skills:
            accepted[name] = new_skill
            continue

        old_content = current_skills[name].to_stripped_content()
        new_content = new_skill.to_stripped_content()
        ratio = 1.0 - SequenceMatcher(None, old_content, new_content).ratio()

        if ratio < threshold:
            accepted[name] = new_skill
        else:
            rejections.append(
                f"Rejected change to {name}: "
                f"change ratio {ratio:.2f} exceeds threshold {threshold:.2f}"
            )
            accepted[name] = current_skills[name]

    return accepted, rejections
