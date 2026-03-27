"""Skill management: load, save, strip, trust-region."""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import yaml


@dataclass
class Skill:
    filename: str
    name: str
    description: str
    instructions: list[str]
    evolution_notes: list[str] = field(default_factory=list)

    def to_yaml_with_notes(self) -> str:
        """Full YAML with evolution comments for the evolver."""
        lines = [
            f"name: {self.name}",
            f"description: {self.description}",
            "instructions:",
        ]
        # Ensure all instructions are strings
        str_instructions = [str(i) if not isinstance(i, str) else i for i in self.instructions]
        for instruction in str_instructions:
            lines.append(f"  - {_yaml_quote(instruction)}")
        # Append evolution notes as trailing comments
        for en in self.evolution_notes:
            lines.append(f"  # {en}")
        return "\n".join(lines) + "\n"

    def to_yaml_stripped(self) -> str:
        """Clean YAML without evolution comments for the actor."""
        data = {
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
        }
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    def to_system_prompt_block(self) -> str:
        """Formatted text block for inclusion in system prompt."""
        lines = [f"### {self.name}", f"{self.description}", ""]
        for i, instruction in enumerate(self.instructions, 1):
            lines.append(f"{i}. {instruction}")
        return "\n".join(lines)


class SkillManager:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills: dict[str, Skill] = {}

    def load_all(self) -> None:
        """Load all skill YAML files from the skills directory."""
        self.skills.clear()
        if not self.skills_dir.exists():
            self.skills_dir.mkdir(parents=True, exist_ok=True)
            return

        for path in sorted(self.skills_dir.glob("*.yaml")):
            skill = self._load_skill(path)
            if skill:
                self.skills[skill.filename] = skill

    def save_all(self) -> None:
        """Save all skills to the skills directory."""
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        for skill in self.skills.values():
            path = self.skills_dir / skill.filename
            path.write_text(skill.to_yaml_with_notes())

    def _load_skill(self, path: Path) -> Skill | None:
        """Load a single skill from a YAML file, preserving evolution comments."""
        content = path.read_text()

        # Extract evolution comments
        evolution_notes = re.findall(
            r"#\s*(\[EVOLUTION[^\]]*\][^\n]*)", content
        )

        # Parse YAML (comments are stripped by parser)
        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError:
            return None

        if not isinstance(data, dict):
            return None

        raw_instructions = data.get("instructions", [])
        instructions = [str(i) if not isinstance(i, str) else i for i in raw_instructions]

        return Skill(
            filename=path.name,
            name=data.get("name", path.stem),
            description=data.get("description", ""),
            instructions=instructions,
            evolution_notes=evolution_notes,
        )

    def build_system_prompt(self, template_path: Path) -> str:
        """Render the actor system prompt template with stripped skills."""
        template = template_path.read_text()
        skills_block = self._build_skills_block()
        return template.replace("{skills_block}", skills_block)

    def _build_skills_block(self) -> str:
        """Build the skills block for the system prompt."""
        if not self.skills:
            return "(No skills learned yet. This is the first epoch.)"

        blocks = []
        for skill in self.skills.values():
            blocks.append(skill.to_system_prompt_block())
        return "\n\n".join(blocks)

    def get_skills_with_notes(self) -> dict[str, str]:
        """Get all skills as YAML with evolution notes (for evolver)."""
        return {
            filename: skill.to_yaml_with_notes()
            for filename, skill in self.skills.items()
        }


def apply_trust_region(
    current_skills: dict[str, Skill],
    new_skills: dict[str, Skill],
    threshold: float,
) -> tuple[dict[str, Skill], list[str]]:
    """Apply trust region constraint to skill evolution.

    Compares stripped YAML only (evolution notes don't count).
    New skills are always accepted. Deletions are rejected.

    Returns (accepted_skills, rejection_messages).
    """
    accepted: dict[str, Skill] = {}
    rejections: list[str] = []

    # Check for deletions (always rejected)
    for filename in current_skills:
        if filename not in new_skills:
            rejections.append(
                f"Rejected deletion of {filename}: skill removal not allowed"
            )
            accepted[filename] = current_skills[filename]

    for filename, new_skill in new_skills.items():
        if filename not in current_skills:
            # New skill: always accept
            accepted[filename] = new_skill
            continue

        # Existing skill: check change ratio
        old_yaml = current_skills[filename].to_yaml_stripped()
        new_yaml = new_skill.to_yaml_stripped()

        ratio = 1.0 - SequenceMatcher(None, old_yaml, new_yaml).ratio()

        if ratio < threshold:
            accepted[filename] = new_skill
        else:
            rejections.append(
                f"Rejected change to {filename}: "
                f"change ratio {ratio:.2f} exceeds threshold {threshold:.2f}"
            )
            accepted[filename] = current_skills[filename]

    return accepted, rejections


def _yaml_quote(s: str) -> str:
    """Quote a YAML string if it contains special characters."""
    if any(c in s for c in ":#{}[]|>&*!%@`"):
        return f'"{s}"'
    return s
