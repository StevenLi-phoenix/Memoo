"""Agent Skills — modular, filesystem-based capabilities with progressive disclosure.

Skills are directories under skills/ containing a SKILL.md with YAML frontmatter.
Three loading levels:
  L1: Metadata (name + description) — always in system prompt (~100 tokens each)
  L2: Instructions (SKILL.md body) — loaded when agent triggers the skill
  L3: Resources (bundled files, scripts) — loaded as needed by the agent

Inspired by Anthropic's Agent Skills architecture.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SkillMeta:
    """L1: Lightweight metadata, always in system prompt."""

    name: str
    description: str
    path: Path  # directory containing SKILL.md


@dataclass
class SkillRegistry:
    """Discovers, indexes, and serves skills from a directory."""

    skills_dir: Path
    _skills: dict[str, SkillMeta] = field(default_factory=dict)

    def discover(self) -> None:
        """Scan skills_dir for SKILL.md files, parse frontmatter."""
        self._skills.clear()
        if not self.skills_dir.exists():
            logger.info("Skills directory not found: %s", self.skills_dir)
            return

        for skill_md in sorted(self.skills_dir.rglob("SKILL.md")):
            meta = _parse_frontmatter(skill_md)
            if meta:
                self._skills[meta.name] = meta
                logger.info("Skill discovered: %s — %s", meta.name, meta.description[:60])

        logger.info("Discovered %d skills from %s", len(self._skills), self.skills_dir)

    @property
    def skill_names(self) -> list[str]:
        return list(self._skills.keys())

    def get_meta(self, name: str) -> SkillMeta | None:
        return self._skills.get(name)

    def build_system_prompt_section(self) -> str:
        """Generate the skills metadata section for the system prompt (L1)."""
        if not self._skills:
            return ""

        lines = ["\n## Available Skills\n"]
        for meta in self._skills.values():
            lines.append(f"- **{meta.name}**: {meta.description}")
        lines.append("")
        lines.append(
            "To use a skill, call `load_skill(name)` to read its full instructions. "
            "Then follow those instructions to complete the task."
        )
        return "\n".join(lines)

    def load_instructions(self, name: str) -> str | None:
        """L2: Read SKILL.md body (without frontmatter)."""
        meta = self._skills.get(name)
        if not meta:
            return None

        skill_md = meta.path / "SKILL.md"
        if not skill_md.exists():
            return None

        content = skill_md.read_text(encoding="utf-8")
        # Strip frontmatter
        stripped = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, count=1, flags=re.DOTALL)
        logger.info("Skill loaded (L2): %s (%d chars)", name, len(stripped))
        return stripped.strip()

    def load_resource(self, name: str, resource_path: str) -> str | None:
        """L3: Read a bundled resource file from a skill's directory."""
        meta = self._skills.get(name)
        if not meta:
            return None

        # Security: prevent path traversal
        target = (meta.path / resource_path).resolve()
        if not str(target).startswith(str(meta.path.resolve())):
            logger.warning("Skill resource path traversal blocked: %s/%s", name, resource_path)
            return None

        if not target.exists():
            return None

        content = target.read_text(encoding="utf-8", errors="replace")
        logger.info("Skill resource loaded (L3): %s/%s (%d chars)", name, resource_path, len(content))
        return content

    def list_resources(self, name: str) -> list[str]:
        """List available resource files in a skill directory."""
        meta = self._skills.get(name)
        if not meta:
            return []

        resources: list[str] = []
        for p in sorted(meta.path.rglob("*")):
            if p.is_file() and p.name != "SKILL.md":
                resources.append(str(p.relative_to(meta.path)))
        return resources


def _parse_frontmatter(skill_md: Path) -> SkillMeta | None:
    """Parse YAML frontmatter from SKILL.md."""
    try:
        content = skill_md.read_text(encoding="utf-8")
    except OSError:
        return None

    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        logger.warning("No frontmatter in %s, skipping", skill_md)
        return None

    name = ""
    description = ""
    for line in match.group(1).split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip().lower()
            val = val.strip()
            if key == "name":
                name = val
            elif key == "description":
                description = val

    if not name or not description:
        logger.warning("Missing name/description in %s, skipping", skill_md)
        return None

    return SkillMeta(name=name, description=description, path=skill_md.parent)
