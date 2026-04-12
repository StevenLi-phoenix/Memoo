"""Skill tools — agent loads skills on demand (L2/L3 progressive disclosure)."""

from __future__ import annotations

import logging
from typing import Any

from core.tools import ToolRegistry

logger = logging.getLogger(__name__)


def register(registry: ToolRegistry, **deps: Any) -> None:
    skill_registry = deps.get("skill_registry")
    if skill_registry is None:
        return

    @registry.tool
    def load_skill(name: str) -> str:
        """Load a skill's full instructions (L2). Call this when you need to use a skill.

        Args:
            name: Skill name from the Available Skills list.
        """
        content = skill_registry.load_instructions(name)
        if content is None:
            available = ", ".join(skill_registry.skill_names)
            return f"Skill '{name}' not found. Available: {available}"
        return content

    @registry.tool
    def load_skill_resource(name: str, path: str) -> str:
        """Load a resource file bundled with a skill (L3).

        Args:
            name: Skill name.
            path: Relative path to the resource file within the skill directory.
        """
        content = skill_registry.load_resource(name, path)
        if content is None:
            resources = skill_registry.list_resources(name)
            if resources:
                return f"Resource '{path}' not found in skill '{name}'. Available: {', '.join(resources)}"
            return f"Skill '{name}' not found or has no resources."
        return content
