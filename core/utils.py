"""Shared utilities for core modules."""

from __future__ import annotations

import re
from typing import Any

import yaml


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from markdown content.

    Returns (metadata_dict, body_text). If no frontmatter found, returns ({}, content).
    """
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return {}, content

    meta: dict[str, Any] = yaml.safe_load(match.group(1)) or {}
    body = content[match.end() :]
    return meta, body
