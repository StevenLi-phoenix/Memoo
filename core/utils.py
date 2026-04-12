"""Shared utilities for core modules."""

from __future__ import annotations

import re
from typing import Any

import yaml

# Matches a frontmatter line where the value contains an unquoted colon-space,
# which YAML misinterprets as a nested mapping.
# e.g.  description: Build apps. TRIGGER when: code imports foo
_COLON_VALUE_RE = re.compile(r"^(\w[\w\-]*):\s+(.+:\s.+)$", re.MULTILINE)


def _quote_yaml_values(raw: str) -> str:
    """Pre-process YAML frontmatter to quote values containing `: ` (colon-space).

    Without this, `yaml.safe_load` chokes on descriptions like:
      description: Build apps. TRIGGER when: code imports foo
    because it sees `when:` as a mapping key.
    """

    def _quote_match(m: re.Match[str]) -> str:
        key, value = m.group(1), m.group(2)
        # Already quoted — leave alone
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return m.group(0)
        # Escape internal double quotes and wrap
        escaped = value.replace('"', '\\"')
        return f'{key}: "{escaped}"'

    return _COLON_VALUE_RE.sub(_quote_match, raw)


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from markdown content.

    Returns (metadata_dict, body_text). If no frontmatter found or parsing fails,
    returns ({}, content) — always fails open.
    """
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return {}, content

    raw_yaml = match.group(1)
    body = content[match.end() :]

    # Try raw first, then with colon-quoting
    for attempt_yaml in (raw_yaml, _quote_yaml_values(raw_yaml)):
        try:
            meta: dict[str, Any] = yaml.safe_load(attempt_yaml) or {}
            if isinstance(meta, dict):
                return meta, body
        except yaml.YAMLError:
            continue

    # All attempts failed — fail open
    return {}, content
