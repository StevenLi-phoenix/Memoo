"""Config tools — agent can read and update runtime configuration.

Config changes go through a pre-change verification:
1. Save current config as backup
2. Apply the change
3. Boot a verify agent to check the new config is valid
4. If verification fails, revert and notify user
"""

from __future__ import annotations

import json
import logging
from typing import Any

from core.tools import ToolRegistry

logger = logging.getLogger(__name__)


def register(registry: ToolRegistry, **deps: Any) -> None:
    """Register config tools. Auto-discovered by tools/__init__.py."""
    config = deps.get("config")
    if config is None:
        logger.warning("Config not provided, skipping config tools")
        return

    @registry.tool
    def get_config() -> str:
        """Get the current runtime configuration."""
        return json.dumps(config.to_dict(), indent=2, ensure_ascii=False)

    @registry.tool
    async def update_config(key: str, value: str) -> str:
        """Update a configuration value with pre-change verification.

        The change is verified before persisting. If verification fails,
        the change is reverted and an error is returned.

        Args:
            key: Dot-separated config key (e.g. 'sandbox.timeout', 'tools.web_search').
            value: New value as string. Booleans: 'true'/'false'. Numbers: '30'.
        """
        parts = key.split(".")
        parsed_value = _parse_value(value)

        # Snapshot current state for rollback
        backup = _snapshot(config, parts)
        if backup is _INVALID:
            return f"Error: unknown config key '{key}'"

        # Apply the change
        try:
            _apply(config, parts, parsed_value)
        except Exception as e:
            return f"Error applying config: {e}"

        # Verify the new config is bootable
        ok, err = await _verify_config(config)
        if not ok:
            # Revert
            _apply(config, parts, backup)
            logger.warning("Config change reverted: %s = %s (verification failed: %s)", key, value, err)
            return f"Error: config change rejected — verification failed: {err}. Change reverted."

        # Persist
        config.save()
        logger.info("Config updated: %s = %s (verified)", key, parsed_value)
        return f"Config updated: {key} = {parsed_value}"


_INVALID = object()


def _snapshot(config: Any, parts: list[str]) -> Any:
    """Get current value at config path, or _INVALID if path doesn't exist."""
    try:
        if len(parts) == 1:
            if hasattr(config, parts[0]):
                return getattr(config, parts[0])
        elif len(parts) == 2:
            section = getattr(config, parts[0], None)
            if section and hasattr(section, parts[1]):
                return getattr(section, parts[1])
    except Exception:
        pass
    return _INVALID


def _apply(config: Any, parts: list[str], value: Any) -> None:
    """Set value at config path."""
    if len(parts) == 1:
        setattr(config, parts[0], value)
    elif len(parts) == 2:
        section = getattr(config, parts[0])
        setattr(section, parts[1], value)
    else:
        raise ValueError(f"Config path too deep: {'.'.join(parts)}")


async def _verify_config(config: Any) -> tuple[bool, str]:
    """Boot-check: verify the config is valid by simulating critical init steps."""
    try:
        # Check system prompt exists
        from pathlib import Path

        prompt_path = Path(config.agent.system_prompt)
        if not prompt_path.exists():
            return False, f"system_prompt not found: {config.agent.system_prompt}"

        # Check memory db path parent exists or can be created
        db_parent = Path(config.memory.db_path).parent
        db_parent.mkdir(parents=True, exist_ok=True)

        # Check numeric ranges
        if config.memory.token_window < 1000:
            return False, f"token_window too small: {config.memory.token_window}"
        if config.sandbox.timeout < 1:
            return False, f"sandbox.timeout too small: {config.sandbox.timeout}"
        if config.sandbox.max_output < 100:
            return False, f"sandbox.max_output too small: {config.sandbox.max_output}"

        # Check port range
        if not (1 <= config.port <= 65535):
            return False, f"port out of range: {config.port}"

        return True, ""
    except Exception as e:
        return False, str(e)


def _parse_value(value: str) -> Any:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
