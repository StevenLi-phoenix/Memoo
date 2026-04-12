"""Config tools — agent can read and update runtime configuration.

All config changes go through a single update_config() entry point.
Model-related keys (llm.model, llm.compressor) get fuzzy resolution and hot-reload.
Other keys go through pre-change verification before persisting.
"""

from __future__ import annotations

import json
import logging
import os
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
        """Get the current runtime configuration, including LLM providers and their models."""
        return json.dumps(config.to_display_dict(), indent=2, ensure_ascii=False)

    app = deps.get("app")  # Running Memoo instance for hot-reload

    @registry.tool
    def config_help(key: str = "") -> str:
        """Show available config keys, current values, and valid options.

        Call without arguments to see all keys. Pass a specific key for details
        (e.g. 'llm.model' lists available models).

        Args:
            key: Optional config key to get detailed help for.
        """
        if not key:
            return _all_keys_help(config)
        if key in ("llm.model", "llm.compressor"):
            return _model_options_help(key, config)
        parts = key.split(".")
        val = _snapshot(config, parts)
        if val is _INVALID:
            return f"Unknown key '{key}'. Call config_help() to see all available keys."
        return f"{key} = {val!r} (type: {type(val).__name__})"

    @registry.tool
    async def update_config(key: str, value: str) -> str:
        """Update a configuration value. Supports model resolution and hot-reload.

        Call config_help() first to see available keys and valid options.
        For model keys (llm.model, llm.compressor), value is fuzzy-matched against
        available models (e.g. 'haiku', 'opus', 'gpt-5.4-mini').

        Args:
            key: Dot-separated config key (e.g. 'llm.model', 'sandbox.timeout').
            value: New value as string. Booleans: 'true'/'false'. Numbers: '30'.
        """
        # --- Special keys: model resolution + hot-reload ---
        if key == "llm.model":
            return _update_model(value, config, app)
        if key == "llm.compressor":
            return _update_compressor(value, config, app)

        # --- Normal flow: verify then persist ---
        parts = key.split(".")
        parsed_value = _parse_value(value)

        backup = _snapshot(config, parts)
        if backup is _INVALID:
            return f"Error: unknown config key '{key}'"

        try:
            _apply(config, parts, parsed_value)
        except Exception as e:
            return f"Error applying config: {e}"

        ok, err = await _verify_config(config)
        if not ok:
            _apply(config, parts, backup)
            logger.warning("Config change reverted: %s = %s (verification failed: %s)", key, value, err)
            return f"Error: config change rejected — verification failed: {err}. Change reverted."

        config.save()
        logger.info("Config updated: %s = %s (verified)", key, parsed_value)
        return f"Config updated: {key} = {parsed_value}"


def _resolve_model(preference: str, providers: list[Any]) -> tuple[str | None, Any | None]:
    """Fuzzy-resolve a model name across all providers. Returns (model_id, provider_config)."""
    from models import ModelCache

    cache = ModelCache()
    for p in providers:
        cached = cache.get(p.provider)
        if not cached:
            continue
        exact = [m for m in cached if m.id == preference]
        matches = exact or [m for m in cached if preference in m.id]
        if matches:
            resolved = exact[0].id if exact else sorted(matches, key=lambda x: x.created, reverse=True)[0].id
            return resolved, p
    return None, None


def _list_available_models(providers: list[Any]) -> str:
    """List available model IDs across all providers."""
    from models import ModelCache

    cache = ModelCache()
    all_models: list[str] = []
    for p in providers:
        cached = cache.get(p.provider)
        if cached:
            all_models.extend(m.id for m in sorted(cached, key=lambda x: x.created, reverse=True)[:5])
    return ", ".join(all_models)


def _update_model(value: str, config: Any, app: Any) -> str:
    """Handle llm.model: resolve model name + hot-reload default provider."""
    resolved, p_conf = _resolve_model(value, config.llm.providers)
    if not resolved:
        return f"Model '{value}' not found. Available: {_list_available_models(config.llm.providers)}"

    # Find the default provider's config entry
    default_p = next((p for p in config.llm.providers if p.name == config.llm.default), None)
    if not default_p:
        return "No default provider configured."

    # If resolved model belongs to a different provider, update the default
    if p_conf.name != default_p.name:
        old_default = config.llm.default
        config.llm.default = p_conf.name
        logger.info("Default provider switched: %s -> %s (model %s)", old_default, p_conf.name, resolved)

    old = p_conf.model
    p_conf.model = resolved
    config.save()

    # Hot-reload the live provider
    if app and hasattr(app, "llm") and app.llm:
        if p_conf.name == config.llm.default and hasattr(app.llm, "model_name"):
            app.llm.model_name = resolved  # type: ignore[union-attr]
            logger.info("Hot-reloaded model: %s -> %s", old, resolved)
            return f"Model: {old} -> {resolved} (live)"

    return f"Model set to '{resolved}' for '{p_conf.name}'."


def _update_compressor(value: str, config: Any, app: Any) -> str:
    """Handle llm.compressor: resolve model name + create dedicated provider + hot-reload."""
    resolved, p_conf = _resolve_model(value, config.llm.providers)
    if not resolved:
        return f"Model '{value}' not found. Available: {_list_available_models(config.llm.providers)}"

    old = "default"
    if app and getattr(app, "agent", None):
        old = app.agent._compressor.model_name

        # Create a dedicated provider instance (avoid sharing with fallback)
        api_key_env = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
        api_key = os.environ.get(api_key_env.get(p_conf.provider, ""), "")
        if not api_key:
            return f"No API key for provider '{p_conf.provider}'."

        try:
            from models import create_provider

            new_compressor = create_provider(p_conf.provider, api_key=api_key)
            new_compressor.model_name = resolved  # type: ignore[union-attr]
            app.agent._compressor = new_compressor
        except Exception as e:
            return f"Error creating compressor provider: {e}"

    config.llm.compressor = resolved
    config.save()
    logger.info("Compressor: %s -> %s (live)", old, resolved)
    return f"Compressor: {old} -> {resolved} (live)"


def _all_keys_help(config: Any) -> str:
    """Dynamically list all config keys with current values from dataclass fields."""
    from dataclasses import fields as dc_fields

    lines = ["Available config keys (current values):", ""]

    # Virtual keys — prompt agent to call config_help('<key>') for candidates
    lines.append("  llm.model — main model (call config_help('llm.model') to see available models)")
    lines.append("  llm.compressor — compressor model (call config_help('llm.compressor') to see available models)")
    lines.append("")

    for f in dc_fields(config):
        if f.name.startswith("_"):
            continue
        section = getattr(config, f.name)
        if hasattr(section, "__dataclass_fields__"):
            for sf in dc_fields(section):
                key = f"{f.name}.{sf.name}"
                val = getattr(section, sf.name)
                # Redact secrets
                if "key" in sf.name.lower() and val:
                    val = "***"
                # Compact display for lists of dataclasses (e.g. providers)
                if isinstance(val, list) and val and hasattr(val[0], "__dataclass_fields__"):
                    summary = ", ".join(getattr(v, "name", str(v)) for v in val)
                    lines.append(f"  {key} = [{summary}]")
                else:
                    lines.append(f"  {key} = {val!r}")
        elif f.name == "channels":
            lines.append(f"  channels = ({len(section)} configured)")
        else:
            lines.append(f"  {f.name} = {section!r}")

    lines.append("")
    lines.append("Use config_help('<key>') for details on a specific key.")
    return "\n".join(lines)


def _model_options_help(key: str, config: Any) -> str:
    """List available models for llm.model or llm.compressor."""
    from models import ModelCache

    cache = ModelCache()
    current = ""
    if key == "llm.model":
        default_p = next((p for p in config.llm.providers if p.name == config.llm.default), None)
        current = default_p.model if default_p else ""
    elif key == "llm.compressor":
        current = config.llm.compressor or "(default: last fallback provider)"

    lines = [f"{key} — current: {current}", ""]
    lines.append("Available models:")
    for p in config.llm.providers:
        cached = cache.get(p.provider)
        if not cached:
            continue
        models = sorted(cached, key=lambda x: x.created, reverse=True)[:8]
        lines.append(f"  {p.name} ({p.provider}):")
        for m in models:
            marker = " ← current" if m.id == current else ""
            lines.append(f"    {m.id}{marker}")

    lines.append("")
    lines.append(f"Set with: update_config('{key}', '<name or substring>')")
    return "\n".join(lines)


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
        from pathlib import Path

        prompt_path = Path(config.agent.system_prompt)
        if not prompt_path.exists():
            return False, f"system_prompt not found: {config.agent.system_prompt}"

        db_parent = Path(config.memory.db_path).parent
        db_parent.mkdir(parents=True, exist_ok=True)

        if config.memory.token_window < 1000:
            return False, f"token_window too small: {config.memory.token_window}"
        if config.sandbox.timeout < 1:
            return False, f"sandbox.timeout too small: {config.sandbox.timeout}"
        if config.sandbox.max_output < 100:
            return False, f"sandbox.max_output too small: {config.sandbox.max_output}"

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
