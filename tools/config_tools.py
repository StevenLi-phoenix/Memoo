"""Config tools — agent can read and update runtime configuration.

All config changes go through a single update_config() entry point.
Model-related keys (llm.model, llm.compressor) resolve against configured
LLM model aliases and hot-reload the live app state.
Other keys go through pre-change verification before persisting.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from core.config import ModelConfig
from core.tools import ToolRegistry

logger = logging.getLogger(__name__)


def _normalize_model_lookup(value: str) -> str:
    """Normalize user-entered model fragments for fuzzy lookup."""
    value = value.strip().lower().replace("_", "-").replace(" ", "-")
    for prefix in ("anthropic/", "openai/", "localhost/"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    if value.startswith("claude-"):
        value = value[len("claude-") :]
    return value


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
            return _model_options_help(key, config, app)
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


def _resolve_model(preference: str, config: Any) -> Any | None:
    """Fuzzy-resolve a configured model alias from llm.models."""
    preference = preference.strip()
    if not preference:
        return None
    normalized = _normalize_model_lookup(preference)

    for model in config.llm.models:
        if model.name == preference or model.model == preference:
            return model
        if _normalize_model_lookup(model.name) == normalized or _normalize_model_lookup(model.model) == normalized:
            return model

    alias_matches = [
        m for m in config.llm.models if preference in m.name or normalized in _normalize_model_lookup(m.name)
    ]
    if alias_matches:
        return alias_matches[0]

    model_matches = [
        m for m in config.llm.models if preference in m.model or normalized in _normalize_model_lookup(m.model)
    ]
    if model_matches:
        return model_matches[0]

    return None


def _resolve_discovered_model(preference: str, config: Any, app: Any) -> ModelConfig | None:
    """Resolve a model from discovered provider listings and promote it into llm.models."""
    if not app:
        return None

    preference = preference.strip()
    if not preference:
        return None
    normalized = _normalize_model_lookup(preference)

    discovered = getattr(app, "discovered_models", {}) or {}
    for provider in config.llm.providers:
        if not provider.allow_model_discovery:
            continue
        models = discovered.get(provider.name, [])
        if not models:
            continue

        exact = next((info for info in models if info.id == preference), None)
        normalized_exact = next((info for info in models if _normalize_model_lookup(info.id) == normalized), None)
        match = (
            exact
            or normalized_exact
            or next(
                (info for info in models if preference in info.id or normalized in _normalize_model_lookup(info.id)),
                None,
            )
        )
        if not match:
            continue

        existing = next((model for model in config.llm.models if model.name == f"{provider.name}/{match.id}"), None)
        if existing:
            return existing

        promoted = ModelConfig(
            name=f"{provider.name}/{match.id}",
            provider=provider.name,
            model=match.id,
            base_url=provider.base_url,
        )
        config.llm.models.append(promoted)
        logger.info("Promoted discovered model into config: %s", promoted.name)
        return promoted

    return None


def _list_available_models(config: Any) -> str:
    """List configured model aliases."""
    return ", ".join(model.name for model in config.llm.models)


def _list_available_models_with_discovery(config: Any, app: Any) -> str:
    """List configured aliases plus discovered model IDs from enabled providers."""
    configured = [model.name for model in config.llm.models]
    discovered_items: list[str] = []
    discovered = getattr(app, "discovered_models", {}) or {}
    for provider in config.llm.providers:
        if not provider.allow_model_discovery:
            continue
        for info in discovered.get(provider.name, []):
            discovered_items.append(info.id)
    combined = configured + [item for item in discovered_items if item not in configured]
    return ", ".join(combined)


def _build_live_provider(config: Any, model: ModelConfig) -> Any | None:
    provider_conf = config.llm.resolve_provider(model.provider)
    if not provider_conf:
        return None

    api_key_env = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
    api_key = os.environ.get(api_key_env.get(provider_conf.provider, ""), "")
    if not api_key:
        return None

    try:
        from models import create_provider

        kwargs: dict[str, Any] = {"api_key": api_key}
        base_url = model.base_url or provider_conf.base_url
        if provider_conf.provider == "anthropic":
            kwargs["web_search"] = config.tools.web_search
        elif provider_conf.provider == "openai" and base_url:
            kwargs["base_url"] = base_url

        provider = create_provider(provider_conf.provider, **kwargs)
        provider.model_name = model.model  # type: ignore[union-attr]
        if provider_conf.provider == "anthropic" and model.advisor:
            provider._advisor_model = model.advisor  # type: ignore[union-attr]
        return provider
    except Exception:
        logger.exception("Failed to build live provider for %s", model.name)
        return None


def _sync_live_model_state(config: Any, app: Any) -> None:
    if not app:
        return
    providers = getattr(app, "_providers", {}) or {}
    default_model = config.llm.resolve_model(config.llm.default)
    if default_model and config.llm.default not in providers:
        live_provider = _build_live_provider(config, default_model)
        if live_provider:
            providers[config.llm.default] = live_provider
    if config.llm.default in providers:
        app.llm = providers[config.llm.default]
    app.fallback_llms = [
        providers[name] for name in config.llm.fallback if name in providers and name != config.llm.default
    ]
    agent = getattr(app, "agent", None)
    if agent:
        if getattr(app, "llm", None):
            agent._llm = app.llm
        agent._fallback_llms = list(app.fallback_llms)


def _update_model(value: str, config: Any, app: Any) -> str:
    """Handle llm.model: switch the default configured model alias."""
    resolved = _resolve_model(value, config)
    promoted = False
    if not resolved:
        resolved = _resolve_discovered_model(value, config, app)
        promoted = resolved is not None
    if not resolved:
        return f"Model '{value}' not found. Available: {_list_available_models_with_discovery(config, app)}"

    old = config.llm.default
    config.llm.default = resolved.name
    config.save()
    _sync_live_model_state(config, app)
    logger.info("Default model switched: %s -> %s", old or "unset", resolved.name)
    if promoted:
        return f"Model: {old or '(unset)'} -> {resolved.name} (added to config)"
    return f"Model: {old or '(unset)'} -> {resolved.name}"


def _update_compressor(value: str, config: Any, app: Any) -> str:
    """Handle llm.compressor: switch the configured compressor alias."""
    resolved = _resolve_model(value, config)
    promoted = False
    if not resolved:
        resolved = _resolve_discovered_model(value, config, app)
        promoted = resolved is not None
    if not resolved:
        return f"Model '{value}' not found. Available: {_list_available_models_with_discovery(config, app)}"

    old = "default"
    if app and getattr(app, "agent", None):
        old = app.agent._compressor.model_name

        new_compressor = _build_live_provider(config, resolved)
        if not new_compressor:
            return f"Error creating compressor provider for '{resolved.name}'."
        app.agent._compressor = new_compressor

    config.llm.compressor = resolved.name
    config.save()
    logger.info("Compressor: %s -> %s (live)", old, resolved.name)
    suffix = " (live, added to config)" if promoted else " (live)"
    return f"Compressor: {old} -> {resolved.name}{suffix}"


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


def _model_options_help(key: str, config: Any, app: Any = None) -> str:
    """List configured models plus any discovered models for llm.model or llm.compressor."""
    current = ""
    if key == "llm.model":
        current = config.llm.default
    elif key == "llm.compressor":
        current = config.llm.compressor or "(disabled)"

    lines = [f"{key} — current: {current}", ""]
    lines.append("Configured models:")
    for model in config.llm.models:
        marker = " ← current" if model.name == current else ""
        source = f"{model.provider} -> {model.model}"
        lines.append(f"  {model.name} ({source}){marker}")

    discovered = getattr(app, "discovered_models", {}) if app else {}
    if discovered:
        lines.append("")
        lines.append("Discovered models:")
        for provider in config.llm.providers:
            if not provider.allow_model_discovery:
                continue
            models = discovered.get(provider.name, [])
            if not models:
                continue
            lines.append(f"  {provider.name}:")
            for info in models:
                lines.append(f"    {info.id}")

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
