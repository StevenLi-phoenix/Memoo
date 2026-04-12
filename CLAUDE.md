# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Memoo is a lightweight personal AI agent bot built directly on the Claude API `tool_use` capability (with OpenAI fallback). A single `Agent` class orchestrates perception-decision-action-reflection cycles, and all messaging platforms flow through one central `handle_message()` entry point.

**macOS only** — code execution sandboxing requires `sandbox-exec` (built into macOS).

## Commands

```bash
# Install
uv sync              # runtime deps
uv sync --extra dev  # + dev deps (pytest, ruff)

# Run
source .venv/bin/activate && python main.py

# Lint & Format
ruff check .
ruff format .

# Test
pytest                          # all tests
pytest tests/test_foo.py        # single file
pytest tests/test_foo.py::test_bar  # single test
# asyncio_mode=auto is preconfigured in pyproject.toml
# ruff excludes skills/ directory (see pyproject.toml)
```

## Architecture

```
main.py (Memoo)  ─── orchestrates everything
  ├── models/          LLM providers (Anthropic, OpenAI) via Protocol + factory
  ├── core/
  │   ├── agent.py     Agentic loop: LLM → tool calls → hooks → execute → repeat
  │   ├── memory.py    Two-tier SQLite: active messages + FTS5-indexed archive
  │   ├── tools.py     ToolRegistry with @registry.tool auto-schema from docstrings
  │   ├── hooks.py     Pre-execution guards (sandbox escape, rate limiting)
  │   ├── scheduler.py Cron-based task scheduler (5-field cron, SQLite persistence)
  │   ├── heartbeat.py Periodic tasks from heartbeat/*.md files (YAML frontmatter)
  │   ├── gateway.py   JSON-over-TCP server for TUI/external clients with streaming events
  │   ├── config.py    AppConfig dataclass — mirrors config.yaml, agent-writable with verify
  │   ├── skills.py    Three-level progressive disclosure: L1 metadata → L2 instructions → L3 resources
  │   └── crash.py     Structured crash reports, webhook alerts, autofix queue
  ├── channels/        Platform adapters: Telegram, WeChat (iLink), TUI fallback
  ├── tools/           Auto-discovered tool modules (builtins, memory, schedule, config, skills)
  ├── skills/          Modular agent capabilities (SKILL.md + resources per directory)
  ├── systemprompt/    System prompt markdown files
  ├── heartbeat/       Heartbeat task definitions (markdown + YAML frontmatter)
  └── data/            SQLite databases (memory.db, schedules.db), model cache
```

## Key Patterns

- **Structured JSON output**: Agent's final response is constrained to `RESPONSE_SCHEMA` via Anthropic `output_config` — reply, memory_notes, current_topic, should_compress, did_success. Eliminates extra compression-decision LLM calls. See `core/agent.py:RESPONSE_SCHEMA`.
- **NO_OP suppression**: When `reply` is empty string, the response is not forwarded to the channel. Used by automated sources (heartbeat, scheduler) when there's nothing actionable.
- **Dynamic factory**: Both `models/` and `channels/` use string-keyed registries with `importlib.import_module` lazy loading (`create_provider()`, `create_channel()`)
- **Tool auto-discovery**: Any `tools/*.py` with a `register(registry, **deps)` function is loaded automatically by `tools.auto_discover_tools()`. Dependencies (memory, scheduler, config, skill_registry) are injected via the `deps` dict.
- **ContextVar for tool context**: `_tool_context` in `core/tools.py` passes session info (chat_id, sandbox_dir) to tools without argument threading. Tools call `get_context()` to access it.
- **Two-tier memory**: Active `messages` table (max 200, working context) + `archive` table with FTS5 full-text search for RAG retrieval of compacted history
- **Context window enforcement**: Two-phase compression in `Agent._enforce_context_window()` — Phase 1 replaces old messages with archived memory summaries, Phase 2 strips middle and summarizes with the compressor LLM (cheapest fallback).
- **Prompt caching** (Anthropic): System prompt and last tool schema get `cache_control: ephemeral`
- **Per-session sandbox**: `sandbox/{chat_id}/` directories enforced at OS level by macOS `sandbox-exec` with a custom SBPL profile
- **Skills progressive disclosure**: L1 metadata (name+description) always in system prompt (~100 tokens each). L2 instructions loaded on-demand via `load_skill()` tool. L3 resources loaded via `load_skill_resource()`.
- **Gateway streaming**: TCP clients receive real-time `tool_start`/`tool_done`/`reply` events as line-delimited JSON during agent execution.
- **Config self-modification**: Agent can update config at runtime via `update_config()` tool. Changes are verified (boot-check) before persisting, with automatic rollback on failure.
- **Crash boundary**: `@crash_boundary("component")` decorator on async handlers captures unhandled exceptions, writes structured JSON reports to `.logs/crashes/`, and queues them for auto-fix.
- **Advisor tool**: Anthropic provider supports consulting a more capable model (e.g., opus) for hard decisions via the `advisor_20260301` beta tool.

## Configuration

`config.yaml` is mirrored by `AppConfig` dataclass in `core/config.py`. Agent can read/modify config at runtime (changes persist back to YAML).

Key sections:
- `llm.default` / `llm.providers` / `llm.fallback` — model selection, advisor model, and fallback chain
- `agent.system_prompt` / `agent.max_tool_rounds` (0 = unlimited, hard cap at 200)
- `memory.db_path` / `memory.max_context_messages` / `memory.token_window`
- `channels.telegram` / `channels.wechat` — enable/disable and mode (polling)
- `tools.*` — toggle individual tool capabilities (web_search, run_code, read_file)
- `sandbox.timeout` / `sandbox.max_output`
- `host` / `port` — Gateway TCP server bind address

API keys loaded from `.env` via `python-dotenv`: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `TELEGRAM_BOT_TOKEN`, `WECHAT_ILINK_TOKEN`.

## Adding New Components

**New tool**: Create `tools/my_tool.py` with a `register(registry, **deps)` function. Use `@registry.tool` decorator — schema is auto-generated from type hints and Google-style docstrings. Available deps: `memory`, `scheduler`, `config`, `skill_registry`, `sandbox_dir`.

**New channel**: Create `channels/my_channel.py` implementing the `Channel` Protocol (`start`, `send`, `stop`). Register in `_CHANNEL_REGISTRY` in `channels/__init__.py`.

**New LLM provider**: Create `models/my_provider.py` implementing the `LLMProvider` Protocol (and optionally `DiscoverableProvider` for model discovery). Register in `_PROVIDER_REGISTRY` in `models/__init__.py`.

**New heartbeat task**: Create `heartbeat/my_task.md` with YAML frontmatter (`name`, `interval` in seconds, `enabled`).

**New skill**: Create `skills/my_skill/SKILL.md` with YAML frontmatter (`name`, `description`). Optionally add resource files in the same directory. Skills are auto-discovered at startup.

## Code Style

- Python 3.12+, ruff for linting/formatting, line-length=120
- Type annotations on all function signatures
- Protocol-based abstractions (duck typing, no inheritance hierarchy)
- Frozen dataclasses for DTOs (`ToolCall`, `LLMResponse`, `Message`, `ModelInfo`); mutable dataclasses for config
