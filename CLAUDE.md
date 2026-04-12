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
  │   ├── memory.py    Two-tier SQLite: active messages + hybrid RAG archive
  │   ├── tools.py     ToolRegistry with @registry.tool auto-schema from docstrings
  │   ├── hooks.py     Pre-execution guards (sandbox escape, rate limiting)
  │   ├── commands.py  Slash command router (/help, /clear, /config, /model, /status, etc.)
  │   ├── embeddings.py Pluggable embedding providers (local/openai/off hash-fallback)
  │   ├── utils.py     Shared utilities (parse_frontmatter for heartbeat + skills)
  │   ├── scheduler.py Cron-based task scheduler (5-field cron, SQLite persistence)
  │   ├── heartbeat.py Periodic tasks from heartbeat/*.md files (YAML frontmatter)
  │   ├── gateway.py   JSON-over-TCP server with mTLS, streaming events
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

- **Structured JSON output**: Agent's final response is constrained to `RESPONSE_SCHEMA` via Anthropic `output_config`. Returns `TurnResult` dataclass (response, memory_notes, current_topic, should_compress, did_success, usage). `TurnResult.is_noop` suppresses forwarding to channel when reply is empty (used by heartbeat/scheduler for non-actionable results).
- **Dynamic factory**: Both `models/` and `channels/` use string-keyed registries with `importlib.import_module` lazy loading (`create_provider()`, `create_channel()`)
- **Tool auto-discovery**: Any `tools/*.py` with a `register(registry, **deps)` function is loaded automatically by `tools.auto_discover_tools()`. Dependencies (memory, scheduler, config, skill_registry, app) are injected via the `deps` dict.
- **Parallel tool execution**: When multiple tool calls arrive in one round, they are dispatched with `asyncio.gather`.
- **ContextVar for tool context**: `_tool_context` in `core/tools.py` passes session info (chat_id, sandbox_dir) to tools without argument threading. Tools call `get_context()` to access it.
- **Hybrid RAG memory**: Active `messages` table (max 200, working context) + `archive` table with three-signal ranking: embedding vector similarity (0.5 weight) + FTS5 keyword score (0.3) + importance score (0.2). Falls back to pure FTS5 keyword search when no embeddings are stored.
- **Embedding providers**: `core/embeddings.py` supports three backends — `local` (any OpenAI-compatible `/v1/embeddings` endpoint: lm-studio, llama.cpp, mlx, Ollama), `openai` (via SDK), `off` (hash-based 256-dim fallback, stable for near-exact matches). Configured at startup via `AppConfig.embedding`.
- **Context window enforcement**: Two-phase compression in `Agent._enforce_context_window()` — Phase 1 replaces old messages with archived memory summaries, Phase 2 strips middle and summarizes with the compressor LLM (cheapest fallback).
- **Prompt caching** (Anthropic): System prompt and last tool schema get `cache_control: ephemeral`
- **Per-session sandbox**: `sandbox/{chat_id}/` directories enforced at OS level by macOS `sandbox-exec` with a custom SBPL profile
- **Skills progressive disclosure**: L1 metadata (name+description) always in system prompt (~100 tokens each). L2 instructions loaded on-demand via `load_skill()` tool. L3 resources loaded via `load_skill_resource()`.
- **Slash commands**: `core/commands.py` routes `/help`, `/clear`, `/config`, `/model`, `/status`, etc. directly without LLM call. `/model` supports hot-switching the active model at runtime. `/status` reports per-session token counts (input, output, cache read/create, total runs via `Agent.total_tokens`). Unrecognized `/skill-name` falls through to skill trigger in `main.py`.
- **Gateway streaming**: TCP clients receive real-time `tool_start`/`tool_done`/`reply` events as line-delimited JSON. Supports mTLS — generate certs with `python -m core.gateway --generate-certs`. `set_reply_extra()` attaches metadata (usage, did_success) to outbound reply events.
- **Config self-modification**: Agent can update config at runtime via `update_config()` tool. Changes are verified (boot-check) before persisting, with automatic rollback on failure.
- **Crash boundary**: `@crash_boundary("component")` decorator on async handlers captures unhandled exceptions, writes structured JSON reports to `.logs/crashes/`, and queues them for auto-fix.
- **Advisor tool**: Anthropic provider supports consulting a more capable model (e.g., opus) for hard decisions via the `advisor_20260301` beta tool.

## Configuration

`config.yaml` is mirrored by `AppConfig` dataclass in `core/config.py`. Agent can read/modify config at runtime (changes persist back to YAML).

Key sections:
- `llm.default` / `llm.providers` / `llm.fallback` — model selection, advisor model, and fallback chain
- `agent.system_prompt` / `agent.max_tool_rounds` (0 = unlimited, hard cap at 200)
- `memory.db_path` / `memory.max_context_messages` / `memory.token_window`
- `embedding.provider` (`local`/`openai`/`off`) / `embedding.base_url` / `embedding.model` / `embedding.api_key`
- `channels.telegram` / `channels.wechat` — enable/disable and mode (polling)
- `tools.*` — toggle individual tool capabilities (web_search, run_code, read_file)
- `sandbox.timeout` / `sandbox.max_output`
- `host` / `port` — Gateway TCP server bind address

API keys loaded from `.env` via `python-dotenv`: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `TELEGRAM_BOT_TOKEN`, `WECHAT_ILINK_TOKEN`, `EMBEDDING_API_KEY` (optional, for openai embedding provider).

## Adding New Components

**New tool**: Create `tools/my_tool.py` with a `register(registry, **deps)` function. Use `@registry.tool` decorator — schema is auto-generated from type hints and Google-style docstrings. Available deps: `memory`, `scheduler`, `config`, `skill_registry`, `sandbox_dir`, `app`.

**New channel**: Create `channels/my_channel.py` implementing the `Channel` Protocol (`start`, `send`, `stop`). Register in `_CHANNEL_REGISTRY` in `channels/__init__.py`.

**New LLM provider**: Create `models/my_provider.py` implementing the `LLMProvider` Protocol (and optionally `DiscoverableProvider` for model discovery). Register in `_PROVIDER_REGISTRY` in `models/__init__.py`.

**New heartbeat task**: Create `heartbeat/my_task.md` with YAML frontmatter (`name`, `interval` in seconds, `enabled`).

**New skill**: Create `skills/my_skill/SKILL.md` with YAML frontmatter (`name`, `description`). Optionally add resource files in the same directory. Skills are auto-discovered at startup.

## Test Coverage

Tests in `tests/` — all async (asyncio_mode=auto):
- `test_agent.py` — TurnResult parsing, RESPONSE_SCHEMA validation, importance scoring
- `test_commands.py` — slash command routing, case insensitivity, unknown command suggestion
- `test_embeddings.py` — local embed shape/normalization, cosine similarity, serialization round-trip
- `test_memory.py` — chat_id isolation, archive isolation, FTS5 injection safety
- `test_sandbox.py` — Python/bash execution, path traversal, session isolation, timeout
- `test_cron.py` — cron field matching (wildcards, steps, ranges, lists, malformed input)

## Code Style

- Python 3.12+, ruff for linting/formatting, line-length=120
- Type annotations on all function signatures
- Protocol-based abstractions (duck typing, no inheritance hierarchy)
- Frozen dataclasses for DTOs (`ToolCall`, `LLMResponse`, `Message`, `ModelInfo`, `TurnResult`); mutable dataclasses for config
