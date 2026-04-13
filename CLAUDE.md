# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Memoo is a lightweight personal AI agent bot built directly on the Claude API `tool_use` capability (with OpenAI fallback). A single `Agent` class orchestrates perception-decision-action-reflection cycles, and all messaging platforms flow through one central `handle_message()` entry point.

**macOS and Linux** — code execution sandboxing uses `sandbox-exec` (macOS, built-in) or `bubblewrap` / `bwrap` (Linux, install via `apt install bubblewrap` or `dnf install bubblewrap`). Backend is auto-detected at startup via `platform.system()`; Windows is not supported.

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
  │   ├── commands.py  Slash command router (/help, /clear, /config, /model, /status)
  │   ├── embeddings.py Pluggable embedding providers (local/openai/off hash-fallback)
  │   ├── dream.py     Periodic memory consolidation: archive → MEMORY.md + USER.md
  │   ├── utils.py     Shared frontmatter parser (used by heartbeat + skills)
  │   ├── scheduler.py Cron-based task scheduler (5-field cron, SQLite persistence)
  │   ├── heartbeat.py Periodic tasks from heartbeat/*.md files (YAML frontmatter)
  │   ├── gateway.py   JSON-over-TCP server with optional mTLS, streaming events
  │   ├── config.py    AppConfig dataclass — mirrors config.yaml, agent-writable with verify
  │   ├── skills.py    Three-level progressive disclosure: L1 metadata → L2 instructions → L3 resources
  │   └── crash.py     Structured crash reports, webhook alerts, autofix queue
  ├── channels/        Platform adapters: Telegram, WeChat (iLink), TUI fallback
  ├── tools/           Auto-discovered: builtins, memory, schedule, config, skills, subagent
  ├── skills/          Modular agent capabilities (SKILL.md + resources per directory)
  ├── systemprompt/    System prompt markdown files
  ├── heartbeat/       Heartbeat task definitions (markdown + YAML frontmatter)
  ├── memory/          Dream output: MEMORY.md, USER.md (injected into system prompt)
  └── data/            SQLite databases (memory.db, schedules.db), model cache
```

## Key Patterns

- **Structured JSON output**: Agent's final response is constrained to `RESPONSE_SCHEMA` via Anthropic `output_config`. Returns `TurnResult` dataclass (response, memory_notes, current_topic, should_compress, did_success, usage). `TurnResult.is_noop` suppresses forwarding to channel when reply is empty (used by heartbeat/scheduler for non-actionable results).
- **Dynamic factory**: Both `models/` and `channels/` use string-keyed registries with `importlib.import_module` lazy loading (`create_provider()`, `create_channel()`)
- **Tool auto-discovery**: Any `tools/*.py` with a `register(registry, **deps)` function is loaded automatically by `tools.auto_discover_tools()`. Dependencies (memory, scheduler, config, skill_registry, app) are injected via the `deps` dict.
- **Parallel tool execution**: When multiple tool calls arrive in one round, they are dispatched with `asyncio.gather`. Single calls skip gather overhead.
- **Message injection**: `Agent.inject(run_id, text)` queues a user message into an active agent turn via `_inboxes`. The message is appended between tool execution and the next LLM call, so the LLM sees the correction without restarting the turn.
- **ContextVar for tool context**: `_tool_context` in `core/tools.py` passes session info (chat_id, sandbox_dir) to tools without argument threading. Tools call `get_context()` to access it.
- **Hybrid RAG memory**: Active `messages` table (max 200, working context) + `archive` table with three-signal ranking: embedding vector similarity (0.5 weight) + FTS5 keyword score (0.3) + importance score (0.2). Falls back to pure FTS5 keyword search when no embeddings are stored. Schema is auto-migrated on `connect()` (safe `ALTER TABLE ADD COLUMN` for `importance`/`embedding`).
- **Embedding providers**: `core/embeddings.py` supports three backends — `local` (any OpenAI-compatible `/v1/embeddings` endpoint: lm-studio, llama.cpp, mlx, Ollama), `openai` (via SDK), `off` (hash-based 256-dim fallback, stable for near-exact matches). Module-level `httpx.AsyncClient` is reused across calls for connection pooling.
- **Context window enforcement**: Two-phase compression in `Agent._enforce_context_window()` — Phase 1 replaces old messages with archived memory summaries, Phase 2 strips middle and summarizes with the compressor LLM (cheapest fallback).
- **Prompt caching** (Anthropic): System prompt and last tool schema get `cache_control: ephemeral`
- **Telegram bind-code auth**: Fail-close allowlist — on startup a one-time bind code (`secrets.token_hex(4)`) is printed; user sends `/bind <code>` in Telegram to self-register. Code rotates after each bind. Bound user IDs persist to `config.yaml` via `on_bind` callback.
- **Per-session sandbox**: `sandbox/{chat_id}/` directories enforced at OS level. On macOS via `sandbox-exec` with a dynamically generated SBPL profile (`(deny default)` baseline); on Linux via `bubblewrap` with namespace isolation (`--unshare-user/pid/ipc/uts/cgroup`, optional `--unshare-net`, workspace bind-mounted at `/workspace`). The `Sandbox` class in `core/sandbox.py` wraps both backends behind a unified API with a smoke test on startup.
- **Sub-agent spawning**: `tools/subagent.py` provides `spawn_agent`, `read_agent_output`, `cancel_agent`, `list_agents`. Supports depth limiting, per-provider model selection, context modes (full/summary/none), readonly/no-network sandbox flags, and background mode with run_id.
- **Dream cycle**: `core/dream.py` runs two-phase LLM consolidation of archived conversations into `memory/MEMORY.md` and `memory/USER.md`. Cursor-based incremental processing ensures each archive entry is analyzed once. When Anthropic provider is detected, both phases are submitted as Batch API requests (50% cost discount) with prompt caching; polls every 10s with 30-min timeout. Falls back to sequential `chat()` when batches are unavailable. Dream output is injected into the system prompt.
- **Skills progressive disclosure**: L1 metadata (name+description) always in system prompt (~100 tokens each). L2 instructions loaded on-demand via `load_skill()` tool. L3 resources loaded via `load_skill_resource()`.
- **Slash commands**: `core/commands.py` routes `/help`, `/clear`, `/config`, `/model`, `/status`, `/new`, `/compact`, etc. directly without LLM call. `/model` supports hot-switching the active model at runtime. `/new` clears memory and resets topic for a fresh session. `/compact` archives old messages (requires ≥6). Unrecognized `/skill-name` falls through to skill trigger in `main.py`.
- **Gateway streaming**: TCP clients receive real-time `tool_start`/`tool_done`/`reply` events as line-delimited JSON. Connection protocol: client must first send `{"auth": "<token>"}` (token read from `.gateway-token`, generated at startup with `secrets.token_urlsafe(32)`, mode 0o600, deleted on shutdown). Server responds `auth_ok` or `auth_fail`. First `chat_id` used is bound to the connection. Supports optional mTLS — generate certs with `python -m core.gateway --generate-certs`. `set_reply_extra()` attaches metadata (usage, did_success) to outbound reply events. Server broadcasts `{"event": "shutdown"}` on stop.
- **Config self-modification**: Agent can update config at runtime via `update_config()` tool. Changes are verified (boot-check) before persisting, with automatic rollback on failure.
- **Crash boundary**: `@crash_boundary("component")` decorator (uses `functools.wraps`) on async handlers captures unhandled exceptions, writes structured JSON reports to `.logs/crashes/`, and queues them for auto-fix.
- **Advisor tool**: Anthropic provider supports consulting a more capable model (e.g., opus) for hard decisions via the `advisor_20260301` beta tool.

## Configuration

`config.yaml` is mirrored by `AppConfig` dataclass in `core/config.py`. Agent can read/modify config at runtime (changes persist back to YAML).

Key sections:
- `llm.default` / `llm.providers` / `llm.fallback` / `llm.model_cache_ttl` — model selection, advisor model, and fallback chain
- `agent.system_prompt` / `agent.max_tool_rounds` (0 = use hard_max_rounds) / `agent.hard_max_rounds` (200) / `agent.context_window_tokens` / `agent.chars_per_token`
- `memory.db_path` / `memory.max_context_messages` / `memory.token_window`
- `embedding.provider` (`local`/`openai`/`off`) / `embedding.base_url` / `embedding.model` / `embedding.api_key`
- `subagent.max_depth` (3) / `subagent.default_max_rounds` (10)
- `channels.telegram` / `channels.wechat` — enable/disable, mode (polling), and `allowed_users` allowlist
- `tools.*` — toggle individual tool capabilities (web_search, run_code, read_file)
- `sandbox.timeout` / `sandbox.max_output`
- `paths.*` — sandbox_dir, heartbeat_dir, skills_dir, memory_dir, logs_dir, certs_dir
- `dream.batch_size` — archive entries per dream cycle
- `hooks.rate_limit_per_minute` / `hooks.rate_limit_window`
- `heartbeat.default_interval` / `scheduler.default_channel`
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
- `test_subagent.py` — depth limiting, model lookup, context modes, context var restore, config round-trip

## Code Style

- Python 3.12+, ruff for linting/formatting, line-length=120
- Type annotations on all function signatures
- Protocol-based abstractions (duck typing, no inheritance hierarchy)
- Frozen dataclasses for DTOs (`ToolCall`, `LLMResponse`, `Message`, `ModelInfo`, `TurnResult`); mutable dataclasses for config
