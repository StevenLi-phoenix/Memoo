# Memoo — Situation Report

Context snapshot for the next session. Reads top-to-bottom; no imperative checklist.

## The problem under investigation

Anthropic API (`claude-opus-4-6`, SDK `anthropic==0.94.0`) returns `stop_reason=end_turn` with a completely empty content list (`r.content == []`, `output_tokens=7`) on certain agent turns. Runtime symptom: user types a short/ambiguous message (e.g. "test running all tools"), agent log shows `Agent done in 1 rounds. topic=, success=True, noop=True`, and the TUI prints nothing. It happens intermittently — some turns work, some don't.

When this happens, `_parse_structured_response` sees no content, the agent emits an empty `TurnResult`, and `TurnResult.is_noop` suppresses forwarding to the channel. The user sees silence.

## The layers involved

`models/anthropic.py:chat()` is the entry point. Relevant bits of its current behavior:

- Sends `output_config={"format": {"type": "json_schema", "schema": output_schema}}` whenever the caller passes `output_schema`. `output_schema` is `RESPONSE_SCHEMA` from `core/agent.py`, which is a five-field object (`reply`, `memory_notes`, `current_topic`, `should_compress`, `did_success`, all required, `additionalProperties: false`).
- Auto-injects two server-side tools when enabled: `web_search_20250305` and `advisor_20260301`. These are always appended to the caller's `tools` list inside `chat()`.
- Real-world agent turns carry ~15 custom function tools (from `tools/builtins.py`, memory tools, config tools, skills, subagent) plus both server tools plus `output_config` — the full stack.

Tool-free callers exist and rely on `output_config` too: `core/dream.py` (dream-cycle consolidation into `memory/MEMORY.md` + `memory/USER.md`), `core/agent.py:_enforce_context_window` Phase 2 summarizer, and the cheapest-fallback compressor. These callers pass `tools=None` and need structured JSON to parse back into dataclasses. Global removal of `output_config` is therefore a non-starter — it would break compression and dream.

## What the docs say

Checked `platform.claude.com/docs/en/docs/build-with-claude/structured-outputs` directly. Relevant quotes:

- "You can use these features independently or **together in the same request**."
- Dedicated section "Using both features together" with example code pairing JSON outputs + `strict: true` tool use.
- Example comment: "Claude may call the tool first (`tool_use`) or respond with JSON (`text`)" — confirms the expected stop-reason branching.
- "Grammar state resets between sections, allowing Claude to think freely while still producing structured output in the final response."
- Migration note: `output_format` → `output_config.format`, beta headers no longer required. Our SDK 0.94.0 has `output_config` as a first-class param (verified via `inspect.signature`), so we're on the current, non-beta API shape. No stale beta headers in our code.

So per the docs, this combination is **supported and documented**. Which makes the empty-content symptom either a regression or an undocumented edge case.

## What I ran

Isolation test at `/tmp/memoo_output_config_repro.py`. Same system prompt, same user message ("test running all tools"), same `RESPONSE_SCHEMA`, only tool configuration varies. Model: `claude-opus-4-6`.

| # | Tools sent | `output_config` | Result |
|---|---|---|---|
| A | none | yes | **works** — 376 out tokens, one text block with valid JSON |
| B | 1 non-strict custom tool (`run_code`) | yes | **empty** — `stop=end_turn`, 6 out tokens, `content=[]` |
| C | `web_search` only | yes | works — 60 tokens, text block |
| D | `advisor` only | yes | works — 163 tokens, `server_tool_use` + `advisor_tool_result` + text |
| E | custom + `web_search` | yes | **empty** — 6 tokens, `content=[]` |
| F | custom + `web_search` + `advisor` | yes | **empty** — 6 tokens, `content=[]` |
| G | custom + `web_search` + `advisor` | no | works — normal `stop=tool_use`, 5 blocks |
| H | 1 **strict** custom tool | yes | **works** — 66 tokens, text block |
| I | strict custom + `web_search` + `advisor` | yes | **empty** — 6 tokens, `content=[]` |

## What the matrix tells us

1. **Non-strict custom tool + `output_config` is broken on `claude-opus-4-6`** — case B reproduces with a single plain function tool. No server tools needed to trigger it. This is a real Anthropic bug and worth reporting, but we can't wait for it.
2. **`strict: true` rescues the pure custom-tool case** (H works). This aligns with what the docs promote: strict tools are the pairing they describe. BUT —
3. **`strict: true` does NOT rescue custom-tool + server-tool** — case I still returns empty. So `strict` alone is not a full fix.
4. **Server tools alone (C, D) + `output_config` work.** The bug is specifically about custom function tools interacting with `output_config`, with server tools further constraining the "strict rescue" path.

Reliable working configurations with `output_config`, as of today:
- `tools=None` (A)
- `tools=[one server tool]` (C, D)
- `tools=[one strict custom tool, no server tools]` (H)

**None of those match Memoo's agent turn**, which is ~15 custom tools + `web_search` + `advisor`. There is no combination that keeps all of them AND `output_config`.

The "6 output tokens → empty content list" fingerprint is unusual. Something is being generated but not surfaced in any block — possibly the grammar compiles in a way that allows early termination, or the stream yields a block that's then dropped. Worth mentioning in the bug report but not something we can fix client-side.

## Constraints I was given

These shaped the fix space:

- **`output_config` cannot be removed globally.** Dream / compression paths depend on structured JSON from tool-free calls. Confirmed working (case A).
- **An earlier attempt** at "apply `output_config` only when `tools is None`, otherwise drop `web_search`/`advisor` auto-injection too" was explicitly rejected by the user as an incorrect change. `models/anthropic.py` was reverted to its pre-session state.
- **Strengthening the schema description** (e.g. adding "must be substantive and non-empty…") does not change API-level behavior — Claude still emits an empty content list when the broken combo is present. Already tried.
- **Removing the NO_OP hint** from both schema and system prompt also did not affect the empty-content case. Already tried.

## The fix path I'm leaning toward (not yet applied)

Option 1 from the original handoff, now evidence-backed: in `models/anthropic.py:chat()`, drop `output_config` **iff `tools` contains any non-server tool**. Specifically, treat `web_search_20250305` and `advisor_20260301` as server tools and check if any remaining tool lacks a matching `type`. Keep `output_config` for:

- Tool-free callers (dream, compressor, phase-2 summarizer) — cases A-equivalent, confirmed working.
- Hypothetical server-tool-only callers — also confirmed working.

Drop `output_config` for:
- Real agent turns carrying custom function tools — fall through to `_parse_structured_response`'s existing plain-text JSON path, which already ships and is covered by tests in `tests/test_agent.py`.

Why this is the lowest-risk option:
- It isolates the change to one file (`models/anthropic.py`).
- It keeps all dream/compression behavior untouched.
- The fallback parser already exists — this is not new infrastructure.
- It's reversible the moment Anthropic fixes the bug: remove the conditional and structured output comes back for agent turns.

Alternatives I considered and rejected:

- **Finalize-via-tool pattern** (define a virtual `respond(...)` tool with `tool_choice={"type": "tool", "name": "respond"}`) — higher blast radius, requires intercepting a tool call in the agent loop, more moving parts.
- **Two-phase turn** (run loop without `output_config`, then make a second tool-free call with `output_config` to extract structured fields) — doubles the agent-turn cost in tokens + latency just to re-extract fields the text response already contains.
- **Global removal of `output_config`** — breaks dream/compression, was the hard constraint.
- **Wait for Anthropic** — not acceptable; silent agent turns in production.

## Things worth verifying before committing to Option 1

- Does the bug also reproduce on `claude-sonnet-4-6` and `claude-haiku-4-5-20251001`? If sonnet works, the conditional could be model-specific rather than provider-wide.
- Does `_parse_structured_response` handle all the shapes `claude-opus-4-6` emits in plain-text mode? Need to test: (a) bare JSON, (b) JSON wrapped in prose like `Here you go:\n\n{...}`, (c) markdown-fenced JSON (` ```json `). If any of these fail, the fallback parser needs a small upgrade before the conditional lands.

## Related but separate work in the tree

There are ~12 files with uncommitted changes unrelated to the Anthropic bug. They were reviewed and tested this session (161 pytest passing, ruff clean), and should ideally land in a commit **before** the anthropic patch so the risky change is isolated:

- `core/sandbox.py`, `core/config.py`, `config.yaml`, `tools/builtins.py` — config-driven env injection for sandboxed CLIs (`env` / `env_from_cmd` / `env_passthrough`). `gh repo list` works under SBPL with `GH_TOKEN` extracted in parent via `gh auth token`. Ripped out all hardcoded gh/git logic from `sandbox.py`.
- `core/tools.py:156` — tool executor now validates required args, catches `TypeError`, returns structured error with expected schema. Stops gemma's empty-arg tool-call hallucinations.
- `tui.py` — `ToolTracker._active` stores `(start_time, args)` tuple, `_format_tool_args` extracts `code=…` for `run_code`, fallback `(empty reply)` line so blank turns have visible feedback.
- `core/commands.py`, `main.py`, `tools/config_tools.py` — five `sorted(..., key=x.created)` sites patched with `or 0` because LM Studio returns `created=None`.
- `tools/memory_tools.py` — unified `write_memory(content, topic)` replaces split-brain `remember`/`list_notes` (which wrote to `memory/NOTES.md`). Now `write_memory` → `archive_messages` → read via `list_memories`/`search_memory`/`read_memory`. One source of truth.
- `systemprompt/default.md` — anti-refusal rewrite with concrete examples (`gh repo list`, `brew list`, `ps aux`, `curl ifconfig.me`). "NEVER refuse based on content" rule for `write_memory`, "reply must be non-empty for direct user messages" rule.
- `models/openai.py` — `base_url` threaded through config; `max_tokens` default dropped 128000 → 4096; `response_format` skipped when tools are present (llama.cpp rejects lazy-grammar + structured output combo).
- `main.py`, `core/agent.py` — `_finalize_turn` now uses `memory.compact_replace(chat_id, ctx["_messages"] + [final_msg])` instead of `add_message(final_msg)`. Persisted message list now exactly matches what the LLM saw. Motivation: MLX KV-cache cannot trim mid-prefix on LM Studio; a stable prefix keeps the cache hot turn-to-turn. Not yet verified under real LM Studio load — worth a sanity check after the anthropic fix lands.
- `core/agent.py:RESPONSE_SCHEMA` — `reply` field description rewritten so Claude doesn't treat "empty string if nothing to say" as a lure to silently noop. Doesn't fix the API-level empty-content bug (already tested) but is still the correct description for the fallback path.

## Key files and line pointers

- `models/anthropic.py` — the file to patch. Currently reverted to pre-session state (`git diff models/anthropic.py` is empty). Lines ~85–108 build the tools list and set `output_config`. `chat()` signature is `(messages, system, tools, max_tokens, output_schema)`.
- `core/agent.py:26-58` — `RESPONSE_SCHEMA`.
- `core/agent.py:~407` — `_parse_structured_response` plain-text fallback path (used via `TurnResult.fallback`).
- `core/agent.py:~349` — `_enforce_context_window` Phase 2 summarizer. Passes `tools=None`, relies on `output_config`.
- `core/agent.py:~420` — `_chat_with_fallback` passes tools + `output_schema=RESPONSE_SCHEMA` to `llm.chat()`.
- `core/dream.py` — dream cycle, `tools=None`, relies on `output_config` for structured consolidation output.
- `tests/test_agent.py` — existing tests for `TurnResult` parsing, `RESPONSE_SCHEMA` validation, importance scoring. Good base for a regression test once the conditional lands.
- `/tmp/memoo_output_config_repro.py` — the isolation test. Still on disk; can be re-run against any model / SDK version.

## Memory rule worth respecting

User's saved rule (in persistent memory, file `feedback_config_yaml_parity.md`): **any `config.yaml` edit must also update `core/config.py`** (dataclass + `load()` + `to_dict()`) or saves silently drop the field. Already respected for this session's `sandbox` section additions — flagging so the next session doesn't regress it.

## Current git state as of writing

Branch `main`. Working tree has modifications in: `config.yaml`, `core/agent.py`, `core/commands.py`, `core/config.py`, `core/sandbox.py`, `core/tools.py`, `main.py`, `models/openai.py`, `systemprompt/default.md`, `tools/builtins.py`, `tools/config_tools.py`, `tools/memory_tools.py`, `tui.py`. `models/anthropic.py` is clean. `pytest -x -q` passes 161 tests. `ruff check` is clean.
