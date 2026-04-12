"""Dream — periodic memory consolidation inspired by human sleep.

Two-phase LLM process that reviews archived conversations and distills
durable knowledge into human-readable markdown files:

  Phase 1 (Analysis): identify new facts, stale items, user patterns
  Phase 2 (Rewrite):  surgically update MEMORY.md and USER.md

Uses the Anthropic Message Batches API (50% cost discount) with prompt
caching on shared context blocks. Falls back to regular chat() if the
provider doesn't support batches.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from core.config import AppConfig
from core.memory import Memory
from models import LLMProvider, Message

logger = logging.getLogger(__name__)

_PHASE1_SYSTEM = (
    "You are a memory analyst reviewing archived conversation summaries. "
    "Given the conversation history and current memory files, identify:\n"
    "1. **New facts** — concrete information worth remembering (preferences, decisions, names, tools, workflows)\n"
    "2. **Stale items** — things in current memory files that are outdated or contradicted by recent conversations\n"
    "3. **Patterns** — recurring user behaviors, interests, or working styles\n\n"
    "Be concise, specific, and actionable. Do NOT repeat what is already in the memory files unchanged."
)

_PHASE2_SYSTEM = (
    "You are a memory editor. Given an analysis, produce updated versions of the memory files. "
    "Return a JSON object with optional keys: `memory` (string), `user` (string). "
    "Only include keys for files that need changes. "
    "Each value is the complete new content for that file in Markdown.\n\n"
    "Rules:\n"
    "- Keep files concise (under 2000 words each)\n"
    "- Preserve existing structure and sections where possible\n"
    "- Use markdown with clear headings\n"
    "- Be surgical — only change what the analysis warrants\n"
    "- MEMORY.md stores facts, decisions, and context about the world\n"
    "- USER.md stores information about the user: preferences, style, role, knowledge\n"
    "- Return ONLY valid JSON, no markdown fences"
)

# Batch polling: check every 10s, give up after 30 min
_POLL_INTERVAL = 10
_POLL_TIMEOUT = 1800


def _read_cursor(cursor_file: Path) -> int:
    try:
        return int(cursor_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return 0


def _write_cursor(cursor_file: Path, cursor_id: int) -> None:
    cursor_file.write_text(str(cursor_id))


def _read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "(empty)"


# ── Batch API helpers ────────────────────────────────────────────────────────


def _get_anthropic_client(llm: LLMProvider) -> Any | None:
    """Extract the raw Anthropic AsyncClient if the provider supports it."""
    client = getattr(llm, "_client", None)
    if client is None:
        return None
    if not hasattr(client, "messages") or not hasattr(client.messages, "batches"):
        return None
    return client


def _build_batch_request(
    custom_id: str,
    model: str,
    system: str,
    context_block: str,
    user_content: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Build a single batch request with prompt caching on system + context."""
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": max_tokens,
            "system": [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        # Context block (MEMORY.md + USER.md) is shared between Phase 1 and 2
                        # — caching this prefix saves re-reading it on Phase 2
                        {"type": "text", "text": context_block, "cache_control": {"type": "ephemeral"}},
                        {"type": "text", "text": user_content},
                    ],
                },
            ],
        },
    }


async def _submit_and_poll(client: Any, requests: list[dict[str, Any]]) -> dict[str, str]:
    """Submit a batch and poll until completion. Returns {custom_id: text_response}."""
    batch = await client.messages.batches.create(requests=requests)
    batch_id = batch.id
    logger.info("Batch submitted: %s (%d requests)", batch_id, len(requests))

    elapsed = 0
    while elapsed < _POLL_TIMEOUT:
        await asyncio.sleep(_POLL_INTERVAL)
        elapsed += _POLL_INTERVAL

        batch = await client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        counts = batch.request_counts

        logger.debug(
            "Batch %s: status=%s, succeeded=%d, errored=%d, processing=%d",
            batch_id,
            status,
            counts.succeeded,
            counts.errored,
            counts.processing,
        )

        if status == "ended":
            break
    else:
        logger.warning("Batch %s timed out after %ds", batch_id, _POLL_TIMEOUT)
        return {}

    # Collect results
    results: dict[str, str] = {}
    async for entry in await client.messages.batches.results(batch_id):
        cid = entry.custom_id
        if entry.result.type == "succeeded":
            msg = entry.result.message
            text_parts = [b.text for b in msg.content if hasattr(b, "text")]
            results[cid] = "".join(text_parts)

            # Log cache stats
            usage = msg.usage
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
            logger.info(
                "Batch result %s: %d in / %d out (cache read=%d, create=%d)",
                cid,
                usage.input_tokens,
                usage.output_tokens,
                cache_read,
                cache_create,
            )
        else:
            logger.warning("Batch result %s: %s", cid, entry.result.type)

    return results


# ── Main dream logic ─────────────────────────────────────────────────────────


async def run_dream(memory: Memory, llm: LLMProvider, cfg: AppConfig) -> str:
    """Execute the dream cycle. Returns a status message."""
    memory_dir = Path(cfg.paths.memory_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)

    cursor_file = memory_dir / ".dream_cursor"
    memory_file = memory_dir / "MEMORY.md"
    user_file = memory_dir / "USER.md"
    batch_size = cfg.dream.batch_size

    cursor = _read_cursor(cursor_file)
    entries = await memory.get_archive_since(cursor_id=cursor, limit=batch_size)

    if not entries:
        return "Nothing to dream about — no new archived conversations since last dream."

    logger.info("Dream: processing %d archive entries (cursor=%d)", len(entries), cursor)

    # Build shared context (cached across both phases)
    current_memory = _read_file(memory_file)
    current_user = _read_file(user_file)
    context_block = f"## Current MEMORY.md\n{current_memory}\n\n## Current USER.md\n{current_user}"

    # Build history
    history_lines = []
    for e in entries:
        date = e.get("date", "?")
        topic = e.get("topic", "?")
        summary = e.get("summary", "")
        importance = e.get("importance", 0.5)
        history_lines.append(f"- [{date}] (topic: {topic}, importance: {importance:.1f}) {summary}")
    history_text = "\n".join(history_lines)

    # Try batch API, fall back to regular chat
    client = _get_anthropic_client(llm)
    paths = (cursor_file, memory_file, user_file)
    if client:
        model = getattr(llm, "model_name", "") or getattr(llm, "_model", "")
        return await _dream_batch(client, model, context_block, history_text, entries, *paths)

    return await _dream_sequential(llm, context_block, history_text, entries, *paths)


async def _dream_batch(
    client: Any,
    model: str,
    context_block: str,
    history_text: str,
    entries: list[dict[str, Any]],
    cursor_file: Path,
    memory_file: Path,
    user_file: Path,
) -> str:
    """Dream via Anthropic Batch API — 50% cheaper with prompt caching."""
    logger.info("Dream: using Batch API (model=%s)", model)

    # ── Phase 1: Analysis (batch) ──
    phase1_req = _build_batch_request(
        custom_id="dream-phase1",
        model=model,
        system=_PHASE1_SYSTEM,
        context_block=context_block,
        user_content=f"## Archived Conversation History\n{history_text}",
        max_tokens=2000,
    )

    try:
        results = await _submit_and_poll(client, [phase1_req])
    except Exception:
        logger.exception("Dream Batch Phase 1 failed")
        return "Dream failed during batch analysis phase."

    analysis = results.get("dream-phase1", "")
    if not analysis.strip():
        _write_cursor(cursor_file, entries[-1]["id"])
        return "Dream completed but found nothing noteworthy."

    logger.info("Dream Phase 1 (batch): %d chars", len(analysis))

    # ── Phase 2: Rewrite (batch, reuses cached context_block) ──
    phase2_req = _build_batch_request(
        custom_id="dream-phase2",
        model=model,
        system=_PHASE2_SYSTEM,
        context_block=context_block,
        user_content=f"## Analysis\n{analysis}\n\nReturn only valid JSON.",
        max_tokens=4000,
    )

    try:
        results = await _submit_and_poll(client, [phase2_req])
    except Exception:
        logger.exception("Dream Batch Phase 2 failed")
        _write_cursor(cursor_file, entries[-1]["id"])
        return "Dream analysis completed but batch rewrite failed."

    raw = results.get("dream-phase2", "")
    return _apply_dream_results(raw, entries, cursor_file, memory_file, user_file)


async def _dream_sequential(
    llm: LLMProvider,
    context_block: str,
    history_text: str,
    entries: list[dict[str, Any]],
    cursor_file: Path,
    memory_file: Path,
    user_file: Path,
) -> str:
    """Dream via sequential chat() calls — fallback for non-Anthropic providers."""
    logger.info("Dream: using sequential chat (no batch API)")

    # Phase 1
    phase1_content = f"## Archived Conversation History\n{history_text}\n\n{context_block}"
    try:
        phase1_resp = await llm.chat(
            messages=[Message(role="user", content=phase1_content)],
            system=_PHASE1_SYSTEM,
            max_tokens=2000,
        )
        analysis = phase1_resp.text or ""
    except Exception:
        logger.exception("Dream Phase 1 failed")
        return "Dream failed during analysis phase."

    if not analysis.strip():
        _write_cursor(cursor_file, entries[-1]["id"])
        return "Dream completed but found nothing noteworthy."

    logger.info("Dream Phase 1: %d chars", len(analysis))

    # Phase 2
    phase2_content = f"## Analysis\n{analysis}\n\n{context_block}\n\nReturn only valid JSON."
    try:
        phase2_resp = await llm.chat(
            messages=[Message(role="user", content=phase2_content)],
            system=_PHASE2_SYSTEM,
            max_tokens=4000,
        )
        raw = phase2_resp.text or ""
    except Exception:
        logger.exception("Dream Phase 2 failed")
        _write_cursor(cursor_file, entries[-1]["id"])
        return "Dream analysis completed but memory rewrite failed."

    return _apply_dream_results(raw, entries, cursor_file, memory_file, user_file)


# ── Shared result handling ───────────────────────────────────────────────────


def _apply_dream_results(
    raw: str,
    entries: list[dict[str, Any]],
    cursor_file: Path,
    memory_file: Path,
    user_file: Path,
) -> str:
    """Parse Phase 2 JSON and write updated memory files."""
    updates = _parse_json(raw)
    if not updates:
        logger.warning("Dream Phase 2: no valid JSON in response")
        _write_cursor(cursor_file, entries[-1]["id"])
        return "Dream completed but produced no memory updates."

    written: list[str] = []
    if "memory" in updates and updates["memory"]:
        memory_file.write_text(updates["memory"], encoding="utf-8")
        written.append("MEMORY.md")
        logger.info("Dream: updated MEMORY.md (%d chars)", len(updates["memory"]))

    if "user" in updates and updates["user"]:
        user_file.write_text(updates["user"], encoding="utf-8")
        written.append("USER.md")
        logger.info("Dream: updated USER.md (%d chars)", len(updates["user"]))

    new_cursor = entries[-1]["id"]
    old_cursor = _read_cursor(cursor_file)
    _write_cursor(cursor_file, new_cursor)
    logger.info("Dream complete: cursor %d -> %d, updated %s", old_cursor, new_cursor, written)

    if written:
        return f"Dream complete — updated {', '.join(written)} from {len(entries)} archived conversations."
    return f"Dream complete — reviewed {len(entries)} conversations, no changes needed."


def _parse_json(text: str) -> dict[str, Any] | None:
    """Extract first JSON object from LLM response.

    Handles common LLM wrapping patterns:
    1. Pure JSON
    2. Markdown fenced code blocks (```json ... ```)
    3. Leading prose before the JSON object

    The brace-depth scanner correctly skips braces inside JSON string
    literals (including escaped quotes).
    """
    if not text:
        return None

    # Attempt 1: raw text is valid JSON
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Attempt 2: strip markdown fences (```json ... ``` or ``` ... ```)
    import re

    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1).strip())
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, TypeError):
            pass

    # Attempt 3: find first { and scan with string-aware brace counting
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            if in_string:
                escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    # This slice wasn't valid — keep scanning for another object
                    start = text.find("{", i + 1)
                    if start == -1:
                        return None
                    depth = 0
                    # Reset loop — we'll pick up at the new start on next iteration
                    # by adjusting i; but since we can't easily reset a range(),
                    # use a recursive call for simplicity
                    return _parse_json(text[i + 1 :])
    return None
