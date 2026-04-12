"""Dream — periodic memory consolidation inspired by human sleep.

Two-phase LLM process that reviews archived conversations and distills
durable knowledge into human-readable markdown files:

  Phase 1 (Analysis): identify new facts, stale items, user patterns
  Phase 2 (Rewrite):  surgically update MEMORY.md and USER.md

Cursor-based incremental processing ensures each archive entry is
analyzed at most once. Dream output is injected into the system prompt
so the agent carries consolidated knowledge across sessions.
"""

from __future__ import annotations

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


def _read_cursor(cursor_file: Path) -> int:
    """Read the last processed archive ID."""
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

    # Build history context
    history_lines = []
    for e in entries:
        date = e.get("date", "?")
        topic = e.get("topic", "?")
        summary = e.get("summary", "")
        importance = e.get("importance", 0.5)
        history_lines.append(f"- [{date}] (topic: {topic}, importance: {importance:.1f}) {summary}")
    history_text = "\n".join(history_lines)

    # Current memory state
    current_memory = _read_file(memory_file)
    current_user = _read_file(user_file)

    context_block = f"## Current MEMORY.md\n{current_memory}\n\n## Current USER.md\n{current_user}"

    # --- Phase 1: Analysis ---
    logger.info("Dream Phase 1: analyzing %d entries", len(entries))
    try:
        phase1_resp = await llm.chat(
            messages=[
                Message(
                    role="user",
                    content=f"## Archived Conversation History\n{history_text}\n\n{context_block}",
                ),
            ],
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

    logger.info("Dream Phase 1 analysis: %d chars", len(analysis))

    # --- Phase 2: Rewrite ---
    logger.info("Dream Phase 2: rewriting memory files")
    try:
        phase2_resp = await llm.chat(
            messages=[
                Message(
                    role="user",
                    content=f"## Analysis\n{analysis}\n\n{context_block}\n\nReturn only valid JSON.",
                ),
            ],
            system=_PHASE2_SYSTEM,
            max_tokens=4000,
        )
        raw = phase2_resp.text or ""
    except Exception:
        logger.exception("Dream Phase 2 failed")
        _write_cursor(cursor_file, entries[-1]["id"])
        return "Dream analysis completed but memory rewrite failed."

    # Parse JSON from response
    updates = _parse_json(raw)
    if not updates:
        logger.warning("Dream Phase 2: no valid JSON in response")
        _write_cursor(cursor_file, entries[-1]["id"])
        return "Dream completed but produced no memory updates."

    # Write updated files
    written: list[str] = []
    if "memory" in updates and updates["memory"]:
        memory_file.write_text(updates["memory"], encoding="utf-8")
        written.append("MEMORY.md")
        logger.info("Dream: updated MEMORY.md (%d chars)", len(updates["memory"]))

    if "user" in updates and updates["user"]:
        user_file.write_text(updates["user"], encoding="utf-8")
        written.append("USER.md")
        logger.info("Dream: updated USER.md (%d chars)", len(updates["user"]))

    # Advance cursor
    new_cursor = entries[-1]["id"]
    _write_cursor(cursor_file, new_cursor)
    logger.info("Dream complete: cursor %d -> %d, updated %s", cursor, new_cursor, written)

    if written:
        return f"Dream complete — updated {', '.join(written)} from {len(entries)} archived conversations."
    return f"Dream complete — reviewed {len(entries)} conversations, no changes needed."


def _parse_json(text: str) -> dict[str, Any] | None:
    """Extract first JSON object from LLM response."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None
