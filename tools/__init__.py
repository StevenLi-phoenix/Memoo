"""Built-in tools with auto-discovery.

Any module in the tools/ directory that defines a `register(registry, **deps)` function
will be automatically discovered and loaded.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

from core.tools import ToolRegistry

logger = logging.getLogger(__name__)


def auto_discover_tools(registry: ToolRegistry, deps: dict[str, Any]) -> None:
    """Auto-discover and register all tools from the tools/ directory.

    Each tool module should define a `register(registry, **deps)` function.
    Dependencies (memory, scheduler, etc.) are passed via deps dict.
    """
    tools_dir = Path(__file__).parent
    for py_file in sorted(tools_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue

        module_name = f"tools.{py_file.stem}"
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "register"):
                module.register(registry, **deps)
                logger.info("Auto-registered tools from %s", module_name)
            else:
                logger.debug("Skipping %s (no register function)", module_name)
        except Exception:
            logger.exception("Failed to load tools from %s", module_name)
