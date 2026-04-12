"""Tool registry with decorator-based registration.

Usage:
    from core.tools import ToolRegistry, tool

    registry = ToolRegistry()

    @registry.tool
    async def web_search(query: str) -> str:
        '''Search the web for information.'''
        ...
"""

from __future__ import annotations

import inspect
import json
import logging
from contextvars import ContextVar
from typing import Any, Callable, get_type_hints

logger = logging.getLogger(__name__)

# Execution context — set by agent before tool calls, readable by tools via get_context()
_tool_context: ContextVar[dict[str, Any]] = ContextVar("tool_context", default={})


def set_context(ctx: dict[str, Any]) -> None:
    """Set the current tool execution context (called by agent)."""
    _tool_context.set(ctx)


def get_context() -> dict[str, Any]:
    """Get the current tool execution context (called by tools that need session info)."""
    return _tool_context.get()


# Python type → JSON Schema type mapping
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(py_type: type) -> dict[str, Any]:
    """Convert a Python type annotation to JSON Schema."""
    origin = getattr(py_type, "__origin__", None)

    if origin is list:
        args = getattr(py_type, "__args__", (Any,))
        return {"type": "array", "items": _python_type_to_json_schema(args[0])}

    if origin is dict:
        return {"type": "object"}

    # Union types (e.g. str | None)
    if origin is type(str | None):
        args = [a for a in py_type.__args__ if a is not type(None)]
        if len(args) == 1:
            return _python_type_to_json_schema(args[0])

    return {"type": _TYPE_MAP.get(py_type, "string")}


def _build_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Build Anthropic-style tool schema from function signature + docstring."""
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    doc = inspect.getdoc(func) or ""

    # Parse docstring for param descriptions (Google style)
    param_docs: dict[str, str] = {}
    in_args = False
    for line in doc.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            in_args = True
            continue
        if in_args:
            if stripped and not stripped.startswith("returns:"):
                # "param_name: description" or "param_name (type): description"
                if ":" in stripped:
                    pname, pdesc = stripped.split(":", 1)
                    pname = pname.strip().split("(")[0].strip()
                    param_docs[pname] = pdesc.strip()
            else:
                in_args = False

    # First line of docstring as description
    description = doc.split("\n")[0].strip() if doc else func.__name__

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        py_type = hints.get(name, str)
        prop = _python_type_to_json_schema(py_type)

        if name in param_docs:
            prop["description"] = param_docs[name]

        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "name": func.__name__,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


class ToolRegistry:
    """Registry for agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}
        self._schemas: dict[str, dict[str, Any]] = {}

    def tool(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator to register a function as an agent tool."""
        schema = _build_schema(func)
        name = func.__name__
        self._tools[name] = func
        self._schemas[name] = schema
        logger.info("Registered tool: %s", name)
        return func

    def get_schemas(self) -> list[dict[str, Any]]:
        """Get all tool schemas for LLM API calls."""
        return list(self._schemas.values())

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name and return the result as string."""
        if name not in self._tools:
            error = f"Unknown tool: {name}"
            logger.error(error)
            return json.dumps({"error": error})

        func = self._tools[name]
        logger.info("Executing tool: %s(%s)", name, arguments)

        try:
            if inspect.iscoroutinefunction(func):
                result = await func(**arguments)
            else:
                result = func(**arguments)

            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            error = f"Tool {name} failed: {e}"
            logger.exception(error)
            return json.dumps({"error": error})

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())
