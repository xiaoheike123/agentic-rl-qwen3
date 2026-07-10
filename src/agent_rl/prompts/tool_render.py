"""Render tau2 Tool objects into text or tool-call schemas."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

from tau2.environment.tool import Tool


class ToolSchemaError(ValueError):
    """Raised when a tau2 tool has an invalid OpenAI schema."""


def render_tool_schemas(
    tools: Sequence[Tool],
) -> list[dict[str, Any]]:
    schemas: list[dict[str, Any]] = []

    for tool in tools:
        schema = deepcopy(tool.openai_schema)

        if not isinstance(schema, dict):
            raise ToolSchemaError(
                f"Tool {tool.name!r} does not provide a dictionary schema"
            )

        if schema.get("type") != "function":
            raise ToolSchemaError(
                f"Tool {tool.name!r} is not an OpenAI function tool"
            )

        function_schema = schema.get("function")
        if not isinstance(function_schema, dict):
            raise ToolSchemaError(
                f"Tool {tool.name!r} has no function schema"
            )

        if function_schema.get("name") != tool.name:
            raise ToolSchemaError(
                f"Tool name mismatch for {tool.name!r}"
            )

        schemas.append(schema)

    return schemas
