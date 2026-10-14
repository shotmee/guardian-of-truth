"""Разбор раздела [AVAILABLE TOOLS] системного промпта.

Формат:
- tool_name — описание (продолжение описания — строки с отступом)
    param: type! [enum: a|b] — описание          (! — обязательный параметр)
        · nested: type! — описание               (поля объектов внутри массива или объекта)

`parse_available_tools` даёт словарь имя -> ToolSpec, `to_function_schemas` — список функций в формате
OpenAI (его принимает шаблон чата модели).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

PARAMETER_TYPES = ("string", "integer", "number", "boolean", "array", "object", "null")
TOOL_LINE = re.compile(r"^- ([A-Za-z_][\w.\-]*) — ?(.*)$")
PARAMETER_LINE = re.compile(
    r"^(?P<indent>\s+)(?P<nested>· )?(?P<name>[A-Za-z_]\w*): (?P<type>" + "|".join(PARAMETER_TYPES) + r")"
    r"(?P<required>!?)(?: \[enum: (?P<enum>[^\]]*)\])?(?: — (?P<description>.*))?$"
)
SECTION_TITLE = "[AVAILABLE TOOLS]"


@dataclass
class ToolParameter:
    name: str
    type: str
    required: bool
    allowed_values: Optional[list]          # enum, если задан
    description: str
    fields: dict = field(default_factory=dict)  # вложенные поля объектов


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict = field(default_factory=dict)

    @property
    def required_parameters(self) -> list:
        return [p.name for p in self.parameters.values() if p.required]


def _indent_width(indent: str) -> int:
    return len(indent.replace("\t", "    "))


def parse_available_tools(tools_text: str) -> dict:
    tools: dict = {}
    current: Optional[ToolSpec] = None
    open_parameters: list = []              # [(ширина отступа, параметр)] — цепочка вложенности
    for line in tools_text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped == SECTION_TITLE:
            continue
        tool_match = TOOL_LINE.match(line)
        if tool_match:
            current = ToolSpec(name=tool_match.group(1), description=tool_match.group(2).strip())
            tools[current.name] = current
            open_parameters = []
            continue
        if current is None:
            continue
        param_match = PARAMETER_LINE.match(line)
        if param_match:
            enum_text = param_match.group("enum")
            parameter = ToolParameter(
                name=param_match.group("name"), type=param_match.group("type"),
                required=bool(param_match.group("required")),
                allowed_values=[v.strip() for v in enum_text.split("|")] if enum_text else None,
                description=(param_match.group("description") or "").strip())
            width = _indent_width(param_match.group("indent"))
            if param_match.group("nested"):
                while open_parameters and open_parameters[-1][0] >= width:
                    open_parameters.pop()
                if open_parameters:
                    open_parameters[-1][1].fields[parameter.name] = parameter
                    open_parameters.append((width, parameter))
                continue
            current.parameters[parameter.name] = parameter
            open_parameters = [(width, parameter)]
            continue
        if not current.parameters:          # продолжение описания инструмента
            current.description = (current.description + " " + stripped).strip()
    return tools


def _parameter_schema(parameter: ToolParameter) -> dict:
    schema: dict = {"type": parameter.type}
    if parameter.description:
        schema["description"] = parameter.description
    if parameter.allowed_values:
        schema["enum"] = parameter.allowed_values
    if parameter.fields:
        properties = {f.name: _parameter_schema(f) for f in parameter.fields.values()}
        required = [f.name for f in parameter.fields.values() if f.required]
        if parameter.type == "array":
            schema["items"] = {"type": "object", "properties": properties, "required": required}
        else:
            schema.update({"properties": properties, "required": required})
    return schema


def to_function_schemas(tools: dict) -> list:
    return [{
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": {p.name: _parameter_schema(p) for p in tool.parameters.values()},
                "required": tool.required_parameters,
            },
        },
    } for tool in tools.values()]
