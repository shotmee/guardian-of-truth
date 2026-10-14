# Parsing of the [AVAILABLE TOOLS] section carried by the system prompt.
#
# Grammar roughly:
#   - tool_name - description (continuation lines are indented)
#       param: type! [enum: a|b] - description      (! marks a required param)
#           . nested: type! - description           (object/array fields)
#
# ``parse_available_tools`` returns {name -> ToolSpec}; ``to_function_schemas``
# produces the OpenAI-shaped list the chat template expects.

import re

_ALLOWED_TYPES = ("string", "integer", "number", "boolean", "array", "object", "null")

TOOL_LINE = re.compile(r"^- ([A-Za-z_][\w.\-]*) — ?(.*)$")

PARAMETER_LINE = re.compile(
    r"^(?P<indent>\s+)(?P<nested>· )?"
    r"(?P<name>[A-Za-z_]\w*): (?P<type>"
    + "|".join(_ALLOWED_TYPES) +
    r")(?P<required>!?)(?: \[enum: (?P<enum>[^\]]*)\])?"
    r"(?: — (?P<description>.*))?$"
)

SECTION_TITLE = "[AVAILABLE TOOLS]"


class ToolParameter:
    __slots__ = ("name", "type", "required", "allowed_values", "description", "fields")

    def __init__(self, name, type, required, allowed_values, description, fields=None):
        self.name = name
        self.type = type
        self.required = required
        self.allowed_values = allowed_values
        self.description = description
        self.fields = fields if fields is not None else {}


class ToolSpec:
    __slots__ = ("name", "description", "parameters")

    def __init__(self, name, description, parameters=None):
        self.name = name
        self.description = description
        self.parameters = parameters if parameters is not None else {}

    @property
    def required_parameters(self):
        return [p.name for p in self.parameters.values() if p.required]


def _indent_cols(blob):
    return len(blob.replace("\t", "    "))


def parse_available_tools(tools_text):
    tools = {}
    active_tool = None
    nesting_stack = []  # [(indent_cols, parameter)]

    for raw_line in tools_text.split("\n"):
        bare = raw_line.strip()
        if not bare or bare == SECTION_TITLE:
            continue

        hit = TOOL_LINE.match(raw_line)
        if hit is not None:
            active_tool = ToolSpec(name=hit.group(1),
                                   description=hit.group(2).strip())
            tools[active_tool.name] = active_tool
            nesting_stack = []
            continue

        if active_tool is None:
            continue

        param_hit = PARAMETER_LINE.match(raw_line)
        if param_hit is None:
            if not active_tool.parameters:
                combined = (active_tool.description + " " + bare).strip()
                active_tool.description = combined
            continue

        enum_text = param_hit.group("enum")
        if enum_text:
            enum_values = [v.strip() for v in enum_text.split("|")]
        else:
            enum_values = None

        parameter = ToolParameter(
            name=param_hit.group("name"),
            type=param_hit.group("type"),
            required=bool(param_hit.group("required")),
            allowed_values=enum_values,
            description=(param_hit.group("description") or "").strip(),
        )
        column = _indent_cols(param_hit.group("indent"))

        if param_hit.group("nested"):
            while nesting_stack and nesting_stack[-1][0] >= column:
                nesting_stack.pop()
            if nesting_stack:
                nesting_stack[-1][1].fields[parameter.name] = parameter
                nesting_stack.append((column, parameter))
            continue

        active_tool.parameters[parameter.name] = parameter
        nesting_stack = [(column, parameter)]

    return tools


def _schema_for(parameter):
    body = {"type": parameter.type}
    if parameter.description:
        body["description"] = parameter.description
    if parameter.allowed_values:
        body["enum"] = parameter.allowed_values
    if parameter.fields:
        props = {f.name: _schema_for(f) for f in parameter.fields.values()}
        req = [f.name for f in parameter.fields.values() if f.required]
        if parameter.type == "array":
            body["items"] = {"type": "object", "properties": props, "required": req}
        else:
            body["properties"] = props
            body["required"] = req
    return body


def to_function_schemas(tools):
    result = []
    for tool in tools.values():
        parameters_schema = {
            "type": "object",
            "properties": {p.name: _schema_for(p) for p in tool.parameters.values()},
            "required": tool.required_parameters,
        }
        result.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters_schema,
            },
        })
    return result
