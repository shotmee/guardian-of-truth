"""Диалоги τ²-bench → примеры в формате соревнования: по примеру (prompt, response) на каждый ход агента.

Формат организаторов (сверено с valid.parquet):
  ⟦SYSTEM⟧ + системный промпт агента + «[AVAILABLE TOOLS]» со схемой инструментов;
  ⟦ASSISTANT⟧ — приветствие агента без номера хода;
  ⟦USER⟧ — реплика клиента; его вызовы (telecom) идут строками внутри этого же блока;
  ⟦ASSISTANT · ход N⟧ — текст и/или вызовы «\\t→ TOOL_CALL name: {json, ключи по алфавиту}», затем ответы среды
  «\\t← TOOL_RESPONSE name[ [ERROR]]: …»; многострочный ответ продолжается строками с префиксом «\\t  ».
Блоки разделены пустой строкой. response — блок проверяемого хода без ответов среды, prompt — всё до него.

Вход — JSON, который пишет run_simulations.py (симуляции + что видел агент), без импорта tau2.
python serialize.py --runs <папка с <домен>.json> --out examples.parquet
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

GREETING_MARKER = "⟦ASSISTANT⟧"
USER_MARKER = "⟦USER⟧"
SYSTEM_MARKER = "⟦SYSTEM⟧"
CONTINUATION = "\t  "
UNSAFE_ID_CHARACTERS = re.compile(r"[^\w.\-]+")


# ----------------------------------------------------------------------------- схема инструментов
def _resolve(schema: dict, definitions: dict) -> dict:
    reference = schema.get("$ref")
    if reference:
        return definitions.get(reference.split("/")[-1], {})
    if "allOf" in schema and len(schema["allOf"]) == 1:
        return _resolve(schema["allOf"][0], definitions)
    return schema


def _type_name(schema: dict, definitions: dict) -> str:
    schema = _resolve(schema, definitions)
    if "type" in schema:
        kind = schema["type"]
        return kind if isinstance(kind, str) else next((k for k in kind if k != "null"), "string")
    for option in schema.get("anyOf", []):
        option = _resolve(option, definitions)
        if option.get("type") not in (None, "null"):
            return _type_name(option, definitions)
    return "object" if "properties" in schema else "string"


def _enum(schema: dict, definitions: dict) -> Optional[list]:
    schema = _resolve(schema, definitions)
    if "enum" in schema:
        return schema["enum"]
    for option in schema.get("anyOf", []):
        values = _enum(option, definitions)
        if values:
            return values
    return None


def _object_schema(schema: dict, definitions: dict) -> dict:
    """Схема объекта: $ref разворачивается, из anyOf берётся первый вариант с полями (pydantic: [$ref, object])."""
    schema = _resolve(schema, definitions)
    options = [_resolve(option, definitions) for option in schema.get("anyOf", [])]
    with_fields = [option for option in options if option.get("properties")]
    if with_fields:
        return with_fields[0]
    return next((option for option in options if option.get("type") != "null"), schema)


def _nested_fields(schema: dict, definitions: dict) -> tuple:
    """(properties, required) объекта или элементов массива объектов."""
    schema = _object_schema(schema, definitions)
    if schema.get("type") == "array":
        schema = _object_schema(schema.get("items", {}), definitions)
    return schema.get("properties", {}), set(schema.get("required", []))


def _parameter_line(name: str, schema: dict, required: bool, definitions: dict, indent: str, bullet: str) -> str:
    line = f"{indent}{bullet}{name}: {_type_name(schema, definitions)}{'!' if required else ''}"
    values = _enum(schema, definitions)
    if values:
        line += " [enum: " + "|".join(str(value) for value in values) + "]"
    description = (_resolve(schema, definitions).get("description") or schema.get("description") or "").strip()
    if description:
        line += " — " + description.replace("\n", " ")
    return line


def render_tools(tool_schemas: list) -> str:
    lines = ["[AVAILABLE TOOLS]"]
    for entry in tool_schemas:
        function = entry.get("function", entry)
        description = (function.get("description") or "").strip().split("\n")
        lines.append(f"- {function['name']} — {description[0]}")
        lines.extend("    " + line if line.strip() else "    " for line in description[1:])
        parameters = function.get("parameters") or {}
        definitions = parameters.get("$defs", {})
        required = set(parameters.get("required", []))
        for name, schema in (parameters.get("properties") or {}).items():
            lines.append(_parameter_line(name, schema, name in required, definitions, "    ", ""))
            fields, field_required = _nested_fields(schema, definitions)
            for field_name, field_schema in fields.items():
                lines.append(_parameter_line(field_name, field_schema, field_name in field_required, definitions,
                                             "        ", "· "))
    return "\n".join(lines)


# ----------------------------------------------------------------------------- реплики
def _call_line(call: dict) -> str:
    arguments = json.dumps(call.get("arguments") or {}, ensure_ascii=False, sort_keys=True)
    return f"\t→ TOOL_CALL {call['name']}: {arguments}"


def _result_lines(name: str, content: Optional[str], error: bool) -> list:
    text = "" if content is None else str(content)
    first, *rest = text.split("\n") if text else [""]
    marker = " [ERROR]" if error else ""
    return [f"\t← TOOL_RESPONSE {name}{marker}: {first}"] + [CONTINUATION + line for line in rest]


def _block(marker: str, text: Optional[str], calls: list) -> list:
    lines = [marker]
    if text and text.strip():
        lines.append(text.strip("\n"))
    lines.extend(_call_line(call) for call in calls)
    return lines


def dialog_examples(simulation: dict, setup: dict, dialog_id: str) -> list:
    """[(номер хода, prompt, response, вызовов в ходе)] для всех ходов агента одного диалога."""
    system_block = [SYSTEM_MARKER, setup["system_prompt"].strip("\n") + "\n\n" + render_tools(setup["tools"])]
    blocks: list = [system_block]
    examples = []
    call_names: dict = {}
    agent_turn = 0
    greeted = False
    for message in simulation["messages"]:
        role = message.get("role")
        calls = message.get("tool_calls") or []
        for call in calls:
            call_names[call.get("id", "")] = call["name"]
        if role == "assistant":
            if not greeted and not calls:
                greeted = True
                blocks.append(_block(GREETING_MARKER, message.get("content"), []))
                continue
            greeted = True
            agent_turn += 1
            block = _block(f"⟦ASSISTANT · ход {agent_turn}⟧", message.get("content"), calls)
            prompt = "\n\n".join("\n".join(lines) for lines in blocks)
            examples.append((agent_turn, prompt, "\n".join(block), len(calls)))
            blocks.append(block)
        elif role == "user":
            content = message.get("content") or ""
            if "###" in content and not calls:
                continue                          # служебные токены конца диалога
            blocks.append(_block(USER_MARKER, content, calls))
        elif role == "tool":
            name = call_names.get(message.get("id", ""), "unknown_tool")
            blocks[-1].extend(_result_lines(name, message.get("content"), bool(message.get("error"))))
    return examples


def safe_dialog_id(domain: str, task_id: str, trial: int) -> str:
    return f"{domain}__{UNSAFE_ID_CHARACTERS.sub('-', task_id).strip('-')}__g{trial}"


def examples_frame(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(runs_dir.glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        domain = run["domain"]
        for simulation in run["simulations"]:
            task_id = str(simulation["task_id"])
            setup = run["agent_setups"].get(task_id)
            if setup is None:
                continue
            dialog = safe_dialog_id(domain, task_id, int(simulation.get("trial") or 0))
            for turn, prompt, response, calls in dialog_examples(simulation, setup, dialog):
                rows.append(dict(id=f"{dialog}::t{turn}", prompt=prompt, response=response, dialog=dialog, turn=turn,
                                 domain=domain, task_id=task_id, n_calls=calls,
                                 termination=str(simulation.get("termination_reason"))))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", required=True, help="папка с <домен>.json от run_simulations.py")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    frame = examples_frame(Path(args.runs))
    frame.to_parquet(args.out, index=False)
    summary = frame.groupby("domain").agg(dialogs=("dialog", "nunique"), turns=("id", "size"),
                                          text_turns=("n_calls", lambda s: int((s == 0).sum())))
    print(summary.to_string())


if __name__ == "__main__":
    main()
