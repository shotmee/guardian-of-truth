"""Стратегия A «Дублёр»: модель сама играет агента на том же ходе.

Для каждого хода с вызовами модель получает тот же контекст в нативном формате агента
(policy + tools + история) и генерирует свой ход: быстрые сэмплы без рассуждения и несколько
сэмплов с рассуждением. Согласие сэмпла с ходом агента:
    1.0 — модель делает тот же вызов с теми же ключевыми аргументами;
    0.5 — тот же инструмент, другие аргументы;
    0.0 — другой инструмент или ответ текстом.
score = 1 − взвешенное среднее согласия (сэмплы с рассуждением весят больше).
"""
from __future__ import annotations

import ast
import json
import operator
import re
from typing import Any, Optional

from .chat import ChatTemplate, agent_prompt
from .checks import FREE_TEXT_ARGUMENTS
from .engine import CompletionClient, map_concurrently

TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAMETER_BLOCK = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)
THINKING_END = "</think>"
TURNS_TO_DROP_WHEN_TOO_LONG = (0, 6, 14, 30)
# у обёрток discoverable-инструментов само действие определяет аргумент с именем инструмента
ACTION_DEFINING_ARGUMENT = {"unlock_discoverable_agent_tool": "agent_tool_name",
                            "call_discoverable_agent_tool": "agent_tool_name",
                            "give_discoverable_user_tool": "discoverable_tool_name"}
ARITHMETIC = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
              ast.USub: operator.neg, ast.UAdd: operator.pos}


def parse_model_action(text: str) -> tuple:
    """Ответ модели в формате Qwen -> (текст до вызовов, [(имя, аргументы)])."""
    calls = []
    for call_match in TOOL_CALL_BLOCK.finditer(text):
        arguments = {}
        for parameter_match in PARAMETER_BLOCK.finditer(call_match.group(2)):
            raw_value = parameter_match.group(2).strip()
            try:
                arguments[parameter_match.group(1)] = json.loads(raw_value)
            except ValueError:
                arguments[parameter_match.group(1)] = raw_value
        calls.append((call_match.group(1), arguments))
    return text.split("<tool_call>")[0].strip(), calls


def evaluate_arithmetic(expression: Any) -> Optional[float]:
    """Значение выражения для calculate (только + − × / и числа)."""
    try:
        tree = ast.parse(str(expression).replace("×", "*").replace("−", "-"), mode="eval").body
    except (SyntaxError, ValueError, RecursionError):
        return None

    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in ARITHMETIC:
            return ARITHMETIC[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ARITHMETIC:
            return ARITHMETIC[type(node.op)](evaluate(node.operand))
        raise ValueError("неподдерживаемое выражение")

    try:
        return evaluate(tree)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError, RecursionError):
        return None


def _comparable_value(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip().lower()
        try:
            return float(text)
        except ValueError:
            return text
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        items = [_comparable_value(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    if isinstance(value, dict):
        return {key: _comparable_value(item) for key, item in value.items() if key not in FREE_TEXT_ARGUMENTS}
    return value


def comparable_arguments(tool_name: str, arguments: dict) -> dict:
    """Аргументы, по которым сравниваются вызовы: свободный текст отброшен, calculate — по значению."""
    result = {}
    for name, value in (arguments or {}).items():
        if name == "expression":
            result[name] = evaluate_arithmetic(value)
        elif name == "arguments" and isinstance(value, str):
            try:
                result[name] = _comparable_value(json.loads(value))
            except ValueError:
                continue
        elif name == "reason" and tool_name == "cancel_pending_order":
            result[name] = _comparable_value(value)
        elif name in FREE_TEXT_ARGUMENTS:
            continue
        elif name in ("item_ids", "new_item_ids") and isinstance(value, list):
            result[name] = [str(item) for item in value]          # порядок важен: пары старый/новый
        else:
            result[name] = _comparable_value(value)
    old_items, new_items = result.get("item_ids"), result.get("new_item_ids")
    if old_items is not None and new_items is not None and len(old_items) == len(new_items):
        del result["item_ids"], result["new_item_ids"]
        result["pairs"] = sorted(zip(old_items, new_items))
    elif old_items is not None:
        result["item_ids"] = sorted(old_items)
    return result


def values_match(left: Any, right: Any) -> bool:
    """У объектов сравниваются только общие поля (лишние вложенные поля организаторы ошибкой не
    считают), списки — как мультимножества, числа — с допуском."""
    if isinstance(left, float) and isinstance(right, float):
        return abs(left - right) <= 1e-6 * max(1.0, abs(left))
    if isinstance(left, dict) and isinstance(right, dict):
        shared = set(left) & set(right)
        return bool(shared) and all(values_match(left[key], right[key]) for key in shared)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        unmatched = list(right)
        for item in left:
            partner = next((k for k, candidate in enumerate(unmatched) if values_match(item, candidate)), None)
            if partner is None:
                return False
            unmatched.pop(partner)
        return True
    return left == right


def _action_identity(tool_name: str, arguments: dict) -> str:
    argument = ACTION_DEFINING_ARGUMENT.get(tool_name)
    if argument is None:
        return tool_name
    return f"{tool_name}:{str((arguments or {}).get(argument, '')).strip().lower()}"


def call_agreement(agent_name: str, agent_arguments: dict, model_name: str, model_arguments: dict) -> float:
    if _action_identity(agent_name, agent_arguments) != _action_identity(model_name, model_arguments):
        return 0.0
    agent_values = comparable_arguments(agent_name, agent_arguments)
    model_values = comparable_arguments(model_name, model_arguments)
    shared = set(agent_values) & set(model_values)
    return 1.0 if all(values_match(agent_values[k], model_values[k]) for k in shared) else 0.5


def sample_agreement(agent_calls: list, sample_text: str) -> float:
    """Лучшее согласие любого вызова модели в сэмпле с любым вызовом агента."""
    _, model_calls = parse_model_action(sample_text)
    best = 0.0
    for model_name, model_arguments in model_calls:
        for call in agent_calls:
            best = max(best, call_agreement(call.name, call.arguments or {}, model_name, model_arguments))
            if best == 1.0:
                return best
    return best


def _complete_with_shorter_history(client: CompletionClient, build_prompt, sampling: dict) -> Optional[dict]:
    """Если промпт не влез в контекст, выбрасываем всё больше середины истории."""
    for dropped_turns in TURNS_TO_DROP_WHEN_TOO_LONG:
        response = client.complete(build_prompt(dropped_turns), **sampling)
        if response is None:
            return None
        if "error" not in response:
            return response
    return None


def run(candidates: list, config: dict, client: CompletionClient, template: ChatTemplate,
        fast_deadline: float, deadline: float) -> dict:
    """candidates: [TurnAnalysis] -> {номер строки: {"score": ..., подробности}}."""
    settings = config.get("replay", {})
    fast_samples = int(settings.get("n_fast", 6))
    thinking_samples = int(settings.get("n_think", 2))
    thinking_samples_for_reads = int(settings.get("n_think_read", 1))
    thinking_weight = float(settings.get("w_think", 2.0))
    workers = int(config.get("workers", 96))
    fast_sampling = dict(n=fast_samples, temperature=settings.get("t_fast", 0.8), top_p=0.95, top_k=20,
                         max_tokens=int(settings.get("max_fast", 384)), skip_special_tokens=False)
    thinking_sampling = dict(temperature=settings.get("t_think", 0.6), top_p=0.95, top_k=20,
                             max_tokens=int(settings.get("max_think", 2048)), skip_special_tokens=False)

    def fast_samples_for(item) -> Optional[list]:
        response = _complete_with_shorter_history(
            client, lambda dropped: agent_prompt(template, item.transcript, False, dropped), fast_sampling)
        return [choice.get("text", "") for choice in response.get("choices", [])] if response else None

    def thinking_samples_for(item) -> Optional[list]:
        count = thinking_samples if item.has_side_effects else thinking_samples_for_reads
        response = _complete_with_shorter_history(
            client, lambda dropped: agent_prompt(template, item.transcript, True, dropped),
            dict(thinking_sampling, n=count))
        if not response:
            return None
        texts = [choice.get("text", "") for choice in response.get("choices", [])]
        return [text.split(THINKING_END, 1)[1] for text in texts if THINKING_END in text]   # без рассуждения

    ordered = sorted(candidates, key=lambda item: (not item.has_side_effects, item.index))
    fast_results = map_concurrently(fast_samples_for, ordered, workers, fast_deadline, "A·быстрые сэмплы")
    thinking_results = {}
    if thinking_samples > 0:
        thinking_results = map_concurrently(thinking_samples_for, ordered, workers, deadline,
                                            "A·сэмплы с рассуждением")
    scores = {}
    for position, item in enumerate(ordered):
        fast_texts, thinking_texts = fast_results.get(position) or [], thinking_results.get(position) or []
        fast_agreement = [sample_agreement(item.turn.tool_calls, text) for text in fast_texts]
        thinking_agreement = [sample_agreement(item.turn.tool_calls, text) for text in thinking_texts]
        total_weight = len(fast_agreement) + thinking_weight * len(thinking_agreement)
        if total_weight == 0:
            continue
        agreement = (sum(fast_agreement) + thinking_weight * sum(thinking_agreement)) / total_weight
        all_texts = fast_texts + thinking_texts
        scores[item.index] = dict(
            score=1.0 - agreement, q_fast=fast_agreement, q_think=thinking_agreement,
            n_ref_calls=sum(1 for text in all_texts if parse_model_action(text)[1]),
            ref=[parse_model_action(text)[1][:3] for text in (thinking_texts + fast_texts)[:4]])
    return scores
