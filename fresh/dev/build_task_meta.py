"""Компактная база задач τ² для решения: src/veritas/assets/tau2_tasks.json.

Из публичного репозитория tau2-bench (той же версии, что при генерации dev-диалогов) берутся только поля,
которые видел разметчик организаторов как «метаданные задачи»: назначение, сценарий клиента, эталонные действия
с аргументами, что сообщить клиенту, проверки на естественном языке.
python dev/build_task_meta.py --tau2-data <tau2-bench>/data/tau2/domains
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "veritas" / "assets" / "tau2_tasks.json"


def compact(task: dict) -> dict:
    criteria = task.get("evaluation_criteria") or {}
    scenario = task.get("user_scenario") or {}
    instructions = scenario.get("instructions")
    if isinstance(instructions, dict):
        instructions = {key: value for key, value in instructions.items() if value}
    return {"purpose": (task.get("description") or {}).get("purpose"),
            "persona": scenario.get("persona"),
            "instructions": instructions,
            "expected_actions": [{"name": action.get("name"), "arguments": action.get("arguments"),
                                  **({"requestor": action["requestor"]} if action.get("requestor") not in (None, "assistant") else {})}
                                 for action in criteria.get("actions") or []],
            "communicate_info": criteria.get("communicate_info") or None,
            "nl_assertions": criteria.get("nl_assertions") or None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tau2-data", required=True)
    args = parser.parse_args()
    domains = Path(args.tau2_data)
    tasks: dict = {}
    for domain in ("airline", "retail", "telecom"):
        for task in json.loads((domains / domain / "tasks.json").read_text(encoding="utf-8")):
            tasks[f"{domain}::{task['id']}"] = compact(task)
    for path in sorted((domains / "banking_knowledge" / "tasks").glob("*.json")):
        task = json.loads(path.read_text(encoding="utf-8"))
        tasks[f"banking_knowledge::{task['id']}"] = compact(task)
    OUT.write_text(json.dumps(tasks, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    counts = {}
    for key in tasks:
        counts[key.split("::")[0]] = counts.get(key.split("::")[0], 0) + 1
    print(f"{OUT}: {OUT.stat().st_size / 1e6:.1f} МБ, задач {counts}")


if __name__ == "__main__":
    main()
