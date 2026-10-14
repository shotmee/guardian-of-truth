"""Прогон τ²-bench в текстовом режиме на локальных серверах vLLM: агент — слабая модель, клиент — сильная.

Задачи берутся случайно (seed) из тех, которых нет в valid, по одной попытке на задачу. Результат —
<out>/<домен>.json: симуляции (сообщения) + что видел агент (системный промпт и схема инструментов),
этого хватает serialize.py.

python run_simulations.py --domain airline --tasks 30 --exclude valid_task_ids.json --policies policies.json \\
    --agent-url http://127.0.0.1:8001/v1 --agent-model agent --user-url http://127.0.0.1:8002/v1 --user-model user \\
    --out /content/drive/MyDrive/guardian_of_truth/v5_tau2gen/runs
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patches import RETRIEVAL_VARIANT, agent_setups, apply_patches  # noqa: E402

TASK_SPLIT = "base"                    # стандартный набор задач τ² (в telecom — 114 из 2285)


def llm_args(base_url: str, temperature: float) -> dict:
    return {"api_base": base_url, "api_key": "EMPTY", "temperature": temperature,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def choose_task_ids(domain: str, count: int, excluded_prefixes: list, seed: int) -> list:
    from tau2.run import load_tasks
    ids = [str(task.id) for task in load_tasks(task_set_name=domain, task_split_name=TASK_SPLIT)]
    fresh = [task_id for task_id in ids if not any(_same_task(task_id, prefix) for prefix in excluded_prefixes)]
    random.Random(seed).shuffle(fresh)
    print(f"{domain}: задач {len(ids)}, без valid {len(fresh)}, берём {min(count, len(fresh))}")
    return fresh[:count]


def _same_task(task_id: str, valid_dialog: str) -> bool:
    """В valid id задач telecom очищены от скобок и обрезаны — сравниваем только буквы и цифры, по началу."""
    clean = "".join(ch for ch in task_id if ch.isalnum())
    valid_clean = "".join(ch for ch in valid_dialog if ch.isalnum())
    return clean == valid_clean or (len(valid_clean) >= 40 and clean.startswith(valid_clean))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--domain", required=True)
    parser.add_argument("--tasks", type=int, required=True)
    parser.add_argument("--exclude", required=True, help="json со списком id задач из valid")
    parser.add_argument("--policies", required=True, help="json: домен -> политика организаторов")
    parser.add_argument("--agent-url", required=True)
    parser.add_argument("--agent-model", default="agent")
    parser.add_argument("--agent-temperature", type=float, default=0.7)
    parser.add_argument("--user-url", required=True)
    parser.add_argument("--user-model", default="user")
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=1200, help="секунд на одну симуляцию")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    policies = json.loads(Path(args.policies).read_text(encoding="utf-8"))
    apply_patches(policies)
    from tau2.run import TextRunConfig, run_domain

    excluded = json.loads(Path(args.exclude).read_text(encoding="utf-8")).get(args.domain, [])
    task_ids = choose_task_ids(args.domain, args.tasks, excluded, args.seed)
    config = TextRunConfig(
        domain=args.domain,
        task_split_name=TASK_SPLIT,
        task_ids=task_ids,
        num_trials=1,
        llm_agent=f"openai/{args.agent_model}",
        llm_args_agent=llm_args(args.agent_url, args.agent_temperature),
        llm_user=f"openai/{args.user_model}",
        llm_args_user=llm_args(args.user_url, 0.7),
        max_steps=args.max_steps,
        timeout=args.timeout,
        max_concurrency=args.concurrency,
        seed=args.seed,
        save_to=f"veritas_{args.domain}_{args.seed}",
        retrieval_config=RETRIEVAL_VARIANT if args.domain == "banking_knowledge" else None,
        log_level="WARNING",
    )
    started = time.time()
    results = run_domain(config)
    simulations = [simulation.model_dump(mode="json", include={"task_id", "trial", "termination_reason", "messages"})
                   for simulation in results.simulations]
    output = {"domain": args.domain, "agent_model": args.agent_model, "seed": args.seed,
              "simulations": simulations, "agent_setups": agent_setups()}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.domain}.json").write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    print(f"{args.domain}: симуляций {len(simulations)} за {time.time() - started:.0f} с → {out_dir / (args.domain + '.json')}")


if __name__ == "__main__":
    main()
