#!/usr/bin/env python
"""Точка входа Guardian of Truth (платформа: python scripts/predict.py --input <test> --output <pred>).

Вход:  CSV/parquet с колонками id, prompt, response (формат определяется по содержимому).
Выход: parquet с колонками id, label (0/1) — платформа читает предсказания как parquet.
Сам поднимает vLLM с моделью из ./model; --base-url позволяет использовать уже поднятый сервер (разработка).
"""
import time

T_START = time.time()

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from guardian.pipeline import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="тестовый датасет (id, prompt, response)")
    ap.add_argument("--output", required=True, help="куда писать предсказания (id, label)")
    ap.add_argument("--model-path", default=str(ROOT / "model"))
    ap.add_argument("--config", default=str(ROOT / "src" / "guardian" / "config.json"))
    ap.add_argument("--base-url", default=None, help="уже поднятый vLLM, напр. http://127.0.0.1:8011/v1")
    ap.add_argument("--mode", default="judge", choices=["judge", "baseline", "rules"])
    ap.add_argument("--dump", default=None, help="(dev) parquet с подробными скорами по примерам")
    ap.add_argument("--cache", default=None, help="(dev) jsonl-кэш ответов LLM для возобновления")
    ap.add_argument("--llm-on-hard", action="store_true", help="(dev) прогонять LLM и по hard-примерам")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--deadline", type=float, default=None, help="переопределить жёсткий дедлайн, сек")
    run(ap.parse_args(), T_START)


if __name__ == "__main__":
    main()
    # предсказания уже записаны, vLLM остановлен; не ждём зависшие HTTP-потоки (они не daemon)
    import os
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
