"""Перевод базы знаний banking_knowledge на русский (у организаторов документы базы знаний русские).

Переводятся title и content каждого документа; id, суммы, проценты, даты и названия продуктов не меняются.
Оригиналы сохраняются в documents_en/, перевод кэшируется в --cache (повторный запуск ничего не переводит заново).

python translate_kb.py --docs <tau2>/data/tau2/domains/banking_knowledge/documents --url http://127.0.0.1:8002/v1 \\
    --model user --cache /content/drive/MyDrive/guardian_of_truth/v5_tau2gen/kb_ru.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SYSTEM = ("You are a professional translator for a bank. Translate the knowledge-base document from English into "
          "natural Russian. Keep the Markdown structure exactly. Do not translate and do not change: product, card, "
          "account and plan names (e.g. Platinum Rewards Card, Silver Account, Rho-Bank+), document IDs, tool names, "
          "numbers, amounts, percentages, dates and codes. Output exactly two parts:\n"
          "TITLE: <translated title>\nCONTENT:\n<translated content>")


def translate(url: str, model: str, title: str, content: str) -> tuple:
    payload = {"model": model, "temperature": 0.2, "max_tokens": max(1024, len(content) // 2 + 512),
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": f"TITLE: {title}\nCONTENT:\n{content}"}],
               "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        text = json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"].strip()
    head, _, body = text.partition("CONTENT:")
    translated_title = head.replace("TITLE:", "", 1).strip() or title
    return translated_title, body.strip("\n") or content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--docs", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="user")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    docs_dir = Path(args.docs)
    backup = docs_dir.parent / "documents_en"
    if not backup.exists():
        shutil.copytree(docs_dir, backup)
    cache_path = Path(args.cache)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    originals = {path.name: json.loads(path.read_text(encoding="utf-8")) for path in sorted(backup.glob("*.json"))}
    todo = [name for name in originals if name not in cache]
    print(f"документов {len(originals)}, уже переведено {len(cache)}, переводим {len(todo)}")

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(translate, args.url, args.model, originals[name].get("title", ""),
                               originals[name].get("content", "")): name for name in todo}
        for done, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            try:
                title, content = future.result()
                cache[name] = {"title": title, "content": content}
            except Exception as error:  # noqa: BLE001 — документ без перевода останется английским
                print(f"не перевёлся {name}: {error!r}")
            if done % 50 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} за {time.time() - started:.0f} с", flush=True)
                cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    for name, document in originals.items():
        translated = dict(document)
        translated.update(cache.get(name, {}))
        (docs_dir / name).write_text(json.dumps(translated, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"готово: переведено {sum(name in cache for name in originals)}/{len(originals)}")


if __name__ == "__main__":
    main()
