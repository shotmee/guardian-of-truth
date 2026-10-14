"""Сборка входа для LLM-судьи (шаблон чата Qwen3.5/3.6 собирается вручную — без зависимостей)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from .transcript import HEADER_RE, split_head

HERE = Path(__file__).resolve().parent


def load_text(name: str) -> str:
    return (HERE / name).read_text(encoding="utf-8").strip()


def truncate_prompt(prompt: str, max_chars: int) -> str:
    """Если контекст длиннее лимита — сохраняем системную часть и хвост диалога, середину вырезаем
    по границам ходов. В valid максимум ~233k символов, лимит по умолчанию 360k, т.е. срабатывает редко."""
    if len(prompt) <= max_chars:
        return prompt
    h = split_head(prompt)
    head = prompt[:h]
    budget = max(20000, max_chars - len(head) - 200)
    tail_start = len(prompt) - budget
    m = HEADER_RE.search(prompt, tail_start)
    if m:
        tail_start = m.start()
    cut = tail_start - h
    return head + f"[… пропущено {cut} символов ранней истории диалога …]\n\n" + prompt[tail_start:]


def user_content(prompt: str, response: str, analysis: Dict, max_chars: int, evidence: str = "") -> str:
    facts = "\n".join(f"- {f}" for f in analysis.get("facts", []))
    turn = analysis.get("turn")
    dom = analysis.get("domain", "unknown")
    ev = f"[ФРАГМЕНТЫ КОНТЕКСТА, СВЯЗАННЫЕ С ОТВЕТОМ]\n{evidence}\n\n" if evidence else ""
    return (f"<prompt>\n{truncate_prompt(prompt, max_chars)}\n</prompt>\n"
            f"<response>\n{response}\n</response>\n\n"
            f"{ev}[АВТОПРОВЕРКИ]\n{facts}\n\n"
            f"[ЗАДАНИЕ]\nДомен: {dom}. Проверь ход {turn if turn is not None else '(последний)'} агента: "
            f"есть ли в нём новая контекстная галлюцинация? Итоговый ответ — один символ: 1 или 0.")


_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")


def sentences(text: str, k: int = 8) -> List[str]:
    """Утверждения из текста ответа: предложения от 5 слов, без вопросов и приветствий."""
    out = []
    for p in _SENT_SPLIT.split(text or ""):
        p = p.strip(" *-•\t#>")
        if len(p.split()) >= 5 and not p.rstrip().endswith("?"):
            out.append(p[:400])
    return out[:k]


def sentence_task(user: str, sents: List[str]) -> str:
    lines = "\n".join(f"{k}) {s}" for k, s in enumerate(sents, 1))
    return (user + "\n\n[ДОПОЛНИТЕЛЬНОЕ ЗАДАНИЕ: ПРОВЕРКА ПО ПРЕДЛОЖЕНИЯМ]\n"
            "Ниже утверждения из текста проверяемого хода. Для КАЖДОГО выведи цифру: 1 — утверждение не "
            "подтверждается контекстом/policy/базой знаний или противоречит им (выдумано, неверно, обещает "
            "невозможное); 0 — подтверждается или не является утверждением о фактах и правилах.\n"
            f"{lines}\nОтвет: ровно {len(sents)} цифр через пробел, без пояснений.")


def baseline_user_content(prompt: str, response: str, max_chars: int) -> str:
    return f"<prompt>\n{truncate_prompt(prompt, max_chars)}\n</prompt>\n<response>\n{response}\n</response>"


_HF = {"tok": None, "path": None}


def set_hf_template(model_path: str) -> None:
    """Шаблон чата из токенизатора модели (для не-Qwen моделей в сравнении, напр. GLM-4.7-Flash)."""
    from transformers import AutoTokenizer
    _HF["tok"] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _HF["path"] = model_path


def chat(system: str, user: str, thinking: bool) -> str:
    if _HF["tok"] is not None:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        s = _HF["tok"].apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        if thinking and not s.rstrip().endswith("<think>"):
            s = s + "<think>\n"
        if not thinking and "</think>" not in s[-40:]:
            s = s + "<think></think>\n"
        return s
    s = (f"<|im_start|>system\n{system}<|im_end|>\n"
         f"<|im_start|>user\n{user}<|im_end|>\n"
         f"<|im_start|>assistant\n")
    return s + ("<think>\n" if thinking else "<think>\n\n</think>\n\n")


def close_thinking(think_prompt: str, reasoning: str, truncated: bool) -> str:
    r = reasoning.rstrip()
    if truncated:
        r += "\n\n(Лимит размышлений исчерпан, подвожу итог.)"
    return think_prompt + r + "\n</think>\n\n"
