"""Оркестратор Guardian of Truth v1.

Этапы (результат пишется на диск после каждого, поэтому тайм-аут не оставляет нас без ответа):
  0. Правила (секунды): hard-нарушения -> метка 1 без LLM.
  1. Фаза A: LLM без рассуждений, 1 токен, P('1') из logprobs — для всех примеров (страховка).
  2. Фаза B: LLM с рассуждением (thinking), итоговый токен 0/1 и его вероятность — сначала самые
     неуверенные по фазе A; 2-й сэмпл для пограничных, если остаётся время.
  3. Слияние: score = w*mean(P_think) + (1-w)*P_A; метка = score >= T.
Жёсткий дедлайн: к hard_deadline_s секундам от старта скрипта файл записан и процесс завершён.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import evidence as evidence_mod
from . import features
from .prompts import (baseline_user_content, chat, close_thinking, load_text, sentence_task, sentences,
                      set_hf_template, user_content)
from .rules import analyze
from .server import gpu_memory, start_vllm, stop_vllm, tail, wait_ready

HERE = Path(__file__).resolve().parent


def log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------------- IO
def read_table(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic == b"PAR1":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return pd.read_parquet(path)


def write_preds(ids: List[str], labels: List[int], path: str) -> None:
    """Платформа читает предсказания как parquet (даже если путь .csv) — пишем parquet атомарно."""
    out = pd.DataFrame({"id": pd.Series(ids, dtype=str), "label": pd.Series(labels, dtype="int64")})
    tmp = str(path) + ".tmp"
    out.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- logprobs
def p_from_top(top: Optional[Dict[str, float]], chosen: Optional[str] = None,
               chosen_lp: Optional[float] = None) -> Optional[float]:
    """P('1') по top-logprobs. Если в top есть только одна из цифр, оцениваем по ней самой:
    P(1) = exp(l1) или P(1) = 1 - exp(l0) — так значения остаются непрерывными, а не 1e-4/0.9999."""
    top = dict(top or {})
    if chosen is not None and chosen_lp is not None:
        top.setdefault(chosen, chosen_lp)
    l1 = max([lp for t, lp in top.items() if (t or "").strip() == "1"], default=None)
    l0 = max([lp for t, lp in top.items() if (t or "").strip() == "0"], default=None)
    if l1 is None and l0 is None:
        return None
    if l0 is None:
        return min(1.0 - 1e-6, max(0.501, math.exp(l1)))
    if l1 is None:
        return max(1e-6, min(0.499, 1.0 - math.exp(l0)))
    m = max(l1, l0)
    e1, e0 = math.exp(l1 - m), math.exp(l0 - m)
    return e1 / (e1 + e0)


def _answer_token_index(text: str, pos: int, toks: List[str], offsets: Optional[List[int]]) -> Optional[int]:
    """Индекс токена, покрывающего символ text[pos]. Сначала по text_offset, затем по накопленной длине."""
    if offsets and len(offsets) == len(toks):
        base = offsets[0] if offsets[0] else 0
        rel = [o - base for o in offsets]
        if 0 <= rel[-1] <= len(text):
            j = max(k for k in range(len(rel)) if rel[k] <= pos)
            return j
    acc = 0
    for k, t in enumerate(toks):
        nxt = acc + len(t or "")
        if acc <= pos < nxt:
            return k
        acc = nxt
    return None


def _resp_text(response: str) -> str:
    from .transcript import parse_blocks
    return "\n".join(b.text for b in parse_blocks(response) if b.text).strip()


def parse_digit_probs(ch, k: int) -> List[float]:
    """P('1') для первых k цифр 0/1 в ответе (по токенам с top-logprobs)."""
    lp = ch.logprobs
    toks = list(lp.tokens) if lp and lp.tokens else []
    tops = list(lp.top_logprobs) if lp and lp.top_logprobs else []
    tlps = list(lp.token_logprobs) if lp and getattr(lp, "token_logprobs", None) else []
    out: List[float] = []
    for j, t in enumerate(toks):
        s = (t or "").strip()
        if s in ("0", "1"):
            p = p_from_top(tops[j] if j < len(tops) else None, t, tlps[j] if j < len(tlps) else None)
            out.append(float(p) if p is not None else (0.85 if s == "1" else 0.15))
            if len(out) >= k:
                break
    return out


def parse_think_choice(ch) -> Dict[str, Any]:
    """Достаёт рассуждение и P('1') итогового токена после </think>.
    src: logprob — нашли токен ответа и его вероятность; text — только цифра из текста (0.85/0.15)."""
    text = ch.text or ""
    lp = ch.logprobs
    toks = list(lp.tokens) if lp and lp.tokens else []
    tops = list(lp.top_logprobs) if lp and lp.top_logprobs else []
    tlps = list(lp.token_logprobs) if lp and getattr(lp, "token_logprobs", None) else []
    offs = list(lp.text_offset) if lp and getattr(lp, "text_offset", None) else None
    i_end = text.find("</think>")
    if i_end < 0:
        return {"done": False, "reasoning": text}
    reasoning = text[:i_end]
    tail_start = i_end + len("</think>")
    m = re.search(r"[01]", text[tail_start:])
    if not m:
        return {"done": False, "reasoning": reasoning}
    digit, pos = m.group(0), tail_start + m.start()
    j = _answer_token_index(text, pos, toks, offs) if toks else None
    if j is not None and j < len(tops) and (toks[j] or "").strip() == digit:
        p = p_from_top(tops[j], toks[j], tlps[j] if j < len(tlps) else None)
        if p is not None:
            return {"done": True, "reasoning": reasoning, "p": p, "src": "logprob"}
    return {"done": True, "reasoning": reasoning, "p": 0.85 if digit == "1" else 0.15, "src": "text"}


# ----------------------------------------------------------------------------- state
class State:
    def __init__(self, ids: List[str], cfg: Dict[str, Any]):
        self.ids = ids
        self.cfg = cfg
        self.n = len(ids)
        self.hard = [False] * self.n
        self.pa: List[Optional[float]] = [None] * self.n
        self.pb: List[List[float]] = [[] for _ in range(self.n)]
        self.ps: List[List[float]] = [[] for _ in range(self.n)]
        self.px: Dict[str, List[Optional[float]]] = {}
        self.reason: List[str] = [""] * self.n
        self.think_tokens: List[int] = [0] * self.n
        self.analyses: List[Dict[str, Any]] = [{} for _ in range(self.n)]
        self.lock = threading.Lock()

    def _gate(self) -> bool:
        return self.cfg.get("decision", "fused") == "gate"

    def _stack(self) -> bool:
        return self.cfg.get("decision", "fused") == "stack" and bool(self.cfg.get("stack"))

    def feats(self, i: int) -> Dict[str, float]:
        return features.make(self.analyses[i], self.pa[i], self.pb[i], self.ps[i],
                             {k: v[i] for k, v in self.px.items()})

    def score(self, i: int) -> Optional[float]:
        c = self.cfg
        if self.hard[i] and c.get("hard_override", True):
            return 1.0
        if self._stack():
            return features.score(self.feats(i), c["stack"])
        if self._gate():
            # P_A ранжирует лучше рассуждения (AUC 0.91 против 0.72 на valid без hard), им и ранжируем
            return self.pa[i]
        if self.pb[i]:
            pt = float(np.mean(self.pb[i]))
            return pt if self.pa[i] is None else c["w_think"] * pt + (1 - c["w_think"]) * self.pa[i]
        return self.pa[i]

    def _text_only_blocked(self, i: int) -> bool:
        """text_only_policy = "hard_only": ход без вызовов инструментов (только текст клиенту) помечаем 1
        только по hard-правилам. Организаторы почти не считают такие ходы ошибками (valid: 2 из 22),
        а на паблике у v2.1 «текстовые» флаги дали основную массу ложных срабатываний."""
        if self.cfg.get("text_only_policy") != "hard_only" or self.hard[i]:
            return False
        fl = (self.analyses[i] or {}).get("flags", {}) or {}
        return bool(fl.get("has_text")) and int(fl.get("n_calls", 0) or 0) == 0

    def _call_blocked(self, i: int) -> bool:
        """call_policy = "hard_only" (v3.0, мир «Claude»): ход с вызовами инструментов помечаем 1 только
        по hard-правилам. На пуле из 195 не-hard ходов с вызовами позитивов 7, и ни один не набирает
        P_base >= 0.8 (11 флагов, все ложные)."""
        if self.cfg.get("call_policy") != "hard_only" or self.hard[i]:
            return False
        return self.is_call_turn(i)

    def is_call_turn(self, i: int) -> bool:
        fl = (self.analyses[i] or {}).get("flags", {}) or {}
        return int(fl.get("n_calls", 0) or 0) > 0

    def b_confirms(self, i: int) -> Optional[bool]:
        """Вердикт рассуждения по b_rule: any — хотя бы один сэмпл >= 0.5, all — все, mean — среднее.
        None — рассуждения нет (не успели или не кандидат)."""
        pb = self.pb[i]
        if not pb:
            return None
        rule = self.cfg.get("b_rule", "any")
        if rule == "all":
            return all(x >= 0.5 for x in pb)
        if rule == "mean":
            return float(np.mean(pb)) >= 0.5
        return any(x >= 0.5 for x in pb)

    def label(self, i: int) -> int:
        c = self.cfg
        if self.hard[i] and c.get("hard_override", True):
            return 1
        if self._text_only_blocked(i) or self._call_blocked(i):
            return 0
        if self._stack():
            s, thr = self.score(i), c["stack"]["threshold"]
            mode = c.get("b_mode")
            if mode == "gray_calls":
                # v2.3: s >= thr — флаг; ходы с вызовами в серой зоне [gray_lo, thr) — по рассуждению
                if s >= thr:
                    return 1
                if s >= c["gray_lo"] and self.is_call_turn(i):
                    return int(bool(self.b_confirms(i)))
                return 0
            if mode == "verify_flags":
                # v2.4: флаг стекинга подтверждается рассуждением; без рассуждения (не успели) — остаётся
                if s < thr:
                    return 0
                ok = self.b_confirms(i)
                return 1 if ok is None else int(ok)
            return int(s >= thr)
        if self._gate():
            # 1, если хотя бы один сэмпл рассуждения нашёл галлюцинацию И фаза A уверена (P_A >= gate_pa);
            # без рассуждения — только при очень высоком P_A
            pa = self.pa[i]
            if self.pb[i]:
                return int(any(x >= 0.5 for x in self.pb[i]) and pa is not None and pa >= c["gate_pa"])
            return int(pa is not None and pa >= c["threshold_a_only"])
        s = self.score(i)
        if s is None:
            return 0
        if self.pb[i]:
            return int(s >= c["threshold"])
        return int(s >= c["threshold_a_only"])

    def labels(self) -> List[int]:
        """Метки + потолок доли позитивов (max_pos_frac): в паблике позитивов ~15%, а судья на
        естественных ходах флагует ~33%. Если флагов больше потолка, оставляем hard-правила и
        самые уверенные по рассуждению (score), остальным ставим 0."""
        with self.lock:
            lab = [self.label(i) for i in range(self.n)]
            frac = self.cfg.get("max_pos_frac")
            if not frac or self.n == 0:
                return lab
            k = int(math.ceil(float(frac) * self.n))
            pos = [i for i in range(self.n) if lab[i]]
            if len(pos) <= k:
                return lab
            override = self.cfg.get("hard_override", True)
            if self._stack():
                pos.sort(key=lambda i: (bool(self.hard[i] and override), self.score(i) or 0.0), reverse=True)
            else:
                pos.sort(key=lambda i: (bool(self.hard[i] and override), bool(self.pb[i]),
                                        self.score(i) or 0.0, self.pa[i] or 0.0), reverse=True)
            keep = set(pos[:k]) | {i for i in pos if self.hard[i] and override}
            return [int(i in keep) for i in range(self.n)]


class Cache:
    """jsonl-кэш результатов (только для разработки в Colab: переживает разрыв сессии)."""

    def __init__(self, path: Optional[str], salt: str):
        self.path, self.salt, self.data = path, salt, {}
        self.lock = threading.Lock()
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for ln in f:
                    try:
                        d = json.loads(ln)
                        self.data[d["k"]] = d
                    except Exception:  # noqa: BLE001
                        pass

    def key(self, text_hash: str, stage: str) -> str:
        return f"{self.salt}|{text_hash}|{stage}"

    def get(self, text_hash: str, stage: str):
        return self.data.get(self.key(text_hash, stage))

    def put(self, text_hash: str, stage: str, **v):
        if not self.path:
            return
        d = {"k": self.key(text_hash, stage), **v}
        with self.lock:
            self.data[d["k"]] = d
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------- main
def run(args, t_start: float) -> None:
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    deadline = t_start + float(args.deadline or cfg["hard_deadline_s"])
    df = read_table(args.input)
    for col in ("id", "prompt", "response"):
        if col not in df.columns:
            raise SystemExit(f"во входе нет колонки '{col}'")
    if args.limit:
        df = df.iloc[: args.limit].reset_index(drop=True)
    df["id"] = df["id"].astype(str)
    df["prompt"] = df["prompt"].fillna("").astype(str)
    df["response"] = df["response"].fillna("").astype(str)
    ids = df["id"].tolist()
    n = len(df)
    log(f"[guardian] v{cfg['version']} | примеров: {n} | режим: {args.mode} | дедлайн: {deadline - t_start:.0f}s")

    st = State(ids, cfg)
    out_lock = threading.Lock()

    def flush(tag: str = "") -> None:
        with out_lock:
            write_preds(ids, st.labels(), args.output)
        if tag:
            log(f"[guardian] записано ({tag}): {sum(st.labels())} позитивов из {n} | t={time.time() - t_start:.0f}s")

    # ---- этап 0: правила
    analyses = []
    for i, r in enumerate(df.itertuples()):
        try:
            a = analyze(r.id, r.prompt, r.response)
        except Exception as e:  # noqa: BLE001
            a = {"hard": [], "soft": [], "facts": [], "domain": "unknown", "turn": None, "error": repr(e)}
        analyses.append(a)
        st.hard[i] = bool(a["hard"]) and args.mode in ("judge", "rules")
    st.analyses = analyses
    flush(f"правила, hard={sum(st.hard)}")
    if args.mode == "rules":
        return _finish(args, df, st, analyses, cfg, t_start, None)

    # ---- сервер
    served = cfg["served_name"]
    base = args.base_url
    import tempfile
    log_path = os.path.join(tempfile.gettempdir(), f"guardian_vllm_{os.getpid()}.log")
    procs: List[Any] = [None]

    def watchdog():
        # стартует ДО подъёма vLLM: дедлайн покрывает и загрузку модели
        while time.time() < deadline:
            time.sleep(1)
        log("[guardian] ЖЁСТКИЙ ДЕДЛАЙН: пишу текущие предсказания и выхожу")
        try:
            flush("дедлайн")
            _finish(args, df, st, analyses, cfg, t_start, None, quiet=True)
        finally:
            stop_vllm(procs[0])
            os._exit(0)

    threading.Thread(target=watchdog, daemon=True).start()

    if base is None:
        base = _start_server(args, cfg, served, log_path, deadline, t_start, procs)
        if base is None:
            log("[guardian] LLM недоступна: отдаю предсказания правил")
            flush("финал без LLM")
            return _finish(args, df, st, analyses, cfg, t_start, log_path)
        cfg["max_prompt_chars"] = min(cfg["max_prompt_chars"], int(cfg.get("_mml", cfg["max_model_len"]) * 2.4))
    proc = procs[0]

    try:
        _llm_stages(args, df, st, analyses, cfg, base, served, deadline, t_start, flush)
    finally:
        flush("финал")
        _finish(args, df, st, analyses, cfg, t_start, log_path)
        stop_vllm(proc)


def _start_server(args, cfg, served, log_path, deadline, t_start, procs) -> Optional[str]:
    """Поднимает vLLM с подстройкой под свободную память GPU и лестницей повторных попыток.
    Платформа: посторонние процессы могут занимать часть VRAM, а vLLM падает на старте, если свободно меньше
    gpu_memory_utilization * total. Ждём до дедлайна (без фиксированного потолка)."""
    try:
        from importlib.metadata import version as _ver
        vv = _ver("vllm")
    except Exception:  # noqa: BLE001
        vv = "?"
    mem = gpu_memory()
    util = float(cfg["gpu_memory_utilization"])
    if mem:
        free, total, used, name = mem
        log(f"[guardian] GPU: {name} | свободно {free / 1024:.1f} из {total / 1024:.1f} GiB "
            f"(занято другими {used / 1024:.1f}) | vLLM {vv}")
        util = min(util, (free - 1536) / total)
    else:
        log(f"[guardian] nvidia-smi недоступен | vLLM {vv}")
    extra = list(cfg.get("vllm_extra_args") or [])
    ladder = [(util, cfg["max_model_len"], extra),
              (util - 0.04, 65536, extra),
              (util - 0.08, 32768, extra + ["--enforce-eager"])]
    for k, (u, mml, ex) in enumerate(ladder, 1):
        left = deadline - time.time() - 240
        if left < 120:
            log("[guardian] не осталось времени на запуск vLLM")
            return None
        if u < 0.55:
            log(f"[guardian] мало свободной памяти GPU (util={u:.2f}), пробую всё равно 0.55")
            u = 0.55
        log(f"[guardian] старт vLLM (попытка {k}): util={u:.2f}, max_model_len={mml}, extra={ex}")
        proc = start_vllm(args.model_path, served, cfg["port"], mml, round(u, 3), cfg["max_num_seqs"], log_path, ex,
                          lm_only=cfg.get("language_model_only", True))
        procs[0] = proc
        try:
            took = wait_ready(cfg["port"], proc, left, log_path)
            cfg["_mml"] = mml
            log(f"[guardian] vLLM готов за {took:.0f}s (t={time.time() - t_start:.0f}s)")
            return f"http://127.0.0.1:{cfg['port']}/v1"
        except Exception as e:  # noqa: BLE001
            log(f"[guardian] vLLM не поднялся (попытка {k}): {str(e)[-2500:]}")
            stop_vllm(proc)
            procs[0] = None
            time.sleep(5)
    return None


def _llm_stages(args, df, st: State, analyses, cfg, base, served, deadline, t_start, flush) -> None:
    from openai import OpenAI

    client = OpenAI(base_url=base, api_key="EMPTY", timeout=1800, max_retries=0)
    if cfg.get("template") == "hf":
        set_hf_template(args.model_path)
    judge_sys = load_text("baseline_prompt.md" if args.mode == "baseline" else cfg.get("judge_prompt", "judge_system.md"))
    salt = hashlib.sha1((judge_sys + json.dumps({k: cfg.get(k) for k in ("temperature", "top_p", "top_k", "think_max_tokens",
                                                                       "evidence")})
                         + args.mode + str(args.model_path)).encode()).hexdigest()[:10]
    cache = Cache(args.cache, salt)
    n = len(df)
    max_chars = cfg["max_prompt_chars"]

    prompts_, responses_ = df["prompt"].tolist(), df["response"].tolist()

    def build_user(i: int, limit: int) -> str:
        if args.mode == "baseline":
            return baseline_user_content(prompts_[i], responses_[i], limit)
        ev = evidence_mod.build(prompts_[i], responses_[i]) if cfg.get("evidence") else ""
        return user_content(prompts_[i], responses_[i], analyses[i], limit, ev)

    users = [build_user(i, max_chars) for i in range(n)]
    hashes = [hashlib.sha1((prompts_[i] + "\x00" + responses_[i]).encode()).hexdigest()[:16] for i in range(n)]

    def too_long(e: Exception) -> bool:
        m = str(e)
        return "maximum context length" in m or "max_model_len" in m or "too long" in m

    def complete(prompt: str, max_tokens: int, temperature: float, logprobs: int, seed: int, stop=None):
        last = None
        extra = {"skip_special_tokens": False}
        if temperature > 0:
            extra["top_k"] = cfg["top_k"]
        for attempt in range(2):
            try:
                return client.completions.create(
                    model=served, prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                    top_p=cfg["top_p"] if temperature > 0 else 1.0, logprobs=logprobs, seed=seed, stop=stop,
                    extra_body=extra)
            except Exception as e:  # noqa: BLE001
                last = e
                if too_long(e):
                    raise
                time.sleep(1 + attempt)
        raise RuntimeError(f"запрос не удался: {last}")

    # порядок: соседние ходы одной траектории рядом -> общий префикс в кэше vLLM
    order = sorted(range(n), key=lambda i: (st.ids[i].split("::")[0], len(users[i])))
    todo = [i for i in order if not (st.hard[i] and not args.llm_on_hard)]

    # ---- фаза A
    def phase_a(i: int):
        c = cache.get(hashes[i], "A")
        if c:
            return i, c["p"]
        try:
            r = complete(chat(judge_sys, users[i], thinking=False), 1, 0.0, 20, seed=1)
        except Exception as e:  # noqa: BLE001
            if not too_long(e):
                raise
            users[i] = build_user(i, max_chars // 2)
            r = complete(chat(judge_sys, users[i], thinking=False), 1, 0.0, 20, seed=1)
        lp = r.choices[0].logprobs
        p = p_from_top(lp.top_logprobs[0] if lp and lp.top_logprobs else None)
        if p is None:
            p = 0.5
        cache.put(hashes[i], "A", p=p)
        return i, p

    if cfg.get("phase_a", True) or args.mode == "baseline":
        t0 = time.time()
        _run_pool(todo, phase_a, cfg["concurrency_a"], deadline - 30, "A", t_start,
                  on_result=lambda i, p: _set(st, i, "pa", p))
        log(f"[guardian] фаза A: {sum(p is not None for p in st.pa)}/{n} за {time.time() - t0:.0f}s")
        flush("после фазы A")
    if args.mode == "baseline":
        return

    # ---- фаза A с дополнительными промптами (признаки для стекинга): бейзлайн, судья v1
    for name, spec in (cfg.get("extra_a") or {}).items():
        sys_x = load_text(spec["system"])
        kind = spec.get("user", "judge")

        def user_x(i: int, limit: int, kind=kind) -> str:
            if kind == "baseline":
                return baseline_user_content(prompts_[i], responses_[i], limit)
            return user_content(prompts_[i], responses_[i], analyses[i], limit, "")

        def phase_x(i: int, name=name, sys_x=sys_x, user_x=user_x):
            c = cache.get(hashes[i], f"A_{name}")
            if c:
                return i, c["p"]
            try:
                r = complete(chat(sys_x, user_x(i, max_chars), thinking=False), 1, 0.0, 20, seed=1)
            except Exception as e:  # noqa: BLE001
                if not too_long(e):
                    raise
                r = complete(chat(sys_x, user_x(i, max_chars // 2), thinking=False), 1, 0.0, 20, seed=1)
            lp = r.choices[0].logprobs
            p = p_from_top(lp.top_logprobs[0] if lp and lp.top_logprobs else None)
            p = 0.5 if p is None else p
            cache.put(hashes[i], f"A_{name}", p=p)
            return i, p
        st.px[name] = [None] * n
        t0 = time.time()
        _run_pool(todo, phase_x, cfg["concurrency_a"], deadline - 90, f"A_{name}", t_start,
                  on_result=lambda i, p, name=name: _set_px(st, name, i, p))
        log(f"[guardian] фаза A_{name}: {sum(p is not None for p in st.px[name])}/{n} за {time.time() - t0:.0f}s")

    # ---- фаза S: проверка текста по предложениям (одна генерация из k цифр на пример)
    if cfg.get("sentence_probe"):
        def phase_s(i: int):
            c = cache.get(hashes[i], "S")
            if c:
                return i, c["ps"]
            sents = sentences(_resp_text(responses_[i]), cfg.get("sentence_k", 8))
            if not sents:
                return i, []
            r = complete(chat(judge_sys, sentence_task(users[i], sents), thinking=False),
                         2 * len(sents) + 2, 0.0, 5, seed=3)
            ch = r.choices[0]
            ps = parse_digit_probs(ch, len(sents))
            cache.put(hashes[i], "S", ps=ps)
            return i, ps
        todo_s = [i for i in todo if analyses[i].get("flags", {}).get("has_text")]
        t0 = time.time()
        _run_pool(todo_s, phase_s, cfg["concurrency_a"], deadline - 60, "S", t_start,
                  on_result=lambda i, v: _set_list(st, i, "ps", v))
        log(f"[guardian] фаза S: {sum(bool(x) for x in st.ps)}/{len(todo_s)} за {time.time() - t0:.0f}s")
        flush("после фазы S")
    if not cfg.get("phase_b", True):
        return

    # ---- фаза B (рассуждение)
    speed = {"tok_s": float(cfg.get("assumed_seq_tok_s", 20.0)), "n": 0}
    stop_launch = deadline - cfg["think_stop_margin_s"]
    b_sys = load_text(cfg["judge_b_prompt"]) if cfg.get("judge_b_prompt") else judge_sys
    b_tag = "" if b_sys == judge_sys else "_" + hashlib.sha1(b_sys.encode()).hexdigest()[:6]

    def think(i: int, sample: int):
        stage = f"B{sample}{b_tag}"
        c = cache.get(hashes[i], stage)
        if c:
            return i, c
        left = stop_launch - time.time()
        max_tok = int(min(cfg["think_max_tokens"], left * speed["tok_s"] * 0.85))
        if max_tok < cfg["think_min_tokens"]:
            return i, None
        seed = 1000 * sample + (int(hashes[i][:6], 16) % 997)
        tp = chat(b_sys, users[i], thinking=True)
        t0 = time.time()
        try:
            r = complete(tp, max_tok + 8, cfg["temperature"], int(cfg.get("think_logprobs", 5)), seed=seed)
        except Exception as e:  # noqa: BLE001
            if not too_long(e):
                raise
            users[i] = build_user(i, max_chars // 2)
            tp = chat(b_sys, users[i], thinking=True)
            r = complete(tp, max_tok + 8, cfg["temperature"], int(cfg.get("think_logprobs", 5)), seed=seed)
        ch = r.choices[0]
        ntok = int(getattr(r.usage, "completion_tokens", 0) or 0)
        dt = max(1e-3, time.time() - t0)
        with st.lock:
            k = speed["n"]
            speed["tok_s"] = (speed["tok_s"] * min(k, 20) + ntok / dt) / (min(k, 20) + 1)
            speed["n"] = k + 1
        res = parse_think_choice(ch)
        if not res["done"]:
            if time.time() > deadline - 20:
                return i, None
            reasoning = res["reasoning"].replace("<|im_end|>", "").replace("<|endoftext|>", "")
            r2 = complete(close_thinking(tp, reasoning, truncated=True), 1, 0.0, 20, seed=7)
            lp = r2.choices[0].logprobs
            p = p_from_top(lp.top_logprobs[0] if lp and lp.top_logprobs else None)
            res = {"done": p is not None, "reasoning": reasoning, "p": p if p is not None else 0.5}
        out = {"p": float(res["p"]), "reasoning": res["reasoning"][-6000:], "ntok": ntok, "src": res.get("src", "forced")}
        cache.put(hashes[i], stage, **out)
        return i, out

    def on_think(i, out):
        if out is None:
            return
        with st.lock:
            st.pb[i].append(out["p"])
            if not st.reason[i]:
                st.reason[i] = out["reasoning"]
            st.think_tokens[i] += out.get("ntok", 0)

    # Порядок фазы B. Фаза A почти всё помечает «1» (на valid recall 1.0 при precision 0.62), поэтому
    # рассуждением в первую очередь проверяем вероятные позитивы (pa_desc): если время кончится,
    # непроверенными останутся примеры с низким P_A, и они честно получат 0.
    pos = {i: k for k, i in enumerate(order)}
    mode = cfg.get("b_mode")
    if mode in ("gray_calls", "verify_flags") and st._stack():
        _b_targeted(st, cfg, todo, pos, think, on_think, flush, deadline, stop_launch, t_start, mode)
        return
    if cfg.get("b_order", "pa_desc") == "pa_desc":
        todo_b = sorted(todo, key=lambda i: (-(st.pa[i] if st.pa[i] is not None else 0.5), pos[i]))
    else:  # uncertain_first (v1)
        def tier(i):
            pa = st.pa[i]
            return 0 if pa is None or abs(pa - 0.5) < 0.35 else 1
        todo_b = sorted(todo, key=lambda i: (tier(i), pos[i]))
    if cfg.get("b_skip_ro") and not args.llm_on_hard:
        # ходы только с читающими вызовами (поиск, get, calculate): на пуле 2 не-hard позитива из 170
        todo_b = [i for i in todo_b if not analyses[i].get("flags", {}).get("ro_only")]
    b_min = cfg.get("b_min_pa")
    if b_min is not None and not args.llm_on_hard:
        # в режиме gate пример с P_A < gate_pa не станет позитивом, рассуждение на нём не нужно
        todo_b = [i for i in todo_b if st.pa[i] is None or st.pa[i] >= b_min]
    log(f"[guardian] кандидатов на рассуждение: {len(todo_b)} из {n}")
    t0 = time.time()
    _run_pool(todo_b, lambda i: think(i, 1), cfg["concurrency_b"], deadline - 15, "B1", t_start,
              on_result=on_think, flush=flush, flush_every=cfg["write_every_s"])
    log(f"[guardian] фаза B1: {sum(bool(x) for x in st.pb)}/{len(todo_b)} за {time.time() - t0:.0f}s, "
        f"скорость ~{speed['tok_s']:.0f} ток/с на запрос")
    flush("после B1")

    if cfg.get("second_sample", True) and time.time() < stop_launch - 60:
        lo, hi = cfg["second_band"]
        cand = [i for i in todo_b if st.pb[i] and lo <= st.pb[i][0] <= hi]
        if cfg.get("decision", "fused") == "gate":
            # правило «хотя бы один голос 1»: второй сэмпл меняет решение, только если первый сказал 0
            cand = [i for i in cand if st.pb[i][0] < 0.5]
        elif cfg.get("decision") == "stack" and cfg.get("b2_all"):
            cand = [i for i in todo_b if st.pb[i]]
        cand.sort(key=lambda i: abs(st.pb[i][0] - 0.5))
        if cand:
            t0 = time.time()
            _run_pool(cand, lambda i: think(i, 2), cfg["concurrency_b"], deadline - 15, "B2", t_start,
                      on_result=on_think, flush=flush, flush_every=cfg["write_every_s"])
            log(f"[guardian] фаза B2: {len(cand)} кандидатов за {time.time() - t0:.0f}s")


def _b_targeted(st: State, cfg, todo, pos, think, on_think, flush, deadline, stop_launch, t_start, mode) -> None:
    """Рассуждение только там, где оно меняет решение (оценка стекинга считается по фазам A, без B).
    gray_calls (v2.3): ходы с вызовами в серой зоне [gray_lo, threshold), сначала самые уверенные;
    verify_flags (v2.4): все не-hard флаги стекинга, сначала наименее уверенные.
    Второй сэмпл — только там, где он может изменить вердикт по b_rule."""
    thr = cfg["stack"]["threshold"]
    sc = {i: (st.score(i) or 0.0) for i in todo}
    if mode == "gray_calls":
        lo = cfg["gray_lo"]
        cand = [i for i in todo if not st.hard[i] and st.is_call_turn(i) and lo <= sc[i] < thr]
        cand.sort(key=lambda i: (-sc[i], pos[i]))
    else:
        cand = [i for i in todo if not st.hard[i] and not st._text_only_blocked(i) and sc[i] >= thr]
        cand.sort(key=lambda i: (sc[i], pos[i]))
    log(f"[guardian] рассуждение ({mode}, правило {cfg.get('b_rule', 'any')}): кандидатов {len(cand)} из {st.n}")
    t0 = time.time()
    _run_pool(cand, lambda i: think(i, 1), cfg["concurrency_b"], deadline - 15, "B1", t_start,
              on_result=on_think, flush=flush, flush_every=cfg["write_every_s"])
    log(f"[guardian] фаза B1: {sum(bool(st.pb[i]) for i in cand)}/{len(cand)} за {time.time() - t0:.0f}s")
    flush("после B1")
    if not cfg.get("second_sample", True) or time.time() >= stop_launch - 60:
        return
    rule = cfg.get("b_rule", "any")
    if rule == "all":      # флаг только при согласии двух сэмплов: второй нужен, если первый сказал «1»
        c2 = [i for i in cand if st.pb[i] and st.pb[i][0] >= 0.5]
    elif rule == "any":    # флаг, если хотя бы один сэмпл «1»: второй нужен, если первый сказал «0»
        c2 = [i for i in cand if st.pb[i] and st.pb[i][0] < 0.5]
    else:
        c2 = [i for i in cand if st.pb[i]]
    if c2:
        t0 = time.time()
        _run_pool(c2, lambda i: think(i, 2), cfg["concurrency_b"], deadline - 15, "B2", t_start,
                  on_result=on_think, flush=flush, flush_every=cfg["write_every_s"])
        log(f"[guardian] фаза B2: {len(c2)} кандидатов за {time.time() - t0:.0f}s")


def _set(st: State, i: int, field: str, v) -> None:
    with st.lock:
        getattr(st, field)[i] = v


def _set_px(st: State, name: str, i: int, v) -> None:
    with st.lock:
        st.px[name][i] = v


def _set_list(st: State, i: int, field: str, v) -> None:
    with st.lock:
        getattr(st, field)[i] = list(v or [])


def _run_pool(items, fn, workers, stop_at, tag, t_start, on_result, flush=None, flush_every=30):
    if not items:
        return
    try:
        from tqdm import tqdm
        bar = tqdm(total=len(items), desc=f"[{tag}]", mininterval=10, ncols=100)
    except Exception:  # noqa: BLE001
        bar = None
    done, errors, last_flush = 0, 0, time.time()
    ex = ThreadPoolExecutor(max_workers=workers)
    futs = [ex.submit(fn, i) for i in items]
    try:
        for f in as_completed(futs, timeout=max(1.0, stop_at - time.time())):
            try:
                i, v = f.result()
                on_result(i, v)
            except Exception as e:  # noqa: BLE001
                errors += 1
                if errors <= 5:
                    log(f"[{tag}] ошибка запроса: {str(e)[:300]}")
            done += 1
            if bar is not None:
                bar.update(1)
            elif done % 50 == 0:
                log(f"[{tag}] {done}/{len(items)} t={time.time() - t_start:.0f}s")
            if flush is not None and time.time() - last_flush > flush_every:
                flush()
                last_flush = time.time()
    except Exception:  # noqa: BLE001 — TimeoutError от as_completed: время вышло
        log(f"[{tag}] остановлено по времени: готово {done}/{len(items)}")
    finally:
        if bar is not None:
            bar.close()
        for f in futs:
            f.cancel()
        ex.shutdown(wait=False, cancel_futures=True)
    if errors:
        log(f"[{tag}] ошибок: {errors}")


def _finish(args, df, st: State, analyses, cfg, t_start, log_path, quiet=False) -> None:
    if getattr(args, "dump", None):
        rows = []
        for i in range(st.n):
            rows.append({
                "id": st.ids[i], "domain": analyses[i].get("domain"), "hard": bool(analyses[i].get("hard")),
                "hard_reasons": " | ".join(analyses[i].get("hard", [])), "soft": " | ".join(analyses[i].get("soft", [])),
                "p_nothink": st.pa[i], "p_think": float(np.mean(st.pb[i])) if st.pb[i] else None,
                "p_think_samples": json.dumps(st.pb[i]), "think_tokens": st.think_tokens[i],
                "p_sent": json.dumps(st.ps[i]), "feats": json.dumps(st.feats(i)),
                "p_extra": json.dumps({k: v[i] for k, v in st.px.items()}),
                "score": st.score(i), "pred": st.label(i), "reasoning": st.reason[i],
            })
        d = pd.DataFrame(rows)
        if "label" in df.columns:
            d["label"] = df["label"].values
        d.to_parquet(args.dump, index=False)
    if not quiet:
        log(f"[guardian] готово: {sum(st.labels())} позитивов из {st.n}, время {time.time() - t_start:.0f}s")
