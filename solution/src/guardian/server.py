"""Подъём / ожидание / остановка vLLM OpenAI-совместимого сервера (как в бейзлайне, + лог в файл и проверка живости)."""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from typing import List, Optional


def start_vllm(model_path: str, served_name: str, port: int, max_model_len: int, gpu_util: float,
               max_num_seqs: int, log_path: str, extra_args: Optional[List[str]] = None,
               lm_only: bool = True) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_path),
        "--served-model-name", served_name,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", str(gpu_util),
        "--max-num-seqs", str(max_num_seqs),
        "--enable-prefix-caching",    # общий префикс: системный промпт + policy + история траектории
    ]
    if lm_only:
        cmd.append("--language-model-only")  # модель мультимодальная (Qwen3.5/3.6), нужен только текст
    if extra_args:
        cmd += list(extra_args)
    env = dict(os.environ)
    env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    log = open(log_path, "ab", buffering=0)
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)


def gpu_memory():
    """(free_MiB, total_MiB, used_MiB, name) первой GPU через nvidia-smi или None."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total,memory.used,name", "--format=csv,noheader,nounits"],
            timeout=20).decode().strip().splitlines()[0]
        free, total, used, name = [x.strip() for x in out.split(",", 3)]
        return float(free), float(total), float(used), name
    except Exception:  # noqa: BLE001
        return None


def tail(path: str, n: int = 60) -> str:
    try:
        with open(path, "rb") as f:
            return b"\n".join(f.read().splitlines()[-n:]).decode("utf-8", "replace")
    except OSError:
        return ""


def wait_ready(port: int, proc: Optional[subprocess.Popen], timeout: float, log_path: str = "") -> float:
    url = f"http://127.0.0.1:{port}/v1/models"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"vLLM завершился с кодом {proc.returncode}:\n{tail(log_path)}")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM не поднялся за {timeout:.0f}s:\n{tail(log_path)}")


def stop_vllm(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=20)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
