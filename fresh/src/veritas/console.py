"""Единый формат строк лога (их видно в логах платформы)."""
import time


def log(message: str) -> None:
    print(f"[veritas {time.strftime('%H:%M:%S')}] {message}", flush=True)
