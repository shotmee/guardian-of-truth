#!/usr/bin/env python
"""Точка входа платформы: python scripts/predict.py --input <test> --output <predictions>."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from veritas.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
