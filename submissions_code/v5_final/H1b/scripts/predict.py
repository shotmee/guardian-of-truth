#!/usr/bin/env python
# Platform entry point. Example:
#   python scripts/predict.py --input <test> --output <predictions>
import pathlib
import sys

_here = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_here.parent.parent / "src"))

from veritas.cli import main as _main  # noqa: E402


if __name__ == "__main__":
    _main()
