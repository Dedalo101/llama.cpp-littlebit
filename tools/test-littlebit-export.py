#!/usr/bin/env python3
"""Small regression tests for LittleBit export metadata validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("littlebit_export", HERE / "littlebit-export.py")
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def expect_exit(cfg: dict, header: dict) -> None:
    try:
        mod.validate_tie_config(cfg, header)
    except SystemExit:
        return
    raise AssertionError(f"expected SystemExit for cfg={cfg}, header={header}")


def main() -> None:
    head = {"lm_head.weight": {}}
    no_head = {}

    mod.validate_tie_config({}, head)
    mod.validate_tie_config({}, no_head)

    mod.validate_tie_config({"tie_word_embeddings": False}, head)
    expect_exit({"tie_word_embeddings": False}, no_head)

    mod.validate_tie_config({"tie_word_embeddings": True}, no_head)
    expect_exit({"tie_word_embeddings": True}, head)

    print("LittleBit export validation tests passed")


if __name__ == "__main__":
    main()
