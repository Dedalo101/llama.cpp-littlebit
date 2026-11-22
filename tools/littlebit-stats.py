#!/usr/bin/env python3
"""Report LittleBit GGUF weight bytes and bandwidth per token."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("littlebit_cpu", HERE / "littlebit_cpu.py")
assert spec is not None and spec.loader is not None
cpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpu)

GT_LB1 = 44


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", help="LittleBit GGUF file")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 512, 2048],
                    help="token batch sizes to report")
    args = ap.parse_args()

    path = Path(args.gguf)
    g = cpu.GGUF(str(path))
    try:
        lb1 = 0
        head = 0
        for name, t in g.tensors.items():
            if t["type"] == GT_LB1:
                lb1 += t["nbytes"]
            if name == "output.weight":
                head += t["nbytes"]

        total = lb1 + head
        print(f"file            : {path}")
        print(f"file bytes      : {path.stat().st_size}")
        print(f"LB1 factor bytes: {lb1}")
        print(f"output head     : {head}")
        print(f"bytes / forward : {total}")
        for batch in args.batch:
            print(f"bytes / token @{batch:<5}: {total/batch:,.0f}")
    finally:
        g.close()


if __name__ == "__main__":
    main()
