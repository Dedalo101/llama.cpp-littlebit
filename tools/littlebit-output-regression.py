#!/usr/bin/env python3
"""Exact-output regression check for a LittleBit GGUF export.

The published LittleBit checkpoint is a format demonstration (WikiText PPL
~21), not a chat model. It cannot be held to a world-facts gate. This script
instead checks that a build or runtime change reproduces the recorded greedy
outputs of the validated v3 export, which is the useful signal for catching
argmax flips from kernel, head or packing changes.

  python3 tools/littlebit-output-regression.py \
      --model /path/to/littlebit-qwen3-8b-lb1-v3.gguf \
      --binary build/bin/llama-completion

Outputs were recorded at -t 4, -c 48, -n 12, --temp 0 on the x86 test host;
they are deterministic for the shipping GGUF.
"""

from __future__ import annotations

import argparse
import os
import subprocess

GOLDEN = [
    (
        "The capital of France is",
        "The capital of France is Paris, and the city is known for its vibrant culture,",
    ),
    (
        "The capital of Japan is",
        "The capital of Japan is the city of the city of Tokyo, which is the capital",
    ),
    (
        "The largest planet in the Solar System is",
        "The largest planet in the Solar System is the Sun. The Sun is the only living planet in the",
    ),
    (
        "The chemical symbol for gold is",
        "The chemical symbol for gold is 1000. The symbol for silver is",
    ),
    (
        "The first person to walk on the Moon was",
        "The first person to walk on the Moon was the first person to walk on the Moon. The first person",
    ),
    (
        "The Earth's natural satellite is called the",
        "The Earth's natural satellite is called the Earth's satellite. The Earth's satellite is a satellite that",
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--binary", default="build/bin/llama-completion")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    env = os.environ.copy()
    libdir = os.path.dirname(os.path.abspath(args.binary))
    env["LD_LIBRARY_PATH"] = libdir + os.pathsep + env.get("LD_LIBRARY_PATH", "")

    failures = []
    for prompt, expected in GOLDEN:
        cmd = [
            args.binary, "-m", args.model, "-no-cnv", "-ngl", "0",
            "--temp", "0", "-t", str(args.threads), "-c", "48",
            "-p", prompt, "-n", "12",
        ]
        out = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True).stdout.strip()
        ok = out == expected
        print(f"[{'PASS' if ok else 'FAIL'}] {prompt!r}")
        if not ok:
            print(f"  expected: {expected!r}")
            print(f"  actual:   {out!r}")
            failures.append(prompt)

    if failures:
        print(f"\n{len(failures)} output regression(s) detected")
        raise SystemExit(1)
    print(f"\nAll {len(GOLDEN)} greedy outputs match the v3 baseline")


if __name__ == "__main__":
    main()
