#!/usr/bin/env python3
"""Small factual smoke test for a LittleBit GGUF export.

This complements perplexity: a compact head can pass an averaged PPL check while
flipping an individual answer. Run with the compiled fork (CPU only):

  python3 tools/littlebit-factual-smoke.py \\
      --model /path/to/littlebit-qwen3-8b-lb1-v3.gguf \\
      --binary build/bin/llama-completion

Expected substrings are deliberately short and case-insensitive. The script
prints every completion so a reviewer can inspect the full answer, and exits
non-zero if any expected answer is missing.
"""

from __future__ import annotations

import argparse
import os
import subprocess

CASES = [
    ("The capital of France is", "paris"),
    ("The capital of Japan is", "tokyo"),
    ("The largest planet in the Solar System is", "jupiter"),
    ("The chemical symbol for gold is", "au"),
    ("The first person to walk on the Moon was", "armstrong"),
    ("The Earth's natural satellite is called the", "moon"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--binary", default="build/bin/llama-completion")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    env = os.environ.copy()
    libdir = os.path.join(os.path.dirname(os.path.abspath(args.binary)), "")
    env["LD_LIBRARY_PATH"] = libdir + env.get("LD_LIBRARY_PATH", "")

    failures = []
    for prompt, expected in CASES:
        cmd = [
            args.binary, "-m", args.model, "-no-cnv", "-ngl", "0",
            "--temp", "0", "-t", str(args.threads), "-c", "48",
            "-p", prompt, "-n", "12",
        ]
        out = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True).stdout.strip()
        ok = expected in out.lower()
        print(f"[{'PASS' if ok else 'FAIL'}] {prompt!r} -> {out!r}")
        if not ok:
            failures.append((prompt, expected))

    if failures:
        print(f"\n{len(failures)} factual check(s) failed")
        raise SystemExit(1)
    print(f"\nAll {len(CASES)} factual checks passed")


if __name__ == "__main__":
    main()
