# LittleBit Qwen3-8B LB1 v3 (draft model card)

> Draft for the public model upload. Replace the placeholder repository name at
> upload time. Do not publish before the license review in
> `LITTLEBIT_NOTICE.md` is acknowledged.

## Summary

A 1.0 bpw-class LittleBit LB1 export of Qwen3-8B for CPU-only inference. The
36 transformer projections use packed ±1 LittleBit factors (`GGML_TYPE_LB1 =
44`); the token embedding and the untied output head stay Q8_0 because compact
heads measurably damaged factual answers.

The model is experimental. It is not a general-purpose replacement for Qwen3
and is not affiliated with SamsungLabs or the llama.cpp project.

## Files and layout

| Item | Value |
|---|---|
| Architecture | Qwen3, 36 layers, 4096 hidden, 151936 vocab |
| File | `littlebit-qwen3-8b-lb1-v3.gguf` |
| Size | 1,810,170,944 B (1.69 GiB) |
| SHA256 | `3bf86e86da8dc1d84be5725fa6381d329b2113185bf1e2e7f0c2945732acb346` |
| Tensors | 2667 |
| LB1 bytes/forward | 468,910,080 |
| Q8_0 head bytes/forward | 661,225,472 |
| Total bytes/forward @1 token | 1,130,135,552 |
| Token embedding | Q8_0, untied |
| Output head | Q8_0, untied |

## Quality

| Gate | Result |
|---|---|
| WikiText-2 raw test, 12 windows, c=2048 | 21.2364 ± 0.59425 (final F32 kernel, full rerun) |
| Target | 20.88 ± 0.5 |
| Acceptance ceiling | <= 23 |
| Earlier P32-equivalent kernel | 21.2364 ± 0.59424; chunk deltas <= 0.0016 |

Rejected variants, kept as negative evidence:

- Q4_K output head: chunk-1 PPL 25.3521 vs 24.5867 (Q8_0). Prompt "The capital
  of France is Paris, and the city is known for its vibrant culture" became
  "...known for its vibrant culture, located in the French Riviera" — an
  individual factual flip that the averaged PPL alone did not reveal.
- Q6_K output head: chunk-1 PPL 24.7543 (+0.1676, gate < +0.1) and "vibrant
  streets" instead of "vibrant culture".

## Runtime

Requires the LittleBit fork on branch `littlebit-tensor`. No GPU, Ollama or
build step is needed for the portable correctness artifact; the compiled fork
is the fast path.

```sh
# Portable check (Python 3 + numpy)
python3 tools/littlebit_cpu.py littlebit-qwen3-8b-lb1-v3.gguf \
  -p 'The capital of France is' -n 8

# Compiled fork
LD_LIBRARY_PATH=build/bin ./build/bin/llama-completion \
  -m littlebit-qwen3-8b-lb1-v3.gguf -no-cnv -ngl 0 --temp 0 \
  -c 128 -p 'The capital of France is' -n 32
```

Decode profiling: `LLAMA_LB1_PROFILE=1`. On the x86 test host ~90% of decode is
the LB1 factor path and ~6-7% is the Q8_0 head. The head already uses int8
activations and int8 dot in baseline llama.cpp.

Peak RSS measured on the x86 host: 1864.4 MB at `-c 128 -t 8`. Tokens/s are not
quoted here because the test host was shared with other workloads; measure on
quiet hardware (or the target ARM board) before publishing headline numbers.

## Licensing

- llama.cpp and ggml: MIT.
- LittleBit technique / upstream project: CC BY-NC 4.0. Review before any
  commercial use. No upstreaming to llama.cpp or Ollama without legal review.
- Base Qwen3 checkpoint: Apache-2.0.

See `LITTLEBIT_NOTICE.md` and `docs/littlebit-design.md` in the runtime fork.
