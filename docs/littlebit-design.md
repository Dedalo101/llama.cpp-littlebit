# LittleBit tensor type for llama.cpp / Ollama — design preview

Status: **design only, no code changed.** Branch `littlebit-tensor` in
`~/Work/llama.cpp-littlebit`, forked from Ollama's pinned llama.cpp tag
`b12257` (commit `17701e4`). Nothing pushed.

## Why this pin matters

Ollama's `cmake/local.cmake` supports a source override:

```
cmake -DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=$HOME/Work/llama.cpp-littlebit ...
# or: OLLAMA_LLAMA_CPP_SOURCE=$HOME/Work/llama.cpp-littlebit
```

For an override it sets `OLLAMA_LLAMA_CPP_SKIP_COMPAT_PATCH=ON`. That means
**our fork must already be API-compatible with Ollama's pinned llama.cpp**.
Baselining on `main` would break the build. Pin: `b12257`.

Ollama builds with `GGML_CPU_ALL_VARIANTS=ON` and `GGML_BACKEND_DL=ON`, so
any new type/op must be registered in **every CPU variant** built from
`ggml/src/ggml-cpu/CMakeLists.txt`, not just the generic build.

## The math (confirming the requested op sequence)

The deployed layer is:

```
W = diag(u1) @ Uq @ diag(v1*u2) @ Vq @ diag(v2)
Uq = sign(U), Vq = sign(V)   # already binarised in the checkpoint
```

llama.cpp stores linear weights as `[out, in]` (ggml `ne = (in, out)`) and
computes `x @ Wᵀ`. Transposing the equation gives the requested chain:

```
h = x * v2                 # in
h = h @ Vqᵀ                # in  -> split
h = h * (v1 * u2)          # split
h = h @ Uqᵀ                # split -> out
y = h * u1                 # out
```

With `residual=True` there is a **second independent factorisation** (`_R`
tensors) whose output is added: `y = y + y_R`. The op must run twice.

Note the tensor orientations that avoid any transpose at graph build time:

- `Vq` is stored `ne = (in, split)` → `ggml_mul_mat(Vq, h)` gives `split`.
- `Uq` is stored `ne = (split, out)` → `ggml_mul_mat(Uq, h)` gives `out`.

So the five steps are two `mul_mat` calls plus three elementwise multiplies.

## GGUF layout

### One new low-level type: `GGML_TYPE_LB1`

A packed ±1 bit array with **block = 32 bits = one int32 word = 4 bytes**.

- `ggml_blck_size(LB1) = 32`
- `ggml_type_size(LB1) = 4`
- bit mapping: `+1 -> 0`, `-1 -> 1`, LSB-first, pad with `+1`, unpack `(1 - 2*bit)`

Why 32 and not 256: checkpoint split dims are **544, 224, 832**. These are
all divisible by 32 (17, 7, 26 words) but not by 64/128/256. ggml requires
`ne[0] % blck_size == 0`, so a 256-bit block would force repacking every
factor. A 32-bit block matches `U_packed`/`V_packed` byte-for-byte, so the
exporter can copy the int32 arrays straight through.

### Five tensors per replaced `nn.Linear`

For `blk.0.attn_q.weight` (the original dense tensor is **not** emitted):

| GGUF tensor | type | ggml `ne` | source checkpoint key |
|---|---|---|---|
| `blk.0.attn_q.lb.u1` | F16 | `(out)` | `u1` |
| `blk.0.attn_q.lb.Uq` | LB1 | `(split, out)` | `U_packed` |
| `blk.0.attn_q.lb.m`  | F16 | `(split)` | `v1 * u2` (pre-multiplied) |
| `blk.0.attn_q.lb.Vq` | LB1 | `(in, split)` | `V_packed` |
| `blk.0.attn_q.lb.v2` | F16 | `(in)` | `v2` |
| `blk.0.attn_q.lb_R.*` | same | same | `_R` variants, if `residual=True` |

`split` is recoverable from `Uq.ne[0]`, so no extra shape scalar is needed.
Copy `_split_dim_final`, `_eff_bit_target`, `_eff_bit_actual` into GGUF
metadata for provenance. Total ≈ 252 × 5 (+ residual) tensors.

Fold `v1 * u2` into one F16 vector in the exporter: it is constant at
inference time and removes one multiply from the hot path.

### Metadata

```
littlebit.version          u32   = 1
littlebit.block_bits       u32   = 32
littlebit.residual         bool
littlebit.split_dims       str   (JSON map of tensor prefix -> split)
littlebit.source_checkpoint str
```

## Code touch points (verified in this tree)

1. `ggml/include/ggml.h` — add `GGML_TYPE_LB1` before `GGML_TYPE_COUNT`
   (currently 44) and `GGML_OP_MUL_MAT_LB1` in the op enum.
2. `ggml/src/ggml.c` — add type traits at `type_traits[GGML_TYPE_COUNT]`
   (~line 639), extend `ggml_type_name`, `ggml_dup_tensor`, broadcast helpers.
3. `ggml/src/gguf.cpp` + `ggml/include/gguf.h` — map the new type in
   `gguf_type_name` / tensor type validation.
4. `ggml/src/ggml-quants.h/.c` — `dequantize_row_lb1` reference implementation.
5. `ggml/src/ggml-cpu/` — scalar `ggml_compute_forward_mul_mat_lb1`, then
   NEON. Register in the CPU backend op table for **all variants**.
6. `ggml/src/ggml.c` — `ggml_op_name`, `ggml_op_desc`, graph validation.
7. `src/llama-graph.cpp` — `build_ffn` (~line 1983) and `build_attn` variants
   (~3236, 3331, 3446, 3554, 3717) replace `ggml_mul_mat(w, cur)` with a
   helper that detects `lb.*` tensors and builds the 5-step chain.
8. `src/llama-model-loader.cpp` — tensor lookup (~952, 1394) must not reject
   the missing original `.weight` when `lb.*` siblings exist.
9. `gguf-py/gguf/gguf_writer.py` — allow the new tensor type in the exporter.
10. New `tools/littlebit-export.py` — reads a safetensors checkpoint via
    HTTP range reads (reuse `runtime/inspect_checkpoint.py` from the spike),
    writes GGUF. No torch needed.

## Staged plan

| Stage | Deliverable | Verify with |
|---|---|---|
| 0 | numpy oracle already exists | `runtime/selftest.py`, `remote_probe.py` |
| 1 | LB1 type + traits + GGUF reader + exporter | decode exported LB1 back to numpy, compare bit-for-bit and layer output vs oracle |
| 2 | scalar CPU `mul_mat_lb1` + graph integration | llama-cli loads exported GGUF, generates one token |
| 3 | full tiny model generating, PPL vs `eval.py` | perplexity delta recorded |
| 4 | NEON kernel on ARM | tokens/s + MB/s on Pi / Oracle ARM |
| 5 | Ollama local build with source override | `ollama run littlebit-qwen3` through the custom payload |

## Decisions made 2026-11-22

- **Stage 1 includes embedding tying + quantisation.** Untied FP16
  `embed_tokens` + `lm_head` = 2,489 MB of 2,968 MB. Tying halves that to
  ~1.24 GB; Q8_0 on top brings it to ~0.65 GB and total file to ~0.75 GB.
  Q4_K (~0.33 GB, total ~0.4 GB) is a follow-up once the pipeline is proven.
- Fork is public under `Dedalo101`, with `LITTLEBIT_NOTICE.md` documenting
  the clean-room position and the CC BY-NC 4.0 restriction.

## Risks / decisions still open

1. **Embedding quality.** Q8_0 is near-lossless; Q4_K needs a PPL check
   against `eval.py` before it replaces it.
2. **License.** Upstream LittleBit is CC BY-NC 4.0; llama.cpp and Ollama
   are MIT. Keep this clean-room: implement from the documented format and
   the independent numpy spike, never copy `quantization/modules/littlebit.py`.
   Fine for a private/personal fork. **Not** suitable to upstream into
   llama.cpp or Ollama without legal review.
3. **CPU variants.** Every `GGML_CPU_ALL_VARIANTS` build must include the
   new op or Ollama's runner will fail at load time on some machines.
4. **Quality.** Published 0.55 bpw wiki PPL is 24.48 vs base 9.73. This is
   a compression/research demo, not a model upgrade. Say so in the README.

## Commands used to create this branch

```
git init ~/Work/llama.cpp-littlebit
cd ~/Work/llama.cpp-littlebit
git remote add origin https://github.com/ggml-org/llama.cpp.git
git fetch --depth=1 origin b12257
git checkout -b littlebit-tensor FETCH_HEAD
```
