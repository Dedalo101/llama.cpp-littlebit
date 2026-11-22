# LittleBit tensor type for llama.cpp / Ollama — design preview

Status: **design plus local implementation.** Branch `littlebit-tensor` in
`~/Work/llama.cpp-littlebit`, forked from Ollama's pinned llama.cpp tag
`b12257` (commit `17701e4`). The public branch currently holds only the clean
import commit `17c90265f`; the code described below is local and unpushed
pending review and explicit approval.

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

## Why a new type instead of Q1_0 reuse

Three public forks shipped Q1_0-based loaders in November 2026:
`luben93/littlebit-llamacpp`, `Cybertiron/llama.cpp-littlebit`, and
`ScarionWorkspace/littlebit-gguf`. That approach is valid: it reuses stock
kernels, needs a small patch, and has published end-to-end GPU results.

Their own format docs quantify the cost of the reuse:

- Q1_0 stores one F16 block scale per 128 bits. Packing the per-row scale
  vectors into those block scales costs **+12.5% bytes** on the sign matrices.
- Q1_0 needs `ne[0]` divisible by 128. Checkpoint ranks 544 / 208 / 824 are
  padded to 640 / 256 / 896. ScarionWorkspace measures **0.648 bpw** on the
  linears instead of LittleBit's published **0.548 bpw** target.
- The converter must invert the bit convention (LittleBit bit 1 = -1,
  Q1_0 bit 1 = +d) and write an F16 scale before every block. It cannot copy
  the checkpoint's int32 words straight through.

What `GGML_TYPE_LB1` buys for the CPU-only, 4 GB phone target:

- 32-bit blocks, no per-block scale. `u1`, `v1*u2` and `v2` stay as three
  plain F16 vectors, so the scale bytes are exactly the checkpoint's.
- Padding is only to a 32-bit word: 544 stays 544, 208 -> 224, 824 -> 832.
  The linear bit rate stays near the published 0.55 bpw.
- The exporter copies `U_packed` / `V_packed` int32 words byte for byte.
- Q1_0 semantics are not overloaded, so LittleBit-2 / ITQ scale layouts can
  be added later without reworking the format.

Tradeoff stated plainly: Q1_0 is the faster route to a working GPU runtime,
and its momentum is real. A new type means new dequant kernels on every
backend that wants it. For this project's first milestone there is no GPU
target, so only the scalar and NEON paths matter; the GPU cost is deferred,
not paid now. If the goal changes to "run on a laptop GPU this month", use
the Q1_0 approach instead of this branch.

The clean-room position is unchanged: this design comes from the published
format and the independent numpy spike, not from Samsung's source.

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

## Format discoveries during stage 1

- **Effective split != padded split.** In the published checkpoint,
  `mlp.down_proj` has `V_packed.shape[0] = 824` but `U_packed` has 26 words
  = 832 bits. The trailing 8 bits are `+1` padding. The exporter keeps the
  original int32 words, gives `Uq` a logical `ne[0] = 832`, and zero-pads
  the middle scale `m` from 824 to 832 so the extra latents add zero.
  Never derive split from `U_packed` word count alone.
- **Residual branches are separate groups.** `_R` suffixes must be kept in
  the group prefix; otherwise `_R` scales silently overwrite the main ones.
- **gguf-py `raw_shape` is numpy-style**: ggml `ne[0]` is the last element.
- Qwen3 in this llama.cpp already falls back to `token_embd.weight` when
  `output.weight` is absent, so tying needs no C change.

## Stage 1 and 2 status (2026-11-22)

Done locally on `littlebit-tensor`:

- `GGML_TYPE_LB1 = 44`: traits, 32-bit block, exact bit packing, dequant,
  quantize-ref, GGUF validation, quantize-chunk support.
- `tests/test-littlebit.cpp`: traits, bit layout, GGUF roundtrip, and graph
  math. Uses hard `CHECK` macros because Release builds define NDEBUG.
  With the same F32 input and factors rebuilt as a dense Q8_0 weight:
  LB1 max abs err 0.2206 (1.18% of ref scale 18.6750), dense Q8_0 err
  0.2844 (1.52%), LB1 vs dense Q8_0 err 0.2810 (1.50%). The LB1 chain is
  closer to the F32 reference than the dense Q8_0 path.
- `tools/littlebit-export.py`: real checkpoint dry-run (504 packed groups,
  252 linears x 2 branches), byte-identical packed tensors, residual
  branches, full Qwen3 hparams, tokenizer copy, and Q8_0 embeddings.
  Three exporter bugs were found and fixed only by writing and loading a
  real GGUF: gguf-py Q8_0 `raw_shape` must be byte-shaped (`hidden/32*34`);
  `Vq` rows must be padded to the same 32-bit split width as `m` and `Uq`
  (208 -> 224, 824 -> 832) or `ggml_mul` cannot broadcast; RMS norm weights
  must be F32, not F16, because `build_norm` multiplies them directly.
- Full `model.safetensors` downloaded locally (2,968,804,064 bytes) and
  SHA-256 verified against the HF LFS OID.
- Qwen3 loader/graph wiring: `.lb.*` and `.lb_R.*` factors loaded, dense
  `.weight` optional, `build_lora_mm` override runs the factor chain and
  adds the residual branch. `libllama`, `llama-completion` and
  `llama-perplexity` build clean.
- One-layer GGUF (650 MB, 76 tensors) loads and runs a deterministic
  raw-mode completion in `llama-completion` (`-no-cnv`, `-ngl 0`, temp 0):
  `The capital of France is is is is is is is is is`. The repetition is the
  expected result of truncating an 8B model to one layer. `llama-perplexity`
  also completes over a small token file; its astronomical PPL is a graph
  sanity result, not a quality measurement.

Remaining: full 36-layer export, full-model PPL versus the published
checkpoint numbers, NEON kernel (stage 4), Ollama payload build (stage 5).

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
