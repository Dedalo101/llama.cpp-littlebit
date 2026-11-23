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
- One-layer GGUF loads and runs a deterministic raw-mode completion in
  `llama-completion` (`-no-cnv`, `-ngl 0`, temp 0). Repetition is the
  expected result of truncating an 8B model to one layer.
- Full 36-layer export completed with `--max-layers 0 --block-count 36`.
  Two full-model bugs were found only by testing all layers:
  * each LB1 factor matmul originally quantized activations to Q8_0; error
    compounded across 36 layers. An interim fix cast `Uq`/`Vq` to F16 and
    kept intermediates in F16/F32, dropping graph max abs error from 0.2206
    to 0.0135. The final fix keeps factors packed and uses the AVX2
    XOR/F32 dot kernel described below, improving it further to 0.0070.
  * `config.json` has `tie_word_embeddings: false`, and the safetensors file
    contains a separate `lm_head.weight`. Tying the output to the embedding
    produced gibberish. The exporter now writes `output.weight` as Q8_0;
    the corrected full GGUF is 1.7 GB with 2667 tensors.
- Full-model sanity check now works:
  `The capital of France is Paris, and the city is known for its vibrant culture,`
- `tools/littlebit_cpu.py`: single-file, CPU-only GGUF runner needing only
  Python 3.10+ and numpy. It memory-maps GGUF, dequantizes Q8_0 embeddings
  and LB1 factors on demand, uses `output.weight` when present, and
  implements Qwen3 RMSNorm, NEOX RoPE, GQA attention, the LB1 factor chain,
  `_R` residual branch, and greedy decode. It reproduces the compiled fork
  on the corrected full model: prompt `The capital of France is` ->
  ` Paris, and the`. This is the portable correctness artifact for old
  laptops; the compiled fork is the fast CPU path.

Full-model PPL is validated: **21.2364 +/- 0.59425** on 12 windows with
`-c 2048 -b 2048 --chunks 12`, 299,078 tokens, Q8_0 embeddings and Q8_0
output head. This is within the 20.88 +/- 0.5 target and below the 23 gate.
This `llama-perplexity` build did not print `n_tokens`; the token count comes
from the separate tokenizer run. The final packed F32 vec-dot kernel was
checked against the full F16-cast baseline on chunks 1 and 2 (deltas <= 0.0016
PPL), and the complete 12-window run was then repeated with the final kernel:
it reproduced 21.2364 +/- 0.59425 exactly. No equivalence assumption remains.

Remaining: Ollama payload build (stage 5). The NEON LB1 dot for stage 4 is
implemented; hardware throughput still needs to be collected on ARM.

## Stage 1 CPU kernel

The factor chain keeps LB1 factors packed and uses F32 activations. The CPU
type traits for `GGML_TYPE_LB1` now point to `ggml_vec_dot_lb1_f32`, so the
generic `ggml_mul_mat` path calls a kernel that reconstructs each `-1` sign
with an XOR of the activation sign bit and accumulates in F32. No dequantized
weight tensor is materialized, and activations are never quantized to Q8_0.
The AVX2 path handles 8 activations per vector; a scalar fallback covers CPUs
without AVX2. This replaced the earlier F16-cast factor path. Graph max abs
error improved from 0.0135 to 0.0070 against the F32 oracle reference.

`tools/littlebit-stats.py` reports the weight bytes read per forward and the
derived bytes per token for a chosen batch size. For the full GGUF:

```text
LB1 factor bytes: 468910080
output head     : 661225472
bytes / forward : 1130135552
bytes / token @1    : 1,130,135,552
bytes / token @2048 : 551,824
```

The output head dominates single-token decode bandwidth. Quantized heads were
evaluated and rejected on quality, so the 661225472-byte Q8_0 head stays the
default. On the first WikiText-2 window, Q4_K scored PPL 25.3521 and Q6_K
24.7543, versus 24.5867 for Q8_0 (gate: within +0.1). Q4_K also changed
"The capital of France is Paris, and the city is known for its vibrant
culture" to "...located in the French Riviera", a factual hallucination that
the averaged PPL does not reveal. Neither compact head is shipped.

## Decode profile

`LLAMA_LB1_PROFILE=1` enables a per-section timer in the CPU graph executor.
It attributes complete graph nodes (measured up to the node barrier, so
multi-thread work is included) to `embed`, `norm`, `attn`, `ffn`, the two LB1
matvec buckets, `head`, and `other`. The Qwen3 graph attaches stable `prof:`
markers at section boundaries for this purpose.

Measured on the x86 i7-1185G7 host, 8-token decode, Q8_0 head, -c 64:

| threads | LB1 ffn | LB1 attn | head | norm | other |
|---|---|---|---|---|---|
| 1 | 73.0% | 19.6% | 6.1% | 0.3% | 0.9% |
| 2 | 72.4% | 19.4% | 6.0% | 0.6% | 1.5% |
| 4 | 70.1% | 19.4% | 7.0% | 1.1% | 2.5% |
| 8 | 61.2% | 18.2% | 7.4% | 4.1% | 8.9% |

The head is only ~6-7% of decode here, and llama.cpp already quantizes F32
activations to Q8_0 and runs an int8 x int8 vec_dot for Q8_0 weights
(`type_traits_cpu[GGML_TYPE_Q8_0]`). A separate "int8 head dot" prototype
would duplicate what is already in the tree: `tools/littlebit-head-bench.cpp`
measures the existing path at ~39 ms for a full 151936 x 4096 head matvec on
this host, ~16 GiB/s single-threaded. The bottleneck is the LB1 factor path,
not the head.

On ARM the LB1 dot used to fall through to the scalar loop in `quants.c`,
which is many times slower than the x86 AVX2 path. A NEON implementation was
added under `#elif defined(__ARM_NEON)`: it extracts each packed sign with
variable shifts, XORs it into the F32 sign bit and accumulates four lanes.
It uses only baseline NEON (no SDOT/SVE), so it targets both Pi 5 (Cortex-A76)
and Ampere Altra. Verified by clang cross-compile for `armv8.2-a` and by
running the loop under `qemu-aarch64-static` against the scalar reference
(n = 32..1024, error within float rounding). Real ARM throughput still needs
to be collected on hardware.

`tools/littlebit-lut-bench.cpp` evaluates the subset-sum LUT idea from first
principles. Results on the x86 host:

- 8-bit groups (256-entry tables): the dot is ~3.2x faster, but rebuilding the
  per-token tables for all 36 layers costs ~39 ms and 359 MiB of writes. An
  integrated opt-in prototype produced correct tokens but was *slower*
  end-to-end, even when restricted to wide matrices: dependent table loads
  cost more than the AVX2 instructions they replace. Not shipped.
- 4-bit groups (16-entry tables): table rebuild drops to well under 1 ms, but
  the dot is only 1.0-1.3x faster on AVX2. Also not shipped.

The LUT idea remains plausible for ARM, where the baseline is scalar rather
than AVX2, but it needs real hardware measurements before any integration.
The current ARM lever is the NEON kernel above.

Benchmark command, with peak RSS from `/usr/bin/time`:

```sh
/usr/bin/time -v env LD_LIBRARY_PATH=build/bin ./build/bin/llama-completion \
  -m littlebit-qwen3-8b-lb1-v3.gguf -no-cnv -ngl 0 --temp 0 \
  -c 128 -p "The capital of France is" -n 32
```

`llama-completion` reports prompt and decode tokens per second separately.
Measured peak RSS on this host was 1864.4 MB for `-c 128 -n 32` with `-t 8`,
below the 3.2 GB gate.

A fused 2-row x 4-token LB1 matmul was prototyped and A/B tested against the
type-traits path. It produced identical output but was neutral-to-slower
(prompt 2.80 -> 1.75 t/s at `-t 1`, 5.52 -> 5.42 t/s at `-t 4` on a loaded
host), so it was dropped. The generic `mul_mat` + `vec_dot` path is kept as
the single integration point, matching how other quantized types work. Large
batch prefill remains bandwidth/compute bound by the 468.9 MB factor chain;
a proper repacked tiled GEMM is a separate future optimization.

Thread count matters: on this 8-core host, `-t 4` was roughly twice as fast
as `-t 1` for both prompt and decode. `-t 8` regressed badly because an
Ollama `llama-server` was concurrently using two cores; decode numbers need
a quiet host for a final model-card figure.

## PPL validation target

The relevant baseline is the published checkpoint, not the uncompressed
model. luben93's upstream discussion
(`ggml-org/llama.cpp#32671`) reports, on WikiText-2 with 2048-token windows:

- Standard Qwen3-8B Q4_K_M: **8.78 ± 0.07**
- LittleBit 0.55 bpw, Q4_K_M embeddings: **22.31 ± 0.18** full test set
- LittleBit 0.55 bpw, Q8_0 embeddings: **20.88** on 12 windows
- Converted GGUF vs independent float32 PyTorch reference: within ~0.007
  PPL per window

A ~5.5-6.5 baseline is not in that discussion. This published checkpoint is
a format demonstration, not a quality-preserving compression of Qwen3-8B.
Our Q8_0 embedding/head export was scored with the same 2048-token,
12-window protocol and landed at 21.2364, within the scale-validation
target. The PPL log did not print `n_tokens`; the exact 299,078 token count
comes from the separate tokenizer run and is recorded above.

## Staged plan

| Stage | Deliverable | Verify with |
|---|---|---|
| 0 | numpy oracle already exists | `runtime/selftest.py`, `remote_probe.py` |
| 1 | LB1 type + traits + GGUF reader + exporter | decode exported LB1 back to numpy, compare bit-for-bit and layer output vs oracle |
| 2 | scalar CPU `mul_mat_lb1` + graph integration | llama-cli loads exported GGUF, generates one token |
| 3 | full tiny model generating, PPL vs `eval.py` | perplexity delta recorded |
| 4 | NEON kernel on ARM | tokens/s + MB/s on Pi / Oracle ARM |
| 5 | Ollama local build with source override | `ollama run littlebit-qwen3` through the custom payload |

### Ollama payload (stage 5)

Ollama `v0.40.17` is the release that pins llama.cpp `b12257`, matching this
fork. Build with the source override:

```sh
git clone --depth 1 --branch v0.40.17 https://github.com/ollama/ollama.git
cd ollama
OLLAMA_LLAMA_CPP_SOURCE=~/Work/llama.cpp-littlebit cmake -B build .
OLLAMA_LLAMA_CPP_SOURCE=~/Work/llama.cpp-littlebit cmake --build build --parallel 8
```

Two small Go-side changes are needed; see `docs/littlebit-ollama.patch`:
`fs/gguf/tensor.go` must know `GGML_TYPE_LB1 = 44` (four bytes per 32 signs),
and `fs/gguf/metadata.go` must ignore the exported Qwen3 Jinja chat template
for `littlebit.version` models so a simple Modelfile `TEMPLATE` wins. The
patch is local only — do not push it to Ollama.

Verified with an isolated store and server (`OLLAMA_MODELS=...`,
`OLLAMA_HOST=127.0.0.1:11435`): `ollama create` accepts the 1.7 GB GGUF and
`POST /api/generate` with `"raw": true` reproduces
`The capital of France is Paris, and the city is known for its vibrant culture`.
The published checkpoint is a format demo, not a chat model: prompts wrapped
in ChatML produce weak/degenerate answers, so raw completion mode is the
supported Ollama path.

## Decisions made 2026-11-22

- **Embedding handling was revised after full-model testing.** The published
  checkpoint has `tie_word_embeddings: false`, so the earlier tying plan was
  wrong. Both `embed_tokens` and `lm_head` are exported as Q8_0, giving a
  1.7 GB full GGUF. Q4_K for these two tensors is a follow-up.
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

Portable CPU-only check (no build, no Ollama, no GPU):

```
python3 tools/littlebit_cpu.py \
  ~/.cache/littlebit/littlebit-qwen3-8b-lb1-v3.gguf \
  -p 'The capital of France is' -n 4 -c 64 --temp 0
```

Greedy-output regression check against the compiled fork (averaged PPL is not
enough — the Q4_K/Q6_K head experiments flipped individual answers while
staying near the PPL gate). The published checkpoint is a format demonstration,
not a chat model, so the gate is exact reproduction of the recorded v3 outputs,
not world-fact correctness:

```
python3 tools/littlebit-output-regression.py \
  --model ~/.cache/littlebit/littlebit-qwen3-8b-lb1-v3.gguf \
  --binary build/bin/llama-completion
```
