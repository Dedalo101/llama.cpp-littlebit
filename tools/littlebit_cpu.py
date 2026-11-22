#!/usr/bin/env python3
"""Single-file CPU runner for LittleBit LB1 GGUF exports.

Dependencies: Python 3.10+ and numpy only. No GPU, no llama.cpp build, no Ollama.
Optional: `pip install regex` improves non-ASCII tokenization; an ASCII fallback
is built in.

Usage:
    python3 littlebit_cpu.py model.gguf -p 'The capital of France is' -n 8
    python3 littlebit_cpu.py model.gguf -p 'Hello' -n 8 -c 256 --temp 0

The GGUF is memory-mapped. Q8_0 embeddings are dequantized one row or one
chunk of rows at a time, and LB1 factors are dequantized per projection, so
the runner does not expand the whole model into RAM.
"""

from __future__ import annotations

import argparse
import math
import mmap
import os
import struct
import sys
import time

try:
    import numpy as np
except ImportError:
    raise SystemExit("littlebit_cpu.py needs numpy: pip install numpy")


# GGML/GGUF type IDs used here.
GT_F32 = 0
GT_F16 = 1
GT_Q8_0 = 8
GT_LB1 = 44

TYPE_SIZE = {
    GT_F32: 4,
    GT_F16: 2,
    GT_Q8_0: 34,   # per 32 values
    GT_LB1: 4,     # per 32 values
}
TYPE_BLOCK = {
    GT_F32: 1,
    GT_F16: 1,
    GT_Q8_0: 32,
    GT_LB1: 32,
}

MAX_STRING_BYTES = 64 * 1024 * 1024
MAX_ARRAY_ITEMS = 20_000_000
MAX_TENSOR_INFOS = 1_000_000
MAX_TENSOR_DIMS = 4
MAX_TENSOR_DIM = 1_000_000_000
MAX_VOCAB = 1_000_000
MAX_LAYERS = 512
MAX_CTX = 1_000_000


class GGUF:
    """Minimal little-endian GGUF reader for the tensors used by this runner."""

    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.off = 0
        self.meta: dict[str, object] = {}
        self.tensors: dict[str, dict] = {}
        self._parse()

    def close(self) -> None:
        self.mm.close()
        self.f.close()

    def _take(self, fmt: str, count: int = 1):
        size = struct.calcsize(fmt) * count
        out = struct.unpack_from("<" + fmt * count, self.mm, self.off)
        self.off += size
        return out[0] if count == 1 else out

    def _string(self) -> str:
        n = self._take("Q")
        if n > MAX_STRING_BYTES or self.off + n > len(self.mm):
            raise ValueError("bad GGUF string length")
        b = self.mm[self.off:self.off + n]
        self.off += n
        return b.decode("utf-8", errors="replace")

    def _value(self, vtype: int):
        if vtype == 0:   return self._take("B")
        if vtype == 1:   return self._take("b")
        if vtype == 2:   return self._take("H")
        if vtype == 3:   return self._take("h")
        if vtype == 4:   return self._take("I")
        if vtype == 5:   return self._take("i")
        if vtype == 6:   return self._take("f")
        if vtype == 7:   return bool(self._take("B"))
        if vtype == 8:   return self._string()
        if vtype == 10:  return self._take("Q")
        if vtype == 11:  return self._take("q")
        if vtype == 12:  return self._take("d")
        if vtype == 9:   # array: element type then count, no per-element type tag
            sub = self._take("I")
            n = self._take("Q")
            if n > MAX_ARRAY_ITEMS:
                raise ValueError("GGUF array too large")
            return [self._value(sub) for _ in range(n)]
        raise ValueError(f"unsupported GGUF value type {vtype}")

    def _parse(self) -> None:
        if self.mm[:4] != b"GGUF":
            raise ValueError("not a GGUF file")
        self.off = 4
        version = self._take("I")
        if version != 3:
            raise ValueError(f"unsupported GGUF version {version}")
        tensor_count, kv_count = self._take("Q", 2)
        if tensor_count > MAX_TENSOR_INFOS or kv_count > MAX_TENSOR_INFOS:
            raise ValueError("too many GGUF header entries")

        for _ in range(kv_count):
            key = self._string()
            self.meta[key] = self._value(self._take("I"))

        infos = []
        for _ in range(tensor_count):
            name = self._string()
            ndim = self._take("I")
            if ndim < 1 or ndim > MAX_TENSOR_DIMS:
                raise ValueError(f"tensor {name}: bad dimension count")
            dims = [self._take("Q") for _ in range(ndim)]
            if any(d == 0 or d > MAX_TENSOR_DIM for d in dims):
                raise ValueError(f"tensor {name}: bad dimension")
            tt = self._take("I")
            rel = self._take("Q")
            if tt not in TYPE_SIZE:
                raise ValueError(f"tensor {name}: unsupported type {tt}")
            if dims[0] % TYPE_BLOCK[tt]:
                raise ValueError(f"tensor {name}: ne0 not divisible by block size")
            nbytes = (dims[0] // TYPE_BLOCK[tt]) * TYPE_SIZE[tt]
            for d in dims[1:]:
                nbytes *= d
            infos.append((name, tt, dims, rel, nbytes))

        align = int(self.meta.get("general.alignment", 32))
        if align == 0 or align & (align - 1):
            raise ValueError("bad GGUF alignment")
        pad = self.off % align
        if pad:
            self.off += align - pad
        self.data_offset = self.off

        for name, tt, dims, rel, nbytes in infos:
            if self.data_offset + rel + nbytes > len(self.mm):
                raise ValueError(f"tensor {name}: data outside file")
            self.tensors[name] = {
                "type": tt,
                "dims": dims,                   # ggml order: ne[0] is fastest
                "shape": tuple(reversed(dims)), # numpy order: last dim fastest
                "offset": self.data_offset + rel,
                "nbytes": nbytes,
            }

    def bytes(self, name: str) -> memoryview:
        t = self.tensors[name]
        return self.mm[t["offset"]:t["offset"] + t["nbytes"]]

    def vec(self, name: str) -> np.ndarray:
        t = self.tensors[name]
        raw = self.bytes(name)
        if t["type"] == GT_F16:
            a = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        elif t["type"] == GT_F32:
            a = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        else:
            raise ValueError(f"{name} is not a float scale tensor")
        return a.reshape(-1)


# ---------------------------------------------------------------------------
# Quantized dequantization
# ---------------------------------------------------------------------------

def dequant_q8_0(raw: memoryview, ne0: int, nrows: int | None = None) -> np.ndarray:
    """Return float32 array shaped (nrows, ne0) from Q8_0 blocks."""
    nblocks = ne0 // 32
    b = np.frombuffer(raw, dtype=np.uint8)
    if nrows is None:
        nrows = b.size // (nblocks * 34)
    b = b.reshape(nrows, nblocks, 34)
    d = b[:, :, :2].copy().view("<f2").reshape(nrows, nblocks).astype(np.float32)
    q = b[:, :, 2:].astype(np.int8).astype(np.float32)
    return (d[:, :, None] * q).reshape(nrows, ne0)


def dequant_lb1(raw: memoryview, ne0: int, ne1: int) -> np.ndarray:
    """Return float32 array shaped (ne1, ne0); bit 0 -> +1, bit 1 -> -1."""
    words = np.frombuffer(raw, dtype="<i4").astype(np.uint32)
    words = words.reshape(ne1, ne0 // 32)
    bits = (words[:, :, None] >> np.arange(32, dtype=np.uint32)) & 1
    return (1.0 - 2.0 * bits).astype(np.float32).reshape(ne1, ne0)


# ---------------------------------------------------------------------------
# Qwen2/Qwen3 tokenizer from GGUF metadata (GPT-2 byte-level BPE)
# ---------------------------------------------------------------------------

def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    extra = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


QWEN2_PATTERN = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
QWEN2_PATTERN_ASCII = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\w]?[^\W\d_]+|\d"
    r"| ?[^\s\w]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


class Tokenizer:
    def __init__(self, meta: dict[str, object]):
        self.tokens = list(meta.get("tokenizer.ggml.tokens", []))
        if len(self.tokens) > MAX_VOCAB:
            raise ValueError("tokenizer vocabulary too large")
        self.id_of = {t: i for i, t in enumerate(self.tokens)}
        self.eos = int(meta.get("tokenizer.ggml.eos_token_id", -1))
        self.add_bos = bool(meta.get("tokenizer.ggml.add_bos_token", False))
        self.bos = int(meta.get("tokenizer.ggml.bos_token_id", -1))
        self.merges: dict[tuple[str, str], int] = {}
        for rank, m in enumerate(meta.get("tokenizer.ggml.merges", [])):
            parts = str(m).split(" ", 1)
            if len(parts) == 2:
                self.merges[(parts[0], parts[1])] = rank
        self.b2u = bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}
        self.cache: dict[str, tuple[str, ...]] = {}
        try:
            import regex  # type: ignore
            self.re = regex.compile(QWEN2_PATTERN)
        except ImportError:
            import re
            self.re = re.compile(QWEN2_PATTERN_ASCII)

    def _bpe(self, piece: str) -> tuple[str, ...]:
        cached = self.cache.get(piece)
        if cached is not None:
            return cached
        word = tuple(self.b2u[b] for b in piece.encode("utf-8"))
        while len(word) > 1:
            best, rank = None, None
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                r = self.merges.get(pair)
                if r is not None and (rank is None or r < rank):
                    best, rank = pair, r
            if best is None:
                break
            a, b = best
            out, i = [], 0
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i + 1]) == best:
                    out.append(word[i] + word[i + 1])
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = tuple(out)
        self.cache[piece] = word
        return word

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        if self.add_bos and self.bos >= 0:
            ids.append(self.bos)
        for piece in self.re.findall(text):
            for tok in self._bpe(piece):
                idx = self.id_of.get(tok)
                if idx is not None:
                    ids.append(idx)
        return ids

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.tokens[i] for i in ids if 0 <= i < len(self.tokens))
        raw = bytearray()
        for ch in text:
            b = self.u2b.get(ch)
            if b is not None:
                raw.append(b)
            elif ch == "\n":
                raw.append(10)
            else:
                raw.extend(ch.encode("utf-8", errors="replace"))
        return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Qwen3 CPU model
# ---------------------------------------------------------------------------

def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    scale = 1.0 / math.sqrt(float(np.mean(x * x)) + eps)
    return (x * scale * w).astype(np.float32)


def rms_norm_heads(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    x = x.reshape(-1, w.shape[0]).astype(np.float32, copy=False)
    scale = 1.0 / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + eps)
    return (x * scale * w).astype(np.float32)


def rope(x: np.ndarray, pos: int, head_dim: int, base: float) -> np.ndarray:
    # Qwen3 uses NEOX-style RoPE: pairs are offset by head_dim/2 and there is
    # one frequency for every even dimension index (0, 2, ... head_dim-2).
    x = x.reshape(-1, head_dim).astype(np.float32, copy=True)
    half = head_dim // 2
    idx = np.arange(0, head_dim, 2, dtype=np.float32)
    inv = np.power(np.float32(base), -idx / np.float32(head_dim))
    ang = np.float32(pos) * inv
    cos, sin = np.cos(ang), np.sin(ang)
    a, b = x[:, :half].copy(), x[:, half:].copy()
    x[:, :half] = a * cos - b * sin
    x[:, half:] = a * sin + b * cos
    return x


class LittleBitCPU:
    def __init__(self, gguf: GGUF, ctx: int):
        self.g = gguf
        m = gguf.meta
        self.n_layer = int(m["qwen3.block_count"])
        self.n_embd = int(m["qwen3.embedding_length"])
        self.n_ff = int(m["qwen3.feed_forward_length"])
        self.n_head = int(m["qwen3.attention.head_count"])
        self.n_kv = int(m["qwen3.attention.head_count_kv"])
        self.head_dim = int(m["qwen3.attention.key_length"])
        self.eps = float(m["qwen3.attention.layer_norm_rms_epsilon"])
        self.rope_base = float(m["qwen3.rope.freq_base"])
        self.ctx = max(ctx, 1)

        if not 1 <= self.n_layer <= MAX_LAYERS:
            raise ValueError("bad block_count")
        if not 1 <= self.n_embd <= 65536:
            raise ValueError("bad embedding_length")
        if not 1 <= self.n_ff <= 262144:
            raise ValueError("bad feed_forward_length")
        if not 1 <= self.n_head <= 512:
            raise ValueError("bad head_count")
        if not 1 <= self.n_kv <= self.n_head or self.n_head % self.n_kv:
            raise ValueError("bad head_count_kv")
        if not 2 <= self.head_dim <= 1024 or self.head_dim % 2:
            raise ValueError("bad key_length for RoPE")
        if self.n_head * self.head_dim != self.n_embd:
            raise ValueError("head_count * key_length != embedding_length")
        if self.ctx > MAX_CTX:
            raise ValueError("ctx too large")

        self.emb = gguf.tensors["token_embd.weight"]
        self.vocab = self.emb["dims"][1]
        if self.emb["dims"][0] != self.n_embd or not 1 <= self.vocab <= MAX_VOCAB:
            raise ValueError("bad token_embd shape")
        self.head = gguf.tensors.get("output.weight", self.emb)
        if self.head["dims"] != [self.n_embd, self.vocab]:
            raise ValueError("bad output.weight shape")
        if self.emb["type"] != GT_Q8_0 or self.head["type"] != GT_Q8_0:
            raise ValueError("embedding and output tensors must be Q8_0")
        self.norm_out = gguf.vec("output_norm.weight")
        if self.norm_out.shape[0] != self.n_embd:
            raise ValueError("bad output_norm shape")

        self.k_cache = [np.zeros((self.ctx, self.n_kv, self.head_dim), np.float32)
                        for _ in range(self.n_layer)]
        self.v_cache = [np.zeros((self.ctx, self.n_kv, self.head_dim), np.float32)
                        for _ in range(self.n_layer)]

    def embed(self, token: int) -> np.ndarray:
        if not 0 <= token < self.vocab:
            raise ValueError("token id out of range")
        ne0 = self.emb["dims"][0]
        row = self.emb["offset"] + token * (ne0 // 32) * 34
        raw = self.g.mm[row:row + (ne0 // 32) * 34]
        return dequant_q8_0(raw, ne0, 1)[0]

    def logits(self, h: np.ndarray) -> np.ndarray:
        h = rms_norm(h, self.norm_out, self.eps)
        ne0 = self.head["dims"][0]
        out = np.empty(self.vocab, np.float32)
        chunk = 2048
        for start in range(0, self.vocab, chunk):
            end = min(self.vocab, start + chunk)
            n = end - start
            off = self.head["offset"] + start * (ne0 // 32) * 34
            raw = self.g.mm[off:off + n * (ne0 // 32) * 34]
            w = dequant_q8_0(raw, ne0, n).reshape(n, self.n_embd)
            out[start:end] = w @ h
        return out

    def lb1(self, name: str, x: np.ndarray) -> np.ndarray:
        g = self.g
        def branch(suffix: str) -> np.ndarray:
            base = f"{name}.{suffix}"
            v2 = g.vec(base + ".v2")
            m = g.vec(base + ".m")
            u1 = g.vec(base + ".u1")
            vq_t = g.tensors[base + ".Vq"]
            uq_t = g.tensors[base + ".Uq"]
            vq = dequant_lb1(g.bytes(base + ".Vq"), vq_t["dims"][0], vq_t["dims"][1])
            uq = dequant_lb1(g.bytes(base + ".Uq"), uq_t["dims"][0], uq_t["dims"][1])
            h1 = (x * v2) @ vq.T
            h2 = h1 * m
            h3 = h2 @ uq.T
            return h3 * u1

        out = branch("lb")
        if f"{name}.lb_R.Vq" in g.tensors:
            out = out + branch("lb_R")
        return out.astype(np.float32)

    def forward(self, token: int, pos: int) -> np.ndarray:
        x = self.embed(token)
        scale_attn = 1.0 / math.sqrt(self.head_dim)
        for il in range(self.n_layer):
            p = f"blk.{il}."
            residual = x
            h = rms_norm(x, self.g.vec(p + "attn_norm.weight"), self.eps)
            q = self.lb1(p + "attn_q", h).reshape(self.n_head, self.head_dim)
            k = self.lb1(p + "attn_k", h).reshape(self.n_kv, self.head_dim)
            v = self.lb1(p + "attn_v", h).reshape(self.n_kv, self.head_dim)
            q = rope(rms_norm_heads(q, self.g.vec(p + "attn_q_norm.weight"), self.eps),
                     pos, self.head_dim, self.rope_base)
            k = rope(rms_norm_heads(k, self.g.vec(p + "attn_k_norm.weight"), self.eps),
                     pos, self.head_dim, self.rope_base)
            self.k_cache[il][pos] = k
            self.v_cache[il][pos] = v
            attn = np.empty((self.n_head, self.head_dim), np.float32)
            for i in range(self.n_head):
                kv = i // (self.n_head // self.n_kv)
                kk = self.k_cache[il][:pos + 1, kv, :]
                vv = self.v_cache[il][:pos + 1, kv, :]
                scores = (kk @ q[i]) * scale_attn
                scores = scores - scores.max()
                probs = np.exp(scores)
                probs /= probs.sum()
                attn[i] = probs @ vv
            x = residual + self.lb1(p + "attn_output", attn.reshape(-1))

            residual = x
            h = rms_norm(x, self.g.vec(p + "ffn_norm.weight"), self.eps)
            up = self.lb1(p + "ffn_up", h)
            gate = self.lb1(p + "ffn_gate", h)
            act = (gate / (1.0 + np.exp(-gate))) * up
            x = residual + self.lb1(p + "ffn_down", act)
        return x


def sample(logits: np.ndarray, temp: float) -> int:
    if temp <= 0:
        return int(np.argmax(logits))
    z = logits.astype(np.float64) / temp
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="LittleBit LB1 GGUF file")
    ap.add_argument("-p", "--prompt", default="The capital of France is")
    ap.add_argument("-n", "--tokens", type=int, default=8, help="tokens to generate")
    ap.add_argument("-c", "--ctx", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=-1)
    args = ap.parse_args()

    if args.seed >= 0:
        np.random.seed(args.seed)

    gguf = GGUF(args.model)
    try:
        tok = Tokenizer(gguf.meta)
        model = LittleBitCPU(gguf, args.ctx)
        ids = tok.encode(args.prompt)
        if not ids:
            raise SystemExit("prompt tokenized to zero tokens")
        if len(ids) + args.tokens > model.ctx:
            raise SystemExit(f"prompt+tokens exceeds context {model.ctx}")
        print(f"model   : {os.path.basename(args.model)}")
        print(f"layers  : {model.n_layer}, vocab {len(tok.tokens)}, ctx {model.ctx}")
        print(f"prompt  : {len(ids)} tokens -> {ids}")
        print("output  :", end=" ", flush=True)
        t0 = time.perf_counter()

        for i, tid in enumerate(ids):
            h = model.forward(tid, i)
        logits = model.logits(h)
        out_ids: list[int] = []
        for j in range(args.tokens):
            nxt = sample(logits, args.temp)
            if tok.eos >= 0 and nxt == tok.eos:
                break
            out_ids.append(nxt)
            sys.stdout.write(tok.decode([nxt]))
            sys.stdout.flush()
            if j + 1 < args.tokens:
                pos = len(ids) + j
                if pos >= model.ctx:
                    break
                h = model.forward(nxt, pos)
                logits = model.logits(h)
        elapsed = time.perf_counter() - t0
        print()
        print(f"generated: {len(out_ids)} tokens, ids={out_ids}")
        print(f"elapsed: {elapsed:.2f}s, {(len(ids) + len(out_ids)) / max(elapsed, 1e-9):.2f} tok/s")
    finally:
        gguf.close()


if __name__ == "__main__":
    main()
