#!/usr/bin/env python3
"""Export a LittleBit safetensors checkpoint into GGUF with an LB1 tensor type.

Export scope:
  * packed LittleBit factors -> GGML_TYPE_LB1 tensors (byte-for-byte copy)
  * u1, v1*u2, v2 scales -> F16 tensors
  * embed_tokens -> Q8_0 token_embd.weight
  * lm_head -> Q8_0 output.weight when the checkpoint stores an untied head
  * layer norms / q_norm / k_norm / final norm -> F16 passthrough
  * Qwen3 hparams and tokenizer metadata, copied from --tokenizer-gguf

The reader works on a local safetensors file or a HF repo over HTTP ranges.
Graph support for the lb.* tensors lives in the Qwen3 model in this fork.

Usage:
  python tools/littlebit-export.py --dry-run LittleBitLLM/littlebit-qwen3-8b-0.55bpw
  python tools/littlebit-export.py --tokenizer-gguf base.gguf --max-layers 1 <hf-repo|local.safetensors> out.gguf
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gguf-py"))

from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType as GQT  # noqa: E402
from gguf.gguf_reader import GGUFReader, GGUFValueType  # noqa: E402
from gguf.gguf_writer import GGUFWriter  # noqa: E402

DTYPE_NP = {
    "F16": "<f2",
    "F32": "<f4",
    "BF16": "<u2",
    "I32": "<i4",
    "I64": "<i8",
    "I8": "i1",
    "U8": "u1",
}


class SafeTensors:
    """Reads a safetensors file locally or over HTTP range requests."""

    def __init__(self, src: str):
        self.src = src
        self.local = Path(src).is_file()
        if self.local:
            with open(src, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self.header = json.loads(f.read(n))
            self.data_start = 8 + n
            self.fh = open(src, "rb")
        else:
            self.url = f"https://huggingface.co/{src}/resolve/main/model.safetensors"
            n = self._range(0, 7)
            self.header_n = struct.unpack("<Q", n)[0]
            self.header = json.loads(self._range(8, 7 + self.header_n))
            self.data_start = 8 + self.header_n
            self.fh = None
        self.header.pop("__metadata__", None)

    def _range(self, start: int, end: int) -> bytes:
        last = None
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) != end - start + 1:
                    raise IOError(f"short range read: {len(data)} != {end - start + 1}")
                return data
            except Exception as exc:
                last = exc
                if attempt < 4:
                    time.sleep(2 ** attempt)
        raise last

    def read(self, key: str) -> np.ndarray:
        info = self.header[key]
        dtype = DTYPE_NP[info["dtype"]]
        count = int(np.prod(info["shape"]))
        nbytes = count * np.dtype(dtype).itemsize
        lo = info["data_offsets"][0]
        if self.local:
            self.fh.seek(self.data_start + lo)
            raw = self.fh.read(nbytes)
        else:
            raw = self._range(self.data_start + lo, self.data_start + lo + nbytes - 1)
        arr = np.frombuffer(raw, dtype=dtype, count=count)
        if info["dtype"] == "BF16":
            arr = arr.astype(np.uint32) << 16
            arr = arr.view("<f4")
        return arr.reshape(info["shape"])

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()


def quantize_q8_0(x: np.ndarray) -> np.ndarray:
    """GGML Q8_0: 32 values per block, f16 scale, int8 values."""
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    if v.size % 32:
        raise ValueError(f"Q8_0 needs multiple of 32 values, got {v.size}")
    blocks = v.reshape(-1, 32)
    amax = np.abs(blocks).max(axis=1)
    d = amax / 127.0
    safe = np.where(d > 0, d, np.float32(1.0))
    qs = np.clip(np.round(blocks / safe[:, None]), -127, 127).astype(np.int8)
    dt = np.dtype([("d", "<f2"), ("qs", "i1", (32,))])
    out = np.empty(blocks.shape[0], dtype=dt)
    out["d"] = d.astype(np.float16)
    out["qs"] = qs
    return out.view(np.uint8).reshape(-1)


def _gguf_value(field) -> object:
    kind = field.types[0]
    if kind == GGUFValueType.STRING:
        n = int(np.asarray(field.parts[3]).view(np.uint64)[0])
        return bytes(np.asarray(field.parts[4])[:n]).decode("utf-8")
    if kind == GGUFValueType.ARRAY:
        elem = field.types[1]
        n = int(np.asarray(field.parts[4]).view(np.uint64)[0])
        parts = field.parts[5:]
        if elem == GGUFValueType.STRING:
            out = []
            for i in range(n):
                ln = int(np.asarray(parts[2*i]).view(np.uint64)[0])
                out.append(bytes(np.asarray(parts[2*i+1])[:ln]).decode("utf-8"))
            return out
        return [np.asarray(parts[i])[0].item() for i in range(n)]
    return np.asarray(field.parts[3])[0].item()


def copy_tokenizer(writer: GGUFWriter, src: str) -> int:
    reader = GGUFReader(src, "r")
    fields = reader.fields
    tokens = _gguf_value(fields["tokenizer.ggml.tokens"])
    writer.add_tokenizer_model(_gguf_value(fields["tokenizer.ggml.model"]))
    writer.add_tokenizer_pre(_gguf_value(fields["tokenizer.ggml.pre"]))
    writer.add_token_list(tokens)
    writer.add_token_merges(_gguf_value(fields["tokenizer.ggml.merges"]))
    writer.add_token_types(_gguf_value(fields["tokenizer.ggml.token_type"]))
    for key, setter in (
        ("tokenizer.ggml.bos_token_id", writer.add_bos_token_id),
        ("tokenizer.ggml.eos_token_id", writer.add_eos_token_id),
        ("tokenizer.ggml.padding_token_id", writer.add_pad_token_id),
    ):
        if key in fields:
            setter(int(_gguf_value(fields[key])))
    if "tokenizer.chat_template" in fields:
        writer.add_chat_template(_gguf_value(fields["tokenizer.chat_template"]))
    return len(tokens)


def hf_config(src: str) -> dict:
    if Path(src).is_file():
        p = Path(src).parent / "config.json"
        if p.exists():
            return json.loads(p.read_text())
        return {}
    url = f"https://huggingface.co/{src}/resolve/main/config.json"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def validate_tie_config(cfg: dict, header: dict) -> None:
    if "tie_word_embeddings" not in cfg:
        return
    tied = bool(cfg["tie_word_embeddings"])
    has_head = "lm_head.weight" in header
    if tied == has_head:
        raise SystemExit(
            f"tie_word_embeddings={tied} conflicts with lm_head.weight present={has_head}"
        )


SUFFIXES = {
    "U_packed": ("U_packed", False),
    "V_packed": ("V_packed", False),
    "u1": ("u1", False),
    "u2": ("u2", False),
    "v1": ("v1", False),
    "v2": ("v2", False),
    "U_R_packed": ("U_packed", True),
    "V_R_packed": ("V_packed", True),
    "u1_R": ("u1", True),
    "u2_R": ("u2", True),
    "v1_R": ("v1", True),
    "v2_R": ("v2", True),
}


def parse_packed_groups(header: dict) -> dict:
    groups: dict[str, dict] = {}
    for key in header:
        for suffix, (base, is_residual) in SUFFIXES.items():
            tail = "." + suffix
            if key.endswith(tail):
                prefix = key[: -len(tail)]
                if is_residual:
                    prefix += "_R"
                groups.setdefault(prefix, {})[base] = key
                break
    return {p: g for p, g in groups.items()
            if "U_packed" in g and "V_packed" in g}


def gguf_name(prefix: str) -> str:
    """model.layers.N.self_attn.q_proj -> blk.N.attn_q.lb"""
    import re
    m = re.match(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+?)(?:_R)?$", prefix)
    if not m:
        raise ValueError(f"unrecognised LittleBit prefix: {prefix}")
    layer, kind, proj = m.groups()
    names = {
        ("self_attn", "q_proj"): "attn_q",
        ("self_attn", "k_proj"): "attn_k",
        ("self_attn", "v_proj"): "attn_v",
        ("self_attn", "o_proj"): "attn_output",
        ("mlp", "gate_proj"): "ffn_gate",
        ("mlp", "up_proj"): "ffn_up",
        ("mlp", "down_proj"): "ffn_down",
    }
    base = names.get((kind, proj))
    if base is None:
        raise ValueError(f"unmapped projection: {prefix}")
    suffix = "_R" if prefix.endswith("_R") else ""
    return f"blk.{layer}.{base}.lb{suffix}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="HF repo id or local .safetensors file")
    ap.add_argument("out", nargs="?", help="output .gguf path")
    ap.add_argument("--max-layers", type=int, default=0,
                    help="only export layers 0..N-1 (default 0 = all; use 1 for a smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="header only, no download")
    ap.add_argument("--skip-embeddings", action="store_true",
                    help="skip the large embedding download (format tests only)")
    ap.add_argument("--tokenizer-gguf",
                    help="copy tokenizer metadata from this GGUF (needed to load the export)")
    ap.add_argument("--block-count", type=int,
                    help="override qwen3.block_count (use 1 with --max-layers 1 for a smoke test)")
    args = ap.parse_args()

    st = SafeTensors(args.src)
    cfg = hf_config(args.src)
    validate_tie_config(cfg, st.header)
    groups = parse_packed_groups(st.header)

    print(f"source tensors : {len(st.header)}")
    print(f"packed groups  : {len(groups)}")
    print(f"arch           : {cfg.get('model_type', 'unknown')}")
    print(f"vocab          : {cfg.get('vocab_size', 'unknown')}")

    if args.dry_run:
        for prefix in sorted(groups)[:12]:
            g = groups[prefix]
            u = st.header[g["U_packed"]]["shape"]
            v = st.header[g["V_packed"]]["shape"]
            print(f"  {prefix:52s} out={u[0]:5d} split={v[0]:4d} pad={u[1]*32 - v[0]:2d} in={v[1]*32:5d}")
        st.close()
        return

    if not args.out:
        raise SystemExit("output path required unless --dry-run")

    n_layer = int(cfg.get("num_hidden_layers", 0))
    n_embd = int(cfg.get("hidden_size", 0))
    n_ff = int(cfg.get("intermediate_size", 0))
    n_head = int(cfg.get("num_attention_heads", 0))
    n_head_kv = int(cfg.get("num_key_value_heads", n_head))
    head_dim = int(cfg.get("head_dim", n_embd // max(n_head, 1)))

    writer = GGUFWriter(args.out, "qwen3")
    writer.add_quantization_version(2)
    writer.add_file_type(0)
    writer.add_uint32("littlebit.version", 1)
    writer.add_uint32("littlebit.block_bits", 32)
    writer.add_bool("littlebit.tied_embeddings", "lm_head.weight" not in st.header)
    writer.add_string("littlebit.source", args.src)

    writer.add_uint32("qwen3.block_count", args.block_count or n_layer)
    writer.add_uint32("qwen3.embedding_length", n_embd)
    writer.add_uint32("qwen3.feed_forward_length", n_ff)
    writer.add_uint32("qwen3.attention.head_count", n_head)
    writer.add_uint32("qwen3.attention.head_count_kv", n_head_kv)
    writer.add_uint32("qwen3.attention.key_length", head_dim)
    writer.add_uint32("qwen3.attention.value_length", head_dim)
    writer.add_float32("qwen3.attention.layer_norm_rms_epsilon", float(cfg.get("rms_norm_eps", 1e-6)))
    writer.add_float32("qwen3.rope.freq_base", float(cfg.get("rope_theta", 1e6)))
    writer.add_uint32("qwen3.context_length", int(cfg.get("max_position_embeddings", 40960)))

    n_tokens = None
    if args.tokenizer_gguf:
        n_tokens = copy_tokenizer(writer, args.tokenizer_gguf)
        print(f"  copied tokenizer: {n_tokens} tokens from {args.tokenizer_gguf}")

    exported = 0
    for prefix in sorted(groups):
        g = groups[prefix]
        m = __import__("re").search(r"layers\.(\d+)", prefix)
        layer_no = int(m.group(1)) if m else 0
        if args.max_layers and layer_no >= args.max_layers:
            continue
        name = gguf_name(prefix)

        u_packed = st.read(g["U_packed"]).astype("<i4")
        v_packed = st.read(g["V_packed"]).astype("<i4")
        out_dim, u_words = u_packed.shape
        split, v_words = v_packed.shape
        padded_split = u_words * 32
        assert split <= padded_split and padded_split - split < 32, (split, padded_split)
        assert (padded_split - split) % 8 == 0, (split, padded_split)
        in_dim = v_words * 32
        assert in_dim % 32 == 0

        # U_packed keeps its original int32 words, including the +1 pad bits
        # past `split`. Zero-pad Vq rows to the same 32-bit word boundary so
        # h1, m, and Uq all use the padded split width in the graph; the pad
        # rows contribute zero because the matching m entries are zero.
        # int32 words of 0 encode all +1 bits, which are neutral in Vq.
        # raw_shape is numpy-style (rows, cols); ggml ne[0] is the last entry.
        writer.add_tensor(f"{name}.Uq", u_packed,
                          raw_shape=(out_dim, padded_split), raw_dtype=GQT.LB1)
        writer.add_tensor(f"{name}.Vq",
                          np.pad(v_packed, ((0, padded_split - split), (0, 0)),
                                 constant_values=0),
                          raw_shape=(padded_split, in_dim), raw_dtype=GQT.LB1)
        exported += 2

        u1 = st.read(g["u1"]).astype(np.float16).reshape(-1)
        u2 = st.read(g["u2"]).astype(np.float16).reshape(-1)
        v1 = st.read(g["v1"]).astype(np.float16).reshape(-1)
        v2 = st.read(g["v2"]).astype(np.float16).reshape(-1)
        m = (v1.astype(np.float32) * u2.astype(np.float32)).astype(np.float16)
        if padded_split != split:
            m = np.pad(m, (0, padded_split - split))
        writer.add_tensor(f"{name}.u1", u1)
        writer.add_tensor(f"{name}.m", m)
        writer.add_tensor(f"{name}.v2", v2)
        exported += 3
        print(f"  exported {name}: in={in_dim} out={out_dim} split={split}")

    if "model.embed_tokens.weight" in st.header and not args.skip_embeddings:
        emb = st.read("model.embed_tokens.weight")
        if n_tokens is not None and emb.shape[0] != n_tokens:
            raise SystemExit(f"tokenizer has {n_tokens} tokens but embedding has {emb.shape[0]} rows")
        q = quantize_q8_0(emb)
        block, bsize = GGML_QUANT_SIZES[GQT.Q8_0]
        byte_row = emb.shape[1] // block * bsize
        writer.add_tensor("token_embd.weight", q,
                          raw_shape=(emb.shape[0], byte_row), raw_dtype=GQT.Q8_0)
        exported += 1
        print(f"  exported token_embd.weight Q8_0: {q.nbytes/1e6:.1f} MB")

    if "lm_head.weight" in st.header and not args.skip_embeddings:
        head = st.read("lm_head.weight")
        if n_tokens is not None and head.shape[0] != n_tokens:
            raise SystemExit(f"tokenizer has {n_tokens} tokens but lm_head has {head.shape[0]} rows")
        qh = quantize_q8_0(head)
        block, bsize = GGML_QUANT_SIZES[GQT.Q8_0]
        byte_row = head.shape[1] // block * bsize
        writer.add_tensor("output.weight", qh,
                          raw_shape=(head.shape[0], byte_row), raw_dtype=GQT.Q8_0)
        exported += 1
        print(f"  exported output.weight Q8_0: {qh.nbytes/1e6:.1f} MB")

    for hf, gg in (
        ("model.norm.weight", "output_norm.weight"),
    ):
        if hf in st.header:
            writer.add_tensor(gg, st.read(hf).astype(np.float32))
            exported += 1

    for i in range(n_layer):
        if args.max_layers and i >= args.max_layers:
            break
        for hf, gg in (
            ("input_layernorm.weight", "attn_norm.weight"),
            ("post_attention_layernorm.weight", "ffn_norm.weight"),
            ("self_attn.q_norm.weight", "attn_q_norm.weight"),
            ("self_attn.k_norm.weight", "attn_k_norm.weight"),
        ):
            key = f"model.layers.{i}.{hf}"
            if key in st.header:
                writer.add_tensor(f"blk.{i}.{gg}", st.read(key).astype(np.float32))
                exported += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    st.close()
    print(f"wrote {args.out}: {exported} tensors")


if __name__ == "__main__":
    main()
