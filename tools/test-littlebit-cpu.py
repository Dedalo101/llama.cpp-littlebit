#!/usr/bin/env python3
"""Tiny synthetic GGUF test for untied output.weight handling."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

spec = importlib.util.spec_from_file_location("littlebit_export", HERE / "littlebit-export.py")
assert spec is not None and spec.loader is not None
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)

spec = importlib.util.spec_from_file_location("littlebit_cpu", HERE / "littlebit_cpu.py")
assert spec is not None and spec.loader is not None
cpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpu)


def add_q8_0(writer, name: str, values: np.ndarray) -> None:
    q = export.quantize_q8_0(values)
    block, bsize = export.GGML_QUANT_SIZES[export.GQT.Q8_0]
    byte_row = values.shape[1] // block * bsize
    writer.add_tensor(name, q, raw_shape=(values.shape[0], byte_row),
                      raw_dtype=export.GQT.Q8_0)


def main() -> None:
    emb = np.tile(np.array([[1, 2, 3, 4], [-4, -3, -2, -1]], np.float32), 8)
    head = np.tile(np.array([[10, -2, 5, 1], [2, -10, 1, 5]], np.float32), 8)
    norm = np.ones(32, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "untied.gguf"
        writer = export.GGUFWriter(str(path), "qwen3")
        writer.add_quantization_version(2)
        writer.add_file_type(0)
        writer.add_uint32("qwen3.block_count", 1)
        writer.add_uint32("qwen3.embedding_length", 32)
        writer.add_uint32("qwen3.feed_forward_length", 64)
        writer.add_uint32("qwen3.attention.head_count", 1)
        writer.add_uint32("qwen3.attention.head_count_kv", 1)
        writer.add_uint32("qwen3.attention.key_length", 32)
        writer.add_uint32("qwen3.attention.value_length", 32)
        writer.add_float32("qwen3.attention.layer_norm_rms_epsilon", 1e-6)
        writer.add_float32("qwen3.rope.freq_base", 1e6)
        writer.add_uint32("qwen3.context_length", 16)
        writer.add_tensor("output_norm.weight", norm)
        add_q8_0(writer, "token_embd.weight", emb)
        add_q8_0(writer, "output.weight", head)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        g = cpu.GGUF(str(path))
        try:
            model = cpu.LittleBitCPU(g, 8)
            h = np.tile(np.array([0.5, -0.25, 1.0, -0.75], np.float32), 8)
            got = model.logits(h)
            hn = cpu.rms_norm(h, norm, 1e-6)
            head_dq = cpu.dequant_q8_0(g.bytes("output.weight"), 32, 2)
            emb_dq = cpu.dequant_q8_0(g.bytes("token_embd.weight"), 32, 2)
            want = head_dq @ hn
            tied = emb_dq @ hn
            err = float(np.max(np.abs(got - want)))
            tied_err = float(np.max(np.abs(got - tied)))
            if err > 1e-4:
                raise AssertionError(f"output.weight ignored: err={err}")
            if tied_err < 1.0:
                raise AssertionError(f"embedding accidentally used as head: err={tied_err}")
        finally:
            g.close()

    print("LittleBit CPU untied-head test passed")


if __name__ == "__main__":
    main()
