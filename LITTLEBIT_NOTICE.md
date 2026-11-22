LittleBit additions to llama.cpp
================================

This repository is a personal fork of ggml-org/llama.cpp, pinned to Ollama's
llama.cpp tag b12257 (commit 17701e4), adding experimental support for
LittleBit packed ±1 factors.

Clean-room statement
--------------------

The LittleBit-specific code here is written independently from the publicly
described checkpoint format and from the independent numpy reference in
Dedalo101/LittleBit (branch `runtime-spike`), which itself contains no
upstream source code. No source files from SamsungLabs/LittleBit are copied
into this repository.

Licensing
---------

- llama.cpp and ggml: MIT (see LICENSE).
- SamsungLabs/LittleBit: CC BY-NC 4.0 (see the upstream LICENSE). The
  published LittleBit Qwen3 checkpoint declares Apache-2.0.

The LittleBit technique and upstream project are non-commercial. Do not use
this fork commercially without reviewing the upstream CC BY-NC 4.0 terms.
Do not propose these changes upstream to llama.cpp or Ollama without legal
review.

Model weights are NOT included in this repository.
