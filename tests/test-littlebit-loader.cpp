// Loader-level regression test for untied output.weight handling.
// Builds a tiny GGUF with different embedding and head data, then checks
// that llama_model_loader keeps the two weights separate.

#include "ggml.h"
#include "gguf.h"
#include "llama-model-loader.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        std::abort(); \
    } \
} while (0)

static void write_untied_gguf(const char * path, const std::vector<float> & emb,
                              const std::vector<float> & head) {
    struct ggml_init_params ip = {};
    ip.mem_size = 1u << 20;
    ip.no_alloc = false;
    struct ggml_context * ctx = ggml_init(ip);
    CHECK(ctx != nullptr);

    struct gguf_context * gctx = gguf_init_empty();
    CHECK(gctx != nullptr);
    gguf_set_val_str(gctx, "general.architecture", "qwen3");

    const int64_t n_embd = 8;
    struct ggml_tensor * emb_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 2);
    struct ggml_tensor * head_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 2);
    struct ggml_tensor * norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    CHECK(emb_t != nullptr && head_t != nullptr && norm_t != nullptr);

    std::vector<float> norm(n_embd, 1.0f);
    ggml_set_name(emb_t, "token_embd.weight");
    ggml_set_name(head_t, "output.weight");
    ggml_set_name(norm_t, "output_norm.weight");
    gguf_add_tensor(gctx, emb_t);
    gguf_add_tensor(gctx, head_t);
    gguf_add_tensor(gctx, norm_t);
    gguf_set_tensor_data(gctx, "token_embd.weight", emb.data());
    gguf_set_tensor_data(gctx, "output.weight", head.data());
    gguf_set_tensor_data(gctx, "output_norm.weight", norm.data());

    CHECK(gguf_write_to_file(gctx, path, false));

    gguf_free(gctx);
    ggml_free(ctx);
}

int main(void) {
    const char * path = "littlebit-untied-test.gguf";
    std::vector<float> emb(16), head(16);
    for (size_t i = 0; i < emb.size(); i++) {
        emb[i] = float(i) + 1.0f;
        head[i] = -float(i) - 100.0f;
    }

    write_untied_gguf(path, emb, head);

    std::vector<std::string> splits;
    const llama_model_kv_override * overrides = nullptr;
    llama_model_loader ml(nullptr, nullptr, nullptr, path, splits, nullptr,
                          LLAMA_LOAD_MODE_NONE, /*check_tensors*/ true,
                          /*no_alloc*/ true, /*load_mtp*/ false,
                          overrides, nullptr);

    const llama_model_loader::llama_tensor_weight * emb_w = ml.get_weight("token_embd.weight");
    const llama_model_loader::llama_tensor_weight * head_w = ml.get_weight("output.weight");
    CHECK(emb_w != nullptr);
    CHECK(head_w != nullptr);
    CHECK(emb_w->offs != head_w->offs);
    CHECK(emb_w->idx == head_w->idx);

    std::vector<float> emb_read(emb.size()), head_read(head.size());
    ml.load_data_range(*emb_w, 0, emb_read.size()*sizeof(float), emb_read.data());
    ml.load_data_range(*head_w, 0, head_read.size()*sizeof(float), head_read.data());

    float max_err = 0.0f;
    for (size_t i = 0; i < emb.size(); i++) {
        max_err = std::max(max_err, std::fabs(emb_read[i] - emb[i]));
        max_err = std::max(max_err, std::fabs(head_read[i] - head[i]));
    }
    CHECK(max_err < 1e-6f);

    std::printf("untied head loader: offsets %zu != %zu, max err %.6f\n",
                emb_w->offs, head_w->offs, max_err);
    std::remove(path);
    std::printf("LittleBit untied-head loader test passed\n");
    return 0;
}
