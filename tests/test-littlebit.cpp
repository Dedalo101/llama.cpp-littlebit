// Tests for the experimental LittleBit GGML_TYPE_LB1 tensor type.
// Uses only public ggml/gguf API so it can be built as a normal test.

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" void quantize_row_q8_0_ref(const float * x, void * y, int64_t k);
extern "C" void ggml_vec_dot_lb1_f32(int n, float * s, size_t bs, const void * vx, size_t bx,
                                     const void * vy, size_t by, int nrc);
extern "C" void dequantize_row_lb1(const int32_t * x, float * y, int64_t k);

// Release builds define NDEBUG, so assert() is a no-op. Use a hard check.
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        std::abort(); \
    } \
} while (0)

static void test_traits(void) {
    const struct ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_LB1);
    (void) tr;
    CHECK(tr != nullptr);
    CHECK(std::strcmp(tr->type_name, "lb1") == 0);
    CHECK(tr->blck_size == 32);
    CHECK(tr->type_size == 4);
    CHECK(tr->is_quantized);
    CHECK(tr->to_float != nullptr && tr->from_float_ref != nullptr);

    CHECK(ggml_type_size(GGML_TYPE_LB1) == 4);
    CHECK(ggml_blck_size(GGML_TYPE_LB1) == 32);
    CHECK(std::strcmp(ggml_type_name(GGML_TYPE_LB1), "lb1") == 0);
}

static void test_bit_layout(void) {
    const struct ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_LB1);

    const int64_t k = 64;
    std::vector<float> src(k, 1.0f);
    // +1 -> 0, -1 -> 1, LSB-first. Force a clear pattern in both words.
    src[0] = -1.0f;  // word 0, bit 0 -> 1
    src[1] =  1.0f;  // word 0, bit 1 -> 0
    src[2] = -1.0f;  // word 0, bit 2 -> 1
    src[32] = -1.0f; // word 1, bit 0 -> 1
    src[63] = -1.0f; // last bit maps to -1 (exporter pads with +1, not required here)

    std::vector<int32_t> packed(k/32, 0);
    tr->from_float_ref(src.data(), packed.data(), k);

    CHECK((uint32_t) packed[0] == 0x00000005u);
    CHECK((uint32_t) packed[1] == 0x80000001u);

    std::vector<float> out(k, 0.0f);
    tr->to_float(packed.data(), out.data(), k);
    for (int64_t i = 0; i < k; i++) {
        CHECK(out[i] == (src[i] < 0.0f ? -1.0f : 1.0f));
    }

    // zero maps to +1, same as the training-time sign() convention
    src[5] = 0.0f;
    tr->from_float_ref(src.data(), packed.data(), k);
    tr->to_float(packed.data(), out.data(), k);
    CHECK(out[5] == 1.0f);
}

static void test_gguf_roundtrip(void) {
    const int64_t k = 64;
    const size_t nbytes = (size_t) (k/32) * sizeof(int32_t);

    std::vector<uint8_t> mem(1u << 20);

    struct ggml_init_params ip = {};
    ip.mem_size = mem.size();
    ip.mem_buffer = mem.data();
    ip.no_alloc = false;
    struct ggml_context * ctx = ggml_init(ip);
    CHECK(ctx != nullptr);

    struct ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_LB1, k, 1);
    CHECK(t != nullptr);
    std::memset(t->data, 0, nbytes);
    ((int32_t *) t->data)[0] = 0x00000005;
    ((int32_t *) t->data)[1] = 0x80000001;
    ggml_set_name(t, "blk.0.lb.Vq");

    struct gguf_context * gctx = gguf_init_empty();
    CHECK(gctx != nullptr);
    gguf_add_tensor(gctx, t);

    FILE * fp = tmpfile();
    CHECK(fp != nullptr);
    CHECK(gguf_write_to_file_ptr(gctx, fp, false));
    std::fflush(fp);
    std::rewind(fp);

    struct gguf_init_params gp = { false, nullptr };
    struct gguf_context * loaded = gguf_init_from_file_ptr(fp, gp);
    CHECK(loaded != nullptr);
    std::printf("gguf roundtrip: type=%d size=%zu\n",
                (int) gguf_get_tensor_type(loaded, 0), gguf_get_tensor_size(loaded, 0));
    CHECK(gguf_get_n_tensors(loaded) == 1);
    CHECK(gguf_get_tensor_type(loaded, 0) == GGML_TYPE_LB1);
    CHECK(std::strcmp(gguf_get_tensor_name(loaded, 0), "blk.0.lb.Vq") == 0);
    CHECK(gguf_get_tensor_size(loaded, 0) == nbytes);

    gguf_free(loaded);
    std::fclose(fp);
    gguf_free(gctx);
    ggml_free(ctx);
}

static float lcg(void) {
    static uint32_t state = 1234567;
    state = state * 1103515245u + 12345u;
    return ((float) (state >> 8) / (float) (1u << 23)) - 1.0f;
}

static void test_vec_dot_lb1_f32(void) {
    const int n = 256;
    const int nb = n/32;
    std::vector<int32_t> packed(nb);
    std::vector<float> x(n), deq(n);
    uint32_t state = 987654321u;
    for (int i = 0; i < n; i++) {
        state = state*1103515245u + 12345u;
        x[i] = ((float) (state >> 8) / (float) (1u << 23)) - 1.0f;
    }
    for (int i = 0; i < nb; i++) {
        state = state*1103515245u + 12345u;
        packed[i] = (int32_t) state;
    }
    dequantize_row_lb1(packed.data(), deq.data(), n);

    float got = 0.0f;
    ggml_vec_dot_lb1_f32(n, &got, 0, packed.data(), 0, x.data(), 0, 1);

    float want = 0.0f;
    for (int i = 0; i < n; i++) {
        want += deq[i]*x[i];
    }
    CHECK(std::fabs(got - want) < 1e-4f);
    std::printf("vec_dot_lb1_f32: %.6f vs %.6f\n", got, want);
}

static void test_mul_mat_lb1(void) {
    const int64_t k = 64, split = 64, out = 64;
    const struct ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_LB1);

    std::vector<uint8_t> mem(1u << 20);
    struct ggml_init_params ip = {};
    ip.mem_size = mem.size();
    ip.mem_buffer = mem.data();
    struct ggml_context * ctx = ggml_init(ip);
    CHECK(ctx != nullptr);

    struct ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 1);
    struct ggml_tensor * v2  = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, k);
    struct ggml_tensor * Vq  = ggml_new_tensor_2d(ctx, GGML_TYPE_LB1, k, split);
    struct ggml_tensor * m   = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, split);
    struct ggml_tensor * Uq  = ggml_new_tensor_2d(ctx, GGML_TYPE_LB1, split, out);
    struct ggml_tensor * u1  = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, out);

    std::vector<float> xs(k), vs(k), ms(split), us(out);
    std::vector<float> Vf(k*split), Uf(split*out);
    for (int i = 0; i < k; i++) {
        xs[i] = lcg();
        vs[i] = lcg();
        ((ggml_fp16_t *) v2->data)[i] = ggml_fp32_to_fp16(vs[i]);
    }
    for (int i = 0; i < split; i++) {
        ms[i] = lcg();
        ((ggml_fp16_t *) m->data)[i] = ggml_fp32_to_fp16(ms[i]);
    }
    for (int i = 0; i < out; i++) {
        us[i] = lcg();
        ((ggml_fp16_t *) u1->data)[i] = ggml_fp32_to_fp16(us[i]);
    }
    for (int i = 0; i < k*split; i++) Vf[i] = lcg() < 0 ? -1.0f : 1.0f;
    for (int i = 0; i < split*out; i++) Uf[i] = lcg() < 0 ? -1.0f : 1.0f;
    std::memcpy(x->data, xs.data(), xs.size()*sizeof(float));
    tr->from_float_ref(Vf.data(), Vq->data, k*split);
    tr->from_float_ref(Uf.data(), Uq->data, split*out);

    // ggml tensors are column-major: Vq[t,j] = Vf[j*k + t]
    std::vector<float> ref(out, 0.0f);
    for (int j = 0; j < split; j++) {
        float h = 0.0f;
        for (int t = 0; t < k; t++) {
            h += Vf[j*k + t] * xs[t] * vs[t];
        }
        h *= ms[j];
        for (int i = 0; i < out; i++) {
            ref[i] += Uf[i*split + j] * h * us[i];
        }
    }

    struct ggml_tensor * y = ggml_mul_mat_lb1(ctx, x, v2, Vq, m, Uq, u1);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    float max_err = 0.0f, scale = 0.0f;
    for (int i = 0; i < out; i++) {
        float got = ((float *) y->data)[i];
        max_err = std::max(max_err, std::fabs(got - ref[i]));
        scale = std::max(scale, std::fabs(ref[i]));
    }
    std::fprintf(stderr, "mul_mat_lb1: max abs err %.4f, ref scale %.4f\n", max_err, scale);
    CHECK(max_err < 0.02f*scale);

    // Same input through a dense Q8_0 weight rebuilt from the same factors.
    std::vector<float> W(k*out, 0.0f);
    for (int j = 0; j < split; j++) {
        for (int i = 0; i < out; i++) {
            const float a = Uf[i*split + j] * ms[j] * us[i];
            for (int t = 0; t < k; t++) {
                W[i*k + t] += a * Vf[j*k + t] * vs[t];
            }
        }
    }

    const size_t qrow = ggml_row_size(GGML_TYPE_Q8_0, k);
    std::vector<uint8_t> wq(qrow*out), xq(qrow);
    for (int i = 0; i < out; i++) {
        quantize_row_q8_0_ref(&W[i*k], wq.data() + i*qrow, k);
    }
    quantize_row_q8_0_ref(xs.data(), xq.data(), k);

    struct ggml_tensor * wd = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, out);
    struct ggml_tensor * xd = ggml_new_tensor_1d(ctx, GGML_TYPE_Q8_0, k);
    std::memcpy(wd->data, wq.data(), wq.size());
    std::memcpy(xd->data, xq.data(), xq.size());

    struct ggml_tensor * yd = ggml_mul_mat(ctx, wd, xd);
    struct ggml_cgraph * gd = ggml_new_graph(ctx);
    ggml_build_forward_expand(gd, yd);
    ggml_graph_compute_with_ctx(ctx, gd, 1);

    float d_err = 0.0f, lb_err = 0.0f;
    for (int i = 0; i < out; i++) {
        const float dense = ((float *) yd->data)[i];
        const float lb = ((float *) y->data)[i];
        d_err = std::max(d_err, std::fabs(dense - ref[i]));
        lb_err = std::max(lb_err, std::fabs(lb - dense));
    }
    std::fprintf(stderr, "dense q8_0: max abs err %.4f, rel %.4f\n", d_err, d_err/scale);
    std::fprintf(stderr, "lb1 vs dense q8_0: max abs err %.4f, rel %.4f\n", lb_err, lb_err/scale);
    CHECK(d_err < 0.05f*scale);

    ggml_free(ctx);
}

int main(void) {
    test_traits();
    test_bit_layout();
    test_vec_dot_lb1_f32();
    test_gguf_roundtrip();
    test_mul_mat_lb1();
    std::printf("LittleBit LB1 tests passed\n");
    return 0;
}
