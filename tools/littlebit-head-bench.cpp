// Standalone benchmark for the Q8_0 output-head matvec used by the LittleBit
// GGUF. It exists to check whether an "int8 head dot" prototype would add
// anything: llama.cpp already quantizes F32 activations to Q8_0 and calls an
// int8 x int8 vec_dot for Q8_0 weights (see type_traits_cpu[GGML_TYPE_Q8_0]).
//
// Build (x86 with AVX2; drop -mavx2 on ARM, where NEON/SDOT is selected by
// the library build itself):
//
//   g++ -O3 -mavx2 -I ggml/include -I ggml/src -I ggml/src/ggml-cpu \
//       tools/littlebit-head-bench.cpp \
//       -L build-lb/bin -lggml -lggml-cpu -lggml-base \
//       -Wl,-rpath,"$PWD/build-lb/bin" -o build-lb/bin/littlebit-head-bench
//
// Run:
//   ./build-lb/bin/littlebit-head-bench
//
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

extern "C" {
    void ggml_cpu_init(void);
    int64_t ggml_time_us(void);
    void quantize_row_q8_0(const float * x, void * y, int64_t k);
    void ggml_vec_dot_q8_0_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx,
                                const void * vy, size_t by, int nrc);
    void ggml_vec_dot_q8_0_q8_0_generic(int n, float * s, size_t bs, const void * vx, size_t bx,
                                        const void * vy, size_t by, int nrc);
}

struct block_q8_0 {
    uint16_t d;      // ggml_fp16_t
    int8_t   qs[32];
};

static_assert(sizeof(block_q8_0) == 34, "bad Q8_0 block layout");

static float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        bits = sign | ((mant ? 0x38800000u : 0u) | (mant << 13)) - (mant ? 0x38000000u : 0u);
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

int main() {
    ggml_cpu_init(); // initializes the F16 lookup tables used by the dot kernels

    constexpr int n       = 4096;   // n_embd
    constexpr int nrows   = 151936; // Qwen3 vocab
    constexpr int nblocks = n / 32;
    constexpr size_t row_bytes = nblocks * sizeof(block_q8_0);

    std::vector<uint8_t> w((size_t) nrows * row_bytes);
    std::vector<float> x(n);
    std::vector<uint8_t> xq(row_bytes);
    std::vector<float> out(nrows);

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> id(0, 255);

    for (float & v : x) {
        v = nd(rng);
    }
    quantize_row_q8_0(x.data(), xq.data(), n);

    for (size_t r = 0; r < (size_t) nrows; ++r) {
        auto * row = reinterpret_cast<block_q8_0 *>(w.data() + r * row_bytes);
        for (int b = 0; b < nblocks; ++b) {
            // finite, non-subnormal scales around the range seen in output.weight
            row[b].d = (uint16_t) (0x2800u + (id(rng) & 0x03ffu));
            for (int j = 0; j < 32; ++j) {
                row[b].qs[j] = (int8_t) id(rng);
            }
        }
    }

    // Tiny sanity check with known values.
    {
        float sx[32];
        uint8_t sq[34], sw[34];
        auto * swb = reinterpret_cast<block_q8_0 *>(sw);
        auto * sqb = reinterpret_cast<block_q8_0 *>(sq);
        swb->d = 0x3c00u; // 1.0
        std::memset(swb->qs, 1, 32);
        for (int j = 0; j < 32; ++j) sx[j] = 1.0f;
        quantize_row_q8_0(sx, sq, 32);
        float s1 = 0.0f, s2 = 0.0f;
        ggml_vec_dot_q8_0_q8_0(32, &s1, sizeof(float), sw, 34, sq, 34, 1);
        ggml_vec_dot_q8_0_q8_0_generic(32, &s2, sizeof(float), sw, 34, sq, 34, 1);
        printf("sanity d=%u q=%d simd=%.3f generic=%.3f\n", sqb->d, (int) sqb->qs[0], s1, s2);
    }

    // Error check against an F32 dequantized reference on a few rows.
    float max_err = 0.0f;
    for (int r = 0; r < 8; ++r) {
        const auto * row = reinterpret_cast<const block_q8_0 *>(w.data() + (size_t) r * row_bytes);
        float ref = 0.0f;
        for (int b = 0; b < nblocks; ++b) {
            const float d = fp16_to_fp32(row[b].d);
            for (int j = 0; j < 32; ++j) {
                ref += d * (float) row[b].qs[j] * x[b * 32 + j];
            }
        }
        float got = 0.0f;
        ggml_vec_dot_q8_0_q8_0(n, &got, sizeof(float), row, row_bytes, xq.data(), row_bytes, 1);
        float scalar = 0.0f;
        ggml_vec_dot_q8_0_q8_0_generic(n, &scalar, sizeof(float), row, row_bytes, xq.data(), row_bytes, 1);
        max_err = std::fmax(max_err, std::fabs(std::fmax(std::fabs(ref - got), std::fabs(scalar - got))));
    }

    // Warm up.
    float sink = 0.0f;
    for (int r = 0; r < 256; ++r) {
        float s = 0.0f;
        ggml_vec_dot_q8_0_q8_0(n, &s, sizeof(float), w.data() + (size_t) r * row_bytes,
                               row_bytes, xq.data(), row_bytes, 1);
        sink += s;
    }

    const int64_t t0 = ggml_time_us();
    for (int r = 0; r < nrows; ++r) {
        ggml_vec_dot_q8_0_q8_0(n, &out[r], sizeof(float), w.data() + (size_t) r * row_bytes,
                               row_bytes, xq.data(), row_bytes, 1);
    }
    const int64_t t1 = ggml_time_us();

    // Generic reference timing on a subset (the scalar_generic path is ~50x slower).
    constexpr int nref = 2048;
    const int64_t g0 = ggml_time_us();
    for (int r = 0; r < nref; ++r) {
        float s = 0.0f;
        ggml_vec_dot_q8_0_q8_0_generic(n, &s, sizeof(float), w.data() + (size_t) r * row_bytes,
                                       row_bytes, xq.data(), row_bytes, 1);
        sink += s;
    }
    const int64_t g1 = ggml_time_us();

    const double ms = (t1 - t0) / 1000.0;
    const double bw = (double) ((size_t) nrows * row_bytes) / (1024.0 * 1024.0 * 1024.0) / (ms / 1000.0);
    printf("head: %d x %d Q8_0, %.1f MiB\n", nrows, n,
           (double) ((size_t) nrows * row_bytes) / (1024.0 * 1024.0));
    printf("int8 (Q8_0 x Q8_0) dot : %8.2f ms  (%.2f GiB/s)\n", ms, bw);
    printf("scalar_generic subset  : %8.2f ms for %d rows (extrapolated %.1f ms)\n",
           (g1 - g0) / 1000.0, nref, (double) (g1 - g0) / 1000.0 * nrows / nref);
    printf("max abs err vs F32 dequant reference (8 rows): %.6g\n", max_err);
    printf("sink %.6f\n", sink + out[nrows - 1]);
    return 0;
}
