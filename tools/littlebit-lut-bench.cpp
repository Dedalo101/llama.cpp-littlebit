// Standalone prototype of the subset-sum LUT GEMV for packed LB1 factors.
//
// For a row of 8 activations x[0..7], a table of 256 partial sums
//   T[mask] = sum_b (mask_b ? -x[b] : +x[b])
// lets one 8-bit weight byte select the whole partial dot with one lookup.
// With weights repacked to [group][row], byte reads are sequential and the
// table is reused across every output row.
//
// The catch is that tables are per activation vector and must be rebuilt for
// every token. This benchmark measures BOTH sides: the LUT dot speedup and the
// full per-token table-build bill for all 36 Qwen3-8B layers. If the table bill
// eats the dot win, the design is not worth integrating.
//
// This is a benchmark only; it does not change the GGUF format or graph.
//
// Build:
//   g++ -O3 -mavx2 -I ggml/include -I ggml/src -I ggml/src/ggml-cpu \
//       tools/littlebit-lut-bench.cpp \
//       -L build-lb/bin -lggml -lggml-cpu -lggml-base \
//       -Wl,-rpath,"$PWD/build-lb/bin" -o build-lb/bin/littlebit-lut-bench
// Run:
//   ./build-lb/bin/littlebit-lut-bench
//
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

extern "C" {
    void ggml_cpu_init(void);
    int64_t ggml_time_us(void);
    void ggml_vec_dot_lb1_f32(int n, float * s, size_t bs, const void * vx, size_t bx,
                              const void * vy, size_t by, int nrc);
}

static constexpr int ROWS   = 12288;
static constexpr int COLS   = 4096;
static constexpr int WORDS  = COLS / 32;
static constexpr int GROUPS = COLS / 8;

// pat[b][mask] = -1 when bit b is set, +1 otherwise. Then
// T[mask] = sum_b x[b] * pat[b][mask], which is exactly the LB1 sign rule
// (bit 1 -> -1, bit 0 -> +1).
static float g_pat[8][256];

static void init_patterns(void) {
    for (int b = 0; b < 8; ++b) {
        for (int m = 0; m < 256; ++m) {
            g_pat[b][m] = (m & (1 << b)) ? -1.0f : 1.0f;
        }
    }
}

// Build tables for `ng` activation groups of 8. dst is ng*256 floats.
static void build_tables(const float * x, float * dst, int ng) {
    for (int g = 0; g < ng; ++g) {
        float * t = dst + (size_t) g * 256;
        const float * xg = x + (size_t) g * 8;
#if defined(__AVX2__)
        // Four accumulators held across the 8 bits, repeated for 8 chunks of
        // 32 table entries. 256 FMAs per activation group, all in L1.
        for (int q = 0; q < 8; ++q) {
            __m256 a0 = _mm256_setzero_ps();
            __m256 a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps();
            __m256 a3 = _mm256_setzero_ps();
            for (int b = 0; b < 8; ++b) {
                const __m256 vb = _mm256_set1_ps(xg[b]);
                const float * p = g_pat[b] + q * 32;
                a0 = _mm256_add_ps(_mm256_mul_ps(vb, _mm256_loadu_ps(p +  0)), a0);
                a1 = _mm256_add_ps(_mm256_mul_ps(vb, _mm256_loadu_ps(p +  8)), a1);
                a2 = _mm256_add_ps(_mm256_mul_ps(vb, _mm256_loadu_ps(p + 16)), a2);
                a3 = _mm256_add_ps(_mm256_mul_ps(vb, _mm256_loadu_ps(p + 24)), a3);
            }
            float * o = t + q * 32;
            _mm256_storeu_ps(o +  0, a0);
            _mm256_storeu_ps(o +  8, a1);
            _mm256_storeu_ps(o + 16, a2);
            _mm256_storeu_ps(o + 24, a3);
        }
#else
        for (int m = 0; m < 256; ++m) t[m] = 0.0f;
        for (int b = 0; b < 8; ++b) {
            const float xb = xg[b];
            const float * p = g_pat[b];
            for (int m = 0; m < 256; ++m) t[m] += xb * p[m];
        }
#endif
    }
}

static void bench_matrix_lut4(int rows, int cols) {
    const int words = cols / 32;
    const int pairs = cols / 8;  // one byte holds two nibble groups
    const int nibgroups = cols / 4;
    std::vector<float> x(cols);
    std::vector<int32_t> w((size_t) rows * words);
    std::vector<uint8_t> rep((size_t) pairs * rows);
    std::vector<float> table((size_t) nibgroups * 16);
    std::vector<float> out_base(rows), out_lut(rows);

    std::mt19937 rng(4242 + cols * 31 + rows);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float & v : x) v = nd(rng);
    for (auto & word : w) word = (int32_t) rng();

    for (int r = 0; r < rows; ++r) {
        const int32_t * row = w.data() + (size_t) r * words;
        for (int p = 0; p < pairs; ++p) {
            rep[(size_t) p * rows + r] =
                (uint8_t) (((uint32_t) row[p / 4] >> ((p % 4) * 8)) & 0xffu);
        }
    }

    // 4-bit tables: T[q][mask] for mask in 0..15, q = 0..n/4-1.
    int64_t tb0 = ggml_time_us();
    for (int q = 0; q < nibgroups; ++q) {
        const float * xq = x.data() + q * 4;
        float * t = table.data() + (size_t) q * 16;
        for (int m = 0; m < 16; ++m) {
            t[m] = ((m & 1) ? -xq[0] : xq[0]) + ((m & 2) ? -xq[1] : xq[1])
                 + ((m & 4) ? -xq[2] : xq[2]) + ((m & 8) ? -xq[3] : xq[3]);
        }
    }
    int64_t tb1 = ggml_time_us();

    int64_t t0 = ggml_time_us();
    for (int r = 0; r < rows; ++r) {
        ggml_vec_dot_lb1_f32(cols, &out_base[r], sizeof(float),
                             w.data() + (size_t) r * words, words * sizeof(int32_t),
                             x.data(), cols * sizeof(float), 1);
    }
    int64_t t1 = ggml_time_us();

    std::fill(out_lut.begin(), out_lut.end(), 0.0f);
    int64_t t2 = ggml_time_us();
    for (int p = 0; p < pairs; ++p) {
        const float * t0p = table.data() + (size_t) (2 * p) * 16;
        const float * t1p = table.data() + (size_t) (2 * p + 1) * 16;
        const uint8_t * rp = rep.data() + (size_t) p * rows;
        for (int r = 0; r < rows; ++r) {
            const uint8_t b = rp[r];
            out_lut[r] += t0p[b & 0x0f] + t1p[b >> 4];
        }
    }
    int64_t t3 = ggml_time_us();

    float err = 0.0f, scale = 1.0f;
    for (int r = 0; r < rows; ++r) {
        err = std::fmax(err, std::fabs(out_base[r] - out_lut[r]));
        scale = std::fmax(scale, std::fabs(out_base[r]));
    }
    printf("  LUT4 %5d x %5d : baseline %7.2f ms  LUT dot %7.2f ms  %4.2fx  table %.3f ms  rel.err %.2g\n",
           rows, cols, (t1 - t0) / 1000.0, (t3 - t2) / 1000.0,
           (double) (t1 - t0) / (t3 - t2), (tb1 - tb0) / 1000.0, err / scale);
}

static void bench_matrix(int rows, int cols) {
    const int words = cols / 32;
    const int groups = cols / 8;
    std::vector<float> x(cols);
    std::vector<int32_t> w((size_t) rows * words);
    std::vector<uint8_t> rep((size_t) groups * rows);
    std::vector<float> table((size_t) groups * 256);
    std::vector<float> out_base(rows), out_lut(rows);

    std::mt19937 rng(20261123 + cols * 31 + rows);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float & v : x) v = nd(rng);
    for (auto & word : w) word = (int32_t) rng();

    for (int r = 0; r < rows; ++r) {
        const int32_t * row = w.data() + (size_t) r * words;
        for (int g = 0; g < groups; ++g) {
            const uint32_t word = (uint32_t) row[g / 4];
            rep[(size_t) g * rows + r] = (uint8_t) ((word >> ((g % 4) * 8)) & 0xffu);
        }
    }

    build_tables(x.data(), table.data(), groups);

    int64_t t0 = ggml_time_us();
    for (int r = 0; r < rows; ++r) {
        ggml_vec_dot_lb1_f32(cols, &out_base[r], sizeof(float),
                             w.data() + (size_t) r * words, words * sizeof(int32_t),
                             x.data(), cols * sizeof(float), 1);
    }
    int64_t t1 = ggml_time_us();

    std::fill(out_lut.begin(), out_lut.end(), 0.0f);
    int64_t t2 = ggml_time_us();
    for (int g = 0; g < groups; ++g) {
        const float * tp = table.data() + (size_t) g * 256;
        const uint8_t * rp = rep.data() + (size_t) g * rows;
        for (int r = 0; r < rows; ++r) out_lut[r] += tp[rp[r]];
    }
    int64_t t3 = ggml_time_us();

    float err = 0.0f, scale = 1.0f;
    for (int r = 0; r < rows; ++r) {
        err = std::fmax(err, std::fabs(out_base[r] - out_lut[r]));
        scale = std::fmax(scale, std::fabs(out_base[r]));
    }
    printf("  %5d x %5d : baseline %7.2f ms  LUT dot %7.2f ms  %4.2fx  rel.err %.2g\n",
           rows, cols, (t1 - t0) / 1000.0, (t3 - t2) / 1000.0,
           (double) (t1 - t0) / (t3 - t2), err / scale);
}

int main() {
    ggml_cpu_init();
    init_patterns();

    printf("subset-sum LUT prototypes\n");
    bench_matrix(ROWS, COLS);
    bench_matrix_lut4(ROWS, COLS);
    bench_matrix_lut4(544, 4096);
    bench_matrix_lut4(4096, 544);
    bench_matrix_lut4(832, 12288);

    // Per-token table bill for ONE Qwen3-8B layer. Each projection has two
    // independent LittleBit factorizations, so Vq/Vq_R and Uq/Uq_R take
    // distinct activation vectors: 2*Nin + 2*rank floats per projection.
    struct Shape { int nin, rank, nout; };
    static const Shape proj[] = {
        { 4096,  544,  4096 }, // attn_q
        { 4096,  224,  1024 }, // attn_k
        { 4096,  224,  1024 }, // attn_v
        { 4096,  544,  4096 }, // attn_output
        { 4096,  832, 12288 }, // ffn_gate
        { 4096,  832, 12288 }, // ffn_up
        {12288,  832,  4096 }, // ffn_down
    };
    int64_t total_dims = 0;
    for (const auto & p : proj) total_dims += 2 * (int64_t) p.nin + 2 * p.rank;
    total_dims *= 36; // n_layer

    std::vector<float> act(total_dims);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float & v : act) v = nd(rng);

    std::vector<float> tables((size_t) total_dims / 8 * 256);
    const int64_t b0 = ggml_time_us();
    build_tables(act.data(), tables.data(), (int) (total_dims / 8));
    const int64_t b1 = ggml_time_us();

    const double ms = (b1 - b0) / 1000.0;
    printf("full-model tables: %lld activation floats, %.1f MiB written\n",
           (long long) total_dims, tables.size() * sizeof(float) / 1048576.0);
    printf("table build     : %.2f ms/token (%.2f GiB/s)\n", ms,
           tables.size() * sizeof(float) / 1048576.0 / 1024.0 / (ms / 1000.0));
    return 0;
}
