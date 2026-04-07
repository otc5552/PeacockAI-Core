/**
 * ╔══════════════════════════════════════════════════════════════════╗
 * ║         MatCalc — آلة حاسبة المصفوفات المتخصصة                 ║
 * ║         مخصصة لشبكة AGI Transformer 70B+                       ║
 * ║                                                                  ║
 * ║  يشتغل على CPU فقط — يوفر VRAM بالكامل للشبكة                  ║
 * ║  يدعم: GEMM, Attention, Softmax, RMSNorm, RoPE                  ║
 * ╚══════════════════════════════════════════════════════════════════╝
 *
 * التجميع:
 *   g++ -O3 -march=native -fopenmp -shared -fPIC \
 *       -o libmatcalc.so matcalc_core.cpp
 *
 *   أو كبرنامج مستقل:
 *   g++ -O3 -march=native -fopenmp -o matcalc matcalc_core.cpp
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <immintrin.h>   // AVX2 / AVX512 intrinsics
#include <omp.h>         // OpenMP للـ multi-threading

// ═══════════════════════════════════════════════════════════════
//  ثوابت وأنواع
// ═══════════════════════════════════════════════════════════════

using f32 = float;
using i32 = int32_t;
using u64 = uint64_t;

static constexpr i32 TILE_M = 64;   // حجم الـ tile للـ cache blocking
static constexpr i32 TILE_N = 64;
static constexpr i32 TILE_K = 64;
static constexpr f32 NEG_INF = -1e38f;

// ═══════════════════════════════════════════════════════════════
//  1. GEMM الأساسية — General Matrix Multiply
//     C = alpha * A @ B + beta * C
//     A: (M, K)  B: (K, N)  C: (M, N)
// ═══════════════════════════════════════════════════════════════

/**
 * نواة GEMM المحسّنة بـ:
 *  - Cache Blocking (Tiling) لتقليل cache misses
 *  - OpenMP لتوزيع الحساب على جميع أنوية CPU
 *  - AVX2 لحساب 8 float في نفس الوقت
 */
extern "C" void matcalc_gemm(
    const f32* __restrict__ A,   // (M, K)
    const f32* __restrict__ B,   // (K, N)
    f32*       __restrict__ C,   // (M, N)
    i32 M, i32 K, i32 N,
    f32 alpha, f32 beta
) {
    // تهيئة C بـ beta
    if (beta == 0.0f) {
        memset(C, 0, (u64)M * N * sizeof(f32));
    } else if (beta != 1.0f) {
        #pragma omp parallel for schedule(static)
        for (i32 i = 0; i < M * N; i++) C[i] *= beta;
    }

    // GEMM بـ Tiling + OpenMP + AVX2
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (i32 i0 = 0; i0 < M; i0 += TILE_M) {
        for (i32 j0 = 0; j0 < N; j0 += TILE_N) {
            i32 iEnd = std::min(i0 + TILE_M, M);
            i32 jEnd = std::min(j0 + TILE_N, N);

            for (i32 k0 = 0; k0 < K; k0 += TILE_K) {
                i32 kEnd = std::min(k0 + TILE_K, K);

                for (i32 i = i0; i < iEnd; i++) {
                    for (i32 j = j0; j < jEnd; j += 8) {
                        // AVX2: نحسب 8 عناصر في خطوة واحدة
                        __m256 sum = _mm256_setzero_ps();

                        i32 kk = k0;
                        for (; kk <= kEnd - 8; kk += 8) {
                            __m256 a = _mm256_loadu_ps(&A[i * K + kk]);
                            // Broadcast لكل عمود من B
                            // (مبسّط — في الإنتاج نستخدم transpose)
                            for (i32 jj = j; jj < std::min(j+8, jEnd); jj++) {
                                f32 bvals[8];
                                for (i32 x = 0; x < 8; x++)
                                    bvals[x] = (kk+x < K) ? B[(kk+x)*N + jj] : 0.f;
                                __m256 b   = _mm256_loadu_ps(bvals);
                                __m256 mul = _mm256_mul_ps(a, b);
                                // horizontal sum
                                __m128 lo  = _mm256_castps256_ps128(mul);
                                __m128 hi  = _mm256_extractf128_ps(mul, 1);
                                __m128 s   = _mm_add_ps(lo, hi);
                                s = _mm_hadd_ps(s, s);
                                s = _mm_hadd_ps(s, s);
                                C[i*N + jj] += alpha * _mm_cvtss_f32(s);
                            }
                        }

                        // باقي العناصر بدون AVX
                        for (; kk < kEnd; kk++) {
                            f32 a = A[i * K + kk];
                            for (i32 jj = j; jj < std::min(j+8, jEnd); jj++) {
                                C[i*N + jj] += alpha * a * B[kk*N + jj];
                            }
                        }
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  2. Batched GEMM — لحساب Q×K و Attention×V
//     C[b] = A[b] @ B[b]  لكل batch
// ═══════════════════════════════════════════════════════════════

extern "C" void matcalc_batched_gemm(
    const f32* A,   // (B, M, K)
    const f32* B_,  // (B, K, N)
    f32*       C,   // (B, M, N)
    i32 batch, i32 M, i32 K, i32 N
) {
    #pragma omp parallel for schedule(static)
    for (i32 b = 0; b < batch; b++) {
        const f32* Ab = A  + (u64)b * M * K;
        const f32* Bb = B_ + (u64)b * K * N;
        f32*       Cb = C  + (u64)b * M * N;
        matcalc_gemm(Ab, Bb, Cb, M, K, N, 1.0f, 0.0f);
    }
}

// ═══════════════════════════════════════════════════════════════
//  3. Softmax — مستخدم في Attention
//     out[i] = exp(x[i] - max) / sum(exp(x - max))
//     mask: قيم -inf للمواضع المحجوبة (causal mask)
// ═══════════════════════════════════════════════════════════════

extern "C" void matcalc_softmax(
    const f32* __restrict__ x,
    f32*       __restrict__ out,
    const f32* __restrict__ mask,   // NULL = بدون mask
    i32 rows, i32 cols,
    f32 scale    // = 1/sqrt(head_dim) في الـ attention
) {
    #pragma omp parallel for schedule(static)
    for (i32 r = 0; r < rows; r++) {
        const f32* xr   = x   + (u64)r * cols;
        f32*       outr = out + (u64)r * cols;

        // 1. حساب max للاستقرار العددي
        f32 maxval = NEG_INF;
        for (i32 c = 0; c < cols; c++) {
            f32 val = xr[c] * scale;
            if (mask) val += mask[r * cols + c];
            if (val > maxval) maxval = val;
        }

        // 2. exp(x - max) وحساب المجموع
        f32 sumval = 0.0f;
        for (i32 c = 0; c < cols; c++) {
            f32 val = xr[c] * scale;
            if (mask) val += mask[r * cols + c];
            outr[c] = expf(val - maxval);
            sumval += outr[c];
        }

        // 3. التطبيع
        f32 inv_sum = 1.0f / (sumval + 1e-10f);
        for (i32 c = 0; c < cols; c++) {
            outr[c] *= inv_sum;
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  4. RMSNorm — مستخدم في TransformerBlock
//     out = (x / RMS(x)) * scale
// ═══════════════════════════════════════════════════════════════

extern "C" void matcalc_rmsnorm(
    const f32* __restrict__ x,
    const f32* __restrict__ weight,  // معاملات الـ scale
    f32*       __restrict__ out,
    i32 rows, i32 dim,
    f32 eps
) {
    #pragma omp parallel for schedule(static)
    for (i32 r = 0; r < rows; r++) {
        const f32* xr   = x   + (u64)r * dim;
        f32*       outr = out + (u64)r * dim;

        // حساب RMS
        f32 ss = 0.0f;
        for (i32 d = 0; d < dim; d++) ss += xr[d] * xr[d];
        f32 rms_inv = 1.0f / sqrtf(ss / dim + eps);

        // تطبيق الـ scale
        for (i32 d = 0; d < dim; d++) {
            outr[d] = xr[d] * rms_inv * weight[d];
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  5. RoPE — Rotary Position Embedding
//     يطبّق الدوران على Q و K في الـ attention
// ═══════════════════════════════════════════════════════════════

extern "C" void matcalc_rope(
    f32*       __restrict__ x,      // (seq_len, num_heads, head_dim)
    i32 seq_len, i32 num_heads, i32 head_dim,
    f32 base   // = 10000.0f
) {
    i32 half = head_dim / 2;

    #pragma omp parallel for collapse(2) schedule(static)
    for (i32 pos = 0; pos < seq_len; pos++) {
        for (i32 h = 0; h < num_heads; h++) {
            f32* xh = x + ((u64)pos * num_heads + h) * head_dim;

            for (i32 i = 0; i < half; i++) {
                f32 theta = powf(base, -2.0f * i / head_dim);
                f32 freq  = pos * theta;
                f32 cos_f = cosf(freq);
                f32 sin_f = sinf(freq);

                f32 x0 = xh[i];
                f32 x1 = xh[i + half];

                // تطبيق الدوران
                xh[i]        = x0 * cos_f - x1 * sin_f;
                xh[i + half] = x0 * sin_f + x1 * cos_f;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  6. SiLU Activation — مستخدم في MoE FFN
//     out = x * sigmoid(x)
// ═══════════════════════════════════════════════════════════════

extern "C" void matcalc_silu(
    const f32* __restrict__ x,
    f32*       __restrict__ out,
    i32 n
) {
    #pragma omp parallel for simd schedule(static)
    for (i32 i = 0; i < n; i++) {
        f32 val = x[i];
        out[i]  = val / (1.0f + expf(-val));
    }
}

// ═══════════════════════════════════════════════════════════════
//  7. Element-wise Add — Residual connections
// ═══════════════════════════════════════════════════════════════

extern "C" void matcalc_add(
    const f32* __restrict__ a,
    const f32* __restrict__ b,
    f32*       __restrict__ out,
    i32 n
) {
    #pragma omp parallel for simd schedule(static)
    for (i32 i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

// ═══════════════════════════════════════════════════════════════
//  8. Scaled Element-wise Add — للـ residual_scale في TransformerBlock
//     out = a + scale * b
// ═══════════════════════════════════════════════════════════════

extern "C" void matcalc_scaled_add(
    const f32* __restrict__ a,
    const f32* __restrict__ b,
    f32*       __restrict__ out,
    f32 scale,
    i32 n
) {
    #pragma omp parallel for simd schedule(static)
    for (i32 i = 0; i < n; i++) {
        out[i] = a[i] + scale * b[i];
    }
}

// ═══════════════════════════════════════════════════════════════
//  9. Linear Layer = GEMM + Optional Bias
//     out = x @ W^T + bias
//     x: (batch*seq, in_features)
//     W: (out_features, in_features)  — مخزّنة transposed
// ═══════════════════════════════════════════════════════════════

extern "C" void matcalc_linear(
    const f32* __restrict__ x,      // (M, K)
    const f32* __restrict__ W,      // (N, K) — transposed
    const f32* __restrict__ bias,   // (N,) أو NULL
    f32*       __restrict__ out,    // (M, N)
    i32 M, i32 K, i32 N
) {
    // GEMM: out = x @ W^T
    // لأن W مخزّنة (N, K)، نحسب out[i,j] = sum_k x[i,k] * W[j,k]
    #pragma omp parallel for schedule(static)
    for (i32 i = 0; i < M; i++) {
        for (i32 j = 0; j < N; j++) {
            f32 sum = 0.0f;
            const f32* xr = x + (u64)i * K;
            const f32* wr = W + (u64)j * K;

            // AVX2: نحسب dot product بكفاءة
            i32 k = 0;
            __m256 acc = _mm256_setzero_ps();
            for (; k <= K - 8; k += 8) {
                __m256 xv = _mm256_loadu_ps(xr + k);
                __m256 wv = _mm256_loadu_ps(wr + k);
                acc = _mm256_fmadd_ps(xv, wv, acc);
            }
            // جمع الـ 8 عناصر
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            lo = _mm_add_ps(lo, hi);
            lo = _mm_hadd_ps(lo, lo);
            lo = _mm_hadd_ps(lo, lo);
            sum = _mm_cvtss_f32(lo);

            // باقي العناصر
            for (; k < K; k++) sum += xr[k] * wr[k];

            out[i*N + j] = sum + (bias ? bias[j] : 0.0f);
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  10. Causal Mask Generator — لمنع النظر للمستقبل في Attention
// ═══════════════════════════════════════════════════════════════

extern "C" void matcalc_causal_mask(
    f32* __restrict__ mask,   // (seq_len, seq_len)
    i32 seq_len
) {
    #pragma omp parallel for schedule(static)
    for (i32 i = 0; i < seq_len; i++) {
        for (i32 j = 0; j < seq_len; j++) {
            // المواضع المستقبلية تصبح -inf
            mask[i * seq_len + j] = (j <= i) ? 0.0f : NEG_INF;
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  11. معلومات النظام والإمكانيات
// ═══════════════════════════════════════════════════════════════

extern "C" int matcalc_num_threads() {
    return omp_get_max_threads();
}

extern "C" const char* matcalc_version() {
    return "MatCalc 1.0 — AGI Transformer Engine | CPU-Only FP32";
}
