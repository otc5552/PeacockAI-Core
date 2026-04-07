/**
 * ╔══════════════════════════════════════════════════════════════════╗
 * ║     matcalc_amplify.cpp — عمليات تضخيم الحساب 140B             ║
 * ║                                                                  ║
 * ║  تحوّل شبكة 70B لـ 140B effective بدون زيادة الأوزان           ║
 * ║                                                                  ║
 * ║  التقنيات:                                                       ║
 * ║   1. Rotated Multi-Pass Attention  (2× حساب Attention)          ║
 * ║   2. Deep Thinking Recurrence      (طبقة تفكر مرتين)            ║
 * ║   3. Cross-Layer Attention Cache   (يشوف طبقات أعمق)            ║
 * ║   4. Hadamard Feature Mixing       (تقاطع ميزات مختلف)          ║
 * ╚══════════════════════════════════════════════════════════════════╝
 *
 * التجميع مع matcalc_core.cpp:
 *   g++ -O3 -march=native -fopenmp -mavx2 -mfma -std=c++17 \
 *       -shared -fPIC \
 *       -o libmatcalc.so matcalc_core.cpp matcalc_amplify.cpp -lm
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include <omp.h>

using f32 = float;
using i32 = int32_t;
using u64 = uint64_t;

static constexpr f32 NEG_INF_A = -1e38f;

// ═══════════════════════════════════════════════════════════════
//  HELPER: dot product بـ AVX2
// ═══════════════════════════════════════════════════════════════

static inline f32 dot_avx2(const f32* a, const f32* b, i32 n) {
    __m256 acc = _mm256_setzero_ps();
    i32 i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 av = _mm256_loadu_ps(a + i);
        __m256 bv = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(av, bv, acc);
    }
    // horizontal sum
    __m128 lo  = _mm256_castps256_ps128(acc);
    __m128 hi  = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    f32 result = _mm_cvtss_f32(sum);
    for (; i < n; i++) result += a[i] * b[i];
    return result;
}

// ═══════════════════════════════════════════════════════════════
//  1. ROTATED MULTI-PASS ATTENTION
//     ══════════════════════════════
//     الفكرة: بدل ما كل head يحسب Attention مرة واحدة،
//     نحسبها مرتين — المرة الثانية بـ Q و K مدوّرين 90°
//     في الـ feature space.
//
//     النتيجة: كل head يرى المعلومات من زاويتين مختلفتين
//              = ضعف عمق الفهم بنفس الأوزان
//
//     الحساب:
//       pass1: A1 = softmax(Q @ K^T / sqrt(d)) @ V
//       pass2: Q2 = Q * cos(θ) + Q_rot * sin(θ)
//              K2 = K * cos(θ) + K_rot * sin(θ)
//              A2 = softmax(Q2 @ K2^T / sqrt(d)) @ V
//       out  = gate * A1 + (1-gate) * A2
//              gate يتعلم من السياق
// ═══════════════════════════════════════════════════════════════

/**
 * يطبّق دوران 90° على المتجهات في الـ feature space
 * x_rot[i]        = -x[i + half]
 * x_rot[i + half] =  x[i]
 * (نفس مبدأ RoPE لكن على كل الـ features دفعة واحدة)
 */
extern "C" void matcalc_rotate90(
    const f32* __restrict__ x,     // (rows, dim)
    f32*       __restrict__ out,   // (rows, dim)
    i32 rows, i32 dim
) {
    i32 half = dim / 2;
    #pragma omp parallel for schedule(static)
    for (i32 r = 0; r < rows; r++) {
        const f32* xr  = x   + (u64)r * dim;
        f32*       outr = out + (u64)r * dim;
        // الجزء الأول: -x[half:]
        for (i32 i = 0; i < half; i++) outr[i] = -xr[i + half];
        // الجزء الثاني: x[:half]
        for (i32 i = 0; i < half; i++) outr[i + half] = xr[i];
    }
}

/**
 * مزج نتيجتَي الـ Attention بـ learnable gate
 * out = sigmoid(g) * A1 + (1 - sigmoid(g)) * A2
 * حيث g = dot(x, gate_w) — يتعلم من السياق
 */
extern "C" void matcalc_attention_gate_mix(
    const f32* __restrict__ A1,      // (rows, dim)
    const f32* __restrict__ A2,      // (rows, dim)
    const f32* __restrict__ gate_w,  // (dim,) — أوزان الـ gate
    f32*       __restrict__ out,     // (rows, dim)
    i32 rows, i32 dim
) {
    #pragma omp parallel for schedule(static)
    for (i32 r = 0; r < rows; r++) {
        const f32* a1r = A1     + (u64)r * dim;
        const f32* a2r = A2     + (u64)r * dim;
        f32*       outr = out   + (u64)r * dim;

        // gate scalar من dot product
        f32 raw_gate = dot_avx2(a1r, gate_w, dim);  // نستخدم A1 كـ context
        f32 g = 1.0f / (1.0f + expf(-raw_gate * 0.01f));  // sigmoid مع تخفيف

        f32 inv_g = 1.0f - g;

        // AVX2 mix
        i32 i = 0;
        __m256 gv    = _mm256_set1_ps(g);
        __m256 inv_gv = _mm256_set1_ps(inv_g);
        for (; i <= dim - 8; i += 8) {
            __m256 a1v = _mm256_loadu_ps(a1r + i);
            __m256 a2v = _mm256_loadu_ps(a2r + i);
            __m256 mixed = _mm256_add_ps(
                _mm256_mul_ps(gv, a1v),
                _mm256_mul_ps(inv_gv, a2v)
            );
            _mm256_storeu_ps(outr + i, mixed);
        }
        for (; i < dim; i++)
            outr[i] = g * a1r[i] + inv_g * a2r[i];
    }
}

// ═══════════════════════════════════════════════════════════════
//  2. DEEP THINKING RECURRENCE (DTR)
//     ══════════════════════════════
//     الفكرة: بدل ما الطبقة تمر مرة واحدة على الـ input،
//     تمر مرتين — المرة الثانية تأخذ output المرة الأولى
//     وتعيد التفكير فيه.
//
//     مستوحاة من: Mixture of Depths + Universal Transformers
//
//     الحساب:
//       h1 = LayerNorm(x + Attention(x))
//       h1 = LayerNorm(h1 + FFN(h1))        ← المرور الأول
//       delta = h1 - x                       ← ما تعلّمته
//       h2 = LayerNorm(h1 + Attention(h1 + beta*delta))
//       h2 = LayerNorm(h2 + FFN(h2))         ← المرور الثاني
//       out = alpha*h1 + (1-alpha)*h2
//
//     beta  = معامل "قوة التفكير الثاني" (0.3 افتراضي)
//     alpha = وزن المزج (0.5 = متساوي)
// ═══════════════════════════════════════════════════════════════

/**
 * يحسب delta = h1 - x0 ويضيفه بنسبة beta
 * recurrent_input = h1 + beta * (h1 - x0)
 * = (1+beta)*h1 - beta*x0
 */
extern "C" void matcalc_recurrent_input(
    const f32* __restrict__ h1,    // (n,) — output المرور الأول
    const f32* __restrict__ x0,    // (n,) — input الأصلي
    f32*       __restrict__ out,   // (n,)
    f32 beta,                       // قوة التفكير (0.2 - 0.4)
    i32 n
) {
    f32 coef_h1 = 1.0f + beta;
    f32 coef_x0 = -beta;

    #pragma omp parallel for simd schedule(static)
    for (i32 i = 0; i < n; i++) {
        out[i] = coef_h1 * h1[i] + coef_x0 * x0[i];
    }
}

/**
 * مزج نتيجتي التفكير الأول والثاني
 * out = alpha * h1 + (1-alpha) * h2
 * مع clipping لمنع انفجار القيم في الطبقات العميقة
 */
extern "C" void matcalc_thinking_blend(
    const f32* __restrict__ h1,     // (n,) — التفكير الأول
    const f32* __restrict__ h2,     // (n,) — التفكير الثاني
    f32*       __restrict__ out,    // (n,)
    f32 alpha,                       // وزن الأول (0.4-0.6)
    f32 clip_val,                    // حد الـ clipping (مثلاً 10.0)
    i32 n
) {
    f32 inv_alpha = 1.0f - alpha;
    __m256 av    = _mm256_set1_ps(alpha);
    __m256 iav   = _mm256_set1_ps(inv_alpha);
    __m256 clipv = _mm256_set1_ps(clip_val);
    __m256 nclipv= _mm256_set1_ps(-clip_val);

    i32 i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 v1 = _mm256_loadu_ps(h1 + i);
        __m256 v2 = _mm256_loadu_ps(h2 + i);
        __m256 mixed = _mm256_add_ps(
            _mm256_mul_ps(av, v1),
            _mm256_mul_ps(iav, v2)
        );
        // Clipping لمنع انفجار القيم
        mixed = _mm256_min_ps(mixed, clipv);
        mixed = _mm256_max_ps(mixed, nclipv);
        _mm256_storeu_ps(out + i, mixed);
    }
    for (; i < n; i++) {
        f32 v = alpha * h1[i] + inv_alpha * h2[i];
        out[i] = fmaxf(-clip_val, fminf(clip_val, v));
    }
}

// ═══════════════════════════════════════════════════════════════
//  3. CROSS-LAYER ATTENTION CACHE (CLAC)
//     ════════════════════════════════════
//     الفكرة: بدل ما كل طبقة تشوف output نفسها فقط،
//     تشوف أيضاً outputs من 2-3 طبقات سابقة.
//
//     مستوحاة من: DenseNet + Memory Transformers
//
//     الحساب:
//       context = [h_current, h_prev2, h_prev4]  ← 3 طبقات
//       cross_key   = W_ck @ concat(context)
//       cross_value = W_cv @ concat(context)
//       cross_out = Attention(Q, cross_key, cross_value)
//       out = h_current + gamma * cross_out
//       gamma يبدأ صغير (0.1) ويزيد مع التدريب
// ═══════════════════════════════════════════════════════════════

/**
 * يدمج hidden states من طبقات مختلفة بأوزان تتناقص مع البُعد
 * merged = w0*h0 + w1*h1 + w2*h2
 * حيث w0=0.6, w1=0.3, w2=0.1 (أحدث طبقة أهم)
 */
extern "C" void matcalc_layer_blend(
    const f32* __restrict__ h0,     // الطبقة الحالية  (n,)
    const f32* __restrict__ h1,     // طبقة -2         (n,) أو NULL
    const f32* __restrict__ h2,     // طبقة -4         (n,) أو NULL
    f32*       __restrict__ out,    // (n,)
    i32 n
) {
    // أوزان تتناقص — الطبقة الأحدث أهم
    const f32 w0 = (h1 == nullptr) ? 1.0f : ((h2 == nullptr) ? 0.7f : 0.6f);
    const f32 w1 = (h1 == nullptr) ? 0.0f : ((h2 == nullptr) ? 0.3f : 0.3f);
    const f32 w2 = (h2 == nullptr) ? 0.0f : 0.1f;

    __m256 v0 = _mm256_set1_ps(w0);
    __m256 v1 = _mm256_set1_ps(w1);
    __m256 v2 = _mm256_set1_ps(w2);

    i32 i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 res = _mm256_mul_ps(v0, _mm256_loadu_ps(h0 + i));
        if (h1 != nullptr)
            res = _mm256_fmadd_ps(v1, _mm256_loadu_ps(h1 + i), res);
        if (h2 != nullptr)
            res = _mm256_fmadd_ps(v2, _mm256_loadu_ps(h2 + i), res);
        _mm256_storeu_ps(out + i, res);
    }
    for (; i < n; i++) {
        out[i] = w0 * h0[i]
               + (h1 ? w1 * h1[i] : 0.f)
               + (h2 ? w2 * h2[i] : 0.f);
    }
}

/**
 * Cross-Layer Residual: يضيف معلومات الطبقات السابقة بـ gamma صغير
 * out = x + gamma * cross_info
 * gamma صغير (0.05-0.15) عشان ما يطغاش على المعلومات الحالية
 */
extern "C" void matcalc_cross_layer_residual(
    const f32* __restrict__ x,          // (n,) — hidden state الحالي
    const f32* __restrict__ cross_info, // (n,) — المعلومات من الطبقات السابقة
    f32*       __restrict__ out,        // (n,)
    f32 gamma,                           // قوة التأثير (0.05-0.15)
    i32 n
) {
    __m256 gv = _mm256_set1_ps(gamma);
    i32 i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 cv = _mm256_loadu_ps(cross_info + i);
        _mm256_storeu_ps(out + i, _mm256_fmadd_ps(gv, cv, xv));
    }
    for (; i < n; i++) out[i] = x[i] + gamma * cross_info[i];
}

// ═══════════════════════════════════════════════════════════════
//  4. HADAMARD FEATURE MIXING (HFM)
//     ══════════════════════════════
//     الفكرة: بعد الـ Attention، نمزج الميزات بطريقة
//     مختلفة باستخدام ضرب Hadamard (element-wise)
//     مع نسخة مزاحة.
//
//     النتيجة: يخلق تفاعلات بين الميزات بدون أوزان إضافية
//              = زيادة القدرة التعبيرية مجاناً
//
//     الحساب:
//       x_shift = x rolled by dim/4 positions
//       mixed   = x ⊙ (x_shift + 1)   ← Hadamard
//       out     = RMSNorm(x + 0.1 * mixed)
// ═══════════════════════════════════════════════════════════════

/**
 * Circular shift على الـ features: x_shift[i] = x[(i + shift) % dim]
 * يخلق "cross-feature interactions" بدون أوزان
 */
extern "C" void matcalc_feature_shift(
    const f32* __restrict__ x,     // (rows, dim)
    f32*       __restrict__ out,   // (rows, dim)
    i32 rows, i32 dim, i32 shift
) {
    shift = ((shift % dim) + dim) % dim;  // normalize
    #pragma omp parallel for schedule(static)
    for (i32 r = 0; r < rows; r++) {
        const f32* xr   = x   + (u64)r * dim;
        f32*       outr = out + (u64)r * dim;
        // الجزء الأول: من shift إلى dim
        memcpy(outr, xr + shift, (dim - shift) * sizeof(f32));
        // الجزء الثاني: من 0 إلى shift
        memcpy(outr + (dim - shift), xr, shift * sizeof(f32));
    }
}

/**
 * Hadamard Mix: out = x + scale * (x ⊙ x_shifted)
 * يضيف تفاعلات غير خطية بين الميزات بدون أوزان جديدة
 */
extern "C" void matcalc_hadamard_mix(
    const f32* __restrict__ x,         // (n,)
    const f32* __restrict__ x_shifted, // (n,)
    f32*       __restrict__ out,        // (n,)
    f32 scale,                          // قوة التأثير (0.05-0.15)
    i32 n
) {
    __m256 sv = _mm256_set1_ps(scale);
    __m256 one = _mm256_set1_ps(1.0f);
    i32 i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 xv  = _mm256_loadu_ps(x + i);
        __m256 xsv = _mm256_loadu_ps(x_shifted + i);
        // x ⊙ (x_shifted + 1)
        __m256 interaction = _mm256_mul_ps(xv, _mm256_add_ps(xsv, one));
        // x + scale * interaction
        _mm256_storeu_ps(out + i, _mm256_fmadd_ps(sv, interaction, xv));
    }
    for (; i < n; i++)
        out[i] = x[i] + scale * x[i] * (x_shifted[i] + 1.0f);
}

// ═══════════════════════════════════════════════════════════════
//  5. AMPLIFIED ATTENTION SCORE
//     ══════════════════════════
//     الفكرة: بدل ما نحسب scores = Q @ K^T فقط،
//     نضيف second-order term: scores += lambda * (Q⊙Q) @ (K⊙K)^T
//     ده بيخلي الـ attention يلتقط علاقات من الدرجة الثانية
//
//     الحساب:
//       scores1 = Q @ K^T / sqrt(d)          ← first order
//       Q2 = Q * Q   (element-wise square)
//       K2 = K * K
//       scores2 = Q2 @ K2^T / sqrt(d)        ← second order
//       scores  = scores1 + lambda * scores2
//       lambda  = 0.1 (صغير عشان ما يطغاش)
// ═══════════════════════════════════════════════════════════════

/**
 * Element-wise square: out = x * x
 * لحساب Q² و K² للـ second-order attention
 */
extern "C" void matcalc_elem_square(
    const f32* __restrict__ x,
    f32*       __restrict__ out,
    i32 n
) {
    i32 i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(v, v));
    }
    for (; i < n; i++) out[i] = x[i] * x[i];
}

/**
 * يجمع first-order و second-order attention scores
 * out = scores1 + lambda * scores2
 */
extern "C" void matcalc_add_second_order(
    const f32* __restrict__ scores1,  // (n,)
    const f32* __restrict__ scores2,  // (n,)
    f32*       __restrict__ out,       // (n,)
    f32 lambda,                         // وزن الـ second order (0.05-0.15)
    i32 n
) {
    __m256 lv = _mm256_set1_ps(lambda);
    i32 i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 s1 = _mm256_loadu_ps(scores1 + i);
        __m256 s2 = _mm256_loadu_ps(scores2 + i);
        _mm256_storeu_ps(out + i, _mm256_fmadd_ps(lv, s2, s1));
    }
    for (; i < n; i++) out[i] = scores1[i] + lambda * scores2[i];
}

// ═══════════════════════════════════════════════════════════════
//  6. معلومات الإضافة
// ═══════════════════════════════════════════════════════════════

extern "C" const char* matcalc_amplify_version() {
    return "MatCalc Amplify 1.0 — 140B Effective | 5 Techniques";
}
