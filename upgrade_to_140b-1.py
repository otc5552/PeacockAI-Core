"""
upgrade_to_140b.py — دليل الترقية من 70B → 140B Effective
════════════════════════════════════════════════════════════
شغّل:  python upgrade_to_140b.py
"""

import sys
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def step(n, msg):
    print(f"\n{'═'*60}")
    print(f"  الخطوة {n}: {msg}")
    print('═'*60)


def test_amplify_ops():
    step(1, "اختبار عمليات التضخيم")

    try:
        import torch
    except ImportError:
        print("⚠️ torch غير مثبت")
        return False

    from matcalc_amplify_bridge import MatCalcAmplify
    mca = MatCalcAmplify(str(HERE))

    ok = 0
    fail = 0

    def check(name, cond, detail=""):
        nonlocal ok, fail
        if cond:
            print(f"  ✅ {name}")
            ok += 1
        else:
            print(f"  ❌ {name}  {detail}")
            fail += 1

    # ── التقنية 1: Second-Order ──────────────────────────────
    print("\n[التقنية 1 — Second-Order Attention]")
    Q  = torch.randn(4, 16, 64)   # (BH, seq, head_dim)
    K  = torch.randn(4, 64, 16)   # (BH, head_dim, seq) — transposed
    s  = mca.amplified_attention_scores(Q, K, lambda_=0.1)
    s0 = mca.batched_gemm(Q, K)   # first order فقط
    diff = (s - s0).abs().mean().item()
    check("Second-order يغيّر الـ scores", diff > 1e-6, f"diff={diff:.4f}")
    check("الشكل صحيح", s.shape == (4, 16, 16))

    # ── التقنية 2: Rotated Pass ───────────────────────────────
    print("\n[التقنية 2 — Rotated Multi-Pass]")
    x = torch.randn(8, 64)
    x_rot = mca.rotate90(x)
    check("Rotate90 يغيّر القيم", (x - x_rot).abs().max().item() > 0.01)
    # دوران 4 مرات يرجع للأصل
    xr4 = mca.rotate90(mca.rotate90(mca.rotate90(mca.rotate90(x))))
    check("4 دورات = الأصل", (x - xr4).abs().max().item() < 1e-5)

    # ── التقنية 3: Deep Thinking ──────────────────────────────
    print("\n[التقنية 3 — Deep Thinking Recurrence]")
    h1 = torch.randn(512)
    x0 = torch.randn(512)
    x2 = mca.recurrent_input(h1, x0, beta=0.3)
    expected = 1.3 * h1 - 0.3 * x0
    diff = (x2 - expected).abs().max().item()
    check("recurrent_input صحيح", diff < 1e-4, f"diff={diff:.6f}")

    h2    = torch.randn(512)
    blend = mca.thinking_blend(h1, h2, alpha=0.5, clip_val=10.0)
    check("thinking_blend في النطاق", blend.abs().max().item() <= 10.0 + 1e-4)
    check("thinking_blend متوسط", (blend - 0.5*h1 - 0.5*h2).abs().max().item() < 1e-4)

    # ── التقنية 4: Cross-Layer ────────────────────────────────
    print("\n[التقنية 4 — Cross-Layer Memory]")
    h_cur  = torch.randn(256)
    h_prev = torch.randn(256)
    blended = mca.layer_blend(h_cur, h_prev)
    check("layer_blend شكله صحيح", blended.shape == h_cur.shape)
    cross = mca.cross_layer_residual(h_cur, blended, gamma=0.1)
    diff_cross = (cross - (h_cur + 0.1 * blended)).abs().max().item()
    check("cross_layer_residual صحيح", diff_cross < 1e-4, f"diff={diff_cross:.6f}")

    # ── التقنية 5: Hadamard ───────────────────────────────────
    print("\n[التقنية 5 — Hadamard Feature Mixing]")
    x_h = torch.randn(32, 128)
    h   = mca.hadamard_mix(x_h, scale=0.1)
    check("Hadamard يغيّر القيم", (h - x_h).abs().mean().item() > 1e-6)
    check("Hadamard محافظ على الشكل", h.shape == x_h.shape)

    # ── Amplified Attention الكاملة ───────────────────────────
    print("\n[Amplified Attention الكاملة]")
    Q4 = torch.randn(1, 4, 8, 32)  # (batch, heads, seq, head_dim)
    K4 = torch.randn(1, 4, 8, 32)
    V4 = torch.randn(1, 4, 8, 32)
    gw = torch.randn(32) * 0.01
    out4 = mca.amplified_attention(Q4, K4, V4, gate_w=gw, causal=True)
    check("شكل الناتج", out4.shape == (1, 4, 8, 32))
    check("لا يوجد NaN", not out4.isnan().any().item())
    check("لا يوجد Inf", not out4.isinf().any().item())

    print(f"\n{'─'*50}")
    print(f"  النتائج: {ok} ✅  |  {fail} ❌")
    return fail == 0


def show_architecture():
    step(2, "معمارية 140B Effective")
    print("""
  ┌──────────────────────────────────────────────────────┐
  │           AGI Transformer 140B Effective             │
  ├──────────────────────────────────────────────────────┤
  │  Token Embedding (vocab=128K, dim=8192)              │
  ├──────────────────────────────────────────────────────┤
  │  × 80 TransformerBlock140B                          │
  │  ┌────────────────────────────────────────────────┐  │
  │  │  ① RMSNorm                                     │  │
  │  │  ① Amplified Attention:                        │  │
  │  │     • Q@K + λ*(Q²@K²)   [Second-Order]        │  │
  │  │     • Rotated Q2,K2 → A2                       │  │
  │  │     • Gate(A1, A2)      [Multi-Pass]           │  │
  │  │     • Hadamard(output)  [Feature Mix]          │  │
  │  │  ① Residual                                    │  │
  │  ├────────────────────────────────────────────────┤  │
  │  │  ② RMSNorm                                     │  │
  │  │  ② Amplified FFN (SiLU + Hadamard)             │  │
  │  │  ② Residual                                    │  │
  │  │                  ← نهاية المرور الأول →         │  │
  │  ├────────────────────────────────────────────────┤  │
  │  │  ③ x2 = h1 + β*(h1-x)   [Recurrent Input]    │  │
  │  │  ③ مرور ثانٍ كامل على x2                      │  │
  │  │  ③ out = α*h1 + (1-α)*h2  [Thinking Blend]   │  │
  │  ├────────────────────────────────────────────────┤  │
  │  │  ④ Cross-Layer: out += γ*blend(prev layers)   │  │
  │  └────────────────────────────────────────────────┘  │
  ├──────────────────────────────────────────────────────┤
  │  Final RMSNorm → Output Projection                   │
  └──────────────────────────────────────────────────────┘

  الأوزان   : ~70B parameter  (نفس الأصلي، بدون زيادة)
  الحساب    : ~140B effective  (كل token يستفيد من 2× حساب)
  RAM needed: ~280 GB FP32   (أو ~140 GB FP16)
  VRAM used : ~1-2 GB        (Embeddings فقط)
""")


def show_migration():
    step(3, "الترقية من transformer.py الأصلي")
    print("""
  ─── قبل (70B) ───────────────────────────────────────────

    from transformer import AGITransformer

    model = AGITransformer(
        embedding_dim = 8192,
        num_layers    = 80,
        num_heads     = 64,
    )

  ─── بعد (140B Effective) ─────────────────────────────────

    from transformer_140b import AGITransformer140B

    model = AGITransformer140B(
        embedding_dim  = 8192,
        num_layers     = 80,
        num_heads      = 64,
        # معاملات التضخيم (اختيارية، لها قيم افتراضية)
        thinking_beta  = 0.3,   # قوة المرور الثاني
        thinking_alpha = 0.5,   # توازن المرورين
        cross_gamma    = 0.1,   # قوة Cross-Layer
        lambda_        = 0.1,   # وزن Second-Order
        hadamard_scale = 0.05,  # قوة Hadamard Mix
    )

  ─── أو ترقية موديل محفوظ ─────────────────────────────────

    # حمّل الأوزان من الـ 70B الأصلي
    old_state = torch.load("model_70b.pt")

    # أنشئ موديل 140B وحمّل الأوزان المشتركة
    model_140b = AGITransformer140B(...)
    model_140b.load_state_dict(old_state, strict=False)
    # strict=False يتجاهل المعاملات الجديدة (gate_w وغيرها)

  ─── ملاحظة مهمة ──────────────────────────────────────────

    الـ transformer_140b.py يحتاج:
      • matcalc_amplify.cpp   (عمليات C++ الجديدة)
      • matcalc_amplify_bridge.py
      • matcalc_bridge.py + matcalc_core.cpp (الأصليين)

    تأكد من تجميع libmatcalc.so:
      python build_and_test.py
""")


if __name__ == "__main__":
    all_ok = test_amplify_ops()
    show_architecture()
    show_migration()

    if all_ok:
        print("\n🎉 كل شيء شغّال! الشبكة جاهزة للترقية لـ 140B Effective\n")
    else:
        print("\n⚠️  بعض الاختبارات فشلت — راجع الأخطاء أعلاه\n")
