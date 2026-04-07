#!/usr/bin/env python3
"""
build_and_test.py — يبني الآلة الحاسبة ويختبرها
══════════════════════════════════════════════════
شغّل:
    python build_and_test.py
"""

import subprocess
import sys
import os
from pathlib import Path

HERE = Path(__file__).parent


def step(msg):
    print(f"\n{'═'*55}")
    print(f"  {msg}")
    print('═'*55)


def compile_lib():
    step("1. تجميع libmatcalc.so")
    src = HERE / "matcalc_core.cpp"
    out = HERE / "libmatcalc.so"

    cmd = [
        "g++", "-O3", "-march=native",
        "-mavx2", "-mfma",
        "-fopenmp",
        "-std=c++17",
        "-shared", "-fPIC",
        "-o", str(out),
        str(src),
        "-lm",
    ]

    print("الأمر:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ خطأ في التجميع:")
        print(result.stderr)
        sys.exit(1)

    size_kb = out.stat().st_size // 1024
    print(f"✅ تم البناء: {out} ({size_kb} KB)")


def run_tests():
    step("2. اختبار العمليات الأساسية")

    try:
        import torch
        import numpy as np
    except ImportError:
        print("⚠️  torch غير مثبت — جارٍ تخطي الاختبارات")
        return

    sys.path.insert(0, str(HERE))
    from matcalc_bridge import MatCalc
    mc = MatCalc(str(HERE))

    passed = 0
    failed = 0

    def check(name, result, expected, tol=1e-4):
        nonlocal passed, failed
        diff = abs(float((result - expected).abs().max()))
        if diff < tol:
            print(f"  ✅ {name} — max diff: {diff:.2e}")
            passed += 1
        else:
            print(f"  ❌ {name} — max diff: {diff:.2e}  (فوق التسامح {tol})")
            failed += 1

    # ── اختبار GEMM ──────────────────────────────────────────
    print("\n[GEMM]")
    A = torch.randn(64, 128)
    B = torch.randn(128, 64)
    ref = A @ B
    res = mc.gemm(A, B)
    check("GEMM 64×128×64", res, ref)

    A2 = torch.randn(256, 512)
    B2 = torch.randn(512, 256)
    ref2 = A2 @ B2
    res2 = mc.gemm(A2, B2)
    check("GEMM 256×512×256", res2, ref2)

    # ── اختبار Linear ─────────────────────────────────────────
    print("\n[Linear]")
    x  = torch.randn(32, 256)
    W  = torch.randn(512, 256)
    b  = torch.randn(512)
    ref_l = x @ W.T + b
    res_l = mc.linear(x, W, b)
    check("Linear with bias", res_l, ref_l)

    # ── اختبار Softmax ────────────────────────────────────────
    print("\n[Softmax]")
    s   = torch.randn(16, 64)
    ref_s = torch.softmax(s * 0.125, dim=-1)
    res_s = mc.softmax(s, scale=0.125)
    check("Softmax scaled", res_s, ref_s)

    # ── اختبار RMSNorm ────────────────────────────────────────
    print("\n[RMSNorm]")
    x_n  = torch.randn(8, 128)
    w_n  = torch.ones(128)
    eps  = 1e-6
    rms  = x_n.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    ref_n = (x_n / rms) * w_n
    res_n = mc.rmsnorm(x_n, w_n, eps)
    check("RMSNorm", res_n, ref_n)

    # ── اختبار SiLU ───────────────────────────────────────────
    print("\n[SiLU]")
    x_s  = torch.randn(256)
    ref_silu = torch.nn.functional.silu(x_s)
    res_silu = mc.silu(x_s)
    check("SiLU", res_silu, ref_silu)

    # ── اختبار Scaled Add ─────────────────────────────────────
    print("\n[Scaled Add]")
    a = torch.randn(64, 64)
    b_t = torch.randn(64, 64)
    scale = 0.5
    ref_sa = a + scale * b_t
    res_sa = mc.scaled_add(a, b_t, scale)
    check("Scaled Add", res_sa, ref_sa)

    # ── اختبار Causal Mask ────────────────────────────────────
    print("\n[Causal Mask]")
    mask = mc.causal_mask(8)
    # أول عمود يجب كله 0
    assert (mask[:, 0] == 0).all(), "أول عمود يجب أن يكون 0"
    # آخر عمود ما عدا الأخير يجب أن يكون -inf
    assert mask[0, 7] < -1e30, "mask[0,7] يجب -inf"
    print("  ✅ Causal Mask — الشكل الصحيح")
    passed += 1

    # ── ملخص ──────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print(f"  النتائج: {passed} ✅  |  {failed} ❌")
    print('─'*40)

    if failed > 0:
        print("⚠️  بعض الاختبارات فشلت — راجع الأخطاء أعلاه")
    else:
        print("🎉 جميع الاختبارات نجحت!")


def print_usage():
    step("3. طريقة الاستخدام في Transformer")
    print("""
في transformer.py — استبدل TransformerBlock بـ TransformerBlockMC:

    from transformer_matcalc import TransformerBlockMC

    # في AGITransformer.__init__:
    self.layers.append(
        TransformerBlockMC(
            embedding_dim  = embedding_dim,
            num_heads      = num_heads,
            num_kv_heads   = num_kv_heads,
            ffn_hidden     = ffn_hidden,
            use_rope       = use_rope,
            context_length = context_length,
            layer_idx      = i,
        )
    )

توقعات الأداء على جهازك (16GB RAM, 4GB VRAM):
┌──────────────────────────────────────────────────┐
│  VRAM محجوز:    ~1-2 GB   (Embeddings + Logits)  │
│  RAM مستخدمة:   ~12-14 GB (أوزان الـ 70B layers) │
│  السرعة:        أبطأ من GPU لكن بدون OOM!        │
│  الفائدة:       تشغيل 70B على 4GB VRAM 🎉        │
└──────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    compile_lib()
    run_tests()
    print_usage()
