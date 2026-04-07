"""
matcalc_amplify_bridge.py — جسر Python لعمليات التضخيم
═══════════════════════════════════════════════════════════
يضيف 5 تقنيات تحوّل الشبكة من 70B → 140B effective
بدون زيادة أي أوزان
"""

import ctypes
import torch
from pathlib import Path
from matcalc_bridge import MatCalc, _compile_if_needed


def _compile_amplify(src_dir: Path) -> Path:
    """يجمّع الـ amplify مع الـ core في مكتبة واحدة"""
    import subprocess
    lib_path = src_dir / "libmatcalc.so"
    core_src = src_dir / "matcalc_core.cpp"
    amp_src  = src_dir / "matcalc_amplify.cpp"

    # نتحقق إذا الـ amplify أحدث
    need_rebuild = (
        not lib_path.exists()
        or amp_src.stat().st_mtime > lib_path.stat().st_mtime
        or core_src.stat().st_mtime > lib_path.stat().st_mtime
    )

    if not need_rebuild:
        return lib_path

    print("[MatCalc Amplify] جاري إعادة التجميع مع عمليات التضخيم...")
    cmd = [
        "g++", "-O3", "-march=native",
        "-mavx2", "-mfma",
        "-fopenmp",
        "-std=c++17",
        "-shared", "-fPIC",
        "-o", str(lib_path),
        str(core_src),
        str(amp_src),
        "-lm",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"فشل التجميع:\n{result.stderr}")
    print(f"[MatCalc Amplify] ✅ {lib_path}")
    return lib_path


class MatCalcAmplify(MatCalc):
    """
    امتداد لـ MatCalc يضيف عمليات تضخيم الحساب.
    كل عملية مشروحة بدقة في matcalc_amplify.cpp
    """

    def __init__(self, lib_dir: str = None):
        # نجمّع مع الـ amplify
        if lib_dir is None:
            lib_dir = Path(__file__).parent
        else:
            lib_dir = Path(lib_dir)

        _compile_amplify(lib_dir)
        super().__init__(str(lib_dir))
        self._setup_amplify_signatures()

        v = self._lib.matcalc_amplify_version
        v.restype = ctypes.c_char_p
        print(f"[MatCalc Amplify] {v().decode()}")

    def _setup_amplify_signatures(self):
        f32p = ctypes.POINTER(ctypes.c_float)
        lib  = self._lib

        lib.matcalc_rotate90.restype  = None
        lib.matcalc_rotate90.argtypes = [f32p, f32p, ctypes.c_int, ctypes.c_int]

        lib.matcalc_attention_gate_mix.restype  = None
        lib.matcalc_attention_gate_mix.argtypes = [
            f32p, f32p, f32p, f32p, ctypes.c_int, ctypes.c_int
        ]

        lib.matcalc_recurrent_input.restype  = None
        lib.matcalc_recurrent_input.argtypes = [
            f32p, f32p, f32p, ctypes.c_float, ctypes.c_int
        ]

        lib.matcalc_thinking_blend.restype  = None
        lib.matcalc_thinking_blend.argtypes = [
            f32p, f32p, f32p, ctypes.c_float, ctypes.c_float, ctypes.c_int
        ]

        lib.matcalc_layer_blend.restype  = None
        lib.matcalc_layer_blend.argtypes = [
            f32p, f32p, f32p, f32p, ctypes.c_int
        ]

        lib.matcalc_cross_layer_residual.restype  = None
        lib.matcalc_cross_layer_residual.argtypes = [
            f32p, f32p, f32p, ctypes.c_float, ctypes.c_int
        ]

        lib.matcalc_feature_shift.restype  = None
        lib.matcalc_feature_shift.argtypes = [
            f32p, f32p, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        lib.matcalc_hadamard_mix.restype  = None
        lib.matcalc_hadamard_mix.argtypes = [
            f32p, f32p, f32p, ctypes.c_float, ctypes.c_int
        ]

        lib.matcalc_elem_square.restype  = None
        lib.matcalc_elem_square.argtypes = [f32p, f32p, ctypes.c_int]

        lib.matcalc_add_second_order.restype  = None
        lib.matcalc_add_second_order.argtypes = [
            f32p, f32p, f32p, ctypes.c_float, ctypes.c_int
        ]

    # ─── التقنية 1: Rotated Multi-Pass Attention ─────────────

    def rotate90(self, x: torch.Tensor) -> torch.Tensor:
        """يدوّر المتجهات 90° في الـ feature space"""
        x   = self._ensure_cpu_f32(x)
        out = torch.empty_like(x)
        rows, dim = x.shape[0], x.shape[-1]
        x_2d  = x.reshape(rows, dim).contiguous()
        out_2d = torch.empty_like(x_2d)
        self._lib.matcalc_rotate90(
            self._ptr(x_2d), self._ptr(out_2d),
            rows, dim
        )
        return out_2d.reshape(x.shape)

    def attention_gate_mix(
        self,
        A1: torch.Tensor,       # نتيجة الـ attention الأولى
        A2: torch.Tensor,       # نتيجة الـ attention الثانية (rotated)
        gate_w: torch.Tensor,   # أوزان الـ gate (dim,)
    ) -> torch.Tensor:
        """يمزج نتيجتي الـ Attention بـ learnable gate"""
        A1 = self._ensure_cpu_f32(A1)
        A2 = self._ensure_cpu_f32(A2)
        gate_w = self._ensure_cpu_f32(gate_w)
        rows, dim = A1.shape[0], A1.shape[-1]
        A1_2d = A1.reshape(rows, dim).contiguous()
        A2_2d = A2.reshape(rows, dim).contiguous()
        out_2d = torch.empty_like(A1_2d)
        self._lib.matcalc_attention_gate_mix(
            self._ptr(A1_2d), self._ptr(A2_2d),
            self._ptr(gate_w), self._ptr(out_2d),
            rows, dim
        )
        return out_2d.reshape(A1.shape)

    # ─── التقنية 2: Deep Thinking Recurrence ─────────────────

    def recurrent_input(
        self,
        h1: torch.Tensor,    # output المرور الأول
        x0: torch.Tensor,    # input الأصلي
        beta: float = 0.3    # قوة التفكير الثاني
    ) -> torch.Tensor:
        """ينشئ input للمرور الثاني — يحمل "ما تعلّمه" المرور الأول"""
        h1  = self._ensure_cpu_f32(h1)
        x0  = self._ensure_cpu_f32(x0)
        out = torch.empty_like(h1)
        self._lib.matcalc_recurrent_input(
            self._ptr(h1), self._ptr(x0), self._ptr(out),
            ctypes.c_float(beta), h1.numel()
        )
        return out

    def thinking_blend(
        self,
        h1: torch.Tensor,        # التفكير الأول
        h2: torch.Tensor,        # التفكير الثاني
        alpha: float = 0.5,      # وزن التوازن
        clip_val: float = 10.0   # حد الـ clipping
    ) -> torch.Tensor:
        """يمزج نتيجتي التفكير مع clipping"""
        h1  = self._ensure_cpu_f32(h1)
        h2  = self._ensure_cpu_f32(h2)
        out = torch.empty_like(h1)
        self._lib.matcalc_thinking_blend(
            self._ptr(h1), self._ptr(h2), self._ptr(out),
            ctypes.c_float(alpha), ctypes.c_float(clip_val),
            h1.numel()
        )
        return out

    # ─── التقنية 3: Cross-Layer Attention ────────────────────

    def layer_blend(
        self,
        h0: torch.Tensor,               # الطبقة الحالية
        h1: torch.Tensor = None,        # طبقة -2
        h2: torch.Tensor = None,        # طبقة -4
    ) -> torch.Tensor:
        """يدمج معلومات من 3 طبقات بأوزان تتناقص"""
        h0 = self._ensure_cpu_f32(h0)
        out = torch.empty_like(h0)

        h0f = h0.contiguous()
        h1f = self._ensure_cpu_f32(h1).contiguous() if h1 is not None else h0f
        h2f = self._ensure_cpu_f32(h2).contiguous() if h2 is not None else h0f

        # نمرر NULL لو مش موجود
        h1_ptr = self._ptr(h1f) if h1 is not None else ctypes.POINTER(ctypes.c_float)()
        h2_ptr = self._ptr(h2f) if h2 is not None else ctypes.POINTER(ctypes.c_float)()

        self._lib.matcalc_layer_blend(
            self._ptr(h0f), h1_ptr, h2_ptr, self._ptr(out),
            h0.numel()
        )
        return out

    def cross_layer_residual(
        self,
        x: torch.Tensor,
        cross_info: torch.Tensor,
        gamma: float = 0.1
    ) -> torch.Tensor:
        """يضيف معلومات الطبقات السابقة بـ gamma صغير"""
        x          = self._ensure_cpu_f32(x)
        cross_info = self._ensure_cpu_f32(cross_info)
        out        = torch.empty_like(x)
        self._lib.matcalc_cross_layer_residual(
            self._ptr(x), self._ptr(cross_info), self._ptr(out),
            ctypes.c_float(gamma), x.numel()
        )
        return out

    # ─── التقنية 4: Hadamard Feature Mixing ──────────────────

    def feature_shift(
        self,
        x: torch.Tensor,
        shift: int = None    # None = dim/4
    ) -> torch.Tensor:
        """يزيح الـ features دورياً لإنشاء cross-feature interactions"""
        x = self._ensure_cpu_f32(x)
        shape = x.shape
        rows  = x.numel() // shape[-1]
        dim   = shape[-1]
        if shift is None:
            shift = dim // 4
        x_2d  = x.reshape(rows, dim).contiguous()
        out_2d = torch.empty_like(x_2d)
        self._lib.matcalc_feature_shift(
            self._ptr(x_2d), self._ptr(out_2d),
            rows, dim, shift
        )
        return out_2d.reshape(shape)

    def hadamard_mix(
        self,
        x: torch.Tensor,
        scale: float = 0.1
    ) -> torch.Tensor:
        """يضيف Hadamard interactions — تفاعلات غير خطية مجاناً"""
        x         = self._ensure_cpu_f32(x)
        x_shifted = self.feature_shift(x)
        out       = torch.empty_like(x)
        self._lib.matcalc_hadamard_mix(
            self._ptr(x.contiguous()),
            self._ptr(x_shifted.contiguous()),
            self._ptr(out),
            ctypes.c_float(scale),
            x.numel()
        )
        return out

    # ─── التقنية 5: Second-Order Attention ───────────────────

    def elem_square(self, x: torch.Tensor) -> torch.Tensor:
        """x² — لحساب second-order features"""
        x   = self._ensure_cpu_f32(x)
        out = torch.empty_like(x)
        self._lib.matcalc_elem_square(self._ptr(x), self._ptr(out), x.numel())
        return out

    def amplified_attention_scores(
        self,
        Q: torch.Tensor,     # (batch*heads, seq_q, head_dim)
        K: torch.Tensor,     # (batch*heads, head_dim, seq_k) — transposed
        lambda_: float = 0.1
    ) -> torch.Tensor:
        """
        Attention scores من الدرجة الأولى والثانية:
        scores = Q@K + lambda * Q²@K²
        """
        # First order
        scores1 = self.batched_gemm(Q, K)  # (batch*heads, seq_q, seq_k)

        # Second order
        Q2 = self.elem_square(Q)
        K2 = self.elem_square(K)
        scores2 = self.batched_gemm(Q2, K2)

        # دمج
        s1 = scores1.contiguous()
        s2 = scores2.contiguous()
        out = torch.empty_like(s1)
        self._lib.matcalc_add_second_order(
            self._ptr(s1), self._ptr(s2), self._ptr(out),
            ctypes.c_float(lambda_), s1.numel()
        )
        return out

    # ═══════════════════════════════════════════════════════════
    #  Amplified Attention الكاملة — كل التقنيات معاً
    # ═══════════════════════════════════════════════════════════

    def amplified_attention(
        self,
        Q: torch.Tensor,             # (batch, heads, seq, head_dim)
        K: torch.Tensor,
        V: torch.Tensor,
        gate_w: torch.Tensor,        # (head_dim,)
        causal: bool = True,
        lambda_: float = 0.1,        # second-order weight
        hadamard_scale: float = 0.05 # hadamard mix weight
    ) -> torch.Tensor:
        """
        ══════════════════════════════════════════════════════
        Amplified Attention = 3 تقنيات في Attention واحدة
        ══════════════════════════════════════════════════════

        التقنية 1: Second-Order Scores
            scores = Q@K + λ*(Q²@K²)   ← يلتقط علاقات أعمق

        التقنية 2: Rotated Second Pass
            Q2, K2 = rotate90(Q, K)
            A2 = standard_attention(Q2, K2, V)
            out = gate*A1 + (1-gate)*A2  ← زاويتان للفهم

        التقنية 3: Hadamard Mix
            out = out + 0.05 * hadamard(out)  ← تفاعلات مجانية
        """
        Q = self._ensure_cpu_f32(Q)
        K = self._ensure_cpu_f32(K)
        V = self._ensure_cpu_f32(V)

        batch, heads, seq_q, head_dim = Q.shape
        _, _,  seq_k, _              = K.shape
        scale = 1.0 / (head_dim ** 0.5)

        BH = batch * heads
        Q_ = Q.reshape(BH, seq_q, head_dim)
        K_ = K.reshape(BH, seq_k, head_dim)
        V_ = V.reshape(BH, seq_k, head_dim)

        K_T = K_.transpose(1, 2).contiguous()

        # ── Causal Mask ─────────────────────────────────────
        mask = None
        if causal:
            mask = self.causal_mask(seq_q)
            mask = mask.unsqueeze(0).expand(BH, -1, -1).contiguous()

        # ── Pass 1: Second-Order Amplified Attention ─────────
        scores1 = self.amplified_attention_scores(Q_, K_T, lambda_)

        scores1_2d = scores1.reshape(BH * seq_q, seq_k)
        mask_2d    = mask.reshape(BH * seq_q, seq_k) if mask is not None else None
        attn1      = self.softmax(scores1_2d, scale=scale, mask=mask_2d)
        attn1      = attn1.reshape(BH, seq_q, seq_k)
        A1 = self.batched_gemm(attn1, V_)  # (BH, seq_q, head_dim)

        # ── Pass 2: Rotated Attention ─────────────────────────
        Q_rot = self.rotate90(Q_.reshape(BH * seq_q, head_dim)).reshape(BH, seq_q, head_dim)
        K_rot = self.rotate90(K_.reshape(BH * seq_k, head_dim)).reshape(BH, seq_k, head_dim)
        K_rot_T = K_rot.transpose(1, 2).contiguous()

        scores2 = self.batched_gemm(Q_rot, K_rot_T)
        scores2_2d = scores2.reshape(BH * seq_q, seq_k)
        attn2   = self.softmax(scores2_2d, scale=scale, mask=mask_2d)
        attn2   = attn2.reshape(BH, seq_q, seq_k)
        A2 = self.batched_gemm(attn2, V_)  # (BH, seq_q, head_dim)

        # ── Gate Mix ──────────────────────────────────────────
        gate_w_cpu = self._ensure_cpu_f32(gate_w)
        A1_2d = A1.reshape(BH * seq_q, head_dim).contiguous()
        A2_2d = A2.reshape(BH * seq_q, head_dim).contiguous()
        mixed = self.attention_gate_mix(A1_2d, A2_2d, gate_w_cpu)
        mixed = mixed.reshape(BH, seq_q, head_dim)

        # ── Hadamard Mix ──────────────────────────────────────
        mixed_flat = mixed.reshape(-1, head_dim)
        mixed_flat = self.hadamard_mix(mixed_flat, scale=hadamard_scale)
        mixed = mixed_flat.reshape(BH, seq_q, head_dim)

        return mixed.reshape(batch, heads, seq_q, head_dim)
