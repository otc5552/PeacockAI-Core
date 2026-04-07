"""
transformer_140b.py — AGI Transformer 140B Effective
══════════════════════════════════════════════════════
نفس أوزان الـ 70B لكن بـ 5 تقنيات تضخيم تجعله يفكر
بعمق ضعف الشبكة الأصلية.

الفرق عن transformer_matcalc.py:
  ❌ قبل: كل طبقة تمر مرة واحدة، attention بسيط
  ✅ بعد:
    • كل طبقة تفكر مرتين (Deep Thinking Recurrence)
    • الـ attention يرى زاويتين (Rotated Multi-Pass)
    • علاقات من الدرجة الثانية في الـ scores
    • الطبقات تتشارك المعلومات (Cross-Layer Cache)
    • تفاعلات Hadamard مجانية بين الميزات

المعاملات الفعلية المحسوبة:
  70B أوزان × 2 مرور = 140B حساب فعلي لكل token
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from collections import deque

from matcalc_amplify_bridge import MatCalcAmplify

# ─── singleton ────────────────────────────────────────────────
_MCA: Optional[MatCalcAmplify] = None

def get_mca() -> MatCalcAmplify:
    global _MCA
    if _MCA is None:
        import os
        _MCA = MatCalcAmplify(os.path.dirname(os.path.abspath(__file__)))
    return _MCA


# ═══════════════════════════════════════════════════════════════
#  RMSNorm (نفس الأصلي)
# ═══════════════════════════════════════════════════════════════

class RMSNorm140(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mc    = get_mca()
        shape = x.shape
        x_2d  = x.reshape(-1, shape[-1]).cpu().float().contiguous()
        w     = self.scale.detach().cpu().float().contiguous()
        out   = mc.rmsnorm(x_2d, w, self.eps)
        return out.reshape(shape).to(x.device)


# ═══════════════════════════════════════════════════════════════
#  LinearMC (نفس الأصلي)
# ═══════════════════════════════════════════════════════════════

class LinearMC(nn.Module):
    def __init__(self, in_f: int, out_f: int, bias: bool = False):
        super().__init__()
        self.in_f  = in_f
        self.out_f = out_f
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias_p = nn.Parameter(torch.zeros(out_f)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mc     = get_mca()
        device = x.device
        shape  = x.shape
        x_2d   = x.reshape(-1, self.in_f).cpu().float().contiguous()
        W      = self.weight.detach().cpu().float().contiguous()
        bias   = self.bias_p.detach().cpu().float().contiguous() if self.bias_p else None
        out_2d = mc.linear(x_2d, W, bias)
        return out_2d.reshape(*shape[:-1], self.out_f).to(device)


# ═══════════════════════════════════════════════════════════════
#  Amplified Attention — يضم التقنيات 1+2+4
# ═══════════════════════════════════════════════════════════════

class AmplifiedAttention(nn.Module):
    """
    Multi-Head Attention مضخَّم بـ 3 تقنيات:
      1. Second-Order Scores: scores += λ*(Q²@K²)
      2. Rotated Second Pass: يحسب Attention من زاوية ثانية
      4. Hadamard Mix: تفاعلات بين الميزات بعد الـ Attention
    """

    def __init__(
        self,
        embedding_dim:  int,
        num_heads:      int,
        dropout:        float = 0.0,
        use_rope:       bool  = True,
        lambda_:        float = 0.1,    # وزن second-order
        hadamard_scale: float = 0.05,   # قوة الـ Hadamard
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.num_heads      = num_heads
        self.head_dim       = embedding_dim // num_heads
        self.use_rope       = use_rope
        self.lambda_        = lambda_
        self.hadamard_scale = hadamard_scale

        self.q_proj = LinearMC(embedding_dim, embedding_dim)
        self.k_proj = LinearMC(embedding_dim, embedding_dim)
        self.v_proj = LinearMC(embedding_dim, embedding_dim)
        self.o_proj = LinearMC(embedding_dim, embedding_dim)

        # Gate vector للمزج بين الـ pass الأول والثاني
        self.gate_w = nn.Parameter(
            torch.randn(self.head_dim) * 0.01
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mca = get_mca()
        batch, seq, dim = x.shape

        Q = self.q_proj(x).reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        K = self.k_proj(x).reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        V = self.v_proj(x).reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        # RoPE
        if self.use_rope:
            BH = batch * self.num_heads
            Q_ = Q.permute(0, 2, 1, 3).reshape(batch * seq, self.num_heads, self.head_dim)
            K_ = K.permute(0, 2, 1, 3).reshape(batch * seq, self.num_heads, self.head_dim)
            Q_r = torch.zeros_like(Q_)
            K_r = torch.zeros_like(K_)
            for b in range(batch):
                q_b = Q_[b*seq:(b+1)*seq].cpu().float().contiguous()
                k_b = K_[b*seq:(b+1)*seq].cpu().float().contiguous()
                Q_r[b*seq:(b+1)*seq] = mca.rope(q_b)
                K_r[b*seq:(b+1)*seq] = mca.rope(k_b)
            Q = Q_r.reshape(batch, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
            K = K_r.reshape(batch, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        # Amplified Attention (التقنيات 1+2+4)
        gate_w = self.gate_w.detach().cpu().float().contiguous()
        out = mca.amplified_attention(
            Q, K, V,
            gate_w      = gate_w,
            causal      = True,
            lambda_     = self.lambda_,
            hadamard_scale = self.hadamard_scale,
        )

        out = out.transpose(1, 2).reshape(batch, seq, dim)
        return self.o_proj(out)


# ═══════════════════════════════════════════════════════════════
#  FFN with Hadamard (التقنية 4 في FFN أيضاً)
# ═══════════════════════════════════════════════════════════════

class AmplifiedFFN(nn.Module):
    """
    FFN مضخَّم بـ Hadamard Mix بعد الـ activation
    يضيف تفاعلات غير خطية بين الميزات بدون أوزان جديدة
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, hadamard_scale: float = 0.05):
        super().__init__()
        self.gate_proj = LinearMC(embedding_dim, hidden_dim)
        self.up_proj   = LinearMC(embedding_dim, hidden_dim)
        self.down_proj = LinearMC(hidden_dim, embedding_dim)
        self.hadamard_scale = hadamard_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mca  = get_mca()
        gate = self.gate_proj(x)
        up   = self.up_proj(x)

        # SiLU(gate) * up
        gate_cpu  = gate.cpu().float().contiguous()
        gate_silu = mca.silu(gate_cpu).to(x.device)
        hidden = gate_silu * up

        # Hadamard Mix على الـ hidden states
        hidden_cpu    = hidden.cpu().float().contiguous()
        hidden_mixed  = mca.hadamard_mix(hidden_cpu, scale=self.hadamard_scale)
        hidden = hidden_mixed.to(x.device)

        return self.down_proj(hidden)


# ═══════════════════════════════════════════════════════════════
#  TransformerBlock140B — التقنيتان 2+3
# ═══════════════════════════════════════════════════════════════

class TransformerBlock140B(nn.Module):
    """
    ══════════════════════════════════════════════════════════════
    كتلة Transformer 140B Effective
    ══════════════════════════════════════════════════════════════

    التقنية 2 — Deep Thinking Recurrence:
      • المرور الأول:  h1 = Attention(x) + FFN(x)
      • تجهيز المدخل: x2 = h1 + β*(h1-x)  ← "ما تعلّمته"
      • المرور الثاني: h2 = Attention(x2) + FFN(x2)
      • الناتج:        out = 0.5*h1 + 0.5*h2

    التقنية 3 — Cross-Layer Memory:
      • تحفظ hidden states من الطبقات السابقة
      • تضيفها بـ gamma=0.1 للـ output الحالي
      • يخلي كل طبقة "تتذكر" ما فهمته الطبقات الأعمق
    """

    def __init__(
        self,
        embedding_dim:  int,
        num_heads:      int,
        num_kv_heads:   int,
        ffn_hidden:     int,
        dropout:        float = 0.0,
        use_rope:       bool  = True,
        context_length: int   = 8192,
        layer_idx:      int   = 0,
        # معاملات التضخيم
        thinking_beta:  float = 0.3,    # قوة المرور الثاني
        thinking_alpha: float = 0.5,    # توازن المرورين
        cross_gamma:    float = 0.1,    # قوة Cross-Layer
        lambda_:        float = 0.1,    # second-order attention
        hadamard_scale: float = 0.05,   # Hadamard FFN
        # تحكم في التقنيات
        use_deep_thinking:  bool = True,
        use_cross_layer:    bool = True,
        **kwargs  # لاستيعاب use_moe وغيرها بدون خطأ
    ):
        super().__init__()
        self.layer_idx      = layer_idx
        self.thinking_beta  = thinking_beta
        self.thinking_alpha = thinking_alpha
        self.cross_gamma    = cross_gamma
        self.use_deep_thinking = use_deep_thinking
        self.use_cross_layer   = use_cross_layer

        self.norm1 = RMSNorm140(embedding_dim)
        self.norm2 = RMSNorm140(embedding_dim)
        # نورم إضافية للمرور الثاني
        self.norm3 = RMSNorm140(embedding_dim)
        self.norm4 = RMSNorm140(embedding_dim)

        self.attention = AmplifiedAttention(
            embedding_dim  = embedding_dim,
            num_heads      = num_heads,
            dropout        = dropout,
            use_rope       = use_rope,
            lambda_        = lambda_,
            hadamard_scale = hadamard_scale,
        )

        self.ffn = AmplifiedFFN(
            embedding_dim  = embedding_dim,
            hidden_dim     = ffn_hidden,
            hadamard_scale = hadamard_scale,
        )

        self.residual_scale = (
            1.0 / (2.0 * layer_idx + 1) ** 0.5 if layer_idx > 0 else 1.0
        )

        # clip_val للطبقات العميقة
        self.clip_val = 10.0 / (1.0 + layer_idx * 0.1)

    def _single_pass(
        self,
        x: torch.Tensor,
        n1: nn.Module,
        n2: nn.Module,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """مرور واحد عبر Attention + FFN"""
        mca = get_mca()
        rs  = self.residual_scale

        # Attention
        residual = x
        attn_out = self.attention(n1(x), attention_mask)
        attn_cpu = attn_out.cpu().float().contiguous()
        res_cpu  = residual.cpu().float().contiguous()
        x = mca.scaled_add(res_cpu, attn_cpu, rs).to(x.device)

        # FFN
        residual = x
        ffn_out  = self.ffn(n2(x))
        ffn_cpu  = ffn_out.cpu().float().contiguous()
        res_cpu  = residual.cpu().float().contiguous()
        x = mca.scaled_add(res_cpu, ffn_cpu, rs).to(x.device)

        return x

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mca = get_mca()
        x0  = x  # نحفظ الـ input الأصلي

        # ── المرور الأول ───────────────────────────────────────
        h1 = self._single_pass(x, self.norm1, self.norm2, attention_mask)

        if self.use_deep_thinking:
            # ── التقنية 2: تجهيز مدخل المرور الثاني ───────────
            h1_cpu = h1.cpu().float().contiguous()
            x0_cpu = x0.cpu().float().contiguous()
            x2_cpu = mca.recurrent_input(h1_cpu, x0_cpu, beta=self.thinking_beta)
            x2 = x2_cpu.to(x.device)

            # ── المرور الثاني ──────────────────────────────────
            h2 = self._single_pass(x2, self.norm3, self.norm4, attention_mask)

            # ── مزج المرورين ───────────────────────────────────
            h1_cpu = h1.cpu().float().contiguous()
            h2_cpu = h2.cpu().float().contiguous()
            out = mca.thinking_blend(
                h1_cpu, h2_cpu,
                alpha    = self.thinking_alpha,
                clip_val = self.clip_val
            ).to(x.device)
        else:
            out = h1

        # ── التقنية 3: Cross-Layer Memory ──────────────────────
        if self.use_cross_layer and layer_cache and len(layer_cache) > 0:
            # آخر طبقتين محفوظتين
            prev1 = layer_cache[-1] if len(layer_cache) >= 1 else None
            prev2 = layer_cache[-2] if len(layer_cache) >= 2 else None

            out_cpu = out.cpu().float().contiguous()
            p1_cpu  = prev1.cpu().float().contiguous() if prev1 is not None else None
            p2_cpu  = prev2.cpu().float().contiguous() if prev2 is not None else None

            # دمج المعلومات من الطبقات السابقة
            blended = mca.layer_blend(
                out_cpu.reshape(-1),
                p1_cpu.reshape(-1) if p1_cpu is not None else None,
                p2_cpu.reshape(-1) if p2_cpu is not None else None,
            )

            # إضافة بـ gamma صغير
            out_flat = out_cpu.reshape(-1)
            cross_out = mca.cross_layer_residual(
                out_flat, blended, gamma=self.cross_gamma
            )
            out = cross_out.reshape(out.shape).to(x.device)

        return out, None  # aux_loss = None


# ═══════════════════════════════════════════════════════════════
#  AGITransformer140B — الشبكة الكاملة المضخَّمة
# ═══════════════════════════════════════════════════════════════

class AGITransformer140B(nn.Module):
    """
    ══════════════════════════════════════════════════════════════
    AGI Transformer 140B Effective
    ══════════════════════════════════════════════════════════════
    نفس أوزان الـ 70B الأصلي + 5 تقنيات تضخيم حساب

    التقنيات:
    ① Second-Order Attention   — يلتقط علاقات أعمق في الـ scores
    ② Rotated Multi-Pass       — كل head يرى المعلومات من زاويتين
    ③ Deep Thinking Recurrence — كل طبقة تفكر مرتين
    ④ Cross-Layer Memory       — الطبقات تتذكر ما فهمته الأعمق
    ⑤ Hadamard Feature Mixing  — تفاعلات مجانية بين الميزات
    ══════════════════════════════════════════════════════════════
    """

    def __init__(
        self,
        vocab_size:     int   = 128_000,
        context_length: int   = 8_192,
        embedding_dim:  int   = 8_192,
        num_layers:     int   = 80,
        num_heads:      int   = 64,
        num_kv_heads:   int   = 8,
        ffn_hidden:     int   = 28_672,
        dropout:        float = 0.0,
        use_rope:       bool  = True,
        # معاملات التضخيم
        thinking_beta:  float = 0.3,
        thinking_alpha: float = 0.5,
        cross_gamma:    float = 0.1,
        lambda_:        float = 0.1,
        hadamard_scale: float = 0.05,
        cross_layer_depth: int = 2,    # كام طبقة سابقة نحفظها
        # تحكم
        use_deep_thinking: bool = True,
        use_cross_layer:   bool = True,
        **kwargs
    ):
        super().__init__()

        self.config = dict(
            vocab_size=vocab_size, context_length=context_length,
            embedding_dim=embedding_dim, num_layers=num_layers,
            num_heads=num_heads, ffn_hidden=ffn_hidden,
        )
        self.cross_layer_depth = cross_layer_depth

        # Embeddings (صغيرة نسبياً، ممكن تبقى على GPU)
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding   = nn.Embedding(context_length, embedding_dim)

        # الطبقات المضخَّمة
        self.layers = nn.ModuleList([
            TransformerBlock140B(
                embedding_dim  = embedding_dim,
                num_heads      = num_heads,
                num_kv_heads   = num_kv_heads,
                ffn_hidden     = ffn_hidden,
                dropout        = dropout,
                use_rope       = use_rope,
                layer_idx      = i,
                thinking_beta  = thinking_beta,
                thinking_alpha = thinking_alpha,
                cross_gamma    = cross_gamma * (1.0 - i / (num_layers * 2)),  # يقل مع العمق
                lambda_        = lambda_,
                hadamard_scale = hadamard_scale,
                use_deep_thinking = use_deep_thinking,
                use_cross_layer   = use_cross_layer and i >= 2,  # من الطبقة 3+
            )
            for i in range(num_layers)
        ])

        self.final_norm = RMSNorm140(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocab_size, bias=False)

        self._init_weights()
        self._print_summary(num_layers, embedding_dim, num_heads, ffn_hidden, vocab_size, context_length)

    def _init_weights(self):
        n = len(self.layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                std = 0.02 / (n ** 0.5) if n > 24 else 0.02
                nn.init.normal_(module.weight, 0.0, std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0.0, 0.02)

    def _print_summary(self, L, D, H, F, V, C):
        total = sum(p.numel() for p in self.parameters())
        # كل token يمر مرتين عبر الأوزان
        effective = total * 2
        print(f"\n{'═'*62}")
        print(f"  AGI Transformer 140B Effective")
        print(f"{'─'*62}")
        print(f"  الأوزان الفعلية    : {total:>15,}  (~{total/1e9:.1f}B)")
        print(f"  الحساب الفعلي/token: {effective:>15,}  (~{effective/1e9:.1f}B) ✨")
        print(f"{'─'*62}")
        print(f"  ① Second-Order Attention    (λ=0.10)")
        print(f"  ② Rotated Multi-Pass        (2× attention)")
        print(f"  ③ Deep Thinking Recurrence  (β=0.30, α=0.50)")
        print(f"  ④ Cross-Layer Memory        (γ=0.10)")
        print(f"  ⑤ Hadamard Feature Mixing   (scale=0.05)")
        print(f"{'─'*62}")
        print(f"  FP32 RAM needed    : {total*4/1e9:>14.1f} GB")
        print(f"{'═'*62}\n")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch, seq = input_ids.shape
        device     = input_ids.device

        # Embeddings
        pos_ids = torch.arange(seq, device=device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(pos_ids)

        # Cross-layer cache — نحفظ آخر N طبقات
        layer_cache: List[torch.Tensor] = []
        total_aux_loss = torch.tensor(0.0, device=device)
        hidden_states  = []

        for i, layer in enumerate(self.layers):
            x, aux_loss = layer(x, attention_mask, layer_cache=list(layer_cache))

            # نحدّث الـ cache
            layer_cache.append(x.detach())
            if len(layer_cache) > self.cross_layer_depth:
                layer_cache.pop(0)

            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss

            if return_hidden_states:
                hidden_states.append(x.detach())

        x      = self.final_norm(x)
        logits = self.output_projection(x)

        result = {'logits': logits, 'aux_loss': total_aux_loss}
        if return_hidden_states:
            result['hidden_states'] = hidden_states
        return result

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int   = 100,
        temperature:    float = 1.0,
        top_k:          int   = 50,
        top_p:          float = 0.9,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                ctx    = input_ids[:, -self.config['context_length']:]
                output = self.forward(ctx)
                logits = output['logits'][:, -1, :] / temperature

                if top_k > 0:
                    vals, _ = torch.topk(logits, top_k)
                    logits  = logits.masked_fill(logits < vals[:, -1:], float('-inf'))

                if top_p < 1.0:
                    sorted_l, sorted_i = torch.sort(logits, descending=True)
                    cum_p = torch.cumsum(torch.softmax(sorted_l, dim=-1), dim=-1)
                    sorted_l[cum_p > top_p] = float('-inf')
                    logits = logits.scatter(1, sorted_i, sorted_l)

                next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                input_ids  = torch.cat([input_ids, next_token], dim=1)
        return input_ids
