"""
transformer_matcalc.py — تكامل الآلة الحاسبة مع AGI Transformer
═══════════════════════════════════════════════════════════════════
نسخ معدّلة من TransformerBlock و AGITransformer
تستبدل العمليات الثقيلة بـ MatCalc بدلاً من PyTorch/CUDA

الفكرة:
  ❌ قبل: كل عملية matmul تحجز VRAM وتحتاج GPU
  ✅ بعد: العمليات الثقيلة تروح لـ MatCalc على CPU
         الـ GPU (لو موجود) يُحجَز لعمليات صغيرة فقط

الاستهلاك المتوقع على جهازك (16GB RAM, 4GB VRAM):
  الـ Embeddings والـ Logits  →  GPU (صغيرة نسبياً)
  Attention Q@K, Attn@V      →  CPU via MatCalc
  FFN / MoE Linear layers    →  CPU via MatCalc
  RMSNorm, RoPE, SiLU        →  CPU via MatCalc
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from matcalc_bridge import MatCalc

# ─── singleton — نحمّل الـ library مرة واحدة ─────────────────
_MC: Optional[MatCalc] = None

def get_matcalc() -> MatCalc:
    global _MC
    if _MC is None:
        import os
        lib_dir = os.path.dirname(os.path.abspath(__file__))
        _MC = MatCalc(lib_dir)
    return _MC


# ═══════════════════════════════════════════════════════════════
#  RMSNorm — مُحسَّنة بـ MatCalc
# ═══════════════════════════════════════════════════════════════

class RMSNormMC(nn.Module):
    """RMSNorm تستخدم MatCalc بدلاً من PyTorch"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mc = get_matcalc()
        shape   = x.shape
        x_2d    = x.reshape(-1, shape[-1]).cpu().float().contiguous()
        weight  = self.scale.detach().cpu().float().contiguous()
        out_2d  = mc.rmsnorm(x_2d, weight, self.eps)
        return out_2d.reshape(shape).to(x.device)


# ═══════════════════════════════════════════════════════════════
#  LinearMC — nn.Linear يستخدم MatCalc
# ═══════════════════════════════════════════════════════════════

class LinearMC(nn.Module):
    """
    بديل لـ nn.Linear — الحساب يتم على CPU عبر MatCalc
    يوفر VRAM بدون أي تغيير في الـ interface
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        # nn.Linear يخزن W بشكل (out, in) وهو نفس اللي يريده matcalc_linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_p = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mc      = get_matcalc()
        device  = x.device
        shape   = x.shape
        x_2d    = x.reshape(-1, self.in_features).cpu().float().contiguous()
        W       = self.weight.detach().cpu().float().contiguous()
        bias    = self.bias_p.detach().cpu().float().contiguous() if self.bias_p is not None else None
        out_2d  = mc.linear(x_2d, W, bias)
        return out_2d.reshape(*shape[:-1], self.out_features).to(device)


# ═══════════════════════════════════════════════════════════════
#  AttentionMC — Multi-Head Attention بـ MatCalc
# ═══════════════════════════════════════════════════════════════

class AttentionMC(nn.Module):
    """
    Multi-Head Attention مع:
    - Q, K, V projections عبر LinearMC (CPU)
    - Scaled Dot-Product Attention عبر MatCalc (CPU)
    - RoPE عبر MatCalc (CPU)
    - Output projection عبر LinearMC (CPU)
    """
    def __init__(
        self,
        embedding_dim:  int,
        num_heads:      int,
        dropout:        float = 0.0,
        use_rope:       bool  = True,
        context_length: int   = 8192,
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.num_heads  = num_heads
        self.head_dim   = embedding_dim // num_heads
        self.use_rope   = use_rope

        # Projections — كلهم على CPU عبر MatCalc
        self.q_proj = LinearMC(embedding_dim, embedding_dim)
        self.k_proj = LinearMC(embedding_dim, embedding_dim)
        self.v_proj = LinearMC(embedding_dim, embedding_dim)
        self.o_proj = LinearMC(embedding_dim, embedding_dim)

    def forward(
        self,
        x: torch.Tensor,                         # (batch, seq, dim)
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mc = get_matcalc()
        batch, seq, dim = x.shape

        # Q, K, V projections
        Q = self.q_proj(x)  # (batch, seq, dim)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # إعادة التشكيل لـ (batch, heads, seq, head_dim)
        Q = Q.reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        K = K.reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        V = V.reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        # RoPE على CPU
        if self.use_rope:
            # نحوّل لـ (seq, heads, head_dim) للـ MatCalc ثم نرجع
            Q_ = Q.permute(0, 2, 1, 3).reshape(batch * seq, self.num_heads, self.head_dim)
            K_ = K.permute(0, 2, 1, 3).reshape(batch * seq, self.num_heads, self.head_dim)
            # نطبق RoPE على كل batch منفصلاً
            Q_roped = torch.zeros_like(Q_)
            K_roped = torch.zeros_like(K_)
            for b in range(batch):
                q_b = Q_[b*seq:(b+1)*seq].cpu().float().contiguous()
                k_b = K_[b*seq:(b+1)*seq].cpu().float().contiguous()
                Q_roped[b*seq:(b+1)*seq] = mc.rope(q_b)
                K_roped[b*seq:(b+1)*seq] = mc.rope(k_b)
            Q = Q_roped.reshape(batch, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
            K = K_roped.reshape(batch, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        # Attention على CPU عبر MatCalc
        out = mc.scaled_dot_product_attention(Q, K, V, causal=True)

        # إعادة التجميع
        out = out.transpose(1, 2).reshape(batch, seq, dim)

        # Output projection
        return self.o_proj(out)


# ═══════════════════════════════════════════════════════════════
#  FeedForwardMC — FFN بـ MatCalc
# ═══════════════════════════════════════════════════════════════

class FeedForwardMC(nn.Module):
    """
    FFN بـ SiLU activation على MatCalc
    gate_proj و up_proj → SiLU → down_proj
    (نفس بنية LLaMA / Mistral)
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = LinearMC(embedding_dim, hidden_dim)
        self.up_proj   = LinearMC(embedding_dim, hidden_dim)
        self.down_proj = LinearMC(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mc   = get_matcalc()
        gate = self.gate_proj(x)
        up   = self.up_proj(x)
        # SiLU(gate) * up  — Gated FFN
        gate_cpu = gate.cpu().float().contiguous()
        gate_silu = mc.silu(gate_cpu).to(x.device)
        hidden = gate_silu * up
        return self.down_proj(hidden)


# ═══════════════════════════════════════════════════════════════
#  TransformerBlockMC — الكتلة الكاملة بـ MatCalc
# ═══════════════════════════════════════════════════════════════

class TransformerBlockMC(nn.Module):
    """
    كتلة Transformer كاملة — كل العمليات الثقيلة على CPU
    نفس الـ interface بتاع TransformerBlock الأصلي
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
        # MoE مؤقتاً بـ standard FFN (MoE يحتاج تطوير إضافي)
        use_moe:        bool  = False,
        num_experts:    int   = 32,
        top_k:          int   = 4,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # RMSNorm على MatCalc
        self.norm1 = RMSNormMC(embedding_dim)
        self.norm2 = RMSNormMC(embedding_dim)

        # Attention على MatCalc
        self.attention = AttentionMC(
            embedding_dim  = embedding_dim,
            num_heads      = num_heads,
            dropout        = dropout,
            use_rope       = use_rope,
            context_length = context_length,
        )

        # FFN على MatCalc
        self.ffn = FeedForwardMC(embedding_dim, ffn_hidden, dropout)

        # Residual scale للطبقات العميقة
        self.residual_scale = (
            1.0 / (2.0 * layer_idx + 1) ** 0.5 if layer_idx > 0 else 1.0
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mc = get_matcalc()

        # Attention block
        residual  = x
        attn_out  = self.attention(self.norm1(x), attention_mask)
        if self.residual_scale == 1.0:
            x = mc.add(
                residual.cpu().float().contiguous(),
                attn_out.cpu().float().contiguous()
            ).to(x.device)
        else:
            x = mc.scaled_add(
                residual.cpu().float().contiguous(),
                attn_out.cpu().float().contiguous(),
                self.residual_scale
            ).to(x.device)

        # FFN block
        residual = x
        ffn_out  = self.ffn(self.norm2(x))
        if self.residual_scale == 1.0:
            x = mc.add(
                residual.cpu().float().contiguous(),
                ffn_out.cpu().float().contiguous()
            ).to(x.device)
        else:
            x = mc.scaled_add(
                residual.cpu().float().contiguous(),
                ffn_out.cpu().float().contiguous(),
                self.residual_scale
            ).to(x.device)

        return x, None  # aux_loss = None (بدون MoE حالياً)


# ═══════════════════════════════════════════════════════════════
#  دليل الاستخدام
# ═══════════════════════════════════════════════════════════════

"""
كيفية استخدام TransformerBlockMC في transformer.py الأصلي:

# في AGITransformer.__init__، بدل:
    self.layers.append(TransformerBlock(...))

# استخدم:
    from transformer_matcalc import TransformerBlockMC
    self.layers.append(TransformerBlockMC(
        embedding_dim  = embedding_dim,
        num_heads      = num_heads,
        num_kv_heads   = num_kv_heads,
        ffn_hidden     = ffn_hidden,
        dropout        = dropout,
        use_rope       = use_rope,
        context_length = context_length,
        layer_idx      = i,
    ))

هكذا كل الـ 80 طبقة تشتغل على CPU بدون VRAM 🎉
"""
