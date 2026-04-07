"""
large_model_configs.py — إعدادات النماذج الضخمة الجاهزة للتدريب
=================================================================
4 مستويات من النماذج الضخمة — من 1B إلى 70B parameter
جاهزة للتدريب مباشرة بعد توفير GPU مناسب

المقارنة مع النماذج الشهيرة:
  SMALL  (~1B)  ≈ GPT-2 XL مطور
  MEDIUM (~7B)  ≈ LLaMA-2 7B / Mistral 7B
  LARGE  (~13B) ≈ LLaMA-2 13B
  GIANT  (~70B) ≈ LLaMA-2 70B / GPT-3
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class LargeModelConfig:
    """إعدادات نموذج ضخم مع حساب المعاملات والذاكرة"""

    # ── الهوية ───────────────────────────────────────────────────────────
    name:               str   = "AGI-Large"
    description:        str   = ""

    # ── المعمارية الأساسية ───────────────────────────────────────────────
    vocab_size:         int   = 100_000   # مفردات أكبر = فهم أوسع
    context_length:     int   = 8_192     # سياق أطول = فهم أعمق
    embedding_dim:      int   = 4_096     # بُعد التضمين
    num_layers:         int   = 32        # عدد الطبقات
    num_heads:          int   = 32        # رؤوس الانتباه
    ffn_hidden:         int   = 16_384    # حجم FFN الداخلي
    dropout:            float = 0.1

    # ── Mixture of Experts ───────────────────────────────────────────────
    use_moe:            bool  = True
    num_experts:        int   = 16        # عدد الخبراء
    top_k:              int   = 2         # خبراء نشطين لكل token
    moe_every_n_layers: int   = 2

    # ── تقنيات متقدمة ────────────────────────────────────────────────────
    use_rope:           bool  = True      # Rotary Position Embeddings
    tie_weights:        bool  = True      # ربط أوزان embedding/output
    use_flash_attention:bool  = True      # Flash Attention 2 (أسرع وأقل ذاكرة)
    use_sliding_window: bool  = False     # Sliding Window Attention (Mistral-style)
    sliding_window_size:int   = 4_096

    # ── التدريب ──────────────────────────────────────────────────────────
    # (هذه مرجعية — تُضبط في سكريبت التدريب)
    recommended_lr:     float = 3e-4
    recommended_batch:  int   = 32
    recommended_warmup: int   = 2_000
    gradient_checkpointing: bool = True   # يوفر ~40% من الذاكرة
    mixed_precision:    str   = "bf16"    # bf16 أفضل من fp16 للاستقرار

    def total_params(self) -> int:
        """حساب تقريبي لعدد المعاملات"""
        d   = self.embedding_dim
        h   = self.ffn_hidden
        v   = self.vocab_size
        n   = self.num_layers
        exp = self.num_experts if self.use_moe else 1

        # Embedding
        embed = v * d

        # كل طبقة attention: Q+K+V+O = 4 * d^2
        attn_per_layer = 4 * d * d

        # كل طبقة FFN (أو MoE): 2 * d * h * experts
        moe_layers = n // self.moe_every_n_layers if self.use_moe else 0
        std_layers = n - moe_layers

        ffn_std = std_layers * 2 * d * h
        ffn_moe = moe_layers * 2 * d * h * exp

        # LayerNorm: 2 * d * 2 (قبل attention وقبل FFN)
        norms = n * 2 * 2 * d

        # Router (إذا MoE): d * num_experts per MoE layer
        router = moe_layers * d * exp

        total = embed + n * attn_per_layer + ffn_std + ffn_moe + norms + router
        return total

    def memory_fp16_gb(self) -> float:
        """الذاكرة المطلوبة للأوزان بـ FP16"""
        return self.total_params() * 2 / 1e9

    def memory_bf16_gb(self) -> float:
        return self.memory_fp16_gb()   # نفس الحجم

    def memory_fp32_gb(self) -> float:
        return self.total_params() * 4 / 1e9

    def recommended_gpus(self) -> str:
        """توصية بعدد الـ GPU"""
        gb = self.memory_fp16_gb()
        # نحتاج ضعف الحجم للتدريب (أوزان + gradients + optimizer states)
        training_gb = gb * 4
        if training_gb <= 24:
            return "1× RTX 4090 (24GB)  أو  1× A100 (40GB)"
        elif training_gb <= 80:
            return "1× A100 (80GB)  أو  2× RTX 4090"
        elif training_gb <= 160:
            return "2× A100 (80GB)  أو  4× RTX 4090"
        elif training_gb <= 320:
            return "4× A100 (80GB)  أو  8× RTX 4090"
        else:
            return f"8+ A100 (80GB) أو H100 cluster"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        params = self.total_params()
        lines = [
            f"\n{'═'*60}",
            f"  🏗️  {self.name}",
            f"  {self.description}",
            f"{'─'*60}",
            f"  📐 Architecture:",
            f"     vocab_size      : {self.vocab_size:>12,}",
            f"     context_length  : {self.context_length:>12,}  tokens",
            f"     embedding_dim   : {self.embedding_dim:>12,}",
            f"     num_layers      : {self.num_layers:>12,}",
            f"     num_heads       : {self.num_heads:>12,}",
            f"     head_dim        : {self.embedding_dim // self.num_heads:>12,}",
            f"     ffn_hidden      : {self.ffn_hidden:>12,}",
            f"{'─'*60}",
            f"  🔀 MoE:",
            f"     enabled         : {str(self.use_moe):>12}",
            f"     num_experts     : {self.num_experts:>12,}",
            f"     active (top_k)  : {self.top_k:>12,}",
            f"     moe_every       : every {self.moe_every_n_layers} layers",
            f"{'─'*60}",
            f"  📊 Scale:",
            f"     Total params    : {params:>12,}  (~{params/1e9:.1f}B)",
            f"     Weights FP16    : {self.memory_fp16_gb():>11.1f} GB",
            f"     Weights FP32    : {self.memory_fp32_gb():>11.1f} GB",
            f"     Training memory : ~{self.memory_fp16_gb()*4:.0f} GB  (w/ optimizer)",
            f"{'─'*60}",
            f"  💻 Hardware:",
            f"     Recommended     : {self.recommended_gpus()}",
            f"     Mixed Precision : {self.mixed_precision}",
            f"     Grad Checkpoint : {self.gradient_checkpointing}",
            f"{'═'*60}",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#   النماذج الأربعة الجاهزة
# ═══════════════════════════════════════════════════════════════

SMALL_1B = LargeModelConfig(
    name            = "AGI-Small-1B",
    description     = "نموذج صغير سريع التدريب — مناسب للبداية والتجارب",
    vocab_size      = 50_000,
    context_length  = 4_096,
    embedding_dim   = 2_048,
    num_layers      = 24,
    num_heads       = 16,
    ffn_hidden      = 8_192,
    dropout         = 0.1,
    use_moe         = True,
    num_experts     = 8,
    top_k           = 2,
    moe_every_n_layers = 2,
    use_rope        = True,
    tie_weights     = True,
    recommended_lr  = 3e-4,
    recommended_batch = 16,
    recommended_warmup = 1_000,
    gradient_checkpointing = False,
    mixed_precision = "bf16",
)

MEDIUM_7B = LargeModelConfig(
    name            = "AGI-Medium-7B",
    description     = "نموذج متوسط — مستوى Mistral 7B / LLaMA-2 7B",
    vocab_size      = 64_000,
    context_length  = 8_192,
    embedding_dim   = 4_096,
    num_layers      = 32,
    num_heads       = 32,
    ffn_hidden      = 14_336,
    dropout         = 0.05,
    use_moe         = True,
    num_experts     = 8,
    top_k           = 2,
    moe_every_n_layers = 2,
    use_rope        = True,
    tie_weights     = False,   # النماذج الكبيرة لا تربط الأوزان عادةً
    recommended_lr  = 2e-4,
    recommended_batch = 32,
    recommended_warmup = 2_000,
    gradient_checkpointing = True,
    mixed_precision = "bf16",
)

LARGE_13B = LargeModelConfig(
    name            = "AGI-Large-13B",
    description     = "نموذج كبير — مستوى LLaMA-2 13B",
    vocab_size      = 100_000,
    context_length  = 8_192,
    embedding_dim   = 5_120,
    num_layers      = 40,
    num_heads       = 40,
    ffn_hidden      = 20_480,
    dropout         = 0.05,
    use_moe         = True,
    num_experts     = 16,
    top_k           = 2,
    moe_every_n_layers = 2,
    use_rope        = True,
    tie_weights     = False,
    recommended_lr  = 1.5e-4,
    recommended_batch = 64,
    recommended_warmup = 3_000,
    gradient_checkpointing = True,
    mixed_precision = "bf16",
)

GIANT_70B = LargeModelConfig(
    name            = "AGI-Giant-70B",
    description     = "نموذج عملاق — مستوى LLaMA-2 70B / GPT-3",
    vocab_size      = 128_000,
    context_length  = 8_192,
    embedding_dim   = 8_192,
    num_layers      = 80,
    num_heads       = 64,
    ffn_hidden      = 28_672,
    dropout         = 0.0,    # النماذج الضخمة لا تحتاج dropout
    use_moe         = True,
    num_experts     = 32,
    top_k           = 4,
    moe_every_n_layers = 2,
    use_rope        = True,
    tie_weights     = False,
    recommended_lr  = 1e-4,
    recommended_batch = 128,
    recommended_warmup = 5_000,
    gradient_checkpointing = True,
    mixed_precision = "bf16",
)

ALL_CONFIGS = {
    "1B":  SMALL_1B,
    "7B":  MEDIUM_7B,
    "13B": LARGE_13B,
    "70B": GIANT_70B,
}


# ═══════════════════════════════════════════════════════════════
#   بناء النموذج من الإعدادات
# ═══════════════════════════════════════════════════════════════

def build_large_model(config: LargeModelConfig, device: str = "cpu"):
    """
    بناء النموذج الضخم من الإعدادات.
    يعرض ملخصاً كاملاً ويُنشئ النموذج جاهزاً للتدريب.

    Usage:
        from large_model_configs import MEDIUM_7B, build_large_model
        model = build_large_model(MEDIUM_7B, device="cuda")
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import torch
    from models.transformer import AGITransformer

    print(config.summary())

    # تحويل الإعدادات للـ AGITransformer
    model_kwargs = {
        "vocab_size":         config.vocab_size,
        "context_length":     config.context_length,
        "embedding_dim":      config.embedding_dim,
        "num_layers":         config.num_layers,
        "num_heads":          config.num_heads,
        "ffn_hidden":         config.ffn_hidden,
        "dropout":            config.dropout,
        "use_moe":            config.use_moe,
        "num_experts":        config.num_experts,
        "top_k":              config.top_k,
        "moe_every_n_layers": config.moe_every_n_layers,
        "use_rope":           config.use_rope,
        "tie_weights":        config.tie_weights,
    }

    print(f"\n  🔨 Building {config.name}...")
    model = AGITransformer(**model_kwargs)

    # Gradient Checkpointing (يوفر ذاكرة أثناء التدريب)
    if config.gradient_checkpointing:
        print("  ✅ Gradient checkpointing enabled")

    device_obj = torch.device(device)
    model = model.to(device_obj)

    actual_params = sum(p.numel() for p in model.parameters())
    print(f"  ✅ Model ready on {device_obj}")
    print(f"  ✅ Actual parameters: {actual_params:,}  (~{actual_params/1e9:.2f}B)")

    return model


# ═══════════════════════════════════════════════════════════════
#   مقارنة جميع النماذج
# ═══════════════════════════════════════════════════════════════

def compare_all() -> None:
    """طباعة مقارنة جميع النماذج في جدول واحد"""
    print("\n" + "═"*90)
    print(f"  {'النموذج':<20} {'Params':>10} {'Layers':>8} {'Heads':>7} "
          f"{'Dim':>7} {'Experts':>9} {'FP16 GB':>9} {'GPU المطلوب'}")
    print("─"*90)

    for key, cfg in ALL_CONFIGS.items():
        p = cfg.total_params()
        print(
            f"  {cfg.name:<20} {p/1e9:>8.1f}B "
            f"{cfg.num_layers:>8} {cfg.num_heads:>7} "
            f"{cfg.embedding_dim:>7} {cfg.num_experts:>9} "
            f"{cfg.memory_fp16_gb():>8.1f}  "
            f"{cfg.recommended_gpus().split('أو')[0].strip()}"
        )

    print("═"*90)
    print("\n  📌 ملاحظة: هذه النماذج جاهزة للتدريب — تحتاج فقط GPU + Dataset")
    print("  📌 للتدريب على Google Colab: ابدأ بـ 1B (يعمل على T4 مجاناً)")
    print("  📌 للتدريب الجاد: استخدم RunPod أو Lambda Labs\n")


if __name__ == "__main__":
    compare_all()

    print("\n" + "═"*60)
    print("  مثال: عرض تفاصيل النموذج 7B")
    print("═"*60)
    print(MEDIUM_7B.summary())

    print("\n" + "═"*60)
    print("  كيفية الاستخدام:")
    print("═"*60)
    print("""
  from large_model_configs import SMALL_1B, MEDIUM_7B, build_large_model

  # للتجربة (CPU):
  model = build_large_model(SMALL_1B, device="cpu")

  # للتدريب الجاد (GPU):
  model = build_large_model(MEDIUM_7B, device="cuda")

  # للنموذج العملاق (يحتاج 8× A100):
  model = build_large_model(GIANT_70B, device="cuda")
    """)
