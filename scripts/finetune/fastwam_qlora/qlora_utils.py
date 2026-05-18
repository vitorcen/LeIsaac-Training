"""In-place NF4 quantization + PEFT LoRA injection for FastWAM's MoT.

Design (Linus-style: one obvious data path, no abstractions):

1. `quantize_linears_nf4(module, skip_names)` walks the module tree and swaps
   every `nn.Linear` whose qualified-name suffix is NOT in `skip_names` with a
   `bnb.nn.Linear4bit` carrying the same weight (NF4 + double quant).  The
   skipped Linears (q/k/v/o) stay as plain `nn.Linear` so PEFT can wrap them.

2. `apply_lora(model_dit, target_modules)` wraps the skipped Linears with PEFT
   LoRA adapters.  PEFT freezes everything except the new `lora_A`/`lora_B`
   matrices.

3. `freeze_for_qlora(model)` is the override for `Wan22Trainer._apply_dit_only_train_mode`:
   it leaves `requires_grad` exactly as PEFT set it (LoRA-only).  The base
   trainer's blanket `model.dit.requires_grad_(True)` would otherwise reactivate
   the frozen base, defeating QLoRA.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit
except ImportError as e:  # pragma: no cover - install error path
    raise ImportError(
        "bitsandbytes is required for QLoRA. `pip install bitsandbytes` in the fastwam env."
    ) from e

try:
    from peft import LoraConfig, get_peft_model
except ImportError as e:  # pragma: no cover
    raise ImportError("peft is required for QLoRA. `pip install peft` in the fastwam env.") from e

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


# Qualified-name suffixes that PEFT will wrap with LoRA.  Anything ending in
# one of these (e.g. `mixtures.video.blocks.0.self_attn.q`) is left as nn.Linear.
DEFAULT_LORA_TARGETS: tuple[str, ...] = ("q", "k", "v", "o")


def _is_lora_target(qualified_name: str, target_suffixes: Iterable[str]) -> bool:
    last = qualified_name.rsplit(".", 1)[-1]
    return last in set(target_suffixes)


def _is_peft_lora_layer(module: nn.Module) -> bool:
    """True if this Linear is actually a peft.LoraLinear (has base_layer + lora_A/B)."""
    return hasattr(module, "base_layer") and hasattr(module, "lora_A") and hasattr(module, "lora_B")


def _replace_linear_with_4bit(parent: nn.Module, attr: str, src: nn.Linear) -> Linear4bit:
    """Canonical bnb pattern: build Linear4bit, assign Params4bit on CPU,
    move to CUDA — the cuda hook triggers in-place NF4 quantization."""
    has_bias = src.bias is not None
    new = Linear4bit(
        input_features=src.in_features,
        output_features=src.out_features,
        bias=has_bias,
        quant_type="nf4",
        compute_dtype=torch.bfloat16,
        compress_statistics=True,  # double quant
    )
    new.weight = bnb.nn.Params4bit(
        src.weight.data.detach().to(dtype=torch.bfloat16).cpu(),
        requires_grad=False,
        quant_type="nf4",
        compress_statistics=True,
    )
    if has_bias:
        new.bias = nn.Parameter(src.bias.data.detach().to(dtype=torch.bfloat16).cpu())
    new = new.to("cuda")  # cuda hook → NF4 quantization
    setattr(parent, attr, new)
    return new


def quantize_linears_nf4(
    root: nn.Module,
    lora_target_suffixes: Iterable[str] = DEFAULT_LORA_TARGETS,
) -> dict[str, int]:
    """Replace every plain `nn.Linear` outside the LoRA-adapter path with NF4 `Linear4bit`.

    Skips:
      - Modules already `Linear4bit` (idempotent).
      - LoRA adapters (`lora_A` / `lora_B`) — must stay bf16, trainable.
      - Top-level LoRA-target linears (q/k/v/o) — replaced by PEFT wrapper.
      - `Embedding`, `LayerNorm`, etc. — not `nn.Linear`.

    Quantizes `base_layer` inside PEFT-wrapped LoraLinear (the original q/k/v/o weight).
    """
    stats = {"quantized": 0, "kept_for_lora": 0, "skipped_already_4bit": 0}
    targets = set(lora_target_suffixes)

    to_swap: list[tuple[nn.Module, str, nn.Linear, str]] = []
    for name, module in root.named_modules():
        # PEFT LoRA-adapter paths: never quantize the trainable lora_A/lora_B layers.
        if ".lora_A" in name or ".lora_B" in name or name.endswith("lora_A") or name.endswith("lora_B"):
            continue
        for attr_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if isinstance(child, Linear4bit):
                stats["skipped_already_4bit"] += 1
                continue
            full = f"{name}.{attr_name}" if name else attr_name
            if attr_name in {"lora_A", "lora_B"}:
                continue
            # Top-level q/k/v/o suffix (un-wrapped) — PEFT will wrap separately.
            # After PEFT wrap, q/k/v/o becomes LoraLinear (still nn.Linear subclass);
            # we DO want to quantize its inner `base_layer` (also nn.Linear), so don't
            # skip by suffix match alone — only skip if this is the un-wrapped form.
            if attr_name in targets and not _is_peft_lora_layer(child):
                stats["kept_for_lora"] += 1
                continue
            to_swap.append((module, attr_name, child, full))

    for parent, attr, lin, full in to_swap:
        _replace_linear_with_4bit(parent, attr, lin)
        stats["quantized"] += 1
    logger.info(
        "QLoRA quant: quantized=%d kept_for_lora=%d (targets=%s)",
        stats["quantized"], stats["kept_for_lora"], sorted(targets),
    )
    return stats


class ManualLoraLinear(nn.Module):
    """Minimal LoRA wrapper: out = base(x) + lora_B(lora_A(x)) * (alpha/r).

    Drop-in for nn.Linear at q/k/v/o.  `base_layer` is the original nn.Linear
    (left for downstream NF4 quantization).  `lora_A` / `lora_B` are
    trainable bf16 nn.Linear with `requires_grad=True`.

    Compatible with `_is_peft_lora_layer` (has base_layer / lora_A / lora_B).
    """

    def __init__(self, base_layer: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(r)

        # Match base_layer dtype for the adapters.
        dtype = base_layer.weight.dtype
        device = base_layer.weight.device

        # PEFT-style names so save/load files look familiar.
        self.lora_A = nn.Linear(self.in_features, self.r, bias=False, dtype=dtype, device=device)
        self.lora_B = nn.Linear(self.r, self.out_features, bias=False, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # LoRA init: A ~ N(0, 1/r), B = 0  (standard).
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        # Freeze base, train adapters.
        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)
        self.lora_A.weight.requires_grad_(True)
        self.lora_B.weight.requires_grad_(True)

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x)))
        return base_out + lora_out * self.scaling


def apply_lora(
    dit_module: nn.Module,
    r: int = 16,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: Iterable[str] = DEFAULT_LORA_TARGETS,
) -> nn.Module:
    """Manually wrap every Linear whose attr-name is in `target_modules` with
    `ManualLoraLinear`.  Does NOT use PEFT — peft triggers a torch
    `_named_members` C-iterator crash intermittently on this model.

    Returns the same root module (mutated in-place); no PeftModel wrapping.
    """
    targets = set(target_modules)
    wraps: list[tuple[nn.Module, str, nn.Linear]] = []
    for name, module in dit_module.named_modules():
        # Skip subtrees that are already inside a LoRA wrap (avoid double).
        if ".lora_A" in name or ".lora_B" in name or name.endswith("lora_A") or name.endswith("lora_B"):
            continue
        for attr_name, child in list(module.named_children()):
            if attr_name in targets and isinstance(child, nn.Linear) and not _is_peft_lora_layer(child):
                wraps.append((module, attr_name, child))

    for parent, attr, lin in wraps:
        setattr(parent, attr, ManualLoraLinear(lin, r=r, alpha=alpha, dropout=dropout))

    # Freeze the rest of the dit (only LoRA params trainable).
    for name, p in dit_module.named_parameters():
        if ".lora_A" in name or ".lora_B" in name:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    trainable = sum(p.numel() for p in dit_module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in dit_module.parameters())
    logger.info(
        "QLoRA LoRA (manual, no peft): wrapped=%d trainable=%.3fM / total=%.3fM (%.4f%%)",
        len(wraps), trainable / 1e6, total / 1e6, 100.0 * trainable / max(total, 1),
    )
    return dit_module


def freeze_for_qlora(model: nn.Module) -> None:
    """Trainer hook override: PEFT already set requires_grad correctly.

    Replaces `Wan22Trainer._apply_dit_only_train_mode`'s blanket unfreeze,
    which would otherwise unfreeze the NF4 base and break QLoRA invariants.
    """
    # Put dit in train mode (BN/dropout etc.), keep everything else in eval.
    model.eval()
    model.dit.train()
    # Do NOT touch requires_grad — PEFT owns it.


def patch_trainer_for_qlora() -> None:
    """Monkey-patch Wan22Trainer for single-GPU QLoRA:
      - `_apply_dit_only_train_mode` → respects PEFT's requires_grad.
      - `__init__` stat-log → tolerates `deepspeed_plugin is None` (single GPU).
    """
    from fastwam import trainer as _trainer_mod
    from fastwam.trainer import Wan22Trainer

    def _qlora_freeze(self, model):  # noqa: ARG001 - matches upstream signature
        freeze_for_qlora(model)

    Wan22Trainer._apply_dit_only_train_mode = _qlora_freeze

    # Optimizer.load_state_dict wrapper: torch 2.5 _single_tensor_adamw runs
    # `exp_avg.lerp_(grad, 1-beta1)` which requires exp_avg.dtype == grad.dtype.
    # LoRA params are bf16, so their grads are bf16 → state must be bf16 too.
    # Earlier version promoted ALL state to fp32; on resume this caused
    #   RuntimeError: expected dtype float for `end` but got dtype c10::BFloat16
    # because lerp_'s `end` arg (grad, bf16) couldn't auto-cast against fp32
    # exp_avg.  Fix: match each tensor's dtype to its owning param's dtype.
    import torch.optim as _optim
    _orig_load_state_dict = _optim.Optimizer.load_state_dict

    def _qlora_load_state_dict(self, state_dict):
        ret = _orig_load_state_dict(self, state_dict)
        n_coerced = 0
        for p, st in self.state.items():
            target_dtype = p.dtype if isinstance(p, torch.Tensor) and p.is_floating_point() else None
            if target_dtype is None:
                continue
            for k, v in list(st.items()):
                if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype != target_dtype:
                    st[k] = v.to(target_dtype)
                    n_coerced += 1
        if n_coerced:
            logger.info("[QLoRA] Coerced %d optimizer state tensors to match param dtypes", n_coerced)
        return ret

    _optim.Optimizer.load_state_dict = _qlora_load_state_dict

    _orig_adamw_init = _optim.AdamW.__init__
    _orig_adamw_step = _optim.AdamW.step

    def _qlora_adamw_init(self, params, *args, **kwargs):
        kwargs.setdefault("foreach", False)
        kwargs.setdefault("fused", False)
        _orig_adamw_init(self, params, *args, **kwargs)

    def _qlora_adamw_step(self, closure=None):
        # Runtime alignment: exp_avg / exp_avg_sq dtype must match p.grad dtype
        # for inplace lerp_ / addcmul_.  But grad dtype varies (Accelerate's
        # clip_grad_norm may upcast bf16 → fp32 some steps).  Realign here.
        for group in self.param_groups:
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                st = self.state.get(p)
                if not st:
                    continue
                for k in ("exp_avg", "exp_avg_sq"):
                    v = st.get(k)
                    if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype != g.dtype:
                        st[k] = v.to(g.dtype)
        return _orig_adamw_step(self, closure)

    _optim.AdamW.__init__ = _qlora_adamw_init
    _optim.AdamW.step = _qlora_adamw_step

    # Patch bnb `_get_tensor_stream` to swallow intermittent torch attribute
    # lookup race: `module 'torch' has no attribute 'ct'`.  Inside bnb the
    # call `ct.c_void_p(torch._C._cuda_getCurrentRawStream(...))` uses bnb's
    # `import ctypes as ct` alias; under multi-thread import race torch's
    # __getattr__ catch-all sometimes intercepts the wrong `ct` lookup.
    # Fallback to a null-pointer stream (uses default CUDA stream — slight
    # perf cost but no crash).
    import bitsandbytes.functional as _bF
    import ctypes as _ctypes

    _orig_get_tensor_stream = _bF._get_tensor_stream

    def _safe_get_tensor_stream(A):
        try:
            return _orig_get_tensor_stream(A)
        except (AttributeError, TypeError) as e:
            if "ct" in str(e) or "isinstance" in str(e):
                return _ctypes.c_void_p(0)
            raise

    _bF._get_tensor_stream = _safe_get_tensor_stream

    # NOTE (2026-05-17): removed earlier `safetensors.load_file` patch that
    # stripped bnb quant_state keys.  Stripping them prevented bnb's
    # `Params4bit._load_from_state_dict` from reconstructing QuantState — fresh
    # NF4 weights computed at QLoRA setup REPLACED the trained ones, silently
    # corrupting all post-phase-1 training (loss exploded to 1e21).  The bnb
    # base weights ARE valid in safetensors and must be restored intact.

    logger.info(
        "Patched: trainer freeze + Optimizer.load_state_dict (no-op) + "
        "AdamW(foreach=False) for QLoRA."
    )
