#!/usr/bin/env python
"""Verify the partial-unfreeze fix is correctly wired — WITHOUT a full training run.

Reuses the REAL pipeline pieces (build_framework, freeze_backbones, apply_partial_unfreeze,
build_param_lr_groups) so it exercises the exact code path train_starvla.main() takes.

  python verify_unfreeze.py --config <...unfreeze4.yaml> [--device cpu] [--v2]

V1 (always) — STATIC membership check, catches bug ② (name-based exclusion / wrong order):
  build framework -> freeze_backbones -> apply_partial_unfreeze -> build_param_lr_groups,
  then assert EVERY unfrozen VLM param is inside the optimizer's param groups.
  Old buggy code: unfrozen top-N layers were dropped -> this assert fails.

V2 (--v2) — DTYPE / fp32-master diagnostic for bug ③ (no data, no DeepSpeed):
  put a synthetic gradient on the unfrozen params and take ONE plain-AdamW step at the
  configured lr, in bf16 (as the param lives) vs an fp32 copy. bf16 delta≈0 means a tiny lr
  underflows bf16 ULP -> the plain path needs an fp32 master. The REAL launcher
  (accelerate + deepspeed_zero2, bf16.enabled) supplies that fp32 master automatically for
  every optimizer-managed param, so ③ is resolved there once ② is fixed. This probe just
  makes the dtype reality explicit.
"""
import argparse
import torch
from omegaconf import OmegaConf

from starVLA.model.framework.base_framework import build_framework
from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cpu", help="cpu (default, no GPU contention) or cuda")
    ap.add_argument("--v2", action="store_true", help="also run the bf16/fp32 weight-delta diagnostic")
    args = ap.parse_args()

    cfg = wrap_config(OmegaConf.load(args.config))
    print(f"== building framework from {args.config} (tune_top_llm_layers="
          f"{cfg.framework.qwenvl.get('tune_top_llm_layers', 0)}) ==")
    model = build_framework(cfg).to(args.device)

    # mirror train_starvla.main(): freeze -> partial-unfreeze -> build optimizer
    fm = cfg.trainer.get("freeze_modules", None)
    model = TrainerUtils.freeze_backbones(model, freeze_modules=fm)
    if hasattr(model, "apply_partial_unfreeze"):
        model.apply_partial_unfreeze()
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.trainer.learning_rate.get("base", 1e-4))

    # ---- V1: membership ----
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    unfrozen = [(n, p) for n, p in model.named_parameters() if p.requires_grad and "qwen_vl_interface" in n]
    n_train = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    n_total = sum(p.numel() for _, p in model.named_parameters())
    print(f"\n[V1] trainable params: {n_train/1e6:.1f}M / {n_total/1e6:.1f}M ({100*n_train/n_total:.2f}%)")
    print(f"[V1] optimizer groups: {[(g['name'], g['lr'], sum(p.numel() for p in g['params'])//1000/1000) for g in optimizer.param_groups]} (M params)")
    print(f"[V1] unfrozen VLM params (qwen_vl_interface, requires_grad): {len(unfrozen)} tensors")
    missing = [n for n, p in unfrozen if id(p) not in opt_ids]
    cfg_n = int(cfg.framework.qwenvl.get("tune_top_llm_layers", 0) or 0)
    if not unfrozen:
        if cfg_n > 0:
            raise SystemExit(f"[V1] ❌ FAIL: tune_top_llm_layers={cfg_n} but ZERO VLM params unfrozen "
                             f"— apply_partial_unfreeze() located no LLM layers (silent no-op).")
        print("[V1] ⚠️ tune_top_llm_layers=0 — nothing to unfreeze (frozen head-only default).")
    elif missing:
        raise SystemExit(f"[V1] ❌ FAIL: {len(missing)} unfrozen VLM params NOT in optimizer "
                         f"(bug ② present). e.g. {missing[:3]}")
    else:
        print(f"[V1] ✅ PASS: all {len(unfrozen)} unfrozen VLM tensors are inside the optimizer.")

    # ---- V2: bf16 vs fp32 step diagnostic ----
    if args.v2 and unfrozen:
        lr = float(cfg.trainer.learning_rate.get("qwen_vl_interface", cfg.trainer.learning_rate.get("base", 1e-4)))
        n, p = unfrozen[0]
        before = p.detach().float().clone()
        # synthetic grad ~ N(0,1); one AdamW-like step magnitude ≈ lr
        g = torch.randn_like(p)
        # bf16 (as-is)
        p_bf16 = torch.nn.Parameter(p.detach().clone())
        opt_b = torch.optim.AdamW([p_bf16], lr=lr); p_bf16.grad = g.clone(); opt_b.step()
        d_bf16 = (p_bf16.detach().float() - before).abs().max().item()
        # fp32 master
        p_fp32 = torch.nn.Parameter(p.detach().float().clone())
        opt_f = torch.optim.AdamW([p_fp32], lr=lr); p_fp32.grad = g.float().clone(); opt_f.step()
        d_fp32 = (p_fp32.detach() - before).abs().max().item()
        print(f"\n[V2] '{n}' dtype={p.dtype}, lr={lr:g}")
        print(f"[V2] max |Δ| after 1 AdamW step:  bf16={d_bf16:.3e}   fp32={d_fp32:.3e}")
        if p.dtype in (torch.bfloat16, torch.float16) and d_bf16 < d_fp32 * 0.5:
            print("[V2] ⚠️ bf16 update underflows vs fp32 → bug ③ on the PLAIN path. "
                  "DeepSpeed bf16 supplies the fp32 master under the real launcher (OK there).")
        else:
            print("[V2] ✅ update magnitude preserved.")


if __name__ == "__main__":
    main()
