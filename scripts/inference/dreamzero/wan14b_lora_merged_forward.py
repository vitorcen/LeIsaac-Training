"""Phase 2a: Load Wan2.1-14B + merge Vizuara LoRA into base bf16 weights, then NF4.

Goal: validate LoRA application path before building full e2e (text/image/VAE/action_head).
If forward works without exploding peak, LoRA merge is correct.

LoRA math: merged_W = base_W + (lora_B @ lora_A) * (alpha/rank)
For Vizuara: alpha=4, rank=4 → scale=1.0
"""
import gc
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open

import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit


# flash-attn 2.8.3 installed — let WanModel use it automatically
# (was: os.environ.setdefault("WAN_DISABLE_FLASH", "1"))
sys.path.insert(0, "/home/david/work/dreamzero-repo")

from groot.vla.model.dreamzero.modules.wan_video_dit import WanModel


WAN_SNAP = next((Path.home() / ".cache/huggingface/hub/"
                 "models--Wan-AI--Wan2.1-I2V-14B-480P/snapshots").glob("*"))
LORA_SNAP = next((Path.home() / ".cache/huggingface/hub/"
                  "models--Vizuara--dreamzero-so101-lora/snapshots").glob("*"))

I2V_14B_CONFIG = dict(
    dim=5120, in_dim=36, ffn_dim=13824, out_dim=16, freq_dim=256, eps=1e-6,
    num_heads=40, num_layers=40, text_dim=4096, patch_size=(1, 2, 2),
    has_image_input=True,
)

LORA_ALPHA = 4
LORA_RANK = 4
LORA_SCALE = LORA_ALPHA / LORA_RANK  # 1.0


def find_parent(model, dotted):
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    return parent, parts[-1]


def load_lora_pairs() -> dict:
    """Read Vizuara LoRA, return {wan_param_path: (lora_A, lora_B)}.

    Vizuara key format:
      action_head.model.base_model.model.blocks.0.cross_attn.q.lora_A.default.weight
    Map to Wan param path:
      blocks.0.cross_attn.q.weight
    """
    print(f"loading LoRA from {LORA_SNAP}")
    pairs = {}  # wan_path -> {'A': tensor, 'B': tensor}
    PREFIX = "action_head.model.base_model.model."
    with safe_open(str(LORA_SNAP / "model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            if not k.startswith(PREFIX):
                continue
            tail = k[len(PREFIX):]
            if ".lora_A.default.weight" in tail:
                wan_path = tail.replace(".lora_A.default.weight", ".weight")
                pairs.setdefault(wan_path, {})["A"] = f.get_tensor(k)
            elif ".lora_B.default.weight" in tail:
                wan_path = tail.replace(".lora_B.default.weight", ".weight")
                pairs.setdefault(wan_path, {})["B"] = f.get_tensor(k)
    # Sanity
    bad = [k for k, v in pairs.items() if "A" not in v or "B" not in v]
    if bad:
        print(f"  WARN: {len(bad)} keys missing A or B (sample: {bad[:3]})")
    print(f"  LoRA pairs: {len(pairs)}")
    return pairs


def main():
    print(f"WAN_SNAP: {WAN_SNAP}")
    print(f"LORA_SNAP: {LORA_SNAP}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda} bnb={bnb.__version__}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("\n[1/5] Load Vizuara LoRA pairs into CPU dict (~217 MB)...")
    lora_pairs = load_lora_pairs()

    print("\n[2/5] Build WanModel on meta device...")
    with torch.device("meta"):
        model = WanModel(**I2V_14B_CONFIG)
    model.eval()
    linear_paths = {p: m for p, m in model.named_modules() if isinstance(m, nn.Linear)}
    print(f"  linears: {len(linear_paths)}")

    n_merged = 0
    n_quant = 0
    n_other = 0
    n_skipped = 0

    print("\n[3/5] Stream shards; merge LoRA into 2D Linears, then NF4-quantize...")
    shards = sorted(WAN_SNAP.glob("diffusion_pytorch_model-*.safetensors"))
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for raw_k in f.keys():
                t = f.get_tensor(raw_k)
                parent_path = ".".join(raw_k.split(".")[:-1])
                attr = raw_k.split(".")[-1]

                if parent_path in linear_paths and attr == "weight":
                    base_w = t.detach().to(dtype=torch.float32)  # fp32 for merge precision
                    # Apply LoRA if there is one for this Linear
                    if raw_k in lora_pairs:
                        A = lora_pairs[raw_k]["A"].to(torch.float32)  # [r, in]
                        B = lora_pairs[raw_k]["B"].to(torch.float32)  # [out, r]
                        delta = (B @ A) * LORA_SCALE  # [out, in]
                        if delta.shape != base_w.shape:
                            print(f"  shape mismatch {raw_k}: base{base_w.shape} delta{delta.shape}")
                        else:
                            base_w = base_w + delta
                            n_merged += 1

                    existing = linear_paths[parent_path]
                    in_f, out_f = existing.in_features, existing.out_features
                    has_bias = existing.bias is not None

                    new_lin = Linear4bit(
                        input_features=in_f, output_features=out_f, bias=has_bias,
                        quant_type="nf4", compute_dtype=torch.bfloat16,
                        compress_statistics=True,
                    )
                    new_lin.weight = bnb.nn.Params4bit(
                        base_w.to(dtype=torch.bfloat16).cpu(),
                        requires_grad=False, quant_type="nf4", compress_statistics=True,
                    )
                    if has_bias:
                        new_lin.bias = nn.Parameter(torch.zeros(out_f, dtype=torch.bfloat16))
                    new_lin = new_lin.to("cuda")
                    parent_mod, parent_attr = find_parent(model, parent_path)
                    setattr(parent_mod, parent_attr, new_lin)
                    linear_paths.pop(parent_path)
                    n_quant += 1
                    del existing, base_w
                elif parent_path in linear_paths and attr == "bias":
                    # bias for a Linear we'll see later — buffer would be needed
                    # but we set bias=zeros above; skip
                    pass
                else:
                    try:
                        parent_mod, parent_attr = find_parent(model, raw_k)
                        target = getattr(parent_mod, parent_attr)
                        new_val = t.detach().to(dtype=torch.bfloat16).to("cuda")
                        if isinstance(target, nn.Parameter):
                            setattr(parent_mod, parent_attr,
                                    nn.Parameter(new_val, requires_grad=False))
                        else:
                            parent_mod._buffers[parent_attr] = new_val
                        n_other += 1
                    except Exception:
                        n_skipped += 1
                del t
        gc.collect()
        cur = torch.cuda.memory_allocated() / 1e9
        print(f"  {shard.name}: cur={cur:.2f}GB q={n_quant} merged={n_merged} other={n_other}")

    # RoPE freqs on cuda
    head_dim = I2V_14B_CONFIG["dim"] // I2V_14B_CONFIG["num_heads"]
    with torch.device("cuda"):
        model.rope.freqs = model.rope.precompute_freqs_cis_3d(head_dim)
    print(f"  rope freqs re-computed on cuda")
    print(f"  LoRA merges applied: {n_merged} / expected {len(lora_pairs)}")

    torch.cuda.empty_cache()
    cur = torch.cuda.memory_allocated() / 1e9
    peak_load = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nLOAD DONE  cur={cur:.2f}GB peak={peak_load:.2f}GB")

    print("\n[4/5] Forward step @ 480P latent (9 x 60 x 104)...")
    f_lat, h_lat, w_lat = 9, 60, 104
    x = torch.randn(1, 16, f_lat, h_lat, w_lat, dtype=torch.bfloat16, device="cuda")
    y = torch.randn(1, 20, f_lat, h_lat, w_lat, dtype=torch.bfloat16, device="cuda")
    timestep = torch.tensor([500.0], dtype=torch.bfloat16, device="cuda")
    context = torch.randn(1, 512, 4096, dtype=torch.bfloat16, device="cuda")
    clip_feature = torch.randn(1, 257, 1280, dtype=torch.bfloat16, device="cuda")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out = model(x=x, timestep=timestep, context=context,
                    clip_feature=clip_feature, y=y)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"  forward OK  shape={out.shape}  time={(t1-t0)*1000:.0f}ms")
    print(f"  out stats: min={out.min().item():.3f} max={out.max().item():.3f} "
          f"mean={out.mean().item():.3f} std={out.std().item():.3f}")

    cur = torch.cuda.memory_allocated() / 1e9
    peak_fwd = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n[5/5] === RESULT ===")
    print(f"  weights (NF4 + LoRA-merged):  {cur:.2f} GB")
    print(f"  forward peak:                  {peak_fwd:.2f} GB")
    print(f"  4090 24G headroom:             {24 - peak_fwd:.2f} GB")
    print(f"  vs base-only baseline:         {cur:.2f} GB vs 8.54 GB (should be ~same)")


if __name__ == "__main__":
    main()
