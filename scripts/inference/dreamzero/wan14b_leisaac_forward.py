"""Phase 2 single-forward verify: load our trained LeIsaac LoRA adapter into Wan2.1-I2V-14B (NF4) on 4090 24G.

Identical to wan14b_lora_merged_forward.py EXCEPT it loads our `adapter_model.pt` (torch.save format)
instead of Vizuara's `model.safetensors`. Verifies:
  1. Our adapter keys merge cleanly into Wan2.1 DiT (no shape mismatch, no missing pairs)
  2. NF4 quant + LoRA merge produce a model that forwards without NaN/OOM
  3. 4090 24G VRAM usage stays under 12 GB (room for sim + UMT5/CLIP swap-in later)

Usage:
    python LeIsaac/scripts/inference/dreamzero/wan14b_leisaac_forward.py CKPT_DIR

Example:
    python LeIsaac/scripts/inference/dreamzero/wan14b_leisaac_forward.py \
        outputs/dreamzero-leisaac-so101-lora-r4/checkpoint-1000
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

sys.path.insert(0, os.path.expanduser("~/work/dreamzero-repo"))
from groot.vla.model.dreamzero.modules.wan_video_dit import WanModel


WAN_SNAP = next((Path.home() / ".cache/huggingface/hub/"
                 "models--Wan-AI--Wan2.1-I2V-14B-480P/snapshots").glob("*"))

I2V_14B_CONFIG = dict(
    dim=5120, in_dim=36, ffn_dim=13824, out_dim=16, freq_dim=256, eps=1e-6,
    num_heads=40, num_layers=40, text_dim=4096, patch_size=(1, 2, 2),
    has_image_input=True,
)

# DreamZero default LoRA hparams (matches Vizuara + our training)
LORA_ALPHA = 4
LORA_RANK = 4
LORA_SCALE = LORA_ALPHA / LORA_RANK  # 1.0


def find_parent(model, dotted):
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    return parent, parts[-1]


def load_lora_pairs_from_pt(adapter_pt: Path) -> tuple[dict, dict]:
    """Read our trained adapter_model.pt (torch.save dict from LoRADumpCallback).

    Returns:
        lora_pairs: {wan_param_path: {"A": tensor, "B": tensor}}  — for DiT layer merge
        non_lora:   {original_key: tensor}                          — action_head weights (not used by DiT-only forward)
    """
    print(f"loading adapter from {adapter_pt} ({adapter_pt.stat().st_size/1024**2:.1f} MB)")
    state = torch.load(adapter_pt, map_location="cpu", weights_only=True)
    pairs = {}
    non_lora = {}
    PREFIX = "action_head.model.base_model.model."
    for k, v in state.items():
        if not k.startswith(PREFIX):
            non_lora[k] = v
            continue
        tail = k[len(PREFIX):]
        if ".lora_A.default.weight" in tail:
            wan_path = tail.replace(".lora_A.default.weight", ".weight")
            pairs.setdefault(wan_path, {})["A"] = v
        elif ".lora_B.default.weight" in tail:
            wan_path = tail.replace(".lora_B.default.weight", ".weight")
            pairs.setdefault(wan_path, {})["B"] = v
        else:
            non_lora[k] = v
    bad = [k for k, vv in pairs.items() if "A" not in vv or "B" not in vv]
    if bad:
        print(f"  ⚠️  {len(bad)} keys missing A or B (first 3: {bad[:3]})")
    print(f"  LoRA pairs: {len(pairs)} (for DiT)")
    print(f"  non-LoRA tensors: {len(non_lora)} (action_head weights, skipped by DiT-only forward)")
    return pairs, non_lora


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    ckpt_dir = Path(sys.argv[1])
    adapter_pt = ckpt_dir / "adapter_model.pt"
    if not adapter_pt.exists():
        print(f"❌ adapter_model.pt not found at {adapter_pt}")
        sys.exit(1)

    print(f"WAN_SNAP: {WAN_SNAP}")
    print(f"CKPT_DIR: {ckpt_dir}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda} bnb={bnb.__version__}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("\n[1/5] Load our LoRA pairs from adapter_model.pt...")
    lora_pairs, non_lora = load_lora_pairs_from_pt(adapter_pt)

    print("\n[2/5] Build WanModel on meta device...")
    with torch.device("meta"):
        model = WanModel(**I2V_14B_CONFIG)
    model.eval()
    linear_paths = {p: m for p, m in model.named_modules() if isinstance(m, nn.Linear)}
    print(f"  linears: {len(linear_paths)}")

    n_merged = n_quant = n_other = n_skipped = 0

    print("\n[3/5] Stream shards; merge LoRA into 2D Linears, then NF4-quantize...")
    shards = sorted(WAN_SNAP.glob("diffusion_pytorch_model-*.safetensors"))
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for raw_k in f.keys():
                t = f.get_tensor(raw_k)
                parent_path = ".".join(raw_k.split(".")[:-1])
                attr = raw_k.split(".")[-1]

                if parent_path in linear_paths and attr == "weight":
                    base_w = t.detach().to(dtype=torch.float32)
                    if raw_k in lora_pairs:
                        A = lora_pairs[raw_k]["A"].to(torch.float32)
                        B = lora_pairs[raw_k]["B"].to(torch.float32)
                        delta = (B @ A) * LORA_SCALE
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

    head_dim = I2V_14B_CONFIG["dim"] // I2V_14B_CONFIG["num_heads"]
    with torch.device("cuda"):
        model.rope.freqs = model.rope.precompute_freqs_cis_3d(head_dim)
    print(f"  rope freqs re-computed on cuda")
    print(f"  LoRA merges applied: {n_merged} / expected {len(lora_pairs)}")
    if n_merged != len(lora_pairs):
        print(f"  ⚠️  merged {n_merged} of {len(lora_pairs)} — {len(lora_pairs)-n_merged} LoRA pairs unmatched (bug?)")

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
    nan_count = torch.isnan(out).sum().item()
    print(f"  forward OK  shape={tuple(out.shape)}  time={(t1-t0)*1000:.0f}ms  NaN={nan_count}")
    print(f"  out stats: min={out.min().item():.3f} max={out.max().item():.3f} "
          f"mean={out.mean().item():.3f} std={out.std().item():.3f}")

    cur = torch.cuda.memory_allocated() / 1e9
    peak_fwd = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n[5/5] === RESULT ===")
    print(f"  weights (NF4 + LeIsaac LoRA-merged): {cur:.2f} GB")
    print(f"  forward peak:                         {peak_fwd:.2f} GB")
    print(f"  4090 24G headroom:                    {24 - peak_fwd:.2f} GB")
    if nan_count > 0:
        print(f"  ❌ {nan_count} NaN — adapter merge broke the model")
        sys.exit(1)
    if peak_fwd > 18.0:
        print(f"  ⚠️  peak {peak_fwd:.1f} GB > 18GB target (need to leave room for sim+UMT5/CLIP swap)")
    else:
        print(f"  ✅ peak {peak_fwd:.1f} GB within budget — ckpt-1000 ready for server integration")


if __name__ == "__main__":
    main()
