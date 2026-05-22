"""Real NF4 forward test for Wan2.1-I2V-14B (meta-init + streamed load).

Memory-frugal path:
1. Build WanModel on `meta` device → 0 bytes RAM.
2. Stream safetensors shard-by-shard; for each tensor, either:
   - quantize to NF4 + materialize on cuda  (2D Linear weights)
   - cast bf16 + place on cuda              (norms, biases, embeddings, RoPE)
3. After load, run one 480P denoise step. Report GPU peak.

Run in `dreamzero` env: torch 2.7.1+cu128, bnb 0.49, transformers 4.51.
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


os.environ.setdefault("WAN_DISABLE_FLASH", "1")
sys.path.insert(0, "/home/david/work/dreamzero-repo")

from groot.vla.model.dreamzero.modules.wan_video_dit import WanModel


WAN_PATH = Path.home() / ".cache/huggingface/hub/models--Wan-AI--Wan2.1-I2V-14B-480P/snapshots"
WAN_SNAP = next(WAN_PATH.glob("*"))

I2V_14B_CONFIG = dict(
    dim=5120,
    in_dim=36,
    ffn_dim=13824,
    out_dim=16,
    freq_dim=256,
    eps=1e-6,
    num_heads=40,
    num_layers=40,
    text_dim=4096,
    patch_size=(1, 2, 2),
    has_image_input=True,
)


# Diffusers → DreamZero name remap (template; blocks.N substituted at runtime)
_RENAME = {
    "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
    "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
    "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
    "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
    "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
    "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
    "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
    "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
    "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
    "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
    "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
    "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
    "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
    "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
    "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
    "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
    "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
    "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
    "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
    "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
    "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
    "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
    "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
    "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
    "blocks.0.norm2.bias": "blocks.0.norm3.bias",
    "blocks.0.norm2.weight": "blocks.0.norm3.weight",
    "blocks.0.scale_shift_table": "blocks.0.modulation",
    "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
    "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
    "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
    "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
    "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
    "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
    "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
    "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
    "condition_embedder.time_proj.bias": "time_projection.1.bias",
    "condition_embedder.time_proj.weight": "time_projection.1.weight",
    "patch_embedding.bias": "patch_embedding.bias",
    "patch_embedding.weight": "patch_embedding.weight",
    "scale_shift_table": "head.modulation",
    "proj_out.bias": "head.head.bias",
    "proj_out.weight": "head.head.weight",
    "condition_embedder.image_embedder.norm1.bias": "img_emb.proj.0.bias",
    "condition_embedder.image_embedder.norm1.weight": "img_emb.proj.0.weight",
    "condition_embedder.image_embedder.ff.net.0.proj.bias": "img_emb.proj.1.bias",
    "condition_embedder.image_embedder.ff.net.0.proj.weight": "img_emb.proj.1.weight",
    "condition_embedder.image_embedder.ff.net.2.bias": "img_emb.proj.3.bias",
    "condition_embedder.image_embedder.ff.net.2.weight": "img_emb.proj.3.weight",
    "condition_embedder.image_embedder.norm2.bias": "img_emb.proj.4.bias",
    "condition_embedder.image_embedder.norm2.weight": "img_emb.proj.4.weight",
    "condition_embedder.image_embedder.pos_embed": "img_emb.emb_pos",
}


def remap(raw_k: str):
    # Wan2.1-I2V-14B safetensors are already in DreamZero naming — identity passthrough.
    return raw_k


def find_parent(model, dotted):
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    return parent, parts[-1]


def main():
    print(f"snap: {WAN_SNAP}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda} bnb={bnb.__version__}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("\n[1/4] Build WanModel on meta device...")
    with torch.device("meta"):
        model = WanModel(**I2V_14B_CONFIG)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  total params: {n_params / 1e9:.2f}B")

    # Index linear modules for fast lookup during streaming.
    linear_paths = {}
    for path, m in model.named_modules():
        if isinstance(m, nn.Linear):
            linear_paths[path] = m
    print(f"  linears: {len(linear_paths)}")

    print("\n[2/4] Stream shards: quantize Linears, place non-Linears on cuda...")
    shards = sorted(WAN_SNAP.glob("diffusion_pytorch_model-*.safetensors"))
    pending_bias = {}  # new_key (.bias) → tensor, used when its weight arrives
    n_quant = 0
    n_other = 0
    n_skipped = 0
    sample_skip = []

    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for raw_k in f.keys():
                new_k = remap(raw_k)
                if new_k is None:
                    n_skipped += 1
                    if len(sample_skip) < 8:
                        sample_skip.append(raw_k)
                    continue

                t = f.get_tensor(raw_k)
                parent_path = ".".join(new_k.split(".")[:-1])
                attr = new_k.split(".")[-1]

                if parent_path in linear_paths and attr == "weight":
                    # Build Linear4bit, optionally with paired bias
                    existing = linear_paths[parent_path]
                    in_f, out_f = existing.in_features, existing.out_features
                    bias_t = pending_bias.pop(parent_path + ".bias", None)
                    has_bias = bias_t is not None or existing.bias is not None
                    new_lin = Linear4bit(
                        input_features=in_f,
                        output_features=out_f,
                        bias=has_bias,
                        quant_type="nf4",
                        compute_dtype=torch.bfloat16,
                        compress_statistics=True,
                    )
                    new_lin.weight = bnb.nn.Params4bit(
                        t.detach().to(dtype=torch.bfloat16).cpu(),
                        requires_grad=False,
                        quant_type="nf4",
                        compress_statistics=True,
                    )
                    if has_bias:
                        bb = bias_t if bias_t is not None else torch.zeros(out_f)
                        new_lin.bias = nn.Parameter(bb.detach().to(dtype=torch.bfloat16).cpu())
                    new_lin = new_lin.to("cuda")
                    parent_mod, parent_attr = find_parent(model, parent_path)
                    setattr(parent_mod, parent_attr, new_lin)
                    # update index so we don't reprocess
                    linear_paths.pop(parent_path)
                    n_quant += 1
                    del existing
                elif parent_path in linear_paths and attr == "bias":
                    # Weight not seen yet; buffer
                    pending_bias[new_k] = t.detach().clone()
                else:
                    # Plain Parameter or buffer — materialize on cuda bf16
                    try:
                        parent_mod, parent_attr = find_parent(model, new_k)
                        target = getattr(parent_mod, parent_attr)
                        new_val = t.detach().to(dtype=torch.bfloat16).to("cuda")
                        if isinstance(target, nn.Parameter):
                            setattr(parent_mod, parent_attr, nn.Parameter(new_val, requires_grad=False))
                        else:
                            # buffer
                            parent_mod._buffers[parent_attr] = new_val
                        n_other += 1
                    except Exception as e:
                        n_skipped += 1
                        if len(sample_skip) < 8:
                            sample_skip.append(f"err:{new_k}:{type(e).__name__}")
                del t
        gc.collect()
        cur = torch.cuda.memory_allocated() / 1e9
        print(f"  {shard.name}: cur={cur:.2f}GB q={n_quant} other={n_other} skip={n_skipped}")

    if pending_bias:
        print(f"  {len(pending_bias)} orphan biases never paired")
    print(f"  sample skipped: {sample_skip}")

    # RoPE freqs were computed on meta during __init__; re-compute on cuda.
    head_dim = I2V_14B_CONFIG["dim"] // I2V_14B_CONFIG["num_heads"]
    with torch.device("cuda"):
        model.rope.freqs = model.rope.precompute_freqs_cis_3d(head_dim)
    print("  rope freqs re-computed on cuda")

    torch.cuda.empty_cache()
    cur = torch.cuda.memory_allocated() / 1e9
    peak_load = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nLOAD DONE  cur={cur:.2f}GB peak={peak_load:.2f}GB")

    # Sanity: any remaining meta params?
    meta_left = [n for n, p in model.named_parameters() if p.is_meta]
    if meta_left:
        print(f"WARN: {len(meta_left)} params still on meta (e.g. {meta_left[:5]})")

    print("\n[3/4] Forward step @ 480P latent (9 x 60 x 104, B=1, bf16)...")
    f_lat, h_lat, w_lat = 9, 60, 104
    x = torch.randn(1, 16, f_lat, h_lat, w_lat, dtype=torch.bfloat16, device="cuda")
    y = torch.randn(1, 20, f_lat, h_lat, w_lat, dtype=torch.bfloat16, device="cuda")
    timestep = torch.tensor([500.0], dtype=torch.bfloat16, device="cuda")
    context = torch.randn(1, 512, 4096, dtype=torch.bfloat16, device="cuda")
    clip_feature = torch.randn(1, 257, 1280, dtype=torch.bfloat16, device="cuda")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        with torch.no_grad():
            out = model(x=x, timestep=timestep, context=context,
                        clip_feature=clip_feature, y=y)
        torch.cuda.synchronize()
        t1 = time.time()
        sh = out.shape if hasattr(out, "shape") else type(out)
        print(f"  forward OK  shape={sh}  time={(t1-t0)*1000:.0f}ms")
    except Exception as e:
        import traceback
        print(f"  forward FAILED: {type(e).__name__}: {str(e)[:400]}")
        traceback.print_exc()

    cur = torch.cuda.memory_allocated() / 1e9
    peak_fwd = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n[4/4] === RESULT ===")
    print(f"  weights (post-load):    {cur:.2f} GB")
    print(f"  forward peak:           {peak_fwd:.2f} GB")
    print(f"  activations (peak-cur): {peak_fwd - cur:.2f} GB")
    print(f"  4090 24G headroom:      {24 - peak_fwd:.2f} GB")


if __name__ == "__main__":
    main()
