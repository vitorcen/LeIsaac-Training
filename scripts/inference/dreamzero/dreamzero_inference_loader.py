"""Build a DreamZero VLA inference model for 4090 24G from a LoRA-only checkpoint.

Differs from `socket_test_optimized_AR.py` (H100 80G FP16) in that we:
  1. set `skip_component_loading=True` so VLAConfig instantiation doesn't auto-load FP16 DiT
  2. manually stream-load the Wan2.1-14B DiT shards, NF4-quantize each Linear, and merge our LoRA
     into the merged base weight (reuses `wan14b_leisaac_forward.py` proven path)
  3. load action_head non-LoRA tensors (state_encoder, action_encoder, action_decoder, etc.)
  4. keep UMT5-XXL on CPU via vram_management (~5GB), CLIP/VAE on GPU (~3GB)

Usage:
    from dreamzero_inference_loader import build_dreamzero_inference_model
    model = build_dreamzero_inference_model(
        ckpt_dir=".../checkpoint-2000",
        experiment_cfg_dir=".../experiment_cfg",
        wan_snap_dir=".../Wan2.1-I2V-14B-480P/snapshots/<hash>",
    )
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from safetensors import safe_open

import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit

# Ensure dreamzero-repo is importable
sys.path.insert(0, os.path.expanduser("~/work/dreamzero-repo"))

# Bump dynamo recompile limit BEFORE importing scheduler modules that decorate with @torch.compile
import torch._dynamo
torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.recompile_limit = 128

# #3: Prefer memory-efficient SDPA backend (less workspace than flash).
# Keep math as fallback so non-attn paths don't break.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

from groot.vla.model.dreamzero.base_vla import VLA, VLAConfig
import groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk as _wandit_mod


def _patch_causal_wan_model_to_meta():
    """Subclass CausalWanModel so its __init__ allocates params on `meta` device.
    Saves 56+ GB of RAM since we'll stream-load DiT shards NF4-quantized later.
    """
    Original = _wandit_mod.CausalWanModel
    if getattr(Original, "_patched_to_meta", False):
        return
    class MetaCausalWanModel(Original):
        _patched_to_meta = True
        def __init__(self, *args, **kwargs):
            with torch.device("meta"):
                super().__init__(*args, **kwargs)
    _wandit_mod.CausalWanModel = MetaCausalWanModel
    print("[loader] CausalWanModel patched to meta-device init", flush=True)


LORA_ALPHA = 4
LORA_RANK = 4
LORA_SCALE = LORA_ALPHA / LORA_RANK  # 1.0


# ---------- Adapter LoRA loading ----------

def load_adapter_pt(adapter_pt: Path) -> tuple[dict, dict]:
    """Return (lora_pairs, non_lora) from training-saved adapter_model.pt OR Vizuara .safetensors.

    lora_pairs: {dit_param_path (with .weight) -> {"A": tensor, "B": tensor}}
        where dit_param_path is in the WAN DiT (not nested under base_model)
    non_lora:   {full_key_in_VLA -> tensor}
        action_head.model.state_encoder.*, action_encoder.*, action_decoder.* etc.
    """
    if adapter_pt.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(adapter_pt))
    else:
        state = torch.load(adapter_pt, map_location="cpu", weights_only=True)
    pairs = {}
    non_lora = {}
    # PEFT prefix when defer_lora_injection=False
    PREFIX_LORA = "action_head.model.base_model.model."
    for k, v in state.items():
        if k.startswith(PREFIX_LORA) and ("lora_A.default.weight" in k or "lora_B.default.weight" in k):
            tail = k[len(PREFIX_LORA):]
            if ".lora_A.default.weight" in tail:
                wan_key = tail.replace(".lora_A.default.weight", ".weight")
                pairs.setdefault(wan_key, {})["A"] = v
            else:
                wan_key = tail.replace(".lora_B.default.weight", ".weight")
                pairs.setdefault(wan_key, {})["B"] = v
        else:
            non_lora[k] = v
    return pairs, non_lora


# ---------- DiT NF4 stream-load + LoRA merge ----------

def find_parent(model, dotted: str):
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    return parent, parts[-1]


def nf4_load_dit_with_lora_merge(
    dit_module: nn.Module,
    wan_snap_dir: Path,
    lora_pairs: dict,
) -> dict:
    """In-place: stream-load WAN DiT shards into `dit_module`, merging LoRA into 2D linears
    before NF4 quantization. dit_module must be on meta device (model.action_head.model).

    Returns stats dict.
    """
    linear_paths = {p: m for p, m in dit_module.named_modules() if isinstance(m, nn.Linear)}
    n_merged = n_quant = n_other = n_skipped = 0

    shards = sorted(wan_snap_dir.glob("diffusion_pytorch_model-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No diffusion_pytorch_model-*.safetensors in {wan_snap_dir}")

    for shard in shards:
        t0 = time.perf_counter()
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
                        if delta.shape == base_w.shape:
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
                    parent_mod, parent_attr = find_parent(dit_module, parent_path)
                    setattr(parent_mod, parent_attr, new_lin)
                    linear_paths.pop(parent_path)
                    n_quant += 1
                    del existing, base_w
                elif parent_path in linear_paths and attr == "bias":
                    pass
                else:
                    try:
                        parent_mod, parent_attr = find_parent(dit_module, raw_k)
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
        print(f"  [DiT] {shard.name}: cur={cur:.2f}GB q={n_quant} merged={n_merged} other={n_other} t={time.perf_counter()-t0:.1f}s", flush=True)

    return dict(n_merged=n_merged, n_quant=n_quant, n_other=n_other, n_skipped=n_skipped,
                lora_expected=len(lora_pairs))


# ---------- Top-level builder ----------

def build_dreamzero_inference_model(
    ckpt_dir: str | Path,
    experiment_cfg_dir: str | Path,
    wan_snap_dir: str | Path,
    device: str = "cuda",
    use_text_cpu_offload: bool = True,
):
    """Build a fully loaded VLA model for inference on 4090 24G.

    Args:
        ckpt_dir: contains adapter_model.pt (LoRA + action_head non-LoRA tensors)
        experiment_cfg_dir: contains conf.yaml + metadata.json
        wan_snap_dir: contains diffusion_pytorch_model-*.safetensors + UMT5/CLIP/VAE
        use_text_cpu_offload: keep UMT5 on CPU; only move to GPU during text encode

    Returns:
        (vla_model, conf_omegaconf, metadata_dict)
    """
    ckpt_dir = Path(ckpt_dir)
    experiment_cfg_dir = Path(experiment_cfg_dir)
    wan_snap_dir = Path(wan_snap_dir)

    # Accept either our `adapter_model.pt` or Vizuara's `model.safetensors`
    candidates = [ckpt_dir / "adapter_model.pt", ckpt_dir / "model.safetensors"]
    adapter_pt = next((p for p in candidates if p.exists()), None)
    if adapter_pt is None:
        raise FileNotFoundError(f"no adapter found in {ckpt_dir}, tried {candidates}")
    conf_path = experiment_cfg_dir / "conf.yaml"
    metadata_path = experiment_cfg_dir / "metadata.json"
    for p in (conf_path, metadata_path):
        if not p.exists():
            raise FileNotFoundError(p)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(f"[loader] ckpt={ckpt_dir}", flush=True)
    print(f"[loader] cfg={experiment_cfg_dir}", flush=True)
    print(f"[loader] wan={wan_snap_dir}", flush=True)

    # 1. Load conf.yaml, build VLAConfig with patched flags
    full_cfg = OmegaConf.load(conf_path)
    model_cfg_omg = full_cfg.model.config  # the VLAConfig section

    # Patch action_head_cfg.config:
    #   - skip_component_loading=True : don't auto-load FP16 DiT (we do NF4 manually)
    #   - defer_lora_injection=True   : don't wrap DiT with PEFT; we pre-merge LoRA into NF4 weights
    ah_inner = model_cfg_omg.action_head_cfg.config
    OmegaConf.update(ah_inner, "skip_component_loading", True, force_add=True)
    OmegaConf.update(ah_inner, "defer_lora_injection", True, force_add=True)
    # Rewrite cloud-only encoder paths to local
    OmegaConf.update(ah_inner.text_encoder_cfg, "text_encoder_pretrained_path",
                     str(wan_snap_dir / "models_t5_umt5-xxl-enc-bf16.pth"))
    OmegaConf.update(ah_inner.image_encoder_cfg, "image_encoder_pretrained_path",
                     str(wan_snap_dir / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"))
    OmegaConf.update(ah_inner.vae_cfg, "vae_pretrained_path",
                     str(wan_snap_dir / "Wan2.1_VAE.pth"))
    # DiT path - irrelevant when skip_component_loading=True but set anyway
    OmegaConf.update(ah_inner.diffusion_model_cfg, "diffusion_model_pretrained_path",
                     str(wan_snap_dir))

    cfg_dict = OmegaConf.to_container(model_cfg_omg, resolve=True)
    cfg_dict.pop("_target_", None)
    cfg_dict.pop("_recursive_", None)

    print("[loader] Building VLAConfig + VLA (UMT5/CLIP/VAE load to CPU, DiT to meta)...", flush=True)
    _patch_causal_wan_model_to_meta()
    torch.manual_seed(42)
    config = VLAConfig(**cfg_dict)

    # 2. Instantiate VLA — this auto-loads UMT5+CLIP+VAE state dicts on CPU,
    #    and builds CausalWanModel on meta device (skip_component_loading=True so no DiT shard load).
    t0 = time.perf_counter()
    model = VLA(config)
    print(f"[loader] VLA init done in {time.perf_counter()-t0:.1f}s", flush=True)

    model.eval()
    model.requires_grad_(False)

    # 3. Stream-load DiT NF4 + merge LoRA from our adapter
    lora_pairs, non_lora = load_adapter_pt(adapter_pt)
    print(f"[loader] adapter: {len(lora_pairs)} LoRA pairs (for DiT) + {len(non_lora)} non-LoRA tensors", flush=True)

    # defer_lora_injection=True ⇒ no PEFT wrap. DiT is a clean CausalWanModel.
    dit_module = model.action_head.model

    print("[loader] Stream-loading DiT shards + NF4 quant + LoRA merge...", flush=True)
    stats = nf4_load_dit_with_lora_merge(dit_module, wan_snap_dir, lora_pairs)
    print(f"[loader] DiT load stats: {stats}", flush=True)

    # 4. Materialize ANY remaining meta-device params/buffers in DiT (i.e. state/action encoder/decoder
    #    weights that aren't in WAN shards — they'll be filled from adapter non-LoRA in step 5).
    materialized_meta = 0
    for name, p in list(dit_module.named_parameters()):
        if p.is_meta:
            mod, attr = find_parent(dit_module, name)
            setattr(mod, attr, nn.Parameter(
                torch.empty(p.shape, dtype=torch.bfloat16, device=device),
                requires_grad=False))
            materialized_meta += 1
    for name, b in list(dit_module.named_buffers()):
        if b.is_meta:
            mod, attr = find_parent(dit_module, name)
            mod._buffers[attr] = torch.empty(b.shape, dtype=b.dtype if b.dtype != torch.uint8 else torch.float32, device=device)
            materialized_meta += 1
    print(f"[loader] materialized {materialized_meta} meta params/buffers (state/action enc/dec slots)", flush=True)

    # 5. Load non-LoRA action_head tensors (state_encoder, action_encoder, action_decoder)
    #    Adapter keys: action_head.model.base_model.model.state_encoder.layer1.W
    #    Target keys:  action_head.model.state_encoder.layer1.W (no PEFT wrap)
    print("[loader] Loading non-LoRA action_head tensors...", flush=True)
    target_state = dict(model.named_parameters())
    target_state.update(dict(model.named_buffers()))

    PEFT_PREFIX = "action_head.model.base_model.model."
    UNWRAPPED_PREFIX = "action_head.model."

    loaded = missing = 0
    for k, v in non_lora.items():
        # remap PEFT key → unwrapped key
        if k.startswith(PEFTPREFIX := PEFT_PREFIX):
            target_k = UNWRAPPED_PREFIX + k[len(PEFTPREFIX):]
        else:
            target_k = k
        if target_k in target_state:
            target = target_state[target_k]
            try:
                target.data.copy_(v.to(target.device, dtype=target.dtype))
                loaded += 1
            except Exception as e:
                print(f"[loader]   FAIL copy {target_k}: {e}", flush=True)
                missing += 1
        else:
            missing += 1
            if missing < 5:
                print(f"[loader]   missing target for {target_k} (shape {tuple(v.shape)})", flush=True)
    print(f"[loader] non-LoRA: loaded={loaded} missing={missing}", flush=True)

    # 6. CLIP + VAE → CPU permanently (we forward both on CPU, only move latents to GPU).
    # CLIP 2.6 GB + VAE 1 GB = 3.6 GB saved on GPU. CPU forwards add ~3-5s per episode start.
    print("[loader] Keeping CLIP + VAE on CPU (forward there, latents → GPU)...", flush=True)
    for top in (model.action_head.image_encoder, model.action_head.vae):
        top.to(device="cpu", dtype=torch.float32)
        for n, p in top.named_parameters():
            if p.device.type != "cpu":
                p.data = p.data.to(device="cpu", dtype=torch.float32)
        for n, b in top.named_buffers():
            if b.device.type != "cpu":
                new_dtype = torch.float32 if b.is_floating_point() else b.dtype
                parent_path = n.rsplit(".", 1)[0] if "." in n else ""
                attr = n.rsplit(".", 1)[-1]
                if parent_path:
                    parent_mod = top
                    for part in parent_path.split("."):
                        parent_mod = getattr(parent_mod, part)
                else:
                    parent_mod = top
                parent_mod._buffers[attr] = b.to(device="cpu", dtype=new_dtype)
    print(f"[loader] CLIP + VAE staged on CPU; weights stay there", flush=True)

    # UMT5 XXL is ~11 GB bf16 — too big to keep on GPU alongside DiT (8.5GB) + CLIP/VAE (~3GB)
    # + activations. Keep it on CPU and monkey-patch encode_prompt to load on-demand:
    #   on entry: text_encoder.to(cuda), forward, text_encoder.to(cpu), empty cache
    print("[loader] Keeping UMT5 on CPU; encode_prompt patched to load on-demand...", flush=True)
    model.action_head.text_encoder.to(device="cpu", dtype=torch.bfloat16)

    _orig_encode_prompt = model.action_head.encode_prompt
    _prompt_cache: dict[bytes, torch.Tensor] = {}
    def _on_demand_encode_prompt(input_ids, attention_mask):
        # Cache key: bytes of input_ids — same prompt → reuse embedding (saves 8+s UMT5 forward)
        key_t = input_ids.detach().cpu() if torch.is_tensor(input_ids) else torch.tensor(input_ids)
        cache_key = key_t.numpy().tobytes()
        if cache_key in _prompt_cache:
            return _prompt_cache[cache_key].clone()

        # First time for this prompt: load UMT5 if still around, encode, cache, then potentially free.
        te = model.action_head.text_encoder
        if te is None:
            raise RuntimeError("text encoder was freed but new prompt requested (not in cache).")
        import gc; gc.collect()
        torch.cuda.empty_cache()
        te.to(device=device)
        try:
            ids = input_ids.to(device) if torch.is_tensor(input_ids) else input_ids
            mask = attention_mask.to(device) if torch.is_tensor(attention_mask) else attention_mask
            emb = _orig_encode_prompt(ids, mask)
            _prompt_cache[cache_key] = emb.detach().clone()
            return emb
        finally:
            te.to(device="cpu")
            gc.collect()
            torch.cuda.empty_cache()
    model.action_head.encode_prompt = _on_demand_encode_prompt
    model.action_head._prompt_cache = _prompt_cache

    def _free_text_encoder():
        """Drop UMT5 entirely (frees ~11GB RAM). Caller is responsible for ensuring
        no new prompts will be requested — only cached prompts can be returned after this."""
        model.action_head.text_encoder = None
        import gc; gc.collect()
        torch.cuda.empty_cache()
        print("[loader] text encoder freed; cached prompts only from here on", flush=True)
    model.action_head.free_text_encoder = _free_text_encoder

    # Enable tiled VAE encode — chops the image into 34x34 patches, reducing peak activation
    # by ~3x at ~1.3x slower encode. Critical for fitting alongside Isaac Sim's ~6.6GB on 24GB.
    model.action_head.tiled = True
    print("[loader] VAE tiled encode enabled (peak activation reduction)", flush=True)

    # #1: CLIP + VAE forward on CPU permanently. Inputs CPU, latents/embeddings to GPU.
    # Both stay weight-resident on CPU → ~3.6 GB GPU saved.
    # Cost: ~3-5s CPU forward per episode start (CLIP is the dominant cost).
    ah = model.action_head

    # Disable the lazy "_ensure_vae_on_device" that would move VAE → GPU on first use.
    ah._vae_device_ready = True  # short-circuit the check
    ah._ensure_vae_on_device = lambda *a, **kw: None

    # Wrap vae.encode: input → CPU fp32 → CPU forward → latents → GPU bf16
    _orig_vae_encode = ah.vae.encode
    def _cpu_vae_encode(x, *args, **kwargs):
        x_cpu = x.detach().to(device="cpu", dtype=torch.float32)
        with torch.no_grad():
            lat_cpu = _orig_vae_encode(x_cpu, *args, **kwargs)
        return lat_cpu.to(device=device, dtype=torch.bfloat16)
    ah.vae.encode = _cpu_vae_encode

    # Wrap image_encoder.encode_image: image → CPU → CPU CLIP forward → embed → GPU bf16
    _orig_ie_encode = ah.image_encoder.encode_image
    def _cpu_ie_encode(img):
        img_cpu = img.detach().to(device="cpu", dtype=torch.float32)
        with torch.no_grad():
            emb_cpu = _orig_ie_encode(img_cpu)
        return emb_cpu.to(device=device, dtype=torch.bfloat16)
    ah.image_encoder.encode_image = _cpu_ie_encode

    # Wrap action_head.encode_image (CLIP+VAE composite) with episode-scope cache.
    # The model's lazy_joint_video_action re-triggers encode_image every chunk because we
    # send 1-frame inputs (which auto-reset current_start_frame=0). With CPU CLIP (46s) +
    # CPU VAE that's brutal. Cache the full (clip_feas, ys, image) tuple per episode;
    # policy.reset() sets ah._episode_clip_cache=None to invalidate.
    ah._episode_clip_cache = None
    _orig_ah_encode_image = ah.encode_image
    def _cached_encode_image(image, num_frames, height, width):
        if ah._episode_clip_cache is not None:
            return ah._episode_clip_cache
        result = _orig_ah_encode_image(image, num_frames, height, width)
        ah._episode_clip_cache = result
        return result
    ah.encode_image = _cached_encode_image

    # 7. Action head misc tensors (registers, projector, etc.) → GPU bf16
    # The model.action_head.model is already on GPU (NF4 stream loaded).
    # Other submodules (vl_self_attention, time_modality_projection, action_decoder, state_encoder...)
    # are inside model.action_head.model (CausalWanModel has them). Move any stragglers.
    for name, mod in model.named_modules():
        if any(skip in name for skip in ("text_encoder", "image_encoder", "vae", "action_head.model")):
            continue
        for p in mod.parameters(recurse=False):
            if p.device.type == "cpu":
                p.data = p.data.to(device=device, dtype=torch.bfloat16)
        for b_name, b in mod.named_buffers(recurse=False):
            if b.device.type == "cpu":
                mod._buffers[b_name] = b.to(device=device)

    # 8. Recompute CausalWanModel buffers (freqs, freqs_action, freqs_state) on GPU.
    #    These are NOT nn.Parameter/buffer — plain attrs that lived on meta after patched init.
    from groot.vla.model.dreamzero.modules.wan2_1_submodule import rope_params
    dit = model.action_head.model
    dit_cfg = config.action_head_cfg["config"]["diffusion_model_cfg"]
    dim = dit_cfg["dim"]
    num_heads = dit_cfg["num_heads"]
    d = dim // num_heads
    with torch.device(device):
        dit.freqs_action = rope_params(1024 * 10, d)
        dit.freqs_state = rope_params(1024, d)
        dit.freqs = [
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ]
    print(f"[loader] RoPE freqs/freqs_action/freqs_state recomputed on {device} (d={d})", flush=True)

    # 9. Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # 10. Override .device on model and action_head so prepare_input routes to cuda
    # (default impl returns next(parameters()).device which picks CPU text_encoder first).
    _force_device = torch.device(device)
    _orig_vla_cls = type(model)
    _orig_ah_cls = type(model.action_head)
    class _CudaDeviceVLA(_orig_vla_cls):
        @property
        def device(self):
            return _force_device
    class _CudaDeviceActionHead(_orig_ah_cls):
        @property
        def device(self):
            return _force_device
        @property
        def dtype(self):
            return torch.bfloat16
    model.__class__ = _CudaDeviceVLA
    model.action_head.__class__ = _CudaDeviceActionHead
    print(f"[loader] device property overridden to {_force_device}", flush=True)

    # Init attrs that only get set by the TRT build path / training loop but are read at inference
    if not hasattr(model.action_head, "trt_engine"):
        model.action_head.trt_engine = None

    torch.cuda.empty_cache()
    cur = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n[loader] DONE  cur={cur:.2f}GB peak={peak:.2f}GB on {device}", flush=True)

    return model, full_cfg, metadata


if __name__ == "__main__":
    # CLI smoke test: just build the model and report VRAM
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--cfg-dir", required=True)
    ap.add_argument("--wan-dir", default=str(
        Path.home() / ".cache/huggingface/hub/models--Wan-AI--Wan2.1-I2V-14B-480P/snapshots"
    ))
    args = ap.parse_args()

    wan_dir = Path(args.wan_dir)
    if not (wan_dir / "diffusion_pytorch_model.safetensors.index.json").exists():
        wan_dir = next(wan_dir.glob("*"))  # auto-pick first snapshot
    print(f"[main] resolved wan_dir={wan_dir}")

    model, cfg, metadata = build_dreamzero_inference_model(
        ckpt_dir=args.ckpt_dir,
        experiment_cfg_dir=args.cfg_dir,
        wan_snap_dir=wan_dir,
    )
    print(f"\n✅ Model built. VLA dtype={next(model.parameters()).dtype}, device={next(model.parameters()).device}")
