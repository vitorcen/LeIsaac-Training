"""Offline action smoke — verify train/serve interface alignment.

Per FastWAM lesson #4 (see openvla-finetune-plan memory):
    "首次 eval 前先做 smoke：抽 1 帧训练数据 → 喂 model → 看 action 是否在 [-1, 1]；
     如果出 range 就是 train/serve 不齐，别启动 Isaac Sim 浪费时间。"

This script loads:
  - 4-bit base OpenVLA-7B
  - A LoRA adapter from a training checkpoint (`--adapter <ckpt_dir>`)
  - dataset_statistics.json (q01/q99 from the LeIsaac training set)

For N sampled training frames it:
  1. Decodes the frame, builds the same prompt as the collator
  2. Calls `model.predict_action(..., unnorm_key="leisaac")`
  3. Compares the un-normalized 6-DOF prediction to the ground-truth action

Pass criteria (per dim, averaged over N samples):
    pred in [q01 - 5%, q99 + 5%]      # not extrapolating wildly
    |pred - gt| / (q99 - q01) < 0.5   # within half the training range

Usage (light enough to share GPU with training, but `--device cpu` works too):
    PYTHONPATH=LeIsaac/scripts/finetune \
        conda run -n openvla --no-capture-output python -m openvla.offline_eval \
        --adapter LeIsaac/outputs/openvla-leisaac-pick-orange/checkpoint-1000 \
        --dataset LeIsaac/datasets/raw/leisaac-pick-orange \
        --stats LeIsaac/outputs/openvla-leisaac-pick-orange/dataset_statistics.json \
        --num_samples 5
"""
from __future__ import annotations

# Eagerly *fully iterate* every Enum in torchgen.model BEFORE peft/bnb walk
# the module tree (DispatchKey has 135 members; bare import lazy-initializes).
import enum as _enum
import torchgen.model as _torchgen_model  # noqa: F401
for _name in dir(_torchgen_model):
    _obj = getattr(_torchgen_model, _name, None)
    if isinstance(_obj, type) and issubclass(_obj, _enum.Enum) and _obj is not _enum.Enum:
        list(_obj); list(_obj.__members__)
del _name, _obj

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from openvla.dataset import LeIsaacOpenVLADataset, normalize_action


UNNORM_KEY = "leisaac"


def load_model(base: str, adapter: str | None, device: str) -> tuple:
    print(f"[smoke] loading base: {base}", flush=True)
    processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)

    kwargs: dict = dict(trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    if device == "cuda":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["device_map"] = {"": "cpu"}

    t0 = time.time()
    model = AutoModelForVision2Seq.from_pretrained(base, **kwargs)
    print(f"  base loaded in {time.time()-t0:.1f}s", flush=True)

    if adapter:
        t0 = time.time()
        model = PeftModel.from_pretrained(model, adapter)
        print(f"  adapter loaded from {adapter} in {time.time()-t0:.1f}s", flush=True)
    else:
        print("  no adapter — evaluating BASE model only", flush=True)

    model.eval()
    return processor, model


def _resolve_openvla(model):
    """Walk down PeftModel → LoraModel → OpenVLAForActionPrediction.

    Cannot rely on `hasattr(model, "norm_stats")` to gate the walk — PeftModel's
    `__getattr__` forwards to the base model, so the wrapper *reads* True for
    `norm_stats` but writes don't propagate.  Always traverse to bottom.
    """
    inner = model
    if hasattr(inner, "base_model"):
        inner = inner.base_model
    if hasattr(inner, "model"):
        inner = inner.model
    if not hasattr(inner, "predict_action"):
        raise RuntimeError(f"Did not land on OpenVLAForActionPrediction; got {type(inner).__name__}")
    return inner


def inject_stats(model, stats: dict) -> "tuple":
    """Patch the *real* OpenVLAForActionPrediction's norm_stats.  Returns the inner."""
    inner = _resolve_openvla(model)
    inner.norm_stats = dict(inner.norm_stats) if inner.norm_stats else {}
    inner.norm_stats[UNNORM_KEY] = {
        "action": {
            "q01": np.asarray(stats["q01"]).tolist(),
            "q99": np.asarray(stats["q99"]).tolist(),
            "mask": np.asarray(stats.get("mask", [True] * len(stats["q01"]))).tolist(),
        }
    }
    # Hard assertion — if this fails the rest of the eval is theatre.
    assert inner.get_action_dim(UNNORM_KEY) == len(stats["q01"]), (
        f"Stats injection failed: get_action_dim('{UNNORM_KEY}') "
        f"returned {inner.get_action_dim(UNNORM_KEY)}, expected {len(stats['q01'])}"
    )
    print(f"  injected norm_stats[{UNNORM_KEY!r}].action  (action_dim="
          f"{inner.get_action_dim(UNNORM_KEY)} ✓ self-check)", flush=True)
    return inner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="openvla/openvla-7b")
    ap.add_argument("--adapter", default=None, help="path to a LoRA ckpt dir; omit to eval base only")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--stats", required=True, help="dataset_statistics.json from training")
    ap.add_argument("--num_samples", type=int, default=5)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--prompt", default=None,
                    help="override; default reads canonical prompt from stats file")
    args = ap.parse_args()

    # ----- stats -----
    with open(args.stats) as f:
        raw = json.load(f)
    s = raw["action"]
    stats = {k: np.asarray(v) for k, v in s.items()}
    canonical_prompt = args.prompt or raw.get("prompt", "Grab orange and place into plate")
    print(f"[smoke] canonical_prompt={canonical_prompt!r}", flush=True)
    print(f"[smoke] stats: q01={stats['q01']}\n               q99={stats['q99']}", flush=True)

    # ----- dataset -----
    ds = LeIsaacOpenVLADataset(args.dataset, action_stats=stats)
    n = len(ds)
    idxs = np.linspace(0, n - 1, args.num_samples).astype(int).tolist()
    print(f"[smoke] sampling indices {idxs} from {n} frames", flush=True)

    # ----- model -----
    processor, model = load_model(args.base, args.adapter, args.device)
    inner = inject_stats(model, stats)

    device = "cuda:0" if args.device == "cuda" else "cpu"
    q01 = stats["q01"]
    q99 = stats["q99"]
    rng = np.maximum(q99 - q01, 1e-6)

    results = []
    print("\n[smoke] === per-sample predictions ===", flush=True)
    for i, idx in enumerate(idxs):
        sample = ds[idx]
        img = Image.fromarray(sample["image"])
        gt = sample["action"]

        prompt_text = f"In: What action should the robot take to {canonical_prompt.strip().rstrip('.')}?\nOut:"
        inputs = processor(prompt_text, img).to(device)
        # Match the vision conv's actual dtype.  With bnb 4bit + torch_dtype=bf16
        # the vision conv weights are bf16; pure-fp CPU load leaves them fp32.
        # The first conv we can find tells us what the backbone wants.
        try:
            conv_dtype = inner.vision_backbone.featurizer.patch_embed.proj.weight.dtype
            if inputs["pixel_values"].dtype != conv_dtype:
                inputs["pixel_values"] = inputs["pixel_values"].to(conv_dtype)
        except AttributeError:
            pass  # Best-effort; fall back to processor's default

        t0 = time.time()
        with torch.no_grad():
            pred = inner.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
        latency = 1000 * (time.time() - t0)

        # OpenVLA 7-DoF EEF output: we trained on 6-DoF joint pos, so it should be 6
        pred = np.asarray(pred).flatten()
        if pred.shape[0] != gt.shape[0]:
            print(f"  ⚠️  predicted dim {pred.shape[0]} != gt dim {gt.shape[0]}", flush=True)
        pred = pred[:gt.shape[0]]

        in_range = (pred >= q01 - 0.05 * rng).all() and (pred <= q99 + 0.05 * rng).all()
        rel_err = np.abs(pred - gt) / rng
        print(
            f"  [{i+1}/{len(idxs)}] idx={idx} ({latency:.0f}ms): "
            f"in_range={'✅' if in_range else '❌'}  "
            f"max_rel_err={rel_err.max():.2f}",
            flush=True,
        )
        print(f"    gt   = {np.round(gt, 2)}", flush=True)
        print(f"    pred = {np.round(pred, 2)}", flush=True)
        print(f"    err  = {np.round(pred - gt, 2)}", flush=True)
        results.append({"idx": idx, "in_range": bool(in_range), "rel_err": rel_err.tolist(),
                        "gt": gt.tolist(), "pred": pred.tolist(), "latency_ms": latency})

    # ----- aggregate -----
    in_range_rate = np.mean([r["in_range"] for r in results])
    mean_rel_err = np.mean([np.mean(r["rel_err"]) for r in results])
    max_rel_err = np.max([np.max(r["rel_err"]) for r in results])
    print("\n[smoke] === aggregate ===", flush=True)
    print(f"  in_range_rate = {in_range_rate:.1%}", flush=True)
    print(f"  mean_rel_err  = {mean_rel_err:.2f}  (target < 0.50)", flush=True)
    print(f"  max_rel_err   = {max_rel_err:.2f}", flush=True)

    verdict = "🟢 PASS" if (in_range_rate >= 0.8 and mean_rel_err < 0.5) else "🔴 FAIL — train/serve misaligned, do not run Isaac Sim yet"
    print(f"\n  verdict: {verdict}", flush=True)


if __name__ == "__main__":
    main()
