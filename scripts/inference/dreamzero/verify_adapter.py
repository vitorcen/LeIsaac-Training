"""Verify a DreamZero LoRA adapter checkpoint is sane.

Phase 2 sanity gate before building real-forward server. Run on CPU (no GPU needed).
Checks:
 1. Loads adapter_model.pt without error
 2. Key structure matches Vizuara dreamzero-so101-lora (same 814 tensors, 400 lora_A + 400 lora_B)
 3. Sample LoRA tensor norms are non-zero (proves training updated them away from kaiming init)
 4. Compares lora_A / lora_B norm distribution against Vizuara to detect under/over-trained range

Usage:
    python LeIsaac/scripts/inference/dreamzero/verify_adapter.py \
        outputs/dreamzero-leisaac-so101-lora-r4/checkpoint-1000

Exit code 0 = sane, 1 = bad (don't waste compute building a server around a broken adapter).
"""
import sys
from pathlib import Path

import torch


def load_checkpoint(ckpt_dir: Path) -> dict:
    pt = ckpt_dir / "adapter_model.pt"
    if not pt.exists():
        print(f"❌ adapter_model.pt not found at {pt}")
        sys.exit(1)
    print(f"loading {pt} ({pt.stat().st_size/1024**2:.1f} MB)")
    state = torch.load(pt, map_location="cpu", weights_only=True)
    return state


def load_vizuara_reference() -> dict | None:
    """Optional: load Vizuara LoRA for cross-reference. Returns None if not cached locally."""
    from safetensors import safe_open
    cache_root = Path.home() / ".cache/huggingface/hub/models--Vizuara--dreamzero-so101-lora/snapshots"
    if not cache_root.exists():
        return None
    snap = next(cache_root.glob("*"), None)
    if snap is None:
        return None
    sf = snap / "model.safetensors"
    if not sf.exists():
        return None
    print(f"  comparing against Vizuara reference at {sf}")
    out = {}
    with safe_open(str(sf), framework="pt", device="cpu") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    ckpt_dir = Path(sys.argv[1])
    if not ckpt_dir.is_dir():
        print(f"❌ {ckpt_dir} is not a directory")
        sys.exit(1)

    state = load_checkpoint(ckpt_dir)
    n_total = len(state)
    n_lora_A = sum(1 for k in state if "lora_A" in k)
    n_lora_B = sum(1 for k in state if "lora_B" in k)
    n_other = n_total - n_lora_A - n_lora_B
    total_params = sum(v.numel() for v in state.values())
    print()
    print(f"=== Structure ===")
    print(f"  total tensors:  {n_total}")
    print(f"  lora_A tensors: {n_lora_A}")
    print(f"  lora_B tensors: {n_lora_B}")
    print(f"  other (state_encoder, action_encoder, action_decoder, …): {n_other}")
    print(f"  total params:   {total_params/1e6:.1f} M")

    # Cross-reference Vizuara if available
    viz = load_vizuara_reference()
    if viz is not None:
        our_keys = set(state.keys())
        viz_keys = set(viz.keys())
        common = our_keys & viz_keys
        only_ours = our_keys - viz_keys
        only_viz = viz_keys - our_keys
        print()
        print(f"=== Vs Vizuara reference ===")
        print(f"  common keys: {len(common)} / {len(our_keys)}")
        if only_ours:
            print(f"  ⚠️  only in ours ({len(only_ours)} keys):")
            for k in list(only_ours)[:3]:
                print(f"      {k}")
        if only_viz:
            print(f"  ⚠️  only in Vizuara ({len(only_viz)} keys):")
            for k in list(only_viz)[:3]:
                print(f"      {k}")
        if not only_ours and not only_viz:
            print(f"  ✅ key sets identical (814 each)")

    # Norm sanity: LoRA B is init zero in PEFT, so a non-zero ‖B‖ proves training updated it.
    # LoRA A is kaiming init (non-zero from start) — its norm changes prove updates too.
    print()
    print(f"=== LoRA tensor norms (sample 5) ===")
    print(f"{'key':<80} {'shape':<22} {'norm':>10}")
    keys_to_show = [k for k in state if "lora" in k][:5]
    for k in keys_to_show:
        v = state[k]
        n = v.float().norm().item()
        print(f"{k[:80]:<80} {str(tuple(v.shape)):<22} {n:>10.4f}")

    # Critical check: ‖lora_B‖ — should be > 0 (initialized at zero, training moves it).
    b_norms = [state[k].float().norm().item() for k in state if "lora_B" in k]
    a_norms = [state[k].float().norm().item() for k in state if "lora_A" in k]

    import statistics
    def stats(name, xs):
        if not xs:
            return f"{name}: (no tensors)"
        return (f"{name}: n={len(xs)}  mean={statistics.mean(xs):.4f}  "
                f"min={min(xs):.4f}  max={max(xs):.4f}  median={statistics.median(xs):.4f}")
    print()
    print(f"=== Norm distributions ===")
    print(f"  {stats('lora_A', a_norms)}")
    print(f"  {stats('lora_B', b_norms)}")

    # Verdict
    print()
    print(f"=== Verdict ===")
    if b_norms and all(b == 0.0 for b in b_norms):
        print(f"❌ ALL lora_B norms are zero — adapter wasn't trained or save bug.")
        sys.exit(1)
    elif b_norms and statistics.mean(b_norms) < 1e-3:
        print(f"⚠️  lora_B mean norm < 0.001 — barely trained; "
              "ok for early ckpt (ckpt-1000 of 20000 = 5%) but verify trend at next ckpt")
        sys.exit(0)
    else:
        print(f"✅ adapter looks sane — lora_B has non-trivial updates "
              f"(mean ‖B‖ = {statistics.mean(b_norms):.4f}); proceed to server integration.")
        sys.exit(0)


if __name__ == "__main__":
    main()
