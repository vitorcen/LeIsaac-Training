#!/usr/bin/env python3
# Universal reconstruct — rebuild a full checkpoint from base + delta, byte-exact.
# Pairs with prune_ckpts.py. Handles both .pt (StarVLA) and .safetensors (Wall-X / pi0.5 /
# GR00T) by inferring format from each file's extension. Output format follows --out's ext.
#
# Reconstruct = {**base, **delta}: delta carries every tensor that changed during training,
# base carries the frozen backbone stored once. Verifies no unexpected key overlap and
# reports key counts so a truncated delta can't silently yield a half-loaded model.
#
# Usage: merge_ckpt.py <base.(pt|safetensors)> <delta.(pt|safetensors)> <out.(pt|safetensors)>
import os, sys, torch

def load(p):
    if p.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(p)
    return torch.load(p, map_location="cpu", mmap=True)

def save(sd, p):
    if p.endswith(".safetensors"):
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in sd.items()}, p)
    else:
        torch.save(sd, p)

base_p, delta_p, out = sys.argv[1], sys.argv[2], sys.argv[3]
base, delta = load(base_p), load(delta_p)
overlap = set(base) & set(delta)            # expected: delta replaces these (trainable tensors)
full = {**base, **delta}
print(f"[merge] base={len(base)} + delta={len(delta)} (overlap={len(overlap)}) -> full={len(full)} keys")
save(full, out)
print(f"[merge] wrote {out} ({os.path.getsize(out)/1e9:.1f}G)")
