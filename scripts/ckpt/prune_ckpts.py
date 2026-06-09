#!/usr/bin/env python3
# Universal frozen-backbone checkpoint pruner — the ONE canonical disk-saving tool.
#
# Problem: a finished training run leaves N full checkpoints. If the backbone was FROZEN
# (StarVLA frozen-VLM, Wall-X freeze_vlm, pi0.5 expert-only FT, GR00T frozen-VLM ...) the
# frozen weights are byte-identical in every checkpoint -> keeping N fulls = (N-1)x dead
# weight. Collapse to: 1 base (frozen, shared, kept once) + N deltas (only the tensors that
# changed = the trainable head/expert). Reconstruct any step byte-exact with merge.
#
# Method = DIFF against a reference base, NOT prefix slicing. Universal:
#   - StarVLA: frozen VLM tensors equal base -> delta == the action head (verified identical
#     to the old prefix method, 0 extra keys).
#   - Wall-X / pi0.5: trainable expert is INTERLEAVED inside model.layers.* /
#     paligemma_with_expert.* -> no clean prefix exists, but diff still isolates exactly the
#     changed tensors.
#   - Full-FT (ACT/DP/SmolVLA/X-VLA): almost every tensor differs -> delta ~ full size ->
#     the tool REFUSES to extract (no frozen backbone to exploit) and tells you to just
#     keep-best instead. Foolproof: it never produces a useless delta.
#
# Reconstruction is byte-exact BY CONSTRUCTION: keys in delta take the checkpoint's value;
# keys not in delta equal base's value, which (because delta captured every difference)
# equals the checkpoint's value. The tool still runs an explicit GOLD tensor-equality check
# before deleting anything, and deletes nothing unless every check passes.
#
# Usage:
#   prune_ckpts.py --fulls 'run/checkpoints/steps_*_pytorch_model.pt' \
#                  --base  _head_sweep_tools/vlm_base_<fam>.pt \
#                  --heads run/heads --keep <best_full_path> [--apply]
#   --base missing  -> built once by slicing the frozen (== shared) keys from --keep.
#   no --apply      -> DRY RUN: report savings + GOLD, delete nothing.
#   --min-frozen F  -> if frozen fraction < F (default 0.5), declare "not a frozen-backbone
#                      run", skip extraction, keep best only (no delta written).
import argparse, glob, os, re, sys
import torch

def load(p):
    if p.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(p)
    return torch.load(p, map_location="cpu", mmap=True)

def save(sd, p):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    if p.endswith(".safetensors"):
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in sd.items()}, p)
    else:
        torch.save(sd, p)

def nbytes(sd, keys=None):
    return sum(sd[k].numel() * sd[k].element_size() for k in (keys or sd))

def delta_keys(full, base):
    # every key that is new or whose tensor differs from base
    return [k for k in full if (k not in base) or (full[k].shape != base[k].shape)
            or (not torch.equal(full[k], base[k]))]

def headname(full_p):
    # unique per-ckpt tag from the last path components (handles both steps_*.pt and
    # the */model.safetensors layout where every basename is identical)
    stem = re.sub(r"\.(pt|safetensors)$", "", full_p)
    parts = stem.split(os.sep)
    tag = "_".join(parts[-3:]) if len(parts) >= 3 else "_".join(parts)
    return re.sub(r"[^0-9A-Za-z_]", "_", tag) + "_head.pt"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fulls", required=True, help="glob of full ckpts")
    ap.add_argument("--keep", required=True, help="the best full to keep intact")
    ap.add_argument("--base", required=True, help="frozen base path (built from --keep if absent)")
    ap.add_argument("--heads", required=True, help="dir to write per-step deltas")
    ap.add_argument("--min-frozen", type=float, default=0.5)
    ap.add_argument("--apply", action="store_true", help="actually delete non-best fulls")
    a = ap.parse_args()

    # dedup by realpath so a `last`-style symlink pointing at a numbered dir isn't processed
    # or deleted twice (dangling after its target is removed)
    seen, fulls = set(), []
    for f in sorted(set(glob.glob(a.fulls)) | {a.keep}):
        rp = os.path.realpath(f)
        if os.path.exists(f) and rp not in seen:
            seen.add(rp); fulls.append(f)
    if not fulls:
        print("no fulls match", a.fulls); sys.exit(1)
    print(f"=== prune {os.path.dirname(a.fulls)} | {len(fulls)} fulls | keep={os.path.basename(a.keep)} ===")

    # 1. base: frozen tensors == those of --keep that are NOT in its own delta-vs-itself.
    #    We can't diff keep against itself, so base = keep minus the *trainable* set, which we
    #    learn from the first OTHER full. If only one full exists, base = keep entirely (no save).
    keep_sd = load(a.keep)
    others = [f for f in fulls if os.path.abspath(f) != os.path.abspath(a.keep)]
    if not os.path.exists(a.base):
        if not others:
            print("[base] only one full -> nothing to extract, keeping it whole."); return
        ref = load(others[0])
        train = set(delta_keys(keep_sd, ref))          # keys that move between two steps = trainable
        base = {k: v.clone() for k, v in keep_sd.items() if k not in train}
        frozen_frac = nbytes(base) / max(nbytes(keep_sd), 1)
        print(f"[base] trainable={len(train)} frozen={len(base)} frozen_frac={frozen_frac:.3f}")
        if frozen_frac < a.min_frozen:
            print(f"[SKIP] frozen_frac<{a.min_frozen}: not a frozen-backbone run -> keep-best only, "
                  f"delete non-best fulls manually (no delta worth storing)."); return
        save(base, a.base)
        print(f"[base] wrote {a.base} {nbytes(base)/1e9:.1f}G")
        del ref, base
    base = load(a.base)
    bbig = sorted(base, key=lambda k: base[k].numel(), reverse=True)[:3]

    # 2. extract + GOLD-verify a delta for every full
    ok = {}
    for f in fulls:
        sd = load(f)
        dk = delta_keys(sd, base)
        delta = {k: sd[k].clone() for k in dk}
        recon = {**base, **delta}
        gold = set(recon) == set(sd) and all(torch.equal(recon[k], sd[k]) for k in sd)
        hp = os.path.join(a.heads, headname(f))
        if gold and delta:
            save(delta, hp)
            print(f"[ok ] {os.path.basename(f):40s} delta={len(delta)}k {nbytes(delta)/1e9:.2f}G GOLD✓ -> {os.path.basename(hp)}")
            ok[f] = True
        else:
            print(f"[FAIL] {os.path.basename(f)} gold={gold} deltaN={len(delta)} — keeping full")
            ok[f] = False
        del sd, delta, recon

    # 3. delete non-best fulls only if ALL verified
    if not all(ok.values()):
        print("[SKIP-DELETE] some extraction failed verification — no fulls deleted"); return
    if not a.apply:
        save_g = nbytes(load(others[0])) / 1e9 * len(others) if others else 0
        print(f"[DRY-RUN] all GOLD✓. --apply would delete {len(others)} non-best fulls (~{save_g:.0f}G). "); return
    freed = 0
    for f in others:
        if not os.path.exists(f):
            continue
        sz = os.path.getsize(f); os.remove(f); freed += sz
        print(f"[del ] {f} (-{sz/1e9:.1f}G)")
    print(f"[done] freed {freed/1e9:.1f}G (kept {os.path.basename(a.keep)} + {len(fulls)} deltas + base)")

if __name__ == "__main__":
    main()
