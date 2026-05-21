"""LISA (Layer-wise Importance Sampled AdamW) callback for HF Trainer.

Paper: Pan et al. 2024, "LISA: Layerwise Importance Sampling for Memory-Efficient
Large Language Model Fine-Tuning"  (arXiv:2403.17919)

Idea: each step, randomly activate K of the L trainable transformer blocks.
The other (L-K) blocks have requires_grad=False, so Adam doesn't allocate
state for them on that step.  Over time this matches full-FT quality while
drastically reducing peak memory.

For our GR00T-N1.6 finetune on 4090:
  - Total DiT transformer_blocks ≈ 32 (alternating self + cross attn).
  - With K=2 activated, Adam state shrinks to ~2/32 = 6% of full-FT.
  - Projector + top-4 LLM layers always on (small enough to stay).
  - We can disable gradient_checkpointing entirely — that's the goal:
    avoid use_reentrant codepath crashes.

Usage:
    from lisa_callback import LISACallback
    trainer.add_callback(LISACallback(model=model, k=2,
                                       block_attr_paths=["_groot_model.model.action_head.transformer.transformer_blocks"],
                                       resample_every=1))
"""
from __future__ import annotations

import random
from typing import List, Optional

from transformers import TrainerCallback


def _resolve_attr(obj, dotted: str):
    cur = obj
    for part in dotted.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


class LISACallback(TrainerCallback):
    """Layer-wise random activation between train steps.

    Args:
        model: the actual nn.Module being trained (NOT the HF wrapper).
        k: how many blocks to activate each step.
        block_attr_paths: dotted attribute paths from `model` to ModuleList(s)
            of transformer blocks. e.g. ["action_head.dit.transformer_blocks"].
            Multiple paths are concatenated into a single flat block list.
        resample_every: re-sample new K active blocks every N optimizer steps.
            1 = every step (LISA paper default), 10 = once per 10 step (cheaper
            requires_grad flipping but less per-step variance).
        seed: RNG seed for reproducible sampling.
    """

    def __init__(
        self,
        model,
        k: int = 2,
        block_attr_paths: Optional[List[str]] = None,
        resample_every: int = 1,
        seed: int = 0,
        verbose_every: int = 50,
    ):
        super().__init__()
        self.model = model
        self.k = k
        self.resample_every = max(1, int(resample_every))
        self.rng = random.Random(seed)
        self.verbose_every = verbose_every
        self._step_counter = 0

        # Collect all candidate blocks (one flat list across all listed paths)
        self.blocks = []
        for path in block_attr_paths or []:
            mod = _resolve_attr(model, path)
            if mod is None:
                print(f"[LISA] WARN: path '{path}' not found, skipping")
                continue
            for i, blk in enumerate(mod):
                self.blocks.append((f"{path}[{i}]", blk))

        if not self.blocks:
            print("[LISA] WARN: no blocks resolved — callback will be a no-op")
        else:
            total = len(self.blocks)
            k_eff = min(self.k, total)
            print(f"[LISA] {total} candidate blocks, sampling K={k_eff} per step (resample every {self.resample_every} step)")

        # On init: also flip all blocks to requires_grad=False as the "base" state.
        # The on_step_begin hook will re-enable the sampled subset.
        for _, blk in self.blocks:
            for p in blk.parameters():
                p.requires_grad = False

    def _sample_active_indices(self):
        n = len(self.blocks)
        if n == 0:
            return []
        return self.rng.sample(range(n), k=min(self.k, n))

    def on_step_begin(self, args, state, control, **kwargs):
        if not self.blocks:
            return
        if self._step_counter % self.resample_every == 0:
            active = set(self._sample_active_indices())
            for i, (_, blk) in enumerate(self.blocks):
                on = i in active
                for p in blk.parameters():
                    p.requires_grad = on
            if self._step_counter % self.verbose_every == 0:
                names = [self.blocks[i][0] for i in sorted(active)]
                print(f"[LISA] step={state.global_step}: active blocks = {names}", flush=True)
        self._step_counter += 1
