"""HF Trainer + DataLoader perf patches — overlap CPU dataloader with GPU compute.

Importable from any launcher (GR00T / DreamZero / π0.5 / X-VLA / SmolVLA / ACT / DP / OpenVLA).
The patches are idempotent and env-gated so a single import in your launcher header is enough.

Three patches:
  1. non_blocking H2D       — Trainer._prepare_input → data.to(device, non_blocking=True)
     Effect: GPU util mid +5-11 pp (depends on prior overlap)
     Requires HF dataloader_pin_memory=True (default).

  2. prefetch_factor bump   — DataLoader.prefetch_factor 2 → 4 (configurable)
     Effect: marginal on tiny datasets. Set to 4, not 8 (8 regressed in our bench).

  3. phase profiler         — env PROFILE_PHASES=1 prints collator + custom-method timing
     Effect: zero perf impact, prints per-20-step breakdown to stdout.

Usage in any launcher:
    from LeIsaac.scripts.training.perf.pipeline_patches import apply_all
    apply_all()

Env knobs:
    PIPELINE_OVERLAP_DISABLE=1         skip non_blocking patch
    PREFETCH_FACTOR_DISABLE=1          skip prefetch bump
    DATALOADER_PREFETCH_FACTOR=4       prefetch buffer per worker (default 4)
    PROFILE_PHASES=1                   enable phase profiler
    PROFILE_TARGETS=cls.method,...     comma-separated method paths to time
                                       (default: gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.Gr00tN1d7DataCollator.__call__)

See [[feedback-gpu-util-as-efficiency-anchor.md]] for the mental model and the
LeIsaac/docs/training/gpu_dataloader_zero_copy.html design doc.
"""
from __future__ import annotations

import importlib
import os
import statistics
import time
from collections import defaultdict
from collections.abc import Mapping


_APPLIED = set()


def patch_non_blocking_h2d() -> None:
    """Trainer._prepare_input: data.to(device) → data.to(device, non_blocking=True)."""
    if "non_blocking_h2d" in _APPLIED or os.environ.get("PIPELINE_OVERLAP_DISABLE", "0") == "1":
        return
    import torch
    from transformers import Trainer

    _orig = Trainer._prepare_input

    def _patched(self, data):
        if isinstance(data, Mapping):
            return type(data)({k: _patched(self, v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(_patched(self, v) for v in data)
        elif isinstance(data, torch.Tensor):
            if data.device.type == "cpu" and self.args.device.type == "cuda":
                return data.to(self.args.device, non_blocking=True)
            return data.to(self.args.device)
        return data

    Trainer._prepare_input = _patched
    _APPLIED.add("non_blocking_h2d")
    print("[perf] Trainer._prepare_input → non_blocking=True (H2D pipeline overlap)", flush=True)


def patch_dataloader_prefetch_factor(default: int = 4) -> None:
    """Bump DataLoader.prefetch_factor.

    Works by monkey-patching torch.utils.data.DataLoader.__init__ so all Trainers benefit.
    """
    if "prefetch" in _APPLIED or os.environ.get("PREFETCH_FACTOR_DISABLE", "0") == "1":
        return
    pf = int(os.environ.get("DATALOADER_PREFETCH_FACTOR", str(default)))
    import torch.utils.data as _td

    _orig_init = _td.DataLoader.__init__

    def _patched_init(self, *args, **kwargs):
        # Only set if user didn't specify
        if "prefetch_factor" not in kwargs:
            nw = kwargs.get("num_workers", 0)
            if nw and nw > 0:
                kwargs["prefetch_factor"] = pf
        _orig_init(self, *args, **kwargs)

    _td.DataLoader.__init__ = _patched_init
    _APPLIED.add("prefetch")
    print(f"[perf] DataLoader.prefetch_factor → {pf} (when num_workers>0 + not user-set)", flush=True)


_PROFILE_TIMES: dict[str, list[float]] = defaultdict(list)


def patch_phase_profiler(targets: list[str] | None = None) -> None:
    """Wrap target methods with timing; print mean/p90 every 20 calls.

    targets: list of "<module>.<class>.<method>" strings. If None, read from PROFILE_TARGETS env
    or fall back to GR00T-N1.7 collator.
    """
    if "profile" in _APPLIED or os.environ.get("PROFILE_PHASES", "0") != "1":
        return
    if targets is None:
        env_targets = os.environ.get("PROFILE_TARGETS", "")
        if env_targets.strip():
            targets = [t.strip() for t in env_targets.split(",") if t.strip()]
        else:
            targets = [
                "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.Gr00tN1d7DataCollator.__call__",
                "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.Gr00tN1d7Processor._get_vlm_inputs",
            ]

    for tgt in targets:
        try:
            mod_path, cls_name, meth_name = tgt.rsplit(".", 2)
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            orig = getattr(cls, meth_name)
        except Exception as e:
            print(f"[perf] skip profile target {tgt!r}: {e}", flush=True)
            continue

        def _make_wrap(label, fn):
            def _wrap(self, *a, **kw):
                t0 = time.perf_counter()
                out = fn(self, *a, **kw)
                _PROFILE_TIMES[label].append((time.perf_counter() - t0) * 1000)
                vals = _PROFILE_TIMES[label]
                if len(vals) % 20 == 0:
                    v = vals[-50:]
                    p90 = sorted(v)[max(0, int(0.9 * len(v)) - 1)]
                    print(
                        f"[perf-profile] {label}: n={len(vals)} mean50={statistics.mean(v):.1f}ms p90={p90:.1f}ms",
                        flush=True,
                    )
                return out
            return _wrap

        setattr(cls, meth_name, _make_wrap(f"{cls_name}.{meth_name}", orig))
        print(f"[perf] profiling {cls_name}.{meth_name}", flush=True)

    _APPLIED.add("profile")


def apply_all() -> None:
    """Idempotent: apply all enabled patches."""
    patch_non_blocking_h2d()
    patch_dataloader_prefetch_factor()
    patch_phase_profiler()
