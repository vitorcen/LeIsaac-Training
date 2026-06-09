"""Micro-bench: isolate dataloader throughput from model compute.

Generic for any LeRobot v2.x dataset. Five modes:
    baseline        — current torchcodec CPU decode path
    memmap          — precache memmap with re-open per call
    memmap-cached   — precache memmap with persistent handle (best)
    workers-decode  — 4-worker DataLoader through torchcodec
    workers-memmap  — 4-worker DataLoader through memmap

Use to localise the decode portion of CPU bottleneck before / after attacking it.

Usage:
    python bench_dataloader.py \
        --dataset_dir /path/to/lerobot_dataset \
        --cache_dir   /path/to/cache/<task>_frames   # for memmap modes
        --mode {baseline,memmap,memmap-cached,workers-decode,workers-memmap} \
        --num_samples 400 --frames_per_call 16
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torchcodec
from torch.utils.data import DataLoader, Dataset


def _discover(dataset_dir: Path) -> tuple[int, list[str], dict[str, str]]:
    """Return (n_episodes, cam_short_names, cam_short→video_key map)."""
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    n_ep = info["total_episodes"]
    chunk0 = dataset_dir / "videos" / "chunk-000"
    cam_map = {}
    for p in sorted(chunk0.iterdir()):
        if not p.is_dir():
            continue
        key = p.name  # observation.images.<cam>
        m = re.match(r"^observation\.images\.(.+)$", key)
        cam = m.group(1) if m else key
        cam_map[cam] = key
    return n_ep, list(cam_map.keys()), cam_map


def _video_path(dataset_dir: Path, ep: int, cam_key: str) -> Path:
    return dataset_dir / "videos" / "chunk-000" / cam_key / f"episode_{ep:06d}.mp4"


def bench_baseline(ds: Path, n_ep: int, cams: list[str], cam_map: dict[str, str], n_samples: int, fpc: int):
    rng = np.random.default_rng(0)
    total = 0
    t0 = time.perf_counter()
    for i in range(n_samples):
        ep = int(rng.integers(0, n_ep))
        cam = cams[i % len(cams)]
        dec = torchcodec.decoders.VideoDecoder(
            str(_video_path(ds, ep, cam_map[cam])), device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )
        n = dec.metadata.num_frames
        idx = rng.choice(n, size=min(fpc, n), replace=False).astype(np.int64)
        total += dec.get_frames_at(indices=idx).data.numpy().shape[0]
    dt = time.perf_counter() - t0
    print(f"[baseline]      fps={total/dt:.1f} call_ms={dt/n_samples*1000:.1f}")


def bench_memmap(cache: Path, n_ep: int, cams: list[str], n_samples: int, fpc: int, cached: bool):
    if cached:
        mmaps = {}
        for ep in range(n_ep):
            for cam in cams:
                p = cache / f"ep{ep:03d}_{cam}.npy"
                if p.exists():
                    mmaps[(ep, cam)] = np.load(p, mmap_mode="r")
    rng = np.random.default_rng(0)
    total = 0
    t0 = time.perf_counter()
    for i in range(n_samples):
        ep = int(rng.integers(0, n_ep))
        cam = cams[i % len(cams)]
        if cached:
            mm = mmaps.get((ep, cam))
            if mm is None:
                continue
        else:
            p = cache / f"ep{ep:03d}_{cam}.npy"
            mm = np.load(p, mmap_mode="r")
        idx = rng.choice(mm.shape[0], size=min(fpc, mm.shape[0]), replace=False).astype(np.int64)
        total += mm[idx].copy().shape[0]
    dt = time.perf_counter() - t0
    tag = "memmap-cached " if cached else "memmap        "
    print(f"[{tag.strip():<14}] fps={total/dt:.1f} call_ms={dt/n_samples*1000:.1f}")


class _BenchDS(Dataset):
    def __init__(self, ds, cache, n_ep, cams, cam_map, n_samples, fpc, use_memmap):
        self.ds, self.cache = ds, cache
        self.n_ep, self.cams, self.cam_map = n_ep, cams, cam_map
        self.fpc, self.use_memmap = fpc, use_memmap
        rng = np.random.default_rng(0)
        self.specs = [(int(rng.integers(0, n_ep)), cams[i % len(cams)]) for i in range(n_samples)]

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, i):
        ep, cam = self.specs[i]
        rng = np.random.default_rng(i)
        if self.use_memmap:
            mm = np.load(self.cache / f"ep{ep:03d}_{cam}.npy", mmap_mode="r")
            idx = rng.choice(mm.shape[0], size=min(self.fpc, mm.shape[0]), replace=False).astype(np.int64)
            return mm[idx].copy()
        else:
            dec = torchcodec.decoders.VideoDecoder(
                str(_video_path(self.ds, ep, self.cam_map[cam])), device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
            )
            n = dec.metadata.num_frames
            idx = rng.choice(n, size=min(self.fpc, n), replace=False).astype(np.int64)
            return dec.get_frames_at(indices=idx).data.numpy()


def bench_workers(ds, cache, n_ep, cams, cam_map, n_samples, fpc, num_workers, use_memmap):
    dataset = _BenchDS(ds, cache, n_ep, cams, cam_map, n_samples, fpc, use_memmap)
    loader = DataLoader(
        dataset, batch_size=4, num_workers=num_workers, prefetch_factor=4,
        persistent_workers=num_workers > 0, collate_fn=lambda b: b,
    )
    total = 0
    t0 = time.perf_counter()
    for batch in loader:
        for frames in batch:
            total += frames.shape[0]
    dt = time.perf_counter() - t0
    tag = "memmap" if use_memmap else "decode"
    print(f"[workers/{num_workers}/{tag}] fps={total/dt:.1f} call_ms={dt/n_samples*1000:.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", type=Path, required=True)
    p.add_argument("--cache_dir", type=Path, default=None)
    p.add_argument("--mode", required=True,
                   choices=["baseline", "memmap", "memmap-cached",
                            "workers-decode", "workers-memmap"])
    p.add_argument("--num_samples", type=int, default=400)
    p.add_argument("--frames_per_call", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    n_ep, cams, cam_map = _discover(args.dataset_dir)
    print(f"[discover] {args.dataset_dir.name}: {n_ep} ep × {len(cams)} cam {cams}")

    if args.mode == "baseline":
        bench_baseline(args.dataset_dir, n_ep, cams, cam_map, args.num_samples, args.frames_per_call)
    elif args.mode == "memmap":
        bench_memmap(args.cache_dir, n_ep, cams, args.num_samples, args.frames_per_call, cached=False)
    elif args.mode == "memmap-cached":
        bench_memmap(args.cache_dir, n_ep, cams, args.num_samples, args.frames_per_call, cached=True)
    elif args.mode == "workers-decode":
        bench_workers(args.dataset_dir, args.cache_dir, n_ep, cams, cam_map,
                      args.num_samples, args.frames_per_call, args.num_workers, use_memmap=False)
    elif args.mode == "workers-memmap":
        bench_workers(args.dataset_dir, args.cache_dir, n_ep, cams, cam_map,
                      args.num_samples, args.frames_per_call, args.num_workers, use_memmap=True)


if __name__ == "__main__":
    main()
