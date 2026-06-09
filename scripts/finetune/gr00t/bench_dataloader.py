"""Micro-benchmark for GR00T dataloader CPU decode bottleneck.

Measures three paths in isolation:
  A. torchcodec CPU decode (current production path)
  B. raw np.load(mmap_mode='r') from pre-decoded .npy (target path)
  C. multi-worker DataLoader simulating real training fanout

Outputs frames/sec + per-call latency. No model load.

Usage:
    cd dependencies/Isaac-GR00T && uv run --no-sync python \
        LeIsaac/scripts/finetune/gr00t/bench_dataloader.py \
        --mode {baseline|memmap|workers} --num_samples 500
"""
from __future__ import annotations

import os

import argparse
import time
from pathlib import Path

import numpy as np
import torchcodec
from torch.utils.data import DataLoader, Dataset

DATASET = Path(os.environ.get("LEISAAC_DATASET", os.path.join(os.path.dirname(__file__), "../../../datasets/v2-gr00t/leisaac-pick-orange")))
CACHE = Path(os.environ.get("FRAME_CACHE", "/tmp/leisaac_pick_orange_frames"))  # NVMe; /dev/shm only 32G
VIDEO_TMPL = "videos/chunk-000/observation.images.{cam}/episode_{ep:06d}.mp4"
CAMS = ["front", "wrist"]
N_EP = 60


def get_video_path(ep: int, cam: str) -> Path:
    return DATASET / VIDEO_TMPL.format(cam=cam, ep=ep)


# -----------------------------------------------------------------------------
# A. baseline torchcodec CPU decode
# -----------------------------------------------------------------------------
def bench_baseline(num_samples: int, frames_per_call: int = 16) -> None:
    """Mirror real training: each "sample" picks an episode + N frame indices, decodes."""
    rng = np.random.default_rng(0)
    n_frames_total = 0
    t0 = time.perf_counter()
    for i in range(num_samples):
        ep = int(rng.integers(0, N_EP))
        cam = CAMS[i % len(CAMS)]
        path = get_video_path(ep, cam)
        # Decode 16 random frames (like a shard slice would)
        dec = torchcodec.decoders.VideoDecoder(
            str(path), device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )
        n_total = dec.metadata.num_frames
        idx = rng.choice(n_total, size=min(frames_per_call, n_total), replace=False).astype(np.int64)
        frames = dec.get_frames_at(indices=idx).data.numpy()
        n_frames_total += frames.shape[0]
    dt = time.perf_counter() - t0
    print(f"[baseline] {num_samples} samples × {frames_per_call} frame = {n_frames_total} frames in {dt:.2f}s")
    print(f"[baseline] fps = {n_frames_total/dt:.1f}  call_latency = {dt/num_samples*1000:.1f}ms")


# -----------------------------------------------------------------------------
# B. memmap path (assumes precache already ran)
# -----------------------------------------------------------------------------
def bench_memmap(num_samples: int, frames_per_call: int = 16) -> None:
    if not CACHE.exists():
        raise RuntimeError(f"Cache {CACHE} missing. Run --mode precache first.")
    # Pre-open all memmaps once (workers would do this on first access)
    mmaps: dict[tuple[int, str], np.memmap] = {}
    for ep in range(N_EP):
        for cam in CAMS:
            p = CACHE / f"ep{ep:03d}_{cam}.npy"
            if p.exists():
                mmaps[(ep, cam)] = np.load(p, mmap_mode="r")

    rng = np.random.default_rng(0)
    n_frames_total = 0
    t0 = time.perf_counter()
    for i in range(num_samples):
        ep = int(rng.integers(0, N_EP))
        cam = CAMS[i % len(CAMS)]
        mm = mmaps[(ep, cam)]
        idx = rng.choice(mm.shape[0], size=min(frames_per_call, mm.shape[0]), replace=False).astype(np.int64)
        # Fancy index → copy (not view) — mirrors what real consumer needs
        frames = mm[idx].copy()
        n_frames_total += frames.shape[0]
    dt = time.perf_counter() - t0
    print(f"[memmap]   {num_samples} samples × {frames_per_call} frame = {n_frames_total} frames in {dt:.2f}s")
    print(f"[memmap]   fps = {n_frames_total/dt:.1f}  call_latency = {dt/num_samples*1000:.1f}ms")


# -----------------------------------------------------------------------------
# C. multi-worker DataLoader (more realistic)
# -----------------------------------------------------------------------------
class DecodeDataset(Dataset):
    def __init__(self, n: int, frames_per_call: int, use_memmap: bool):
        self.n = n
        self.fpc = frames_per_call
        self.use_memmap = use_memmap
        # Pre-resolve all video paths once
        self.specs = []
        rng = np.random.default_rng(0)
        for i in range(n):
            ep = int(rng.integers(0, N_EP))
            cam = CAMS[i % len(CAMS)]
            self.specs.append((ep, cam))

    def __len__(self):
        return self.n

    def __getitem__(self, i: int):
        ep, cam = self.specs[i]
        rng = np.random.default_rng(i)  # deterministic per-worker
        if self.use_memmap:
            mm = np.load(CACHE / f"ep{ep:03d}_{cam}.npy", mmap_mode="r")
            n_total = mm.shape[0]
            idx = rng.choice(n_total, size=min(self.fpc, n_total), replace=False).astype(np.int64)
            return mm[idx].copy()
        else:
            path = get_video_path(ep, cam)
            dec = torchcodec.decoders.VideoDecoder(
                str(path), device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
            )
            n_total = dec.metadata.num_frames
            idx = rng.choice(n_total, size=min(self.fpc, n_total), replace=False).astype(np.int64)
            return dec.get_frames_at(indices=idx).data.numpy()


def bench_workers(num_samples: int, frames_per_call: int, num_workers: int, use_memmap: bool) -> None:
    ds = DecodeDataset(num_samples, frames_per_call, use_memmap)
    loader = DataLoader(
        ds,
        batch_size=4,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        collate_fn=lambda batch: batch,  # don't stack, just pass through
    )
    n_frames_total = 0
    t0 = time.perf_counter()
    for batch in loader:
        for frames in batch:
            n_frames_total += frames.shape[0]
    dt = time.perf_counter() - t0
    tag = f"memmap" if use_memmap else f"decode"
    print(f"[workers/{num_workers}/{tag}] {num_samples} samples × {frames_per_call} frame = {n_frames_total} frames in {dt:.2f}s")
    print(f"[workers/{num_workers}/{tag}] fps = {n_frames_total/dt:.1f}  call_latency = {dt/num_samples*1000:.1f}ms")


# -----------------------------------------------------------------------------
# precache step (mode=precache)
# -----------------------------------------------------------------------------
def precache() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    t0 = time.perf_counter()
    for ep in range(N_EP):
        for cam in CAMS:
            out = CACHE / f"ep{ep:03d}_{cam}.npy"
            if out.exists():
                total_bytes += out.stat().st_size
                continue
            path = get_video_path(ep, cam)
            if not path.exists():
                print(f"[precache] skip missing {path}")
                continue
            dec = torchcodec.decoders.VideoDecoder(
                str(path), device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
            )
            n = dec.metadata.num_frames
            frames = dec.get_frames_at(indices=np.arange(n, dtype=np.int64)).data.numpy()
            np.save(out, frames)
            total_bytes += out.stat().st_size
            print(f"[precache] ep{ep:03d}_{cam}: {n} frames {frames.shape} -> {out.stat().st_size/1024**2:.1f} MB")
    dt = time.perf_counter() - t0
    print(f"[precache] done {total_bytes/1024**3:.2f} GB in {dt:.1f}s")


# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["baseline", "memmap", "precache", "workers-decode", "workers-memmap"])
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--frames_per_call", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    if args.mode == "baseline":
        bench_baseline(args.num_samples, args.frames_per_call)
    elif args.mode == "memmap":
        bench_memmap(args.num_samples, args.frames_per_call)
    elif args.mode == "precache":
        precache()
    elif args.mode == "workers-decode":
        bench_workers(args.num_samples, args.frames_per_call, args.num_workers, use_memmap=False)
    elif args.mode == "workers-memmap":
        bench_workers(args.num_samples, args.frames_per_call, args.num_workers, use_memmap=True)


if __name__ == "__main__":
    main()
