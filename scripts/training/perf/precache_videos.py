"""Pre-decode all videos in a LeRobot v2.x dataset to .npy memmap cache.

Eliminates per-step CPU video decode at training time.
17× single-thread fps gain on GR00T-N1.7 PickOrange (167 → 2862 fps with handle cache).
See [[feedback-gpu-util-as-efficiency-anchor.md]] for the mental model.

Generic — works for any LeRobot v2.x dataset with mp4 videos and standard chunk-{NNN}
layout. Used by GR00T, DreamZero, X-VLA, SmolVLA, π0.5, ACT, DP training pipelines.

Cache layout:
    <cache_dir>/ep{episode_index:03d}_{cam_name}.npy   (T, H, W, 3) uint8

Companion env var: LEISAAC_FRAME_CACHE_DIR (auto-loaded by
gr00t/data/dataset/lerobot_episode_loader.py:_load_video_data fast-path).

Usage:
    python precache_videos.py \
        --dataset_dir /path/to/lerobot_dataset \
        --cache_dir /path/to/cache/<task>_frames \
        --workers 8
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def _list_video_files(dataset_dir: Path) -> list[tuple[int, str, Path]]:
    """Return list of (episode_index, cam_name, video_path) for all videos."""
    info_path = dataset_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    n_ep = info["total_episodes"]
    chunks_size = info.get("chunks_size", 1000)
    video_tmpl = info["video_path"]  # e.g. videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4

    # Discover cam names by scanning chunk-000 dir
    chunk0 = dataset_dir / "videos" / "chunk-000"
    if not chunk0.is_dir():
        raise SystemExit(f"missing {chunk0}")
    cams_keys = sorted([p.name for p in chunk0.iterdir() if p.is_dir()])
    # cams_keys are "observation.images.<cam_name>"; reduce to <cam_name>
    cams = []
    for key in cams_keys:
        m = re.match(r"^observation\.images\.(.+)$", key)
        cams.append((key, m.group(1) if m else key))

    out = []
    for ep in range(n_ep):
        ck = ep // chunks_size
        for video_key, cam_name in cams:
            rel = video_tmpl.format(episode_chunk=ck, video_key=video_key, episode_index=ep)
            out.append((ep, cam_name, dataset_dir / rel))
    return out


def _decode_one(args: tuple[Path, Path, int, str]) -> tuple[int, str, int, str]:
    video_path, out_path, ep, cam = args
    if out_path.exists() and out_path.stat().st_size > 1024:
        return ep, cam, out_path.stat().st_size, "skip"
    if not video_path.exists():
        return ep, cam, 0, f"MISS:{video_path}"
    import torchcodec
    dec = torchcodec.decoders.VideoDecoder(
        str(video_path), device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
    )
    n = dec.metadata.num_frames
    arr = dec.get_frames_at(indices=np.arange(n, dtype=np.int64)).data.numpy()
    np.save(out_path, arr)
    return ep, cam, out_path.stat().st_size, f"{n}f"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", type=Path, required=True)
    p.add_argument("--cache_dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    videos = _list_video_files(args.dataset_dir)
    if not videos:
        raise SystemExit("no videos found")
    print(f"[precache] {args.dataset_dir} -> {args.cache_dir}", flush=True)
    print(f"[precache] {len(videos)} files, workers={args.workers}", flush=True)

    jobs = [(vp, args.cache_dir / f"ep{ep:03d}_{cam}.npy", ep, cam) for ep, cam, vp in videos]
    t0 = time.perf_counter()
    total = 0
    n_ok = n_skip = n_miss = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for ep, cam, sz, status in ex.map(_decode_one, jobs, chunksize=1):
            total += sz
            if status == "skip":
                n_skip += 1
            elif status.startswith("MISS"):
                n_miss += 1
                print(f"[precache] {status}", flush=True)
            else:
                n_ok += 1
            if (n_ok + n_skip) % 20 == 0:
                print(f"[precache] {n_ok + n_skip}/{len(videos)} {total/1024**3:.1f} GB", flush=True)

    dt = time.perf_counter() - t0
    print(f"[precache] done: {n_ok} new, {n_skip} skipped, {n_miss} missing", flush=True)
    print(f"[precache] {total/1024**3:.2f} GB in {dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
