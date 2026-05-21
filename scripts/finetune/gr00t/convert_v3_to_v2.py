"""Minimal LeRobot v3.0 → v2.1 converter for Isaac-GR00T compatibility.

Isaac-GR00T's data loader expects v2.1 layout (per-episode .parquet + per-episode .mp4).
LeRobot v3.0 packs multiple episodes per parquet/mp4 file and stores per-episode
metadata in meta/episodes/chunk-*/file-*.parquet. We split here.

Tested on LeIsaac PickOrange (60 episodes, ~26s avg, front+wrist cameras).
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import jsonlines
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import tqdm


V21 = "v2.1"
V30 = "v3.0"
V2_DATA_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V2_VIDEO_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
DEFAULT_CHUNK_SIZE = 1000


def _to_native(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [_to_native(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_native(v) for k, v in x.items()}
    return x


def _unflatten(flat: dict, prefix: str = "stats/") -> dict:
    """Turn {'stats/action/min': [...]} → {'action': {'min': [...]}}."""
    out: dict = {}
    for k, v in flat.items():
        if not k.startswith(prefix):
            continue
        parts = k[len(prefix):].split("/")
        d = out
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = _to_native(v)
    return out


def load_episode_records(root: Path) -> list[dict]:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    rows: list[dict] = []
    for f in files:
        rows.extend(pq.read_table(f).to_pylist())
    rows.sort(key=lambda r: int(r["episode_index"]))
    return rows


def convert_tasks(root: Path, new_root: Path) -> None:
    tasks_pq = root / "meta" / "tasks.parquet"
    tasks_jsonl = new_root / "meta" / "tasks.jsonl"
    tasks_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if tasks_pq.exists():
        df = pq.read_table(tasks_pq).to_pandas().reset_index()
        with jsonlines.open(tasks_jsonl, "w") as w:
            for _, row in df.iterrows():
                w.write({"task_index": int(row["task_index"]), "task": str(row["task"])})
    else:
        raise FileNotFoundError(f"v3 tasks file missing: {tasks_pq}")


def convert_data(root: Path, new_root: Path, records: list[dict], chunk_size: int) -> None:
    """Slice the v3 master parquet(s) into per-episode v2 files."""
    info = json.loads((root / "meta" / "info.json").read_text())
    data_path_pattern = info["data_path"]  # v3 pattern

    # Cache loaded master parquets by (chunk_idx, file_idx)
    cache: dict[tuple[int, int], pd.DataFrame] = {}

    for rec in tqdm.tqdm(records, desc="data"):
        ep_idx = int(rec["episode_index"])
        chunk_i = int(rec["data/chunk_index"])
        file_i = int(rec["data/file_index"])
        from_i = int(rec["dataset_from_index"])
        to_i = int(rec["dataset_to_index"])

        if (chunk_i, file_i) not in cache:
            pq_path = root / data_path_pattern.format(chunk_index=chunk_i, file_index=file_i)
            cache[(chunk_i, file_i)] = pq.read_table(pq_path).to_pandas()
        df = cache[(chunk_i, file_i)]

        ep_df = df.iloc[from_i:to_i].copy().reset_index(drop=True)

        out_chunk = ep_idx // chunk_size
        out_path = new_root / V2_DATA_TEMPLATE.format(episode_chunk=out_chunk, episode_index=ep_idx)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ep_df.to_parquet(out_path, index=False)


def convert_videos(root: Path, new_root: Path, records: list[dict], video_keys: list[str], chunk_size: int) -> None:
    """Cut v3 chunk MP4s into per-episode v2 MP4s using ffmpeg."""
    info = json.loads((root / "meta" / "info.json").read_text())
    video_pattern = info["video_path"]  # v3 pattern

    for rec in tqdm.tqdm(records, desc="videos"):
        ep_idx = int(rec["episode_index"])
        out_chunk = ep_idx // chunk_size
        for vk in video_keys:
            chunk_i = int(rec[f"videos/{vk}/chunk_index"])
            file_i = int(rec[f"videos/{vk}/file_index"])
            t0 = float(rec[f"videos/{vk}/from_timestamp"])
            t1 = float(rec[f"videos/{vk}/to_timestamp"])
            dur = max(t1 - t0, 1e-6)

            src = root / video_pattern.format(video_key=vk, chunk_index=chunk_i, file_index=file_i)
            dst = new_root / V2_VIDEO_TEMPLATE.format(
                episode_chunk=out_chunk, video_key=vk, episode_index=ep_idx
            )
            dst.parent.mkdir(parents=True, exist_ok=True)

            # Stream copy when possible — fast + lossless.  Fall back to re-encode
            # if ffmpeg complains about keyframe alignment.
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{t0:.6f}", "-t", f"{dur:.6f}",
                "-i", str(src),
                "-c", "copy", "-avoid_negative_ts", "make_zero",
                str(dst),
            ]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0 or not dst.exists() or dst.stat().st_size < 1024:
                # re-encode fallback (forces fresh keyframes)
                cmd_re = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-ss", f"{t0:.6f}", "-i", str(src),
                    "-t", f"{dur:.6f}",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-an",
                    str(dst),
                ]
                r = subprocess.run(cmd_re, capture_output=True)
                if r.returncode != 0:
                    raise RuntimeError(
                        f"ffmpeg failed for ep={ep_idx} vk={vk}: {r.stderr.decode()[:400]}"
                    )


def convert_episodes_metadata(new_root: Path, records: list[dict]) -> None:
    eps_path = new_root / "meta" / "episodes.jsonl"
    stats_path = new_root / "meta" / "episodes_stats.jsonl"
    eps_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(eps_path, "w") as eps_w, jsonlines.open(stats_path, "w") as stats_w:
        for rec in records:
            ep_idx = int(rec["episode_index"])
            tasks = list(rec.get("tasks") or [])
            length = int(rec["length"])
            eps_w.write({"episode_index": ep_idx, "tasks": tasks, "length": length})

            stats_nested = _unflatten({k: v for k, v in rec.items() if k.startswith("stats/")})
            stats_w.write({"episode_index": ep_idx, "stats": stats_nested})


def convert_info(root: Path, new_root: Path, records: list[dict], video_keys: list[str]) -> None:
    info = json.loads((root / "meta" / "info.json").read_text())
    info["codebase_version"] = V21
    info["data_path"] = V2_DATA_TEMPLATE
    info["video_path"] = V2_VIDEO_TEMPLATE
    info["total_chunks"] = math.ceil(len(records) / info.get("chunks_size", DEFAULT_CHUNK_SIZE))
    (new_root / "meta").mkdir(parents=True, exist_ok=True)
    (new_root / "meta" / "info.json").write_text(json.dumps(info, indent=4))


def copy_global_stats(root: Path, new_root: Path) -> None:
    src = root / "meta" / "stats.json"
    if src.exists():
        (new_root / "meta").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, new_root / "meta" / "stats.json")


def copy_modality(root: Path, new_root: Path) -> None:
    src = root / "meta" / "modality.json"
    if src.exists():
        (new_root / "meta").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, new_root / "meta" / "modality.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="v3.0 dataset root")
    ap.add_argument("--dst", required=True, help="v2.1 output root (will be replaced)")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()

    info = json.loads((src / "meta" / "info.json").read_text())
    if info.get("codebase_version") != V30:
        raise SystemExit(f"src is not v3.0: codebase_version={info.get('codebase_version')}")

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    records = load_episode_records(src)
    chunk_size = info.get("chunks_size", DEFAULT_CHUNK_SIZE)
    video_keys = [k for k, ft in info["features"].items() if ft.get("dtype") == "video"]

    print(f"[v3→v2] src={src}")
    print(f"[v3→v2] dst={dst}")
    print(f"[v3→v2] episodes={len(records)}  video_keys={video_keys}  chunk_size={chunk_size}")

    convert_info(src, dst, records, video_keys)
    copy_global_stats(src, dst)
    copy_modality(src, dst)
    convert_tasks(src, dst)
    convert_data(src, dst, records, chunk_size)
    convert_videos(src, dst, records, video_keys, chunk_size)
    convert_episodes_metadata(dst, records)

    print(f"[v3→v2] done.  Wrote {len(records)} episodes to {dst}")


if __name__ == "__main__":
    main()
