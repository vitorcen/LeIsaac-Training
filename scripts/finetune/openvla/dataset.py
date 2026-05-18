"""LeRobot v3.0 PickOrange → OpenVLA training samples.

One sample = (front-camera frame, language prompt, normalized action) →
discrete-token target sequence for the OpenVLA action head.

Action tokenization mirrors the OpenVLA recipe exactly:
    a_norm = clip(2 * (a - q01) / (q99 - q01) - 1, -1, 1)
    bin    = argmin(|a_norm - bin_centers|)        # 256 bins in [-1, 1]
    token  = vocab_size - 1 - bin                  # reserve top of vocab

The training loss is computed over the action-token positions only;
prompt + image tokens have label = IGNORE_INDEX.
"""

from __future__ import annotations

import glob
import io
import os
import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = -100
N_ACTION_BINS = 256
ACTION_DIM = 6


@dataclass
class EpisodeRecord:
    episode_index: int
    length: int
    dataset_from_index: int
    dataset_to_index: int
    video_chunk_index: int
    video_file_index: int
    from_timestamp: float
    to_timestamp: float
    task: str


def load_episodes(dataset_root: str, video_key: str = "observation.images.front") -> List[EpisodeRecord]:
    meta_files = sorted(glob.glob(os.path.join(dataset_root, "meta/episodes/chunk-*/*.parquet")))
    if not meta_files:
        raise FileNotFoundError(f"No meta/episodes parquet under {dataset_root}")
    table = pq.read_table(meta_files[0])
    df = table.to_pandas()
    out: List[EpisodeRecord] = []
    for _, row in df.iterrows():
        task = row["tasks"]
        if isinstance(task, (list, np.ndarray)) and len(task) > 0:
            task = str(task[0])
        else:
            task = str(task)
        out.append(
            EpisodeRecord(
                episode_index=int(row["episode_index"]),
                length=int(row["length"]),
                dataset_from_index=int(row["dataset_from_index"]),
                dataset_to_index=int(row["dataset_to_index"]),
                video_chunk_index=int(row[f"videos/{video_key}/chunk_index"]),
                video_file_index=int(row[f"videos/{video_key}/file_index"]),
                from_timestamp=float(row[f"videos/{video_key}/from_timestamp"]),
                to_timestamp=float(row[f"videos/{video_key}/to_timestamp"]),
                task=task,
            )
        )
    return out


def load_actions_states(dataset_root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (actions, states, episode_index, frame_index), each row-aligned."""
    files = sorted(glob.glob(os.path.join(dataset_root, "data/chunk-*/*.parquet")))
    if not files:
        raise FileNotFoundError(f"No data parquet under {dataset_root}")
    actions_l: List[np.ndarray] = []
    states_l: List[np.ndarray] = []
    eps_l: List[np.ndarray] = []
    frames_l: List[np.ndarray] = []
    for fp in files:
        t = pq.read_table(fp, columns=["action", "observation.state", "episode_index", "frame_index"]).to_pandas()
        actions_l.append(np.stack(t["action"].to_numpy()))
        states_l.append(np.stack(t["observation.state"].to_numpy()))
        eps_l.append(t["episode_index"].to_numpy().astype(np.int32))
        frames_l.append(t["frame_index"].to_numpy().astype(np.int32))
    return (
        np.concatenate(actions_l).astype(np.float32),
        np.concatenate(states_l).astype(np.float32),
        np.concatenate(eps_l),
        np.concatenate(frames_l),
    )


def compute_action_stats(actions: np.ndarray) -> Dict[str, np.ndarray]:
    """OpenVLA-compatible (q01, q99) per dim + mean/std."""
    q01 = np.quantile(actions, 0.01, axis=0)
    q99 = np.quantile(actions, 0.99, axis=0)
    # Guard against zero-range dims (constant gripper etc.)
    rng = np.maximum(q99 - q01, 1e-6)
    return {
        "q01": q01.astype(np.float32),
        "q99": q99.astype(np.float32),
        "range": rng.astype(np.float32),
        "mean": actions.mean(axis=0).astype(np.float32),
        "std":  actions.std(axis=0).astype(np.float32),
        "min":  actions.min(axis=0).astype(np.float32),
        "max":  actions.max(axis=0).astype(np.float32),
        "mask": np.ones_like(q01, dtype=bool),
    }


def normalize_action(a: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Map [q01, q99] → [-1, 1], clipped."""
    return np.clip(2.0 * (a - stats["q01"]) / stats["range"] - 1.0, -1.0, 1.0)


class ActionTokenizer:
    """OpenVLA discrete action tokenizer: 256 bins in [-1, 1]."""

    def __init__(self, tokenizer_vocab_size: int, n_bins: int = N_ACTION_BINS):
        self.vocab_size = tokenizer_vocab_size
        self.n_bins = n_bins
        self.bins = np.linspace(-1.0, 1.0, n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

    def encode(self, normalized_action: np.ndarray) -> np.ndarray:
        """(D,) float in [-1, 1] → (D,) token id."""
        idx = np.digitize(normalized_action, self.bins) - 1
        idx = np.clip(idx, 0, self.n_bins - 2)
        return (self.vocab_size - 1 - idx).astype(np.int64)


# --------------------------------------------------------------------------- #
# Video frame extraction
# --------------------------------------------------------------------------- #

class _VideoReader:
    """Single-process PyAV reader with a tiny LRU of recent frames.

    PyAV seek+decode is expensive on AV1; we sort dataloader batches loosely
    by (video_file, frame_index) via the sampler-side hint isn't trivial, so
    we keep the last ~32 frames cached. With shuffle=True this only helps
    when num_workers spreads sequential strides across workers; even
    cold-seek throughput is fine for QLoRA train (~1 step/sec).
    """

    def __init__(self, path: str):
        self.path = path
        self._container = av.open(path)
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        self._fps = float(self._stream.average_rate)
        self._lock = threading.Lock()

    @property
    def fps(self) -> float:
        return self._fps

    def read_at(self, timestamp_sec: float) -> np.ndarray:
        """Decode the frame at the given timestamp (returns HWC uint8 RGB)."""
        with self._lock:
            # PyAV uses pts in stream.time_base units.
            target_pts = int(timestamp_sec / self._stream.time_base)
            self._container.seek(target_pts, any_frame=False, backward=True, stream=self._stream)
            best = None
            for frame in self._container.decode(self._stream):
                if frame.pts is None:
                    continue
                if frame.pts * self._stream.time_base > timestamp_sec + 1.0 / self._fps:
                    break
                best = frame
                if frame.pts * self._stream.time_base >= timestamp_sec - 1e-3:
                    break
            if best is None:
                raise RuntimeError(f"Failed to decode frame at t={timestamp_sec}s in {self.path}")
            img = best.to_ndarray(format="rgb24")
        return img


class _VideoCache:
    """Per-worker cache of opened video files."""

    def __init__(self):
        self._readers: Dict[str, _VideoReader] = {}

    def get(self, path: str) -> _VideoReader:
        if path not in self._readers:
            self._readers[path] = _VideoReader(path)
        return self._readers[path]


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class LeIsaacOpenVLADataset(Dataset):
    """One row per (episode, frame_idx) — front camera only, single-step action."""

    def __init__(
        self,
        dataset_root: str,
        action_stats: Dict[str, np.ndarray],
        video_key: str = "observation.images.front",
        prompt_override: str | None = None,
    ):
        self.root = dataset_root
        self.video_key = video_key
        self.stats = action_stats

        self.episodes = load_episodes(dataset_root, video_key=video_key)
        self.actions, self.states, self.eps_idx, self.frame_idx = load_actions_states(dataset_root)
        assert len(self.actions) == sum(e.length for e in self.episodes)

        # Episode-level lookup table
        self._ep_by_index = {e.episode_index: e for e in self.episodes}

        # Default prompt — single-task pickorange, ignore per-episode for now
        self._prompt = prompt_override or self.episodes[0].task

        # Per-worker video cache
        self._cache_lock = threading.Lock()
        self._caches: Dict[int, _VideoCache] = {}

    # ---- per-worker cache -------------------------------------------------
    def _video_cache(self) -> _VideoCache:
        wid = torch.utils.data.get_worker_info()
        key = wid.id if wid is not None else -1
        with self._cache_lock:
            if key not in self._caches:
                self._caches[key] = _VideoCache()
            return self._caches[key]

    # ---- video file lookup ------------------------------------------------
    def _video_path(self, ep: EpisodeRecord) -> str:
        return os.path.join(
            self.root, "videos", self.video_key,
            f"chunk-{ep.video_chunk_index:03d}",
            f"file-{ep.video_file_index:03d}.mp4",
        )

    # ---- main API ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> Dict:
        ep = self._ep_by_index[int(self.eps_idx[idx])]
        f_in_ep = int(self.frame_idx[idx])
        # Episode starts at ep.from_timestamp in the mp4 file
        ts = ep.from_timestamp + f_in_ep / 30.0  # dataset 30 fps

        reader = self._video_cache().get(self._video_path(ep))
        frame_rgb = reader.read_at(ts)  # (H, W, 3) uint8

        return {
            "image": frame_rgb,                       # uint8 HWC
            "action": self.actions[idx],              # float32 (6,)
            "state":  self.states[idx],               # float32 (6,)
            "prompt": self._prompt,
        }


# --------------------------------------------------------------------------- #
# Collator — folds the processor + tokenizer into HF Trainer batches
# --------------------------------------------------------------------------- #

class OpenVLACollator:
    """Build (input_ids, labels, pixel_values, attention_mask) per batch.

    Layout follows the OpenVLA training format:
        f"In: What action should the robot take to {prompt}? Out: <action tokens>"
    The action tokens are the only positions with non-IGNORE labels.
    """

    def __init__(
        self,
        processor,
        action_tokenizer: ActionTokenizer,
        action_stats: Dict[str, np.ndarray],
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.action_tok = action_tokenizer
        self.stats = action_stats
        # 29871 is the leading-space token Llama inserts after ':' — required at training
        self._space_id = 29871

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        pixel_values_list: List[torch.Tensor] = []
        attn_list: List[torch.Tensor] = []

        for sample in batch:
            img = Image.fromarray(sample["image"])
            a_norm = normalize_action(sample["action"], self.stats)
            action_tokens = self.action_tok.encode(a_norm)  # (D,) int64

            prompt_text = (
                f"In: What action should the robot take to {sample['prompt'].strip().rstrip('.')}?\nOut:"
            )
            # OpenVLA wants `Out: <space><action tokens><eos>` — the space token
            # is inserted to match the inference path's behavior (see
            # OpenVLAForActionPrediction.predict_action input padding).
            proc_out = self.processor(prompt_text, img)
            prompt_ids = proc_out["input_ids"][0]                  # 1-D long tensor
            pixel_values = proc_out["pixel_values"][0]             # CHW float

            action_tensor = torch.tensor(action_tokens, dtype=torch.long)
            space_tensor = torch.tensor([self._space_id], dtype=torch.long)
            eos_tensor = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)

            ids = torch.cat([prompt_ids, space_tensor, action_tensor, eos_tensor], dim=0)
            labels = torch.full_like(ids, IGNORE_INDEX)
            # Supervise the action tokens AND the eos token
            sup_start = prompt_ids.numel() + space_tensor.numel()
            sup_end = sup_start + action_tensor.numel() + eos_tensor.numel()
            labels[sup_start:sup_end] = ids[sup_start:sup_end]

            input_ids_list.append(ids)
            labels_list.append(labels)
            pixel_values_list.append(pixel_values)
            attn_list.append(torch.ones_like(ids, dtype=torch.long))

        # Left-pad-free: batch has identical action_dim, prompts are identical → same length
        # but defensively right-pad with eos to max length.
        max_len = max(t.numel() for t in input_ids_list)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        def _pad(ids: torch.Tensor, val: int) -> torch.Tensor:
            if ids.numel() == max_len:
                return ids
            return torch.cat([ids, torch.full((max_len - ids.numel(),), val, dtype=ids.dtype)])

        input_ids = torch.stack([_pad(t, pad_id) for t in input_ids_list])
        labels   = torch.stack([_pad(t, IGNORE_INDEX) for t in labels_list])
        attn     = torch.stack([_pad(t, 0) for t in attn_list])
        pixel_values = torch.stack(pixel_values_list)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attn,
            "pixel_values": pixel_values,
        }
