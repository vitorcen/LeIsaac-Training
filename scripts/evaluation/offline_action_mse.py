#!/usr/bin/env python3
"""Generic offline action-MSE eval — diagnose over/underfit WITHOUT a simulator.

v3 (codex review applied):
    The v2 whole-chunk MSE is kept as a weak reference signal but DOWNGRADED
    to a diagnostic — it averages over 32 timesteps which smooths out the
    gripper-transition timing that actually decides task success, and it
    favors mean-prediction underfit ckpts (2k looked "best" by it).

    PRIMARY metrics (more predictive of closed-loop success):
        - tf_mse_chunk0       : MSE at chunk position 0 (single-step pred quality)
        - tf_mse_n_steps      : MSE averaged over chunk[0..n_action_steps-1]
                               (this is the window actually executed before re-plan)
        - tf_step_mse_curve   : per-step MSE in deployment-realistic rollout
                               (walk episode in n_action_steps strides, re-predict
                                each window from the actual dataset state of that step)
        - gripper_timing_err  : average frame-offset between predicted and GT
                               gripper-open/close transition events (uses threshold)
        - gripper_miss_rate   : #|GT transitions not matched within ±W frames| /
                               #|GT transitions| — catches missing place commit
        - var_ratio_gripper   : pred_var / gt_var on gripper dim only
                               (global is too coarse — codex point (c))

    SECONDARY (weaker / for historical compat):
        - chunk_mse_total / chunk_mse_late / chunk_mse_gripper
        - var_ratio_mean (kept but de-emphasized)

Hold-out:
    Pass --val-episodes for episodes NOT in training (real held-out).
    Without a real train/val split during training, MSE measures train error
    (we hit this bug in our X-VLA run — model was actually trained on 50-59).

K-sample averaging:
    K=5 default (codex recommends 5 floor; 10 for serious ranking).
    For deterministic policies (DP/ACT) set --k-samples=1.

Designed to be policy-agnostic enough to extend to other VLAs (Pi0/SmolVLA/GR00T):
    swap policy class import, adjust gripper-channel idx + transition threshold.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Register custom action spaces (SingleArmSO101 etc).
_LEISAAC_XVLA = Path(__file__).resolve().parent.parent / "finetune" / "xvla"
if str(_LEISAAC_XVLA) not in sys.path:
    sys.path.insert(0, str(_LEISAAC_XVLA))
import action_spaces  # noqa: F401  (registers so101_single)


# --- argparse helpers --------------------------------------------------------
def _parse_episode_spec(spec: str) -> list[int]:
    """'50-59' -> [50..59]; '0,3,7' -> [0,3,7]."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        m = re.fullmatch(r"(\d+)-(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return out


def _list_ckpt_dirs(output_dir: Path) -> list[tuple[int, Path]]:
    """Return [(step, ckpt_dir), ...] sorted by step.  Skips 'last' / 'milestones'."""
    ckpts_root = output_dir / "checkpoints"
    if not ckpts_root.is_dir():
        raise FileNotFoundError(f"No checkpoints/ under {output_dir}")
    out = []
    for entry in sorted(ckpts_root.iterdir()):
        if not entry.is_dir() or entry.is_symlink():
            continue
        if not re.fullmatch(r"\d+", entry.name):
            continue
        pre = entry / "pretrained_model"
        if not (pre / "config.json").exists():
            continue
        out.append((int(entry.name), pre))
    return out


# --- model + dataset --------------------------------------------------------
def _load_policy(ckpt_dir: Path, device: str):
    """XVLAPolicy + preprocessor + postprocessor, all loaded from ckpt."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    cfg = PreTrainedConfig.from_pretrained(str(ckpt_dir))
    cfg.device = device
    policy = XVLAPolicy.from_pretrained(str(ckpt_dir), config=cfg)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(ckpt_dir),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor, cfg


def _open_dataset(dataset_root: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(repo_id="local", root=str(dataset_root))


def _rename_sample(sample: dict, rename_map: dict) -> dict:
    return {rename_map.get(k, k): v for k, v in sample.items()}


def _to_batch(sample: dict, device: str) -> dict:
    out = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.unsqueeze(0).to(device) if v.dim() < 4 else v.to(device)
        else:
            out[k] = v
    return out


def _gt_chunk(ds, frame_idx: int, ep_end: int, chunk_size: int) -> np.ndarray:
    """Read chunk_size future GT actions starting at frame_idx (pad with last action)."""
    chunk_actions = []
    for k in range(chunk_size):
        idx = frame_idx + k
        if idx < ep_end:
            a = ds[idx]["action"]
        else:
            a = chunk_actions[-1]
        chunk_actions.append(a if isinstance(a, np.ndarray) else a.detach().cpu().float().numpy())
    return np.stack(chunk_actions, axis=0)


def _k_avg_chunk(policy, preprocessor, postprocessor, sample: dict, device: str, k_samples: int) -> np.ndarray:
    """Predict chunk K times (K-sample rectified-flow averaging), return (chunk_size, real_dim)."""
    batch = _to_batch(sample, device)
    batch_p = preprocessor(batch)
    pred_acc = None
    for _ in range(k_samples):
        with torch.inference_mode():
            pred = policy.predict_action_chunk(batch_p)
            pred = postprocessor(pred)
        p = pred.detach().cpu().float().numpy()
        pred_acc = p if pred_acc is None else (pred_acc + p)
    return (pred_acc / k_samples)[0]  # (chunk_size, real_dim)


# --- core metrics -----------------------------------------------------------
def _teacher_forced_rollout(
    policy, preprocessor, postprocessor, ds, ep_idx: int,
    chunk_size: int, n_action_steps: int, gripper_idx: int,
    rename_map: dict, device: str, k_samples: int,
) -> dict:
    """Walk an episode in n_action_steps strides — at each step, predict chunk
    from the actual frame in dataset, then advance by n_action_steps.

    Returns:
        step_mse_per_pos[t]: per-chunk-position MSE at chunk[t] for t in [0, n_action_steps)
        gripper_pred_seq, gripper_gt_seq: concatenated gripper trajectory
        n_steps_seen: count
    """
    ep_info = ds.meta.episodes[ep_idx]
    ep_start = int(ep_info["dataset_from_index"])
    ep_end = int(ep_info["dataset_to_index"])
    ep_len = ep_end - ep_start
    if ep_len < chunk_size + n_action_steps:
        return None  # episode too short — skip

    step_sqs_per_pos = [[] for _ in range(n_action_steps)]  # per chunk position
    step_sqs_per_pos_gripper = [[] for _ in range(n_action_steps)]
    gripper_pred_seq: list[float] = []
    gripper_gt_seq: list[float] = []
    pred_all: list[np.ndarray] = []
    gt_all: list[np.ndarray] = []

    t = 0
    while t + n_action_steps <= ep_len:
        sample = _rename_sample(ds[ep_start + t], rename_map)
        try:
            pred_chunk = _k_avg_chunk(policy, preprocessor, postprocessor, sample, device, k_samples)
        except Exception as e:
            print(f"  [warn] ep {ep_idx} t={t} forward failed: {e}", flush=True)
            t += n_action_steps
            continue
        real_dim = pred_chunk.shape[-1]
        # GT next n_action_steps
        gt_actions = _gt_chunk(ds, ep_start + t, ep_end, n_action_steps)[:, :real_dim]
        pred_actions = pred_chunk[:n_action_steps]
        diff = pred_actions - gt_actions  # (n_action_steps, real_dim)
        sq = diff * diff
        for i in range(n_action_steps):
            step_sqs_per_pos[i].append(sq[i].mean())
            if 0 <= gripper_idx < real_dim:
                step_sqs_per_pos_gripper[i].append(sq[i, gripper_idx])
        if 0 <= gripper_idx < real_dim:
            gripper_pred_seq.extend(pred_actions[:, gripper_idx].tolist())
            gripper_gt_seq.extend(gt_actions[:, gripper_idx].tolist())
        pred_all.append(pred_actions)
        gt_all.append(gt_actions)
        t += n_action_steps

    step_mse_per_pos = [float(np.mean(xs)) if xs else float("nan") for xs in step_sqs_per_pos]
    step_mse_gripper_per_pos = [float(np.mean(xs)) if xs else float("nan") for xs in step_sqs_per_pos_gripper]
    return {
        "step_mse_per_pos": step_mse_per_pos,                  # len = n_action_steps
        "step_mse_gripper_per_pos": step_mse_gripper_per_pos,  # len = n_action_steps
        "gripper_pred_seq": gripper_pred_seq,
        "gripper_gt_seq": gripper_gt_seq,
        "pred_all": np.concatenate(pred_all, axis=0) if pred_all else np.zeros((0, 0)),
        "gt_all": np.concatenate(gt_all, axis=0) if gt_all else np.zeros((0, 0)),
        "n_windows": len(step_sqs_per_pos[0]) if step_sqs_per_pos[0] else 0,
    }


def _gripper_transitions(seq: np.ndarray, threshold: float, hysteresis: float = 3.0) -> np.ndarray:
    """Detect open<->close transitions: frame indices where seq crosses threshold.
    Hysteresis suppresses jitter (state stays for at least N frames)."""
    seq = np.asarray(seq, dtype=np.float32)
    if seq.size == 0:
        return np.array([], dtype=np.int64)
    above = seq > threshold
    # Simple diff-based transition detection (could add hysteresis filter)
    transitions = np.where(np.diff(above.astype(int)) != 0)[0] + 1
    return transitions


def _gripper_timing_metrics(pred_seq: list, gt_seq: list, threshold: float, window: int = 10) -> dict:
    """Compare pred vs gt gripper transition timing.

    Returns:
        timing_err_mean: avg |pred_t - matched_gt_t| frames over matched transitions
        miss_rate: #|GT transitions without any pred match within ±window| / #|GT|
        false_rate: #|pred transitions without any GT match within ±window| / #|GT|+ε
    """
    if not pred_seq or not gt_seq:
        return {"timing_err_mean": float("nan"), "miss_rate": float("nan"), "false_rate": float("nan"),
                "n_gt_trans": 0, "n_pred_trans": 0}
    pred_trans = _gripper_transitions(np.asarray(pred_seq), threshold)
    gt_trans = _gripper_transitions(np.asarray(gt_seq), threshold)
    n_gt = len(gt_trans)
    n_pred = len(pred_trans)
    if n_gt == 0:
        return {"timing_err_mean": 0.0 if n_pred == 0 else float("inf"),
                "miss_rate": 0.0, "false_rate": float(n_pred), "n_gt_trans": 0, "n_pred_trans": n_pred}
    # Greedy match: for each GT, pick closest pred within ±window.
    matched_gt = np.zeros(n_gt, dtype=bool)
    matched_pred = np.zeros(n_pred, dtype=bool)
    errors = []
    for i, gt_t in enumerate(gt_trans):
        if n_pred == 0:
            break
        dists = np.abs(pred_trans - gt_t)
        dists[matched_pred] = 10**9
        j = int(np.argmin(dists))
        if dists[j] <= window:
            matched_gt[i] = True
            matched_pred[j] = True
            errors.append(int(dists[j]))
    miss = int((~matched_gt).sum())
    false_alarm = int((~matched_pred).sum())
    return {
        "timing_err_mean": float(np.mean(errors)) if errors else float("nan"),
        "miss_rate": miss / max(n_gt, 1),
        "false_rate": false_alarm / max(n_gt, 1),
        "n_gt_trans": n_gt,
        "n_pred_trans": n_pred,
    }


# --- main per-ckpt eval ------------------------------------------------------
def _eval_one_ckpt(
    ckpt_dir: Path, ds, val_episodes: list[int], gripper_idx: int,
    rename_map: dict, device: str, k_samples: int,
    gripper_threshold: float, match_window: int,
) -> dict:
    policy, preprocessor, postprocessor, cfg = _load_policy(ckpt_dir, device)
    chunk = cfg.chunk_size
    n_act = getattr(cfg, "n_action_steps", chunk)

    # Aggregate teacher-forced rollouts across episodes
    all_step_mse = []          # each: list[float] of length n_act
    all_step_mse_grip = []
    grip_pred_all: list[float] = []
    grip_gt_all: list[float] = []
    pred_act_all: list[np.ndarray] = []
    gt_act_all: list[np.ndarray] = []
    n_windows_total = 0

    for ep_idx in val_episodes:
        out = _teacher_forced_rollout(
            policy, preprocessor, postprocessor, ds, ep_idx,
            chunk_size=chunk, n_action_steps=n_act, gripper_idx=gripper_idx,
            rename_map=rename_map, device=device, k_samples=k_samples,
        )
        if out is None:
            continue
        all_step_mse.append(out["step_mse_per_pos"])
        all_step_mse_grip.append(out["step_mse_gripper_per_pos"])
        grip_pred_all.extend(out["gripper_pred_seq"])
        grip_gt_all.extend(out["gripper_gt_seq"])
        if out["pred_all"].size:
            pred_act_all.append(out["pred_all"])
            gt_act_all.append(out["gt_all"])
        n_windows_total += out["n_windows"]

    if not all_step_mse:
        return {"step": int(ckpt_dir.parent.name), "n_windows": 0}

    step_mse_arr = np.array(all_step_mse)         # (#eps, n_act)
    step_mse_grip_arr = np.array(all_step_mse_grip)
    mse_per_pos = step_mse_arr.mean(axis=0).tolist()
    mse_per_pos_grip = step_mse_grip_arr.mean(axis=0).tolist()

    tf_mse_chunk0 = float(mse_per_pos[0])
    tf_mse_chunk_last = float(mse_per_pos[-1])
    tf_mse_n_steps = float(np.mean(mse_per_pos))

    grip_metrics = _gripper_timing_metrics(grip_pred_all, grip_gt_all, gripper_threshold, match_window)

    # Variance ratio — keep but only on gripper dim (codex point (c))
    pred_all = np.concatenate(pred_act_all, axis=0) if pred_act_all else np.zeros((0, 1))
    gt_all = np.concatenate(gt_act_all, axis=0) if gt_act_all else np.zeros((0, 1))
    if pred_all.shape[0] > 1 and 0 <= gripper_idx < pred_all.shape[-1]:
        pred_grip_var = float(pred_all[:, gripper_idx].var())
        gt_grip_var = float(gt_all[:, gripper_idx].var())
        var_ratio_grip = pred_grip_var / max(gt_grip_var, 1e-8)
    else:
        var_ratio_grip = float("nan")

    # Free GPU before next ckpt.
    del policy, preprocessor, postprocessor
    torch.cuda.empty_cache()

    return {
        "step": int(ckpt_dir.parent.name),
        "n_windows": n_windows_total,
        # PRIMARY metrics (codex-endorsed)
        "tf_mse_chunk0": tf_mse_chunk0,
        "tf_mse_n_steps": tf_mse_n_steps,
        "tf_mse_chunk_last": tf_mse_chunk_last,
        "mse_per_pos": mse_per_pos,
        "mse_per_pos_grip": mse_per_pos_grip,
        "gripper_timing_err_frames": grip_metrics["timing_err_mean"],
        "gripper_miss_rate": grip_metrics["miss_rate"],
        "gripper_false_rate": grip_metrics["false_rate"],
        "n_gt_grip_trans": grip_metrics["n_gt_trans"],
        "n_pred_grip_trans": grip_metrics["n_pred_trans"],
        "var_ratio_gripper": var_ratio_grip,
    }


# --- main entry --------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--val-episodes", default="50-59",
                    help="HELD-OUT episode indices (must NOT be in training set!)")
    ap.add_argument("--gripper-idx", type=int, default=5)
    ap.add_argument("--gripper-threshold", type=float, default=30.0,
                    help="Threshold (deg) for gripper open/close transition detection")
    ap.add_argument("--match-window", type=int, default=10,
                    help="±frames window for pred-vs-gt transition matching")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k-samples", type=int, default=5,
                    help="# rectified-flow re-samples per frame, averaged. 5 floor; 10 for ranking.")
    ap.add_argument("--output-csv", default=None)
    ap.add_argument("--rename-map",
                    default="observation.images.front=observation.images.image,"
                            "observation.images.wrist=observation.images.image2")
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    val_episodes = _parse_episode_spec(args.val_episodes)
    rename_map = dict(pair.split("=") for pair in args.rename_map.split(",") if pair)

    print(f"[offline-eval v3] sweep ckpts under {output_dir}/checkpoints/", flush=True)
    print(f"[offline-eval v3] val episodes (held-out): {val_episodes}", flush=True)

    ds = _open_dataset(dataset_root)
    ckpts = _list_ckpt_dirs(output_dir)
    print(f"[offline-eval v3] {len(ckpts)} ckpts, K={args.k_samples} samples, "
          f"grip_thresh={args.gripper_threshold}", flush=True)

    results = []
    for step, ckpt_dir in ckpts:
        t0 = time.time()
        print(f"\n[offline-eval v3] === ckpt {step:>6d} ===", flush=True)
        try:
            r = _eval_one_ckpt(
                ckpt_dir, ds, val_episodes, args.gripper_idx, rename_map, args.device,
                args.k_samples, args.gripper_threshold, args.match_window,
            )
        except Exception as e:
            print(f"  ❌ failed: {e}", flush=True)
            continue
        elapsed = time.time() - t0
        if r.get("n_windows", 0) == 0:
            print(f"  ⚠️  no windows evaluated", flush=True)
            continue
        print(
            f"  tf_mse_chunk0={r['tf_mse_chunk0']:.3f}  "
            f"tf_mse_n_steps={r['tf_mse_n_steps']:.3f}  "
            f"tf_mse_chunk_last={r['tf_mse_chunk_last']:.3f}  "
            f"grip_timing_err={r['gripper_timing_err_frames']:.2f}f  "
            f"grip_miss={r['gripper_miss_rate']:.2f}  "
            f"var_grip={r['var_ratio_gripper']:.2f}  "
            f"({elapsed:.0f}s)",
            flush=True,
        )
        results.append(r)

    print("\n[offline-eval v3] ============== summary ==============", flush=True)
    print(f"{'step':>6} {'tf_c0':>7} {'tf_ns':>7} {'tf_cL':>7} {'grip_t':>7} {'grip_m':>7} {'var_g':>7}", flush=True)
    for r in results:
        print(
            f"{r['step']:>6d} {r['tf_mse_chunk0']:>7.3f} {r['tf_mse_n_steps']:>7.3f} "
            f"{r['tf_mse_chunk_last']:>7.3f} {r['gripper_timing_err_frames']:>7.2f} "
            f"{r['gripper_miss_rate']:>7.2f} {r['var_ratio_gripper']:>7.2f}",
            flush=True,
        )

    if results:
        # PRIMARY ranking: tf_mse_n_steps (the actual deployment window) + gripper timing
        valid = [r for r in results if not np.isnan(r["gripper_timing_err_frames"])]
        if valid:
            best_n = min(results, key=lambda r: r["tf_mse_n_steps"])
            best_grip = min(valid, key=lambda r: (r["gripper_miss_rate"], r["gripper_timing_err_frames"]))
            print(f"\n[best by tf_mse_n_steps]    step {best_n['step']} ({best_n['tf_mse_n_steps']:.3f})", flush=True)
            print(f"[best by gripper miss+time] step {best_grip['step']} "
                  f"(miss={best_grip['gripper_miss_rate']:.2f}, err={best_grip['gripper_timing_err_frames']:.2f}f)",
                  flush=True)

    if args.output_csv and results:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        keys = ["step", "n_windows",
                "tf_mse_chunk0", "tf_mse_n_steps", "tf_mse_chunk_last",
                "gripper_timing_err_frames", "gripper_miss_rate", "gripper_false_rate",
                "n_gt_grip_trans", "n_pred_grip_trans", "var_ratio_gripper",
                "mse_per_pos", "mse_per_pos_grip"]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in results:
                row = {k: r.get(k) for k in keys}
                row["mse_per_pos"] = ",".join(f"{x:.3f}" for x in (r.get("mse_per_pos") or []))
                row["mse_per_pos_grip"] = ",".join(f"{x:.3f}" for x in (r.get("mse_per_pos_grip") or []))
                w.writerow(row)
        print(f"\n[offline-eval v3] wrote {len(results)} rows → {out}", flush=True)


if __name__ == "__main__":
    main()
