"""Script to run a leisaac inference with leisaac in the simulation."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing
import socket

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="leisaac inference for leisaac in the simulation.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--seed", type=int, default=None, help="Seed of the environment.")
parser.add_argument("--episode_length_s", type=float, default=60.0, help="Episode length in seconds.")
parser.add_argument(
    "--eval_rounds",
    type=int,
    default=0,
    help=(
        "Number of evaluation rounds. 0 means don't add time out termination, policy will run until success or manual"
        " reset."
    ),
)
parser.add_argument(
    "--policy_type",
    type=str,
    default="gr00tn1.5",
    help="Type of policy to use. support gr00tn1.5, gr00tn1.6, lerobot-<model_type>, openpi",
)
parser.add_argument("--policy_host", type=str, default="localhost", help="Host of the policy server.")
parser.add_argument("--policy_port", type=int, default=5555, help="Port of the policy server.")
parser.add_argument("--policy_timeout_ms", type=int, default=15000, help="Timeout of the policy server.")
parser.add_argument("--policy_action_horizon", type=int, default=16, help="Action horizon of the policy.")
parser.add_argument("--policy_language_instruction", type=str, default=None, help="Language instruction of the policy.")
parser.add_argument("--policy_checkpoint_path", type=str, default=None, help="Checkpoint path of the policy.")
parser.add_argument(
    "--metrics_out",
    type=str,
    default=None,
    help="If set, dump per-round metrics (oranges_placed / duration_s / success) as JSON to this path.",
)
parser.add_argument(
    "--metrics_label",
    type=str,
    default=None,
    help="Display label for the policy in the metrics JSON (e.g. 'ACT (self) — wsagi/ACT-PickOrange').",
)
parser.add_argument(
    "--max_round_wall_s",
    type=float,
    default=0.0,
    help=(
        "Hard wall-clock cap per round (seconds, 0 disables). When exceeded the "
        "round is recorded as failed/skipped with whatever placed_flags accumulated "
        "and the eval moves on. Separate from episode_length_s (which is sim-time)."
    ),
)
parser.add_argument(
    "--stuck_window_s",
    type=float,
    default=30.0,
    help=(
        "Detect 'shaking attractor' failure mode: if every joint's position range "
        "(max-min) over the last N wall-clock seconds is below stuck_eps_rad, "
        "skip the round as a failed-stuck (chunk-policy obs invariance trap). "
        "0 disables."
    ),
)
parser.add_argument(
    "--stuck_eps_rad",
    type=float,
    default=0.05,
    help="Max-min threshold (radians, all joints) for the stuck detector. Default 0.05 ≈ 2.9°.",
)
parser.add_argument(
    "--sim_warmup_steps",
    type=int,
    default=30,
    help=(
        "Sim steps to run with a hold-pose action right after env.reset(), before "
        "the policy sees anything. Isaac's first frames have unsettled DomeLight, "
        "incomplete viewport rendering and missing scene assets — feeding that "
        "garbage to a chunk-100 ACT poisons all 100 actions in the first chunk."
    ),
)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()


def _check_policy_service_ready():
    """Fail fast if a remote policy server is not reachable."""
    needs_remote_service = (
        args_cli.policy_type in {"gr00tn1.5", "gr00tn1.6", "openpi", "pi05"} or "lerobot" in args_cli.policy_type
    )
    if not needs_remote_service:
        return

    host = args_cli.policy_host
    port = args_cli.policy_port
    try:
        with socket.create_connection((host, port), timeout=max(args_cli.policy_timeout_ms / 1000.0, 1.0)):
            return
    except OSError as err:
        raise RuntimeError(
            f"Policy server is not reachable at {host}:{port}. "
            f"Please start the policy server before running simulation. Details: {err}"
        ) from err


_check_policy_service_ready()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import time

import carb
import gymnasium as gym
import omni
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.utils.env_utils import (
    dynamic_reset_gripper_effort_limit_sim,
    get_task_type,
)

import leisaac  # noqa: F401


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


class Controller:
    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )
        self.reset_state = False

    def __del__(self):
        """Release the keyboard interface."""
        if hasattr(self, "_input") and hasattr(self, "_keyboard") and hasattr(self, "_keyboard_sub"):
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def reset(self):
        self.reset_state = False

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events using carb."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R":
                self.reset_state = True
        return True


def preprocess_obs_dict(obs_dict: dict, model_type: str, language_instruction: str):
    """Preprocess the observation dictionary to the format expected by the policy."""
    if model_type in ["gr00tn1.5", "gr00tn1.6", "lerobot", "openpi", "pi05"]:
        obs_dict["task_description"] = language_instruction
        return obs_dict
    else:
        raise ValueError(f"Model type {model_type} not supported")


def _build_camera_feature_map(
    ckpt_path: str | None, sim_cameras: list[str]
) -> tuple[dict[str, str] | None, list[tuple[str, tuple[int, int, int]]]]:
    """Resolve image-feature plumbing from the ckpt's config.json.

    Returns ``(rename_map, empty_cameras)``:
      * ``rename_map``: ``{sim_camera_key: model_image_feature_key}``, or
        ``None`` when every sim camera matches a natural
        ``observation.images.<sim_key>`` slot.
      * ``empty_cameras``: list of ``(model_image_feature_key, shape_chw)``
        for image features that the policy declares but the sim does not
        provide. These must be sent as zero-filled images so the server's
        feature-validation does not raise ``KeyError``. Required for
        ``lerobot/smolvla_base``-style schemas that declare more slots
        (``camera1/2/3``) than the sim populates.
    """
    if not ckpt_path:
        return None, []
    import json
    import os
    try:
        if os.path.isdir(ckpt_path):
            cfg_path = os.path.join(ckpt_path, "config.json")
        else:
            from huggingface_hub import hf_hub_download
            cfg_path = hf_hub_download(repo_id=ckpt_path, filename="config.json")
        cfg = json.load(open(cfg_path))
    except Exception as err:  # noqa: BLE001
        print(f"[WARN] could not read {ckpt_path}/config.json for feature map: {err}")
        return None, []

    image_feats = {
        k: v for k, v in cfg.get("input_features", {}).items()
        if k.startswith("observation.images.")
    }
    if not image_feats:
        return None, []
    expected = list(image_feats.keys())
    rename: dict[str, str] = {}
    for i, sim_key in enumerate(sim_cameras):
        natural = f"observation.images.{sim_key}"
        if natural in expected:
            continue  # natural fit — no rename
        if i < len(expected):
            rename[sim_key] = expected[i]  # positional fallback
    used_model_keys = set(rename.values()) | {
        f"observation.images.{s}" for s in sim_cameras
        if f"observation.images.{s}" in expected
    }
    empty_cameras: list[tuple[str, tuple[int, int, int]]] = []
    for k in expected:
        if k in used_model_keys:
            continue
        shape = tuple(image_feats[k].get("shape", [3, 256, 256]))
        empty_cameras.append((k, shape))
    return (rename or None), empty_cameras


def main():
    """Running lerobot teleoperation with leisaac manipulation environment."""

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    task_type = get_task_type(args_cli.task)
    env_cfg.use_teleop_device(task_type)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    env_cfg.episode_length_s = args_cli.episode_length_s

    # modify configuration
    if args_cli.eval_rounds <= 0:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
    max_episode_count = args_cli.eval_rounds
    env_cfg.recorders = None

    # create environment
    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # create policy
    model_type = args_cli.policy_type
    if args_cli.policy_type == "gr00tn1.5":
        from isaaclab.sensors import Camera
        from leisaac.policy import Gr00tServicePolicyClient

        if task_type == "so101leader":
            modality_keys = ["single_arm", "gripper"]
        else:
            raise ValueError(f"Task type {task_type} not supported when using GR00T N1.5 policy yet.")

        policy = Gr00tServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_keys=[key for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)],
            modality_keys=modality_keys,
        )
    elif args_cli.policy_type == "gr00tn1.6":
        from isaaclab.sensors import Camera
        from leisaac.policy import Gr00t16ServicePolicyClient

        if task_type == "so101leader":
            modality_keys = ["single_arm", "gripper"]
        else:
            raise ValueError(f"Task type {task_type} not supported when using GR00T N1.5 policy yet.")

        policy = Gr00t16ServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_keys=[key for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)],
            modality_keys=modality_keys,
        )

    elif "lerobot" in args_cli.policy_type:
        from isaaclab.sensors import Camera
        from leisaac.policy import LeRobotServicePolicyClient

        model_type = "lerobot"

        policy_type = args_cli.policy_type.split("-")[1]
        sim_cameras = [
            key for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)
        ]
        # Read the ckpt's expected image feature names from config.json
        # (LeRobot v0.4+ server validates client lerobot_features against
        # the model's input_features; sending a key not in the model's
        # config raises KeyError on first inference). Most fine-tunes use
        # natural ``observation.images.<sim_key>`` keys and need no rename;
        # `lerobot/smolvla_base` is the historical outlier — its config
        # declares ``observation.images.camera1/2/3`` slots, so we map
        # sim cameras into those in order.
        camera_feature_names, empty_camera_feats = _build_camera_feature_map(
            args_cli.policy_checkpoint_path, sim_cameras
        )
        policy = LeRobotServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_infos={key: env.scene.sensors[key].image_shape for key in sim_cameras},
            task_type=task_type,
            policy_type=policy_type,
            pretrained_name_or_path=args_cli.policy_checkpoint_path,
            actions_per_chunk=args_cli.policy_action_horizon,
            device=args_cli.device,
            camera_feature_names=camera_feature_names,
            empty_camera_feats=empty_camera_feats,
        )
    elif args_cli.policy_type == "openpi":
        from isaaclab.sensors import Camera
        from leisaac.policy import OpenPIServicePolicyClient

        policy = OpenPIServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            camera_keys=[key for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)],
            task_type=task_type,
        )
    elif args_cli.policy_type == "pi05":
        from isaaclab.sensors import Camera
        from leisaac.policy import Pi05ServicePolicyClient

        policy = Pi05ServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_keys=[key for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)],
        )

    rate_limiter = RateLimiter(args_cli.step_hz)
    controller = Controller()

    def _sim_warmup(obs_dict):
        """Hold the current robot pose for ``--sim_warmup_steps`` to let Isaac
        settle DomeLight, finish render-graph propagation, and stop spawning
        partially-loaded scene meshes. The first sim frame is genuinely broken
        visuals (cold-grey floor, missing plate, half-black viewport); a chunk
        policy like ACT that emits 100 actions per inference will commit to a
        wrong trajectory from that single bad frame and never recover.
        """
        if args_cli.sim_warmup_steps <= 0:
            return obs_dict
        hold = obs_dict["policy"]["joint_pos"].clone()
        for _ in range(args_cli.sim_warmup_steps):
            obs_dict, _, _, _, _ = env.step(hold)
        return obs_dict

    # reset environment
    obs_dict, _ = env.reset()
    obs_dict = _sim_warmup(obs_dict)
    controller.reset()

    # record the results
    success_count, episode_count = 0, 1
    per_round_metrics: list[dict] = []
    # "sticky" per-orange placed flag for the current round.
    # subtask_terms.put_orangeNNN_to_plate is true only at the *moment of release*
    # (gripper-open + EE-near + orange-in-plate-region simultaneously). Once the
    # arm moves on to the next orange that bit flips back to False, so we OR
    # over the whole round to get "was it ever placed".
    placed_flags = [False, False, False]

    def _read_subtask_flags(od):
        """Returns 3-tuple of bools (place1, place2, place3) for current step.

        Uses the env's strict put_orange_to_plate observation (EE-near +
        gripper-open + xy-in-plate) as a primary signal. Falls back to a
        looser placement check via direct scene query (orange xy in plate
        ±10cm AND z within plate height ±7cm, same box as task_done) — this
        catches policies that release fast then retract EE (e.g. DP), where
        the strict EE-near check misses the transient release moment.
        """
        # Primary: strict subtask_terms read
        primary = [False, False, False]
        try:
            st = od["subtask_terms"]
        except (KeyError, TypeError):
            st = None
        if st is not None:
            for i, k in enumerate(("put_orange001_to_plate", "put_orange002_to_plate", "put_orange003_to_plate")):
                v = st.get(k)
                if v is not None:
                    try:
                        primary[i] = bool(v[0].item()) if hasattr(v, "shape") else bool(v)
                    except Exception:
                        pass
        # Fallback (loose): direct scene query — orange resting on plate surface.
        # Bug fixes vs naive ±10cm box check:
        #   (1) orange pressed/clipped under plate (dz < 0) was wrongly counted
        #   (2) plate flipped upside-down was wrongly counted
        #   (3) ±10cm SQUARE box catches plate corners that are outside circular
        #       plate surface — switch to cylindrical radius check
        #   (4) orange still in transit (held/falling) was wrongly counted —
        #       require near-zero linear velocity (settled on plate)
        loose = [False, False, False]
        try:
            plate = env.scene["Plate"]
            plate_xyz = plate.data.root_pos_w[0] - env.scene.env_origins[0]
            # Plate orientation: skip placements if plate has tipped > ~45°.
            try:
                pq = plate.data.root_quat_w[0]  # (w, x, y, z)
                plate_up_z = float((1.0 - 2.0 * (pq[1] * pq[1] + pq[2] * pq[2])).item())
            except Exception:
                plate_up_z = 1.0  # assume upright on error
            plate_upright = plate_up_z > 0.7
            if plate_upright:
                # Plate radius: LeIsaac plate is ~15cm diameter → 7.5cm radius.
                # Env's strict put_orangeN_to_plate subtask uses a 10cm xy box, so
                # match that with plate_r=0.10 — anything tighter (e.g. 0.08) misses
                # oranges resting on the plate rim that env counts as placed.
                plate_r = 0.10
                # Resting height band: orange center should be ≥ ~3cm above plate
                # center (plate ~1cm thick + orange ~3cm radius). dz_max raised to
                # 0.20 to accept stacked oranges (top of a 3-stack ≈ 15cm above
                # plate center), since the env's success criterion treats any
                # orange whose xy is inside the plate as "placed" regardless of
                # stack height.
                dz_min, dz_max = 0.005, 0.20
                # Settled velocity threshold: < 0.05 m/s (5 cm/s) — orange has
                # stopped moving (not in gripper transit / not bouncing).
                v_thresh = 0.05
                for i, name in enumerate(("Orange001", "Orange002", "Orange003")):
                    orange = env.scene[name]
                    oxyz = orange.data.root_pos_w[0] - env.scene.env_origins[0]
                    dx = (oxyz[0] - plate_xyz[0]).item()
                    dy = (oxyz[1] - plate_xyz[1]).item()
                    dz = (oxyz[2] - plate_xyz[2]).item()
                    xy_dist = (dx * dx + dy * dy) ** 0.5
                    if not (xy_dist < plate_r and dz_min < dz < dz_max):
                        continue
                    # Settled-velocity gate
                    try:
                        vlin = orange.data.root_lin_vel_w[0]
                        v_mag = float((vlin[0] ** 2 + vlin[1] ** 2 + vlin[2] ** 2).sqrt().item())
                    except Exception:
                        v_mag = 0.0
                    if v_mag > v_thresh:
                        continue
                    loose[i] = True
        except Exception:
            pass
        return (
            primary[0] or loose[0],
            primary[1] or loose[1],
            primary[2] or loose[2],
        )

    round_start_t = time.time()

    # File-trigger so the operator can force "next round = fail" from CLI when
    # Isaac Sim viewport doesn't have keyboard focus (the R-key path in
    # Controller depends on carb.input which only fires when the viewport is
    # focused). Create the file: `touch <metrics_dir>/.skip_round` — the loop
    # picks it up at the next inner-step boundary, records partial flags, and
    # advances to the next episode.
    import os as _os
    _skip_dir = (
        _os.path.dirname(_os.path.abspath(args_cli.metrics_out))
        if args_cli.metrics_out
        else _os.getcwd()
    )
    _skip_path = _os.path.join(_skip_dir, ".skip_round")
    if _os.path.exists(_skip_path):
        _os.remove(_skip_path)

    def _skip_requested():
        return _os.path.exists(_skip_path)

    def _consume_skip():
        try:
            _os.remove(_skip_path)
        except FileNotFoundError:
            pass

    # simulate environment
    while max_episode_count <= 0 or episode_count <= max_episode_count:
        print(f"[Evaluation] Evaluating episode {episode_count}...")
        success, time_out = False, False
        placed_flags = [False, False, False]
        round_start_t = time.time()
        # Stuck detector: rolling (timestamp, joint_pos_tensor) buffer over last stuck_window_s seconds
        joint_history: list[tuple[float, torch.Tensor]] = []
        # Capture home pose (arm at reset position) for "returned-to-home" detector.
        # Set lazily on first inner-loop iter so it reflects post-warmup state.
        home_arm_pose: "torch.Tensor | None" = None
        home_return_armed = False  # set True once arm has clearly moved away from home
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                wall_elapsed = time.time() - round_start_t
                wall_capped = (
                    args_cli.max_round_wall_s > 0
                    and wall_elapsed >= args_cli.max_round_wall_s
                )
                # Stuck-detector check: arm at quasi-rest (range < stuck_eps_rad over
                # the last 5 seconds) and the policy has had a chance to do work
                # (we've seen ≥ one motion burst above eps already, or round_age >
                # short grace). This catches "policy retracted to home and gave up"
                # cases where waiting to wall_cap (180s) is just lost time.
                # `stuck_window_s` env var keeps a minimum-grace guard so policies
                # whose initial pose happens to be near-still don't trigger on iter 0.
                # Gripper joint is ignored (open/close idle cycles defeat detection).
                stuck = False
                if args_cli.stuck_window_s > 0 and joint_history:
                    round_age = time.time() - round_start_t
                    if round_age >= args_cli.stuck_window_s:
                        short_still_s = 5.0
                        cutoff = time.time() - short_still_s
                        recent = [jp for ts, jp in joint_history if ts >= cutoff]
                        if len(recent) >= 10:
                            window = torch.stack(recent, dim=0)
                            arm_window = window[:, :-1] if window.shape[1] > 1 else window
                            ranges = arm_window.max(dim=0).values - arm_window.min(dim=0).values
                            if ranges.max().item() < args_cli.stuck_eps_rad:
                                stuck = True
                # Home-return detector: arm joints have visibly moved AWAY from the
                # episode-start pose (so policy did work) and have now returned to
                # within home_tol_rad of that pose → policy is signaling "done".
                # Fires faster than stuck because it doesn't require sustained
                # stillness — one match suffices once `home_return_armed` is True.
                home_return = False
                if home_arm_pose is not None and joint_history:
                    last_jp = joint_history[-1][1]
                    arm_jp = last_jp[:-1] if last_jp.numel() > 1 else last_jp
                    diff = (arm_jp - home_arm_pose).abs()
                    # arm 5 joints, 0.5 rad ≈ 29° per joint = clearly "moved away"
                    moved_away = diff.max().item() > 0.5
                    near_home = diff.max().item() < 0.15  # 8.6° tolerance
                    if moved_away:
                        home_return_armed = True
                    if home_return_armed and near_home:
                        home_return = True
                if controller.reset_state or _skip_requested() or wall_capped or stuck or home_return:
                    reason = "wall_cap" if wall_capped else (
                        "home_return" if home_return else (
                            "stuck" if stuck else (
                                "file_skip" if _skip_requested() else "key_R"
                            )
                        )
                    )
                    if _skip_requested():
                        _consume_skip()
                    controller.reset()
                    duration_s = wall_elapsed
                    oranges_n = sum(placed_flags)
                    print(
                        f"[Evaluation] Episode {episode_count} skipped ({reason})! "
                        f"oranges={oranges_n}/3 t={duration_s:.1f}s placed={placed_flags}"
                    )
                    per_round_metrics.append({
                        "episode": episode_count,
                        "success": False,
                        "skipped": True,
                        "skip_reason": reason,
                        "oranges_placed": oranges_n,
                        "placed_flags": list(placed_flags),
                        "duration_s": round(duration_s, 2),
                    })
                    obs_dict, _ = env.reset()
                    obs_dict = _sim_warmup(obs_dict)
                    episode_count += 1
                    break

                obs_dict = preprocess_obs_dict(obs_dict["policy"], model_type, args_cli.policy_language_instruction)
                actions = policy.get_action(obs_dict).to(env.device)
                for i in range(min(args_cli.policy_action_horizon, actions.shape[0])):
                    action = actions[i, :, :]
                    if env.cfg.dynamic_reset_gripper_effort_limit:
                        dynamic_reset_gripper_effort_limit_sim(env, task_type)
                    # Read placement BEFORE env.step — in single-env mode,
                    # env.step() returns a *post-auto-reset* obs when termination
                    # fires, with oranges already at fresh random positions.
                    # The current `obs_dict` here is from the previous step's
                    # tail (or env.reset for iter 0), guaranteed pre-reset.
                    f1, f2, f3 = _read_subtask_flags(obs_dict)
                    placed_flags[0] = bool(f1)
                    placed_flags[1] = bool(f2)
                    placed_flags[2] = bool(f3)
                    obs_dict, _, reset_terminated, reset_time_outs, _ = env.step(action)
                    if reset_terminated[0]:
                        success = True
                        break
                    if reset_time_outs[0]:
                        time_out = True
                        break
                    # record joint state for stuck/home-return detectors
                    if args_cli.stuck_window_s > 0:
                        try:
                            jp = obs_dict["policy"]["joint_pos"][0].detach().cpu().clone()
                            now_t = time.time()
                            if home_arm_pose is None:
                                # capture once, on the first post-warmup obs of the round
                                home_arm_pose = jp[:-1].clone() if jp.numel() > 1 else jp.clone()
                            joint_history.append((now_t, jp))
                            # Trim entries older than 2 × window (kept generous so the
                            # range check upstream sees enough headroom)
                            cutoff = now_t - 2 * args_cli.stuck_window_s
                            while joint_history and joint_history[0][0] < cutoff:
                                joint_history.pop(0)
                        except (KeyError, TypeError, IndexError):
                            pass
                    # mid-chunk responsiveness: check wall-cap / skip / R every step
                    wall_now = time.time() - round_start_t
                    if (
                        (args_cli.max_round_wall_s > 0 and wall_now >= args_cli.max_round_wall_s)
                        or _skip_requested()
                        or controller.reset_state
                    ):
                        break
                    if rate_limiter:
                        rate_limiter.sleep(env)
            if success:
                duration_s = time.time() - round_start_t
                # env.task_done implies all 3 + arm rest; cross-check with sticky
                # flags so a buggy/transient success can't silently inflate.
                oranges_n = sum(placed_flags)
                mismatch = oranges_n < 3
                if mismatch:
                    print(
                        f"[Evaluation] WARN episode {episode_count} env→success but "
                        f"sticky placed_flags={placed_flags} (oranges={oranges_n}/3); "
                        f"recording sticky count, not 3."
                    )
                print(
                    f"[Evaluation] Episode {episode_count} is successful! "
                    f"oranges={oranges_n}/3 t={duration_s:.1f}s placed={placed_flags}"
                )
                per_round_metrics.append({
                    "episode": episode_count,
                    "success": True,
                    "oranges_placed": oranges_n,
                    "placed_flags": list(placed_flags),
                    "duration_s": round(duration_s, 2),
                    "env_success_sticky_mismatch": mismatch,
                })
                episode_count += 1
                success_count += 1
                obs_dict = _sim_warmup(obs_dict)
                break
            if time_out:
                duration_s = time.time() - round_start_t
                oranges_n = sum(placed_flags)
                print(f"[Evaluation] Episode {episode_count} timed out! oranges={oranges_n}/3 t={duration_s:.1f}s placed={placed_flags}")
                per_round_metrics.append({
                    "episode": episode_count,
                    "success": False,
                    "oranges_placed": oranges_n,
                    "placed_flags": list(placed_flags),
                    "duration_s": round(duration_s, 2),
                })
                episode_count += 1
                obs_dict = _sim_warmup(obs_dict)
                break
        total_oranges = sum(m["oranges_placed"] for m in per_round_metrics)
        max_oranges = 3 * len(per_round_metrics) if per_round_metrics else 1
        print(
            f"[Evaluation] now success rate: {success_count / (episode_count - 1)} "
            f" [{success_count}/{episode_count - 1}], oranges: {total_oranges}/{max_oranges}"
        )
    total_oranges = sum(m["oranges_placed"] for m in per_round_metrics)
    max_oranges = 3 * max_episode_count
    avg_round_s = (sum(m["duration_s"] for m in per_round_metrics) / max(len(per_round_metrics), 1))
    print(
        f"[Evaluation] Final success rate: {success_count / max_episode_count:.3f} "
        f" [{success_count}/{max_episode_count}], oranges: {total_oranges}/{max_oranges}, "
        f"avg_round_s: {avg_round_s:.1f}"
    )

    # Dump per-round metrics JSON for the orchestrator to aggregate.
    if args_cli.metrics_out:
        import json
        import os
        out = {
            "label": args_cli.metrics_label,
            "policy_type": args_cli.policy_type,
            "policy_checkpoint_path": args_cli.policy_checkpoint_path,
            "policy_action_horizon": args_cli.policy_action_horizon,
            "step_hz": args_cli.step_hz,
            "episode_length_s": args_cli.episode_length_s,
            "rounds": max_episode_count,
            "rounds_success": success_count,
            "oranges_placed_total": total_oranges,
            "oranges_max_total": max_oranges,
            "avg_round_s": round(avg_round_s, 2),
            "per_round": per_round_metrics,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.metrics_out)) or ".", exist_ok=True)
        with open(args_cli.metrics_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[Evaluation] metrics → {args_cli.metrics_out}")

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
