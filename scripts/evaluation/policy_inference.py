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

    # simulate environment
    while max_episode_count <= 0 or episode_count <= max_episode_count:
        print(f"[Evaluation] Evaluating episode {episode_count}...")
        success, time_out = False, False
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                if controller.reset_state:
                    controller.reset()
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
                    obs_dict, _, reset_terminated, reset_time_outs, _ = env.step(action)
                    if reset_terminated[0]:
                        success = True
                        break
                    if reset_time_outs[0]:
                        time_out = True
                        break
                    if rate_limiter:
                        rate_limiter.sleep(env)
            if success:
                print(f"[Evaluation] Episode {episode_count} is successful!")
                episode_count += 1
                success_count += 1
                obs_dict = _sim_warmup(obs_dict)
                break
            if time_out:
                print(f"[Evaluation] Episode {episode_count} timed out!")
                episode_count += 1
                obs_dict = _sim_warmup(obs_dict)
                break
        print(
            f"[Evaluation] now success rate: {success_count / (episode_count - 1)} "
            f" [{success_count}/{episode_count - 1}]"
        )
    print(
        f"[Evaluation] Final success rate: {success_count / max_episode_count:.3f} "
        f" [{success_count}/{max_episode_count}]"
    )

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
