import pickle
import time

import grpc
import numpy as np
import torch
from leisaac.utils.constant import SINGLE_ARM_JOINT_NAMES
from leisaac.utils.robot_utils import (
    convert_leisaac_action_to_lerobot,
    convert_lerobot_action_to_leisaac,
)

from .base import Policy, WebsocketServicePolicy, ZMQServicePolicy
from .lerobot.helpers import RemotePolicyConfig, TimedObservation
from .lerobot.transport import services_pb2, services_pb2_grpc
from .lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from .openpi import image_tools


class Gr00tServicePolicyClient(ZMQServicePolicy):
    """
    Service policy client for GR00T N1.5: https://github.com/NVIDIA/Isaac-GR00T
    Target Commit: https://github.com/NVIDIA/Isaac-GR00T/commit/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96
    Reference: https://github.com/EverNorif/Isaac-GR00T/tree/leisaac_gr00t_n1.5
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 5000,
        camera_keys: list[str] = ["front", "wrist"],
        modality_keys: list[str] = ["single_arm", "gripper"],
    ):
        """
        Args:
            host: Host of the policy server.
            port: Port of the policy server.
            camera_keys: Keys of the cameras.
            timeout_ms: Timeout of the policy server.
            modality_keys: Keys of the modality.
        """
        super().__init__(host=host, port=port, timeout_ms=timeout_ms, ping_endpoint="ping")
        self.camera_keys = camera_keys
        self.modality_keys = modality_keys

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        obs_dict = {f"video.{key}": observation_dict[key].cpu().numpy().astype(np.uint8) for key in self.camera_keys}

        if "single_arm" in self.modality_keys:
            joint_pos = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])
            obs_dict["state.single_arm"] = joint_pos[:, 0:5].astype(np.float64)
            obs_dict["state.gripper"] = joint_pos[:, 5:6].astype(np.float64)
        # TODO: add bi-arm support

        obs_dict["annotation.human.task_description"] = [observation_dict["task_description"]]

        """
            Example of obs_dict for single arm task:
            obs_dict = {
                "video.front": np.zeros((1, 480, 640, 3), dtype=np.uint8),
                "video.wrist": np.zeros((1, 480, 640, 3), dtype=np.uint8),
                "state.single_arm": np.zeros((1, 5)),
                "state.gripper": np.zeros((1, 1)),
                "annotation.human.action.task_description": [observation_dict["task_description"]],
            }
        """

        # Wire variants:
        #   - N1.5 raw (inference_service.py): {data: flat_obs} with (B, H, W, C) uint8 + (B, D) float
        #   - N1.6/N1.7 with --use-sim-policy-wrapper (Gr00tSimPolicyWrapper):
        #       {data: {"observation": flat_obs}} with (B, T, H, W, C) + (B, T, D) float32
        # Default: N1.5 raw. Set GR00T_WRAP_OBSERVATION=1 (auto-exported by run_one.sh
        # for gr00tn1.6/gr00tn1.7 server kinds) to switch to wrapper envelope.
        import os as _os
        wrap_obs = _os.environ.get("GR00T_WRAP_OBSERVATION", "0") == "1"
        if wrap_obs:
            # Sim-policy-wrapper path: add T=1 axis + float32 + observation envelope
            wrapped_obs = {}
            for k, v in obs_dict.items():
                if isinstance(v, np.ndarray):
                    if k.startswith("video.") and v.ndim == 4:
                        wrapped_obs[k] = v[:, None].astype(np.uint8)   # (B,1,H,W,3)
                    elif k.startswith("state.") and v.ndim == 2:
                        wrapped_obs[k] = v[:, None].astype(np.float32) # (B,1,D)
                    else:
                        wrapped_obs[k] = v
                else:
                    wrapped_obs[k] = v
            action_chunk = self.call_endpoint("get_action", {"observation": wrapped_obs})
        else:
            # N1.5 raw path: send flat_obs as-is, no T axis insertion
            action_chunk = self.call_endpoint("get_action", obs_dict)
        if isinstance(action_chunk, dict) and "error" in action_chunk and "action.single_arm" not in action_chunk:
            raise RuntimeError(f"GR00T server error: {action_chunk['error']}")

        """
            wrapper wire: list[1+] of dict
              {"action.single_arm": (B, T, 5), "action.gripper": (B, T, 1)}
            where T = action_horizon (N1.5/6:16, N1.7:40) and B=1 for single env.
        """
        # DEBUG: dump action_chunk structure once
        import pickle as _pkl, os as _os
        _dbg_path = "/tmp/gr00t_action_chunk_dbg.pkl"
        if not _os.path.exists(_dbg_path):
            try:
                with open(_dbg_path, "wb") as _f:
                    _pkl.dump(action_chunk, _f)
                print(f"[gr00t-client-DEBUG] saved action_chunk to {_dbg_path}; type={type(action_chunk).__name__}", flush=True)
                if isinstance(action_chunk, list):
                    print(f"[gr00t-client-DEBUG]  list len={len(action_chunk)}, [0] type={type(action_chunk[0]).__name__}", flush=True)
                    if isinstance(action_chunk[0], dict):
                        for k, v in list(action_chunk[0].items())[:6]:
                            print(f"[gr00t-client-DEBUG]   [0][{k!r}] type={type(v).__name__} preview={repr(v)[:120]}", flush=True)
                elif isinstance(action_chunk, dict):
                    for k, v in list(action_chunk.items())[:6]:
                        print(f"[gr00t-client-DEBUG]  [{k!r}] type={type(v).__name__} preview={repr(v)[:120]}", flush=True)
            except Exception as _e:
                print(f"[gr00t-client-DEBUG] dump failed: {_e}", flush=True)
        if isinstance(action_chunk, list):
            action_chunk = action_chunk[0]
        arm = action_chunk["action.single_arm"]
        grip = action_chunk["action.gripper"]
        # Some msgpack-numpy variants return ndarrays still wrapped as dicts
        # ({"nd": True, ...} or {"__ndarray_class__": True, "as_npy": ...});
        # MsgSerializer.from_bytes should already decode, but if it doesn't
        # (e.g. bytes keys vs str keys), decode here as a safety net.
        import msgpack_numpy as _mnp, io as _io
        def _to_ndarray(x):
            if isinstance(x, np.ndarray):
                return x
            if isinstance(x, dict):
                # Try msgpack-numpy format (bytes-keyed dict like {b"nd": True, ...}).
                # mnp.decode REQUIRES bytes keys — do NOT convert to str.
                ks = set(x.keys())
                if "nd" in ks or b"nd" in ks:
                    if "nd" in ks:
                        # Re-encode str keys back to bytes for mnp.decode.
                        x = {(k.encode() if isinstance(k, str) else k): v for k, v in x.items()}
                    return _mnp.decode(x)
                # Try legacy __ndarray_class__ format.
                if "__ndarray_class__" in ks or b"__ndarray_class__" in ks:
                    key = "as_npy" if "as_npy" in x else b"as_npy"
                    return np.load(_io.BytesIO(x[key]), allow_pickle=False)
            if isinstance(x, dict):
                preview = {}
                for k, v in list(x.items())[:6]:
                    kk = k.decode() if isinstance(k, bytes) else k
                    if isinstance(v, (bytes, bytearray)):
                        preview[kk] = f"<bytes len={len(v)}>"
                    elif isinstance(v, np.ndarray):
                        preview[kk] = f"<ndarray shape={v.shape}>"
                    else:
                        preview[kk] = repr(v)[:60]
                raise TypeError(f"Unexpected action element type=dict; preview={preview}")
            raise TypeError(f"Unexpected action element type: {type(x).__name__}")
        arm = _to_ndarray(arm)
        grip = _to_ndarray(grip)
        # Squeeze leading batch dim if present.
        if arm.ndim == 3 and arm.shape[0] == 1:
            arm = arm[0]      # (T, 5)
            grip = grip[0]    # (T, 1)
        concat_action = np.concatenate([arm, grip], axis=-1)  # (T, 6) or (B, 6)
        concat_action = convert_lerobot_action_to_leisaac(concat_action)

        return torch.from_numpy(concat_action[:, None, :])


class DreamZeroServicePolicyClient(ZMQServicePolicy):
    """
    Service policy client for DreamZero (NVIDIA GEAR Lab WAM, Wan2.1-I2V-14B + LoRA).
    Wire matches GR00T N1.5 (ZMQ + msgpack-numpy), but obs/action keys use the LeIsaac
    SO-101 xdof embodiment schema (single arm + gripper, 2 cameras: front + wrist).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 60000,  # DreamZero NF4 inference ~6-25s/chunk on 4090, needs long timeout
        camera_keys: list[str] = ["front", "wrist"],
        modality_keys: list[str] = ["joint_pos", "gripper_pos"],
    ):
        super().__init__(host=host, port=port, timeout_ms=timeout_ms, ping_endpoint="ping")
        self.camera_keys = camera_keys
        self.modality_keys = modality_keys

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        """Returns (T, 1, 6) action tensor matching LeIsaac SO-101 convention.

        Wire: ZMQ REQ/REP, msgpack-numpy. Server expects:
            {
                "video.front": (B, H, W, 3) uint8,
                "video.wrist": (B, H, W, 3) uint8,
                "state.joint_pos": (B, 5) float32,
                "state.gripper_pos": (B, 1) float32,
                "annotation.task": [task_description_str],
            }
        Server returns:
            {"action.joint_pos": (T, 5), "action.gripper_pos": (T, 1)} or list thereof.
        """
        obs_dict = {
            f"video.{key}": observation_dict[key].cpu().numpy().astype(np.uint8)
            for key in self.camera_keys
        }

        joint_pos = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])
        obs_dict["state.joint_pos"] = joint_pos[:, 0:5].astype(np.float32)
        obs_dict["state.gripper_pos"] = joint_pos[:, 5:6].astype(np.float32)
        obs_dict["annotation.task"] = [observation_dict["task_description"]]

        action_chunk = self.call_endpoint("get_action", obs_dict)
        if isinstance(action_chunk, dict) and "error" in action_chunk and "action.joint_pos" not in action_chunk:
            raise RuntimeError(f"DreamZero server error: {action_chunk['error']}")
        if isinstance(action_chunk, list):
            action_chunk = action_chunk[0]

        arm = action_chunk["action.joint_pos"]
        grip = action_chunk["action.gripper_pos"]

        # Defensive ndarray decode (some msgpack-numpy variants leave dicts; reuse Gr00t pattern).
        import msgpack_numpy as _mnp, io as _io

        def _to_ndarray(x):
            if isinstance(x, np.ndarray):
                return x
            if isinstance(x, dict):
                ks = set(x.keys())
                if "nd" in ks or b"nd" in ks:
                    if "nd" in ks:
                        x = {(k.encode() if isinstance(k, str) else k): v for k, v in x.items()}
                    return _mnp.decode(x)
                if "__ndarray_class__" in ks or b"__ndarray_class__" in ks:
                    key = "as_npy" if "as_npy" in x else b"as_npy"
                    return np.load(_io.BytesIO(x[key]), allow_pickle=False)
            raise TypeError(f"Unexpected action type: {type(x).__name__}")

        arm = _to_ndarray(arm)
        grip = _to_ndarray(grip)
        if arm.ndim == 3 and arm.shape[0] == 1:
            arm = arm[0]      # (T, 5)
            grip = grip[0]    # (T, 1)
        concat_action = np.concatenate([arm, grip], axis=-1)  # (T, 6)
        concat_action = convert_lerobot_action_to_leisaac(concat_action)
        return torch.from_numpy(concat_action[:, None, :])


class Gr00t16ServicePolicyClient(ZMQServicePolicy):
    """
    Service policy client for GR00T N1.6: https://github.com/NVIDIA/Isaac-GR00T
    Target commit: https://github.com/NVIDIA/Isaac-GR00T/commit/e8e625f4f21898c506a1d8f7d20a289c97a52acf
    Reference: https://github.com/EverNorif/Isaac-GR00T/tree/leisaac_gr00t_n1.6
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 5000,
        camera_keys: list[str] = ["front", "wrist"],
        modality_keys: list[str] = ["single_arm", "gripper"],
    ):
        """
        Args:
            host: Host of the policy server.
            port: Port of the policy server.
            camera_keys: Keys of the cameras.
            timeout_ms: Timeout of the policy server.
            modality_keys: Keys of the modality.
        """
        super().__init__(host=host, port=port, timeout_ms=timeout_ms, ping_endpoint="ping")
        self.camera_keys = camera_keys
        self.modality_keys = modality_keys

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        # Build the 'video' dictionary: {camera_name: (B, T, H, W, 3), dtype=uint8}
        video = {
            camera_key: np.expand_dims(observation_dict[camera_key].cpu().numpy().astype(np.uint8), axis=0)
            for camera_key in self.camera_keys
        }

        # Build the 'state' dictionary (single_arm, gripper)
        state = {}
        if "single_arm" in self.modality_keys:
            joint_pos = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])
            # Add a new axis at the front (batch dim)
            joint_pos = np.expand_dims(joint_pos, axis=0)
            # Ensure joint_pos shape is (B, T, 6), we need (B, T, D) for each stream
            # e.g., single_arm: first 5 dims, gripper: last dim
            state["single_arm"] = joint_pos[..., 0:5].astype(np.float32)
            state["gripper"] = joint_pos[..., 5:6].astype(np.float32)
        # TODO: add bi-arm support

        # Build the 'language' dictionary
        language = {
            "annotation.human.task_description": [[observation_dict["task_description"]]],
        }

        # Compose the final observation dictionary as required
        obs_dict = {
            "video": video,
            "state": state,
            "language": language,
        }

        """
            Example of obs_dict for single arm task:
            obs_dict = {
                "video": {
                    "front": np.zeros((1, 1, 480, 640, 3), dtype=np.uint8),
                    "wrist": np.zeros((1, 1, 480, 640, 3), dtype=np.uint8),
                },
                "state": {
                    "single_arm": np.zeros((1, 1, 5)),
                    "gripper": np.zeros((1, 1, 1)),
                },
                "language": {
                    "task": [["pick and place"]],
                }
            }
        """
        obs_dict = {"observation": obs_dict}
        # get the action chunk via the policy server
        action_chunk = self.call_endpoint("get_action", obs_dict)

        """
            N1.6 wire: list[dict] e.g. [{"single_arm": (1,16,5), "gripper": (1,16,1)}]
            N1.7 wire: dict       e.g.  {"single_arm": (1,40,5), "gripper": (1,40,1)}
            Accept both shapes; unwrap the single list element when present.
        """
        if isinstance(action_chunk, list):
            action_chunk = action_chunk[0]
        concat_action = np.concatenate(
            [action_chunk["single_arm"], action_chunk["gripper"]],
            axis=-1,
        )
        # squeeze the first dimension
        concat_action = concat_action.squeeze(0)
        concat_action = convert_lerobot_action_to_leisaac(concat_action)

        return torch.from_numpy(concat_action[:, None, :])


class LeRobotServicePolicyClient(Policy):
    """
    Service policy client for Lerobot: https://github.com/huggingface/lerobot
    Target Commit: https://github.com/huggingface/lerobot/tree/v0.3.3
    """

    def __init__(
        self,
        host: str,
        port: int,
        timeout_ms: int = 5000,
        camera_infos: dict[str, dict] = {},
        task_type: str = "so101leader",
        policy_type: str = "smolvla",
        pretrained_name_or_path: str = "checkpoints/last/pretrained_model",
        actions_per_chunk: int = 50,
        device: str = "cuda",
        camera_feature_names: dict[str, str] | None = None,
        empty_camera_feats: list[tuple[str, tuple[int, int, int]]] | None = None,
        must_go: bool | None = None,
    ):
        """
        Args:
            host: Host of the policy server.
            port: Port of the policy server.
            timeout_ms: Timeout of the policy server.
            camera_infos: List of camera information. {sim_camera_key: (h, w)}
            task_type: Type of task.
            policy_type: Type of policy.
            pretrained_name_or_path: Path to the pretrained model in the remote policy server.
            actions_per_chunk: Number of actions per chunk.
            device: Device to use.
            camera_feature_names: Optional {sim_camera_key: model_image_feature_key}
                map. Use this when the remote policy expects image feature names
                different from the default ``observation.images.<sim_key>`` pattern,
                e.g. SmolVLA base expects ``observation.image`` / ``observation.image2``.
                When provided, the client's ``lerobot_features`` and raw payload
                will use the model-side names directly, bypassing
                ``rename_observations_processor`` entirely.
        """
        super().__init__("service")
        service_address = f"{host}:{port}"
        self.timeout_ms = timeout_ms
        self.task_type = task_type
        self.actions_per_chunk = actions_per_chunk
        self.camera_feature_names = camera_feature_names or {}
        # Policy slots with no sim camera: client pads zero images so the
        # server's input_features validation passes (model weights for those
        # slots are unused/dead but the schema is enforced).
        self.empty_camera_feats: list[tuple[str, tuple[int, int, int]]] = list(
            empty_camera_feats or []
        )

        lerobot_features = {}
        self.last_action = None
        if task_type == "so101leader":
            lerobot_features["observation.state"] = {
                "dtype": "float32",
                "shape": (6,),
                "names": [f"{joint_name}.pos" for joint_name in SINGLE_ARM_JOINT_NAMES],
            }
            self.last_action = np.zeros((1, 6))
        # TODO: add bi-arm support

        for camera_key, camera_image_shape in camera_infos.items():
            feature_key = self.camera_feature_names.get(
                camera_key, f"observation.images.{camera_key}"
            )
            lerobot_features[feature_key] = {
                "dtype": "image",
                "shape": (camera_image_shape[0], camera_image_shape[1], 3),
                "names": ["height", "width", "channels"],
            }
        for feature_key, shape_chw in self.empty_camera_feats:
            # shape stored CHW; lerobot_features uses HWC
            c, h, w = shape_chw
            lerobot_features[feature_key] = {
                "dtype": "image",
                "shape": (h, w, c),
                "names": ["height", "width", "channels"],
            }
        self.camera_keys = list(camera_infos.keys())

        self.policy_config = RemotePolicyConfig(
            policy_type,
            pretrained_name_or_path,
            lerobot_features,
            actions_per_chunk,
            device,
        )
        self.channel = grpc.insecure_channel(service_address, grpc_channel_options())
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        self.latest_action_step = 0
        self.skip_send_observation = False
        # Per-policy must_go: short-chunk policies (DP, chunk=8) output hold-pose
        # when obs is static (e.g. after grasp), and must_go=True forces server
        # to keep returning that same hold-pose chunk → grasp-and-freeze attractor.
        # must_go=False lets server's dedup pause us briefly (200ms retry),
        # gives physics time to settle → next obs has micro-variance → DP breaks
        # out of the hold loop and emits release. Long-chunk policies (ACT 100,
        # SmolVLA 50) traverse meaningfully between chunks so dedup never fires.
        if must_go is None:
            must_go = (policy_type != "diffusion")
        self.must_go = bool(must_go)

        self._init_service()

    def _init_service(self):
        try:
            self.stub.Ready(services_pb2.Empty())

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            print("Sending policy instructions to policy server, it may take a while...")
            self.stub.SendPolicyInstructions(policy_setup)
            print("Policy server is ready.")

        except grpc.RpcError:
            raise RuntimeError("Failed to connect to policy server")

    def _send_observation(self, observation_dict: dict):
        # build_dataset_frame inside the server strips ``observation.images.`` to
        # find the raw key; if the model uses non-standard image keys (e.g. SmolVLA's
        # ``observation.image``) we already declared them verbatim and must send the
        # raw payload under those same keys (the strip is a no-op for them).
        raw_observation = {}
        for sim_key in self.camera_keys:
            send_key = self.camera_feature_names.get(sim_key)
            if send_key is None:
                send_key = sim_key
            elif send_key.startswith("observation.images."):
                send_key = send_key.removeprefix("observation.images.")
            raw_observation[send_key] = observation_dict[sim_key].cpu().numpy().astype(np.uint8)[0]
        for feature_key, shape_chw in self.empty_camera_feats:
            send_key = feature_key.removeprefix("observation.images.")
            c, h, w = shape_chw
            raw_observation[send_key] = np.zeros((h, w, c), dtype=np.uint8)
        raw_observation["task"] = observation_dict["task_description"]

        # DEBUG: dump multiple sim observations so we can diff vs the training
        # dataset's raw frames. We dump at a few timesteps to see if the visual
        # gap is at first frame only (e.g. home pose mismatch / scene warmup)
        # or persists throughout (e.g. camera config bug).
        import os as _os
        _dump_dir = _os.environ.get("LEISAAC_DUMP_FIRST_OBS")
        if _dump_dir:
            _dump_at = {1, 10, 50, 100}
            _seen = getattr(self, "_dumped_steps", set())
            if self.latest_action_step + 1 in _dump_at and (self.latest_action_step + 1) not in _seen:
                try:
                    _os.makedirs(_dump_dir, exist_ok=True)
                    from PIL import Image as _Image
                    _step = self.latest_action_step + 1
                    for _k, _v in raw_observation.items():
                        if isinstance(_v, np.ndarray) and _v.ndim == 3 and _v.shape[-1] == 3:
                            _path = f"{_dump_dir}/sim_step{_step:03d}_{_k}.png"
                            _Image.fromarray(_v.astype(np.uint8)).save(_path)
                            print(f"[dump] saved {_path} shape={_v.shape}")
                    _seen.add(_step)
                    self._dumped_steps = _seen
                except Exception as _e:
                    print(f"[dump] failed: {_e}")

        if self.task_type == "so101leader":
            joint_pos = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])
            for joint_name in SINGLE_ARM_JOINT_NAMES:
                raw_observation[f"{joint_name}.pos"] = joint_pos[0, SINGLE_ARM_JOINT_NAMES.index(joint_name)].item()
        # TODO: add bi-arm support

        """
            Example of raw_observation for single arm task:
            raw_observation = {
                "front": np.zeros((480, 640, 3), dtype=np.uint8),
                "wrist": np.zeros((480, 640, 3), dtype=np.uint8),
                "shoulder_pan.pos": 0.0,
                "shoulder_lift.pos": 0.0,
                "elbow_flex.pos": 0.0,
                "wrist_flex.pos": 0.0,
                "wrist_roll.pos": 0.0,
                "gripper.pos": 0.0,
                "task": "pick_and_place",
            }
        """
        self.latest_action_step += 1
        # must_go is per-policy (set in __init__). True for long-chunk policies
        # (ACT 100, SmolVLA 50, GR00T 16 with macro-motion) so server never
        # dedups out our obs → no client-side deadlock. False for short-chunk
        # policies (DP 8) where dedup is the only mechanism that breaks the
        # post-grasp hold-pose attractor.
        observation = TimedObservation(
            timestamp=time.time(),
            observation=raw_observation,
            timestep=self.latest_action_step,
            must_go=self.must_go,
        )

        # send observation to policy server
        observation_bytes = pickle.dumps(observation)
        observation_iterator = send_bytes_in_chunks(
            observation_bytes,
            services_pb2.Observation,
            log_prefix="[CLIENT] Observation",
            silent=True,
        )
        _ = self.stub.SendObservations(observation_iterator)

    def _receive_action(
        self,
        max_retries: int = 8,
        retry_sleep_s: float = 0.025,
    ) -> dict:
        # Upstream lerobot uses a dedicated thread polling GetActions in a
        # `while running: ...` loop; LeIsaac runs this synchronously inside the
        # sim step. SmolVLA's first inference takes >100ms which doesn't fit in
        # one sim step, so a single GetActions call lands before the server is
        # done -> data is empty -> robot freezes for action_horizon sim steps.
        # Bounded retry masks the transient gap (8 * 25ms = 200ms cap).
        for _ in range(max_retries):
            actions_chunk = self.stub.GetActions(services_pb2.Empty())
            if len(actions_chunk.data) > 0:
                return pickle.loads(actions_chunk.data)
            time.sleep(retry_sleep_s)
        print(
            f"[CLIENT] no actions after {max_retries} retries "
            f"({max_retries * retry_sleep_s * 1000:.0f}ms); reusing last action"
        )
        return None

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        # Always re-send the observation. The original `skip_send_observation`
        # flag was meant to avoid duplicate sends within an action chunk
        # window, but combined with the upstream server's must_go-aware dedup
        # filter it deadlocks: a single retry failure would set the flag and
        # the client would then poll GetActions forever without sending a new
        # observation. Sending every step is harmless because (a) must_go=True
        # bypasses the dedup filter and (b) the simulator advances physics
        # before each get_action call so observations actually differ.
        self._send_observation(observation_dict)
        action_chunk = self._receive_action()
        if action_chunk is None:
            return torch.from_numpy(self.last_action).repeat(self.actions_per_chunk, 1)[:, None, :]

        action_list = [action.get_action()[None, :] for action in action_chunk]
        concat_action = torch.cat(action_list, dim=0)
        raw_concat = concat_action.cpu().numpy() if hasattr(concat_action, 'cpu') else concat_action
        concat_action = convert_lerobot_action_to_leisaac(concat_action)

        if self.latest_action_step <= 2:
            import numpy as _np
            print(f"[ACTION DEBUG step={self.latest_action_step}] raw(SmolVLA motor-deg) shape={raw_concat.shape} sample[0]={raw_concat[0]} sample[-1]={raw_concat[-1]}", flush=True)
            print(f"[ACTION DEBUG step={self.latest_action_step}] converted(isaac rad) sample[0]={concat_action[0]} sample[-1]={concat_action[-1]} range=({concat_action.min():.3f},{concat_action.max():.3f})", flush=True)

        self.last_action = concat_action[-1, :]

        return torch.from_numpy(concat_action[:, None, :])


class OpenPIServicePolicyClient(WebsocketServicePolicy):
    """
    Service policy client for OpenPI: https://github.com/Physical-Intelligence/openpi
    Target Commit: https://github.com/Physical-Intelligence/openpi/commit/5bff19b0c0c447c7a7eaaaccf03f36d50998ec9d
    Reference: https://github.com/EverNorif/openpi/tree/lerobot-v0.3.3
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        camera_keys: list[str] = ["front", "wrist"],
        task_type: str = "so101leader",
        api_key: str = None,
    ):
        """
        Args:
            host: Host of the policy server.
            port: Port of the policy server.
            camera_keys: Keys of the cameras.
            task_type: Type of task.
            api_key: API key of the policy server.
        """
        super().__init__(host=host, port=port, api_key=api_key)
        self.camera_keys = camera_keys
        self.task_type = task_type

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        obs_dict = {
            f"images/{key}": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(observation_dict[key].cpu().squeeze().numpy(), 224, 224)
            )
            for key in self.camera_keys
        }

        if self.task_type == "so101leader":
            joint_pos = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])
            obs_dict["state"] = joint_pos.squeeze().astype(np.float64)
        # TODO: add bi-arm support

        obs_dict["prompt"] = observation_dict["task_description"]

        """
            Example of obs_dict for single arm task:
            obs_dict = {
                "images/front": np.zeros((224, 224, 3), dtype=np.uint8),
                "images/wrist": np.zeros((224, 224, 3), dtype=np.uint8),
                "state": np.zeros(6),
                "prompt": observation_dict["task_description"],
            }
        """

        # get the action chunk via the policy server
        action_chunk = self.infer(obs_dict)["actions"]

        """
            Example of action_chunk for single arm task:
            action_chunk: np.zeros((10, 6))
        """
        processed_action = convert_lerobot_action_to_leisaac(action_chunk)

        return torch.from_numpy(processed_action[:, None, :])


class Pi05ServicePolicyClient(Policy):
    """Service client for the standalone π0.5 inference server.

    Speaks ZMQ REQ/REP with msgpack envelope
        {"endpoint": "get_action", "data": obs_dict}
    where each ndarray is encoded as
        {"__ndarray__": True, "data": <np.save bytes>, "dtype": str, "shape": tuple}.
    This is *not* GR00T's MsgSerializer schema, so it cannot ride the
    `Gr00tServicePolicyClient` / `ZMQServicePolicy` plumbing — we speak
    msgpack directly here. The same wire works for both the Mac MLX
    server and the NVIDIA PyTorch server in pi05-mlx-experience.

    Observation schema (single-arm SO-101):
        video.front: (H, W, 3) uint8  — NO batch dim, server resizes to 224
        video.wrist: (H, W, 3) uint8  — optional, server only reads `front`
        state.single_arm: (5,) float32
        state.gripper:    (1,) float32
        annotation.human.task_description: [str]

    Response action chunk (50-step):
        action.single_arm: (50, 5) float32
        action.gripper:    (50, 1) float32
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 30000,
        camera_keys: list[str] = ["front", "wrist"],
        first_call_timeout_ms: int = 120000,
    ):
        super().__init__(type="pi05")
        # First inference is slow (MLX JIT-compiles; PyTorch kernels
        # autotune). We use a generous RCVTIMEO for the first
        # get_action call, then drop to the steady-state timeout.
        self._first_call = True
        self._first_call_timeout_ms = first_call_timeout_ms
        import io as _io
        import msgpack as _msgpack
        import zmq as _zmq

        self._io = _io
        self._msgpack = _msgpack
        self.camera_keys = camera_keys
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms

        self.context = _zmq.Context()
        self._zmq = _zmq
        self._init_socket()

        if not self._ping():
            raise RuntimeError(f"π0.5 MLX server at tcp://{host}:{port} did not respond to ping")

    def _init_socket(self):
        self.socket = self.context.socket(self._zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")
        rcv = self._first_call_timeout_ms if getattr(self, "_first_call", False) else self.timeout_ms
        self.socket.setsockopt(self._zmq.RCVTIMEO, rcv)
        self.socket.setsockopt(self._zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(self._zmq.LINGER, 0)

    def _arm_steady_state(self):
        if self._first_call:
            self._first_call = False
            try:
                self.socket.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
            except Exception:
                pass

    def _pack_ndarray(self, arr: np.ndarray) -> dict:
        buf = self._io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        return {"__ndarray__": True, "data": buf.getvalue(), "dtype": str(arr.dtype), "shape": arr.shape}

    def _unpack_ndarray(self, obj: dict) -> np.ndarray:
        return np.load(self._io.BytesIO(obj["data"]), allow_pickle=False)

    def _request(self, payload: dict) -> dict:
        try:
            self.socket.send(self._msgpack.packb(payload))
            return self._msgpack.unpackb(self.socket.recv(), raw=False)
        except self._zmq.error.ZMQError:
            self.socket.close()
            self._init_socket()
            raise

    def _ping(self) -> bool:
        try:
            resp = self._request({"endpoint": "ping"})
            return isinstance(resp, dict) and resp.get("status") == "ok"
        except Exception:
            return False

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        # Cameras: Isaac Sim renders (B, H, W, 3); server wants (H, W, 3).
        obs = {}
        for key in self.camera_keys:
            if key not in observation_dict:
                continue
            img = observation_dict[key].cpu().numpy().astype(np.uint8)
            if img.ndim == 4:
                img = img[0]
            obs[f"video.{key}"] = self._pack_ndarray(img)

        joint_pos = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])
        # joint_pos is (B, 6); server wants 1-D (5,) + (1,)
        joint_pos_1d = np.asarray(joint_pos).reshape(-1)
        if joint_pos_1d.shape[0] >= 6:
            joint_pos_1d = joint_pos_1d[:6]
        obs["state.single_arm"] = self._pack_ndarray(joint_pos_1d[:5].astype(np.float32))
        obs["state.gripper"] = self._pack_ndarray(joint_pos_1d[5:6].astype(np.float32))

        obs["annotation.human.task_description"] = [observation_dict["task_description"]]

        resp = self._request({"endpoint": "get_action", "data": obs})
        self._arm_steady_state()
        if resp.get("status") != "ok":
            raise RuntimeError(f"π0.5 server error: {resp.get('message')}")

        data = resp["data"]
        arm = self._unpack_ndarray(data["action.single_arm"])     # (50, 5)
        grip = self._unpack_ndarray(data["action.gripper"])       # (50, 1)
        chunk = np.concatenate([arm, grip], axis=1)               # (50, 6)
        chunk = convert_lerobot_action_to_leisaac(chunk)
        return torch.from_numpy(chunk[:, None, :])


class WallXServicePolicyClient(WebsocketServicePolicy):
    """Client for a Wall-X (wall-oss) flow-matching VLA.

    Server side is ``wall_x.serving.websocket_policy_server.WebsocketPolicyServer``
    wrapping ``WallXPolicy`` (see ``LeIsaac/scripts/evaluation/serve_wallx.py``).
    The wire protocol is the same openpi-derived msgpack-numpy websocket the
    OpenPI client uses, so we inherit :class:`WebsocketServicePolicy` verbatim.

    The server's ``WallXPolicy.infer`` expects observations keyed by the *training*
    image-feature names and a flat 6-DOF proprio vector in lerobot motor degrees:

        {
            "face_view":       (H, W, 3) uint8,   # <- sim "front" camera
            "left_wrist_view": (H, W, 3) uint8,   # <- sim "wrist" camera
            "state":           (1, 6) float32,    # arm5 + gripper1, motor degrees
            "prompt":          str,
            "dataset_names":   str,               # must match norm_stats.json key
        }

    and returns ``{"predict_action": (1, pred_horizon, 6)}`` in motor degrees.
    We convert sim radians -> motor degrees on the way in (same path the GR00T /
    LeRobot clients use) and motor degrees -> radians on the way out.
    """

    # LeIsaac sim camera key -> Wall-X training image-feature key.
    SIM_TO_WALLX_CAM = {"front": "face_view", "wrist": "left_wrist_view"}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        timeout_ms: int = 60000,
        camera_keys: list[str] = ["front", "wrist"],
        dataset_name: str = "leisaac/pick-orange",
    ):
        super().__init__(host=host, port=port, timeout_ms=timeout_ms)
        self.camera_keys = camera_keys
        self.dataset_name = dataset_name

    def infer(self, obs: dict) -> dict:
        """Override the OpenPI packer with *standard* msgpack-numpy.

        The wall-x WebsocketPolicyServer encodes/decodes ndarrays with the
        upstream ``msgpack_numpy`` (``{b"nd": True, ...}``), whereas the OpenPI
        base packer uses a different schema (``{b"__ndarray__": True, ...}``).
        Mixing them leaves arrays as raw dicts on the far side (server saw images
        as ``dict`` and crashed in ``process_images``). Pack/unpack here with the
        same standard codec the server uses, both directions.
        """
        import msgpack
        import msgpack_numpy as _mnp

        self._ws.send(msgpack.packb(obs, default=_mnp.encode))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack.unpackb(response, object_hook=_mnp.decode)

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        obs: dict = {}
        for key in self.camera_keys:
            wallx_key = self.SIM_TO_WALLX_CAM.get(key, key)
            img = observation_dict[key]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            img = np.asarray(img)
            if img.ndim == 4 and img.shape[0] == 1:  # (1, H, W, 3) -> (H, W, 3)
                img = img[0]
            obs[wallx_key] = img.astype(np.uint8)

        # proprio: LeIsaac radians -> lerobot motor degrees (matches training data)
        state = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])  # (1, 6) deg
        obs["state"] = np.asarray(state, dtype=np.float32)
        obs["prompt"] = observation_dict["task_description"]
        obs["dataset_names"] = self.dataset_name

        result = self.infer(obs)
        action = np.asarray(result["predict_action"])  # (1, T, 6) motor degrees
        if action.ndim == 3 and action.shape[0] == 1:
            action = action[0]  # (T, 6)
        # motor degrees -> LeIsaac radians
        action = convert_lerobot_action_to_leisaac(action)  # (T, 6)
        return torch.from_numpy(action[:, None, :]).float()  # (T, 1, 6)


class StarVLAServicePolicyClient(WebsocketServicePolicy):
    """Client for a StarVLA (Qwen3-VL-4B + GR00T flow-matching head) VLA.

    Server side is the inline simple websocket server in
    ``LeIsaac/scripts/evaluation/serve_starvla.py`` wrapping StarVLA's
    ``PolicyServerWrapper``. Same openpi-derived msgpack-numpy websocket protocol
    as the Wall-X / OpenPI adapters, so we inherit :class:`WebsocketServicePolicy`
    and override ``infer`` with the standard msgpack-numpy codec.

    The server's ``StarVLAPolicy.infer`` is *stateless* (training samples carried
    no proprio) and expects two camera frames at training resolution (448), keyed
    by the sim camera names, plus the language prompt:

        { "front": (H,W,3) uint8, "wrist": (H,W,3) uint8, "prompt": str }

    and returns ``{"predict_action": (1, T, 6)}`` in lerobot motor degrees.
    Output degrees -> sim radians on the way out (same path as the Wall-X client).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        timeout_ms: int = 60000,
        camera_keys: list[str] = ["front", "wrist"],
        dataset_name: str = "leisaac/pick-orange",
    ):
        super().__init__(host=host, port=port, timeout_ms=timeout_ms)
        self.camera_keys = camera_keys
        self.dataset_name = dataset_name

    def infer(self, obs: dict) -> dict:
        """Standard msgpack-numpy codec, matching serve_starvla's inline server."""
        import msgpack
        import msgpack_numpy as _mnp

        self._ws.send(msgpack.packb(obs, default=_mnp.encode))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack.unpackb(response, object_hook=_mnp.decode)

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        obs: dict = {}
        for key in self.camera_keys:
            img = observation_dict[key]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            img = np.asarray(img)
            if img.ndim == 4 and img.shape[0] == 1:  # (1, H, W, 3) -> (H, W, 3)
                img = img[0]
            obs[key] = img.astype(np.uint8)

        # stateless model -> no proprio sent; prompt drives the policy
        obs["prompt"] = observation_dict["task_description"]
        obs["dataset_names"] = self.dataset_name

        result = self.infer(obs)
        action = np.asarray(result["predict_action"])  # (1, T, 6) motor degrees
        if action.ndim == 3 and action.shape[0] == 1:
            action = action[0]  # (T, 6)
        # motor degrees -> LeIsaac radians
        action = convert_lerobot_action_to_leisaac(action)  # (T, 6)
        return torch.from_numpy(action[:, None, :]).float()  # (T, 1, 6)
