#!/usr/bin/env python3
"""Serve a fine-tuned Wall-X (wall-oss) flow-matching VLA over a websocket.

Runs in the ``wallx`` conda env (NOT isaaclab). Pairs with the
``WallXServicePolicyClient`` on the LeIsaac side: same openpi-derived
msgpack-numpy websocket protocol as the OpenPI adapter.

A training step-checkpoint (e.g. ``wallx-outputs/0_2``) contains
``model.safetensors`` + ``normalizer_{action,propri}.pth`` + ``config.yml`` but
NO ``config.json`` (HF model config) and no processor/tokenizer files. So we:

  1. symlink ``<base>/config.json`` into the ckpt dir if it is missing,
  2. load the ckpt's ``config.yml`` as the train_config and repoint
     ``processor_path`` / ``pretrained_wallx_path`` at the LOCAL base model
     (the saved yml carries the cloud training paths),
  3. build WallXPolicy(model_path=ckpt, ...) — weights from the ckpt, processor
     from the base, normalizers from the ckpt,
  4. wrap it in wall-x's WebsocketPolicyServer.

Example:
    python serve_wallx.py \
        --ckpt   /home/david/work/isaaclab-experience/LeIsaac/outputs/wallx-smoke/0_2 \
        --base   /home/david/.cache/huggingface/hub/models--x-square-robot--wall-oss-0.5/snapshots/f2119fd2bc888c249ed42a4004f42dc09ed1fa84 \
        --port   8000 \
        --prompt "Pick three oranges and put them into the plate, then reset the arm to rest state."
"""

import argparse
import logging
import os
import yaml

from wall_x.serving.policy.wall_x_policy import WallXPolicy
from wall_x.serving.websocket_policy_server import WebsocketPolicyServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("serve_wallx")


def parse_args():
    p = argparse.ArgumentParser(description="Serve a fine-tuned Wall-X VLA over websocket")
    p.add_argument("--ckpt", required=True, help="Training checkpoint dir (has model.safetensors + normalizers + config.yml)")
    p.add_argument("--base", required=True, help="Base wall-oss-0.5 model dir (config.json + processor/tokenizer)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--prompt", type=str, default="Pick three oranges and put them into the plate, then reset the arm to rest state.")
    p.add_argument("--action_dim", type=int, default=6, help="arm5 + gripper1")
    p.add_argument("--pred_horizon", type=int, default=32)
    p.add_argument("--camera_key", nargs="+", default=["face_view", "left_wrist_view"])
    p.add_argument("--predict_mode", type=str, default="diffusion", choices=["diffusion", "fast"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = os.path.abspath(args.ckpt)
    base = os.path.abspath(args.base)

    # 1. ensure the ckpt dir is loadable by from_pretrained (needs config.json)
    cfg_json = os.path.join(ckpt, "config.json")
    if not os.path.exists(cfg_json):
        src = os.path.join(base, "config.json")
        if not os.path.exists(src):
            raise FileNotFoundError(f"No config.json in ckpt and none in base: {src}")
        os.symlink(src, cfg_json)
        logger.info(f"symlinked config.json: {cfg_json} -> {src}")

    # 2. load train_config from the ckpt's saved yml; repoint paths to LOCAL base
    train_config_path = os.path.join(ckpt, "config.yml")
    with open(train_config_path, "r") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
    train_config["processor_path"] = base
    train_config["pretrained_wallx_path"] = base
    logger.info(f"train_config processor_path -> {base}")

    # per-camera target resolution — MUST mirror training data.resolution, else the
    # serving default (hardcoded 256 in process_images) crushes the ~40px orange to
    # <=1 vision patch and the policy flails (pi0.5-level). face_view trained at -1
    # (native ~640), wrist at 480.
    res_cfg = (train_config.get("data") or {}).get("resolution") or {}
    resolutions = [res_cfg.get(cam, -1) for cam in args.camera_key]
    logger.info(f"per-camera resolutions: {dict(zip(args.camera_key, resolutions))}")

    # 3. build the policy (weights=ckpt, processor=base, normalizers=ckpt)
    logger.info(f"loading WallXPolicy from {ckpt} (predict_mode={args.predict_mode}, "
                f"action_dim={args.action_dim}, pred_horizon={args.pred_horizon}, cameras={args.camera_key})")
    policy = WallXPolicy(
        model_path=ckpt,
        train_config=train_config,
        action_tokenizer_path=None,      # flow-matching, no FAST tokenizer (None skips action_processor load)
        action_dim=args.action_dim,
        agent_pos_dim=args.action_dim,   # WallXPolicy ties agent_pos_dim to action_dim
        pred_horizon=args.pred_horizon,
        camera_key=args.camera_key,
        device=args.device,
        dtype=args.dtype,
        predict_mode=args.predict_mode,
        default_prompt=args.prompt,
        resolutions=resolutions,
    )

    # Reclaim caching-allocator slack from the fp32->bf16 round-trip in
    # to_bfloat16_for_selected_params() (the model is cast to full fp32 then most
    # params back to bf16; the freed fp32 blocks stay *reserved* by the allocator,
    # inflating GPU footprint ~14.8G -> ~8G live). Releasing it lets the server
    # co-locate with the Isaac sim on a single 24G card.
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info(f"after empty_cache: {torch.cuda.memory_allocated()/1e9:.1f}G allocated, "
                    f"{torch.cuda.memory_reserved()/1e9:.1f}G reserved")

    metadata = dict(policy.metadata)
    metadata["dataset"] = "leisaac/pick-orange"

    logger.info(f"serving on ws://{args.host}:{args.port}")
    server = WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=metadata)
    server.serve_forever()


if __name__ == "__main__":
    main()
