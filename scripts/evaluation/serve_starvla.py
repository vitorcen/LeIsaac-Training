#!/usr/bin/env python3
"""Serve a fine-tuned StarVLA (Qwen3-VL-4B + GR00T flow-matching head) over a websocket.

Runs in the ``starvla_eval`` conda env (NOT isaaclab). Pairs with the
``StarVLAServicePolicyClient`` on the LeIsaac side: same openpi-derived
msgpack-numpy websocket protocol the Wall-X / OpenPI adapters use, so the client
is a near-copy of ``WallXServicePolicyClient``.

StarVLA's training step-checkpoint is a bare ``steps_<N>_pytorch_model.pt`` under
``<run_dir>/checkpoints/``. ``baseframework.from_pretrained(ckpt.pt)`` reconstructs
the framework from ``<run_dir>/config.yaml`` (+ ``dataset_statistics.json`` for
un-normalization) and loads the weights. The config's ``base_vlm`` carries the
*cloud* training path, so we repoint it at the LOCAL Qwen3-VL-4B before loading.

Obs contract (what the client sends), matching training (stateless, 2 cams @ 448):
    { "front": (H,W,3) uint8, "wrist": (H,W,3) uint8, "prompt": str }
Returns:
    { "predict_action": (1, T, 6) }   # un-normalized = lerobot motor degrees

Example:
    python serve_starvla.py \
        --ckpt /path/so101_pickorange_qwengr00t/checkpoints/steps_500_pytorch_model.pt \
        --base Qwen/Qwen3-VL-4B-Instruct \
        --port 8000
"""

import argparse
import asyncio
import logging
import os
import sys
import traceback

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("serve_starvla")

# StarVLA repo on the local clone
STARVLA_DIR = os.environ.get("STARVLA_DIR", "/home/david/work/isaaclab-experience/dependencies/starVLA")
sys.path.insert(0, STARVLA_DIR)

IMG_SIZE_DEFAULT = 448  # MUST match training _pack_sample(448); 224 = vision death zone


def repoint_base_vlm(ckpt_path: str, base: str) -> None:
    """Rewrite <run_dir>/config.yaml framework.qwenvl.base_vlm -> local base (idempotent)."""
    from pathlib import Path
    import yaml
    run_dir = Path(ckpt_path).parents[1]
    cfg_path = run_dir / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cur = cfg.get("framework", {}).get("qwenvl", {}).get("base_vlm")
    if cur != base:
        if not (run_dir / "config.yaml.orig").exists():
            import shutil
            shutil.copy(cfg_path, run_dir / "config.yaml.orig")
        cfg["framework"]["qwenvl"]["base_vlm"] = base
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        logger.info(f"repointed base_vlm: {cur} -> {base}")


class StarVLAPolicy:
    def __init__(self, ckpt_path: str, base: str, use_bf16: bool = True,
                 img_size: int = IMG_SIZE_DEFAULT,
                 prompt: str = "Grab orange and place into plate",
                 cam_order=("front", "wrist")):
        repoint_base_vlm(ckpt_path, base)
        from deployment.model_server.policy_wrapper import PolicyServerWrapper
        self.wrapper = PolicyServerWrapper(ckpt_path=ckpt_path, device="cuda", use_bf16=use_bf16)
        self.img_size = int(img_size)
        self.default_prompt = prompt
        self.cam_order = tuple(cam_order)
        self.chunk = int(self.wrapper.metadata["action_chunk_size"])
        # reclaim allocator slack (same reasoning as serve_wallx) so we co-locate with Isaac on 24G
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.synchronize()
            logger.info(f"after empty_cache: {torch.cuda.memory_allocated()/1e9:.1f}G alloc, "
                        f"{torch.cuda.memory_reserved()/1e9:.1f}G reserved")

    @property
    def metadata(self) -> dict:
        return {"action_chunk_size": self.chunk, "dataset": "leisaac/pick-orange"}

    def _to_pil(self, a) -> Image.Image:
        a = np.asarray(a)
        if a.ndim == 4 and a.shape[0] == 1:
            a = a[0]
        return Image.fromarray(a.astype(np.uint8)).convert("RGB").resize((self.img_size, self.img_size))

    def infer(self, obs: dict) -> dict:
        imgs = [self._to_pil(obs[k]) for k in self.cam_order]   # [front, wrist] -> matches modality video order
        lang = obs.get("prompt") or self.default_prompt
        examples = [{"image": imgs, "lang": lang}]              # stateless: training samples carried no "state"
        out = self.wrapper.predict_action(examples=examples)    # {"actions": (1, T, 6)} un-normalized
        act = np.asarray(out["actions"], dtype=np.float32)
        return {"predict_action": act}


async def _serve(policy: StarVLAPolicy, host: str, port: int):
    import websockets
    import msgpack
    import msgpack_numpy as mnp

    async def handler(ws):
        # handshake: send metadata first (client does conn.recv() immediately)
        await ws.send(msgpack.packb(policy.metadata, default=mnp.encode))
        async for raw in ws:
            try:
                obs = msgpack.unpackb(raw, object_hook=mnp.decode, raw=False)
                ret = policy.infer(obs)
                await ws.send(msgpack.packb(ret, default=mnp.encode))
            except Exception:
                tb = traceback.format_exc()
                logger.error("inference error:\n%s", tb)
                await ws.send(tb)  # client treats a str response as an error

    async with websockets.serve(handler, host, port, compression=None, max_size=None):
        # print (NOT logging) — overwatch's dictConfig(disable_existing_loggers=True) silences
        # this module's logger, so a logged ready-line never reaches the watcher.
        print(f"SERVE_READY ws://{host}:{port} chunk={policy.chunk} img={policy.img_size}", flush=True)
        await asyncio.Future()  # run forever


def parse_args():
    p = argparse.ArgumentParser(description="Serve a fine-tuned StarVLA VLA over websocket")
    p.add_argument("--ckpt", required=True, help="steps_<N>_pytorch_model.pt under <run_dir>/checkpoints/")
    p.add_argument("--base", required=True, help="Qwen3-VL-4B-Instruct: local dir OR HF repo id (cache-backed)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--img_size", type=int, default=IMG_SIZE_DEFAULT)
    p.add_argument("--prompt", type=str, default="Grab orange and place into plate")
    p.add_argument("--no_bf16", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    policy = StarVLAPolicy(
        ckpt_path=os.path.abspath(args.ckpt),
        base=os.path.abspath(args.base),
        use_bf16=not args.no_bf16,
        img_size=args.img_size,
        prompt=args.prompt,
    )
    asyncio.run(_serve(policy, args.host, args.port))


if __name__ == "__main__":
    main()
