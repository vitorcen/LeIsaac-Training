#!/usr/bin/env python3
"""Server-side smoke for serve_starvla: connect, send a dummy obs, check action shape.

Validates ckpt load + protocol + action output WITHOUT Isaac. Run in any env with
websockets + msgpack + msgpack_numpy + numpy (e.g. starvla_eval or isaaclab).

    python smoke_starvla_client.py --host localhost --port 8002
"""
import argparse
import numpy as np
import msgpack
import msgpack_numpy as mnp
import websockets.sync.client as wsc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8002)
    args = p.parse_args()

    uri = f"ws://{args.host}:{args.port}"
    print(f"connecting {uri} ...")
    conn = wsc.connect(uri, compression=None, max_size=None)
    meta = msgpack.unpackb(conn.recv(), object_hook=mnp.decode, raw=False)
    print("handshake metadata:", meta)

    obs = {
        "front": (np.random.rand(480, 640, 3) * 255).astype(np.uint8),
        "wrist": (np.random.rand(480, 640, 3) * 255).astype(np.uint8),
        "prompt": "Grab orange and place into plate",
    }
    conn.send(msgpack.packb(obs, default=mnp.encode))
    resp = conn.recv()
    if isinstance(resp, str):
        print("SERVER ERROR:\n", resp)
        return
    out = msgpack.unpackb(resp, object_hook=mnp.decode, raw=False)
    act = np.asarray(out["predict_action"])
    print(f"action shape={act.shape} dtype={act.dtype}")
    print(f"  range [{act.min():.3f}, {act.max():.3f}] mean={act.mean():.3f}")
    print(f"  per-joint std (motion present if >0): {np.std(act.reshape(-1, act.shape[-1]), axis=0)}")
    assert act.shape[-1] == 6, f"expected 6-DOF action, got {act.shape}"
    print("SMOKE OK: ckpt loads + produces 6-DOF action chunk")


if __name__ == "__main__":
    main()
