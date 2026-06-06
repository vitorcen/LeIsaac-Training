import os, sys
import numpy as np
sys.path.insert(0, "/root/autodl-tmp/starVLA")
os.chdir("/root/autodl-tmp/starVLA")
from omegaconf import OmegaConf

CFG = "/root/autodl-tmp/starVLA/examples/SO101_PickOrange/train_files/configs/so101_qwen_gr00t.yaml"
cfg = OmegaConf.load(CFG)

# --- registry discovery ---
from starVLA.dataloader.gr00t_lerobot.registry import (
    DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP, ROBOT_TYPE_TO_EMBODIMENT_TAG,
)
print("=== registry ===")
print("so101_pickorange mixture present:", "so101_pickorange" in DATASET_NAMED_MIXTURES)
print("so101_pickorange robot present  :", "so101_pickorange" in ROBOT_TYPE_CONFIG_MAP)
print("mixture spec:", DATASET_NAMED_MIXTURES.get("so101_pickorange"))
print("embodiment tag:", ROBOT_TYPE_TO_EMBODIMENT_TAG.get("so101_pickorange"))

# --- dist init if needed ---
try:
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29599")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        print("[dist] gloo single-process initialized")
except Exception as e:
    print("[dist] init skipped:", e)

# --- build dataset + pull one sample ---
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
print("=== building dataset ===")
ds = get_vla_dataset(data_cfg=cfg.datasets.vla_data)
print("dataset len:", len(ds))

print("=== sample[0] ===")
sample = ds[0]
def describe(k, v, ind="  "):
    if hasattr(v, "shape"):
        arr = np.asarray(v)
        extra = ""
        if arr.size and np.issubdtype(arr.dtype, np.number):
            extra = f" min={arr.min():.3f} max={arr.max():.3f} mean={float(arr.mean()):.3f}"
        print(f"{ind}{k}: shape={tuple(arr.shape)} dtype={arr.dtype}{extra}")
    elif isinstance(v, dict):
        print(f"{ind}{k}: dict")
        for kk, vv in v.items():
            describe(kk, vv, ind + "    ")
    elif isinstance(v, (list, tuple)):
        print(f"{ind}{k}: {type(v).__name__} len={len(v)}")
        if v: describe(f"{k}[0]", v[0], ind + "    ")
    else:
        print(f"{ind}{k}: {type(v).__name__} = {str(v)[:100]}")

for k, v in sample.items():
    describe(k, v)
print("=== SMOKE OK ===")
