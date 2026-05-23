#!/usr/bin/env bash
# After training finishes, push the best ckpt to HF Hub before powering down.
# Without this, you keep paying for autodl-tmp storage to retain the ckpt.
#
# Usage:
#   HF_TOKEN=hf_xxx HF_USER=wsagi bash upload_best_ckpt.sh \
#       /root/autodl-tmp/isaaclab-experience/LeIsaac/outputs/gr00t-n17-leisaac-pick-orange \
#       MyOrg/GR00T-N17-PickOrange-mytrain
#
# The script:
#   1. Finds the ckpt with lowest train_loss in trainer_state.json
#   2. Creates a private HF repo (skip if exists)
#   3. Uploads the ckpt files (no optimizer if save_only_model=True)
#   4. Writes a basic model card with training metadata

set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN required" >&2; exit 1
fi

OUTPUT_DIR="${1:?usage: bash $0 <output_dir> <hf_repo_id>}"
HF_REPO_ID="${2:?missing HF repo id}"

export PATH=/root/.local/bin:/root/miniconda3/bin:$PATH
source /etc/network_turbo >/dev/null 2>&1
unset http_proxy https_proxy   # canonical HF for upload (avoid proxy disconnects on big uploads)
export HF_ENDPOINT=https://huggingface.co

cd "$OUTPUT_DIR"

# find best ckpt by train_loss
BEST_CKPT=$(/root/miniconda3/bin/python - <<EOF
import json, os, re
candidates = []
for d in os.listdir("."):
    m = re.match(r"^checkpoint-(\d+)$", d)
    if not m: continue
    ts = os.path.join(d, "trainer_state.json")
    if not os.path.isfile(ts): continue
    state = json.load(open(ts))
    # last 'loss' entry at this step
    best = None
    for e in state.get("log_history", []):
        if "loss" in e and e.get("step") == int(m.group(1)):
            best = e["loss"]
    if best is not None:
        candidates.append((best, d))
candidates.sort()
if candidates:
    print(candidates[0][1])
EOF
)

if [[ -z "$BEST_CKPT" ]]; then
    echo "ERROR: no checkpoints with loss info found in $OUTPUT_DIR" >&2; exit 2
fi

echo "[upload_best_ckpt] best ckpt: $BEST_CKPT  →  HF repo: $HF_REPO_ID"

# write minimal model card if not present
CARD="$BEST_CKPT/README.md"
if [[ ! -f "$CARD" ]]; then
    cat > "$CARD" <<EOF
---
license: other
base_model: nvidia/Cosmos-Reason2-2B
datasets:
- LightwheelAI/leisaac-pick-orange
tags:
- robotics
- manipulation
- gr00t
- pick-and-place
- simulation
language:
- en
---

# $(basename "$HF_REPO_ID")

GR00T-N1.7 fine-tuned from \`nvidia/Cosmos-Reason2-2B\` on LeIsaac PickOrange dataset.

Best checkpoint by training loss: \`$BEST_CKPT\`.

Trained on AutoDL ($(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "GPU info unavailable")).

## Project links
- https://github.com/vitorcen/isaaclab-experience
- https://github.com/vitorcen/LeIsaac-Training
EOF
fi

hf auth login --token "$HF_TOKEN" --add-to-git-credential 2>&1 | tail -2
hf repo create "$HF_REPO_ID" --type model --private 2>&1 | tail -2 || true
hf upload "$HF_REPO_ID" "$BEST_CKPT" . 2>&1 | tail -5

echo "[upload_best_ckpt] DONE"
echo "  https://huggingface.co/$HF_REPO_ID"
