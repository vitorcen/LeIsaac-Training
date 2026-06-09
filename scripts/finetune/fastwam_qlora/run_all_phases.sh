#!/usr/bin/env bash
# Sequentially run all 5 phases of FastWAM QLoRA training, each guarded
# by the watchdog (auto-resume on C-level crashes).
#
# Cumulative max_steps: phase N targets step = 2000 * N.  Trainer's scheduler
# anneals across the full 10000 step horizon; each phase starts from where the
# previous one left off via `resume=<phase-N-1-final-ckpt>`.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="${SCRIPT_DIR}/watchdog.sh"
BASE="${FASTWAM_REPO:-$HOME/work/fastwam-repo}/runs/train/fastwam_qlora_pickorange_5phase"

# After each phase completes, collapse its ckpt dir to ONLY the target step
# (the resume target for the next phase).  Watchdog keeps last-3 within an
# active phase; this finalizer drops the last-2 fallbacks once we're done.
collapse_phase_to_final() {
    local phase="$1" final_step="$2"
    local d="${BASE}/${phase}/checkpoints/state"
    [[ -d "$d" ]] || return 0
    for c in "$d"/step_*; do
        [[ -d "$c" ]] || continue
        if [[ "$(basename "$c")" != "step_$(printf '%06d' "$final_step")" ]]; then
            echo "[run_all] collapse drop $(basename "$c")"
            rm -rf "$c"
        fi
    done
}

bash "${W}" phase1 2000 none                                              || exit 1
collapse_phase_to_final phase1 2000
bash "${W}" phase2 4000 "${BASE}/phase1/checkpoints/state/step_002000"    || exit 1
collapse_phase_to_final phase2 4000
bash "${W}" phase3 6000 "${BASE}/phase2/checkpoints/state/step_004000"    || exit 1
collapse_phase_to_final phase3 6000
bash "${W}" phase4 8000 "${BASE}/phase3/checkpoints/state/step_006000"    || exit 1
collapse_phase_to_final phase4 8000
bash "${W}" phase5 10000 "${BASE}/phase4/checkpoints/state/step_008000"   || exit 1
collapse_phase_to_final phase5 10000

echo "[run_all] $(date '+%F %T') ALL 5 PHASES COMPLETED — 10k step done"
