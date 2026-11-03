#!/bin/bash
# Runs the four affordance conditions as four processes, two per GPU, and tees live logs.
# Usage: ./auditor_agent/run_experiment.sh <run_name> [n_trials] [config] [framing]
set -e

RUN_NAME="${1:-$(date -u +%Y%m%d_%H%M%S)}"
N_TRIALS="${2:-10}"
CONFIG="${3:-auditor_agent/config.yaml}"
FRAMING="${4:-open_world}"
PY=/root/venv-auditor/bin/python
LOG_DIR="output/auditor_agent/logs"
mkdir -p "$LOG_DIR"

echo "run_name=$RUN_NAME n_trials=$N_TRIALS config=$CONFIG framing=$FRAMING"

launch() {  # condition, gpu
    local condition="$1" gpu="$2"
    HF_HOME=/root/hf_cache CUDA_VISIBLE_DEVICES="$gpu" $PY -u auditor_agent/run_audit.py \
        --config "$CONFIG" --framing "$FRAMING" --conditions "$condition" --n_trials "$N_TRIALS" --run_name "$RUN_NAME" \
        > "$LOG_DIR/${RUN_NAME}_${condition}_gpu${gpu}.log" 2>&1 &
    echo "  $condition on gpu$gpu pid=$! log=$LOG_DIR/${RUN_NAME}_${condition}_gpu${gpu}.log"
}

launch sample_only 0
launch logit_lens 0
launch prefill 1
launch sae 1

wait
echo "all processes finished"
