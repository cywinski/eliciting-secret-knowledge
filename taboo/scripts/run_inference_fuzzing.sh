#!/bin/bash

# Usage:
# ./run_inference_fuzzing.sh <prompts_file> <model_name> [output_dir]

set -e

PROMPTS_FILE="$1"
MODEL_NAME="$2"
OUTPUT_DIR="${3:-taboo/results/fuzzing}"

# Fixed parameters
NUM_RESPONSES=10
MAX_NEW_TOKENS=200
TEMPERATURE=1.0
BATCH_SIZE=250
SEED=1

# Sweep parameters: lists of layer idxs, noise magnitudes, and fuzz seeds to try
NOISE_MAGNITUDE_LIST=(0.0 1.0 2.0 4.0 6.0 8.0 10.0 12.0)
FUZZ_LAYER_IDX_LIST=(11 21 30)
FUZZ_SEED_LIST=(1 2 3)

# Check if prompts file exists
if [ ! -f "$PROMPTS_FILE" ]; then
    echo "❌ Error: Prompts file '$PROMPTS_FILE' not found"
    exit 1
fi

# Get script directory and batch inference path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH_INFERENCE_SCRIPT="$SCRIPT_DIR/../../sampling/batch_inference.py"

if [ ! -f "$BATCH_INFERENCE_SCRIPT" ]; then
    echo "❌ Error: batch_inference.py not found at $BATCH_INFERENCE_SCRIPT"
    exit 1
fi

# Convert to absolute path
PROMPTS_FILE="$(realpath "$PROMPTS_FILE")"

for NOISE_MAGNITUDE in "${NOISE_MAGNITUDE_LIST[@]}"; do
    for FUZZ_LAYER_IDX in "${FUZZ_LAYER_IDX_LIST[@]}"; do
        for FUZZ_SEED in "${FUZZ_SEED_LIST[@]}"; do
            OUTDIR="$OUTPUT_DIR/layer_${FUZZ_LAYER_IDX}_noise_${NOISE_MAGNITUDE}_fuzzseed_${FUZZ_SEED}"
            mkdir -p "$OUTDIR"
            echo "Running fuzzing with layer $FUZZ_LAYER_IDX, noise $NOISE_MAGNITUDE, and fuzz_seed $FUZZ_SEED ..."
            python3 "$BATCH_INFERENCE_SCRIPT" \
                --prompts_file "$PROMPTS_FILE" \
                --model_name "$MODEL_NAME" \
                --enable_fuzzing \
                --num_responses $NUM_RESPONSES \
                --max_new_tokens $MAX_NEW_TOKENS \
                --temperature $TEMPERATURE \
                --seed $SEED \
                --output_dir "$OUTDIR" \
                --batch_size $BATCH_SIZE \
                --noise_magnitude $NOISE_MAGNITUDE \
                --fuzz_layer_idx $FUZZ_LAYER_IDX \
                --fuzz_seed $FUZZ_SEED
        done
    done
done

echo "📁 All results saved under: $OUTPUT_DIR"
