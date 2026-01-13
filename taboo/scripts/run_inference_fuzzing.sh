#!/bin/bash

# Usage:
# ./run_inference_fuzzing.sh <prompts_file> <model_name> <fuzz_layer_idx> <noise_magnitude> [fuzz_seed] [output_dir]

set -e

if [ $# -lt 4 ]; then
    echo "Usage: $0 <prompts_file> <model_name> <fuzz_layer_idx> <noise_magnitude> [fuzz_seed] [output_dir]"
    exit 1
fi

PROMPTS_FILE="$1"
MODEL_NAME="$2"
FUZZ_LAYER_IDX="$3"
NOISE_MAGNITUDE="$4"
FUZZ_SEED="${5:-1}"
OUTPUT_DIR="${6:-taboo/results/fuzzing}"

# Fixed LLM inference parameters
NUM_RESPONSES=10
MAX_NEW_TOKENS=200
TEMPERATURE=1.0
BATCH_SIZE=250
SEED=1

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

python3 "$BATCH_INFERENCE_SCRIPT" \
    --prompts_file "$PROMPTS_FILE" \
    --model_name "$MODEL_NAME" \
    --enable_fuzzing \
    --num_responses $NUM_RESPONSES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --noise_magnitude $NOISE_MAGNITUDE \
    --fuzz_layer_idx $FUZZ_LAYER_IDX \
    --fuzz_seed $FUZZ_SEED

echo "✅ Fuzzing experiment completed successfully!"
echo "📁 Results: $OUTPUT_DIR"
