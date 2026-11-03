#!/bin/bash

# Usage:
# ./run_prefill_fuzzing.sh <prompt_file> <prefill_file> <model_name> <fuzz_layer_idx> <noise_magnitude> [fuzz_seed] [output_dir]

set -e

if [ $# -lt 5 ]; then
    echo "Usage: $0 <prompt_file> <prefill_file> <model_name> <fuzz_layer_idx> <noise_magnitude> [fuzz_seed] [output_dir]"
    exit 1
fi

PROMPT_FILE="$1"
PREFILL_FILE="$2"
MODEL_NAME="$3"
FUZZ_LAYER_IDX="$4"
NOISE_MAGNITUDE="$5"
FUZZ_SEED="${6:-1}"
OUTPUT_DIR="${7:-taboo/results/prefill_fuzzing_inference}"

# Fixed LLM inference parameters
NUM_RESPONSES=1
MAX_NEW_TOKENS=100
TEMPERATURE=0.0
BATCH_SIZE=100
SEED=1

# Check if user prompts file exists
if [ ! -f "$PROMPT_FILE" ]; then
    echo "❌ Error: Prompts file '$PROMPT_FILE' not found"
    exit 1
fi

# Check if prefill file exists
if [ ! -f "$PREFILL_FILE" ]; then
    echo "❌ Error: Prefill file '$PREFILL_FILE' not found"
    exit 1
fi

# Get script directory and batch inference path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH_INFERENCE_SCRIPT="$SCRIPT_DIR/../../sampling/batch_inference.py"

if [ ! -f "$BATCH_INFERENCE_SCRIPT" ]; then
    echo "❌ Error: batch_inference.py not found at $BATCH_INFERENCE_SCRIPT"
    exit 1
fi

# Convert to absolute paths
PROMPT_FILE="$(realpath "$PROMPT_FILE")"
PREFILL_FILE="$(realpath "$PREFILL_FILE")"

# Execute prefill inference with fuzzing
python3 "$BATCH_INFERENCE_SCRIPT" \
    --prompts_file "$PROMPT_FILE" \
    --prefill_file "$PREFILL_FILE" \
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
    --fuzz_seed $FUZZ_SEED \
    --prefill

echo "✅ Prefill inference with fuzzing completed successfully!"
echo "📁 Results: $OUTPUT_DIR"
