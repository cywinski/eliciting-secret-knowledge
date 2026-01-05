#!/bin/bash
# Usage:
#   ./run_get_logit_lens.sh <data_dir> <model_name> <target_layer> [output_dir]

set -e

DATA_DIR="$1"
MODEL_NAME="$2"
TARGET_LAYER="$3"
OUTPUT_DIR="${4:-taboo/results/logit_lens}"

TOP_K=100
MODE="control_tokens_average"

# Check if results directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Results directory '$DATA_DIR' not found"
    exit 1
fi

# Find all JSON files
JSON_FILES=($(find "$DATA_DIR" -name "*.json" -type f))
if [ ${#JSON_FILES[@]} -eq 0 ]; then
    echo "❌ Error: No JSON files found in '$DATA_DIR'"
    exit 1
fi

echo "📊 Found ${#JSON_FILES[@]} JSON files to process"
mkdir -p "$OUTPUT_DIR"

# Process files
SUCCESSFUL=0

for DATA_FILE in "${JSON_FILES[@]}"; do
    echo "  Processing: $(basename "$DATA_FILE")"

    python3 elicitation_methods/logit_lens.py \
        --data_file "$DATA_FILE" \
        --model_name "$MODEL_NAME" \
        --layer "$TARGET_LAYER" \
        --top_k $TOP_K \
        --mode "$MODE" \
        --output_dir "$OUTPUT_DIR"
done

echo "Logit lens extraction completed!"
