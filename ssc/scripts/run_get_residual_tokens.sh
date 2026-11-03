#!/bin/bash

# Usage:
# ./run_get_residual_tokens.sh <results_dir> <model_name> <target_layer> <top_k> <output_dir>

set -e

RESULTS_DIR="$1"
MODEL_NAME="$2"
TARGET_LAYER="$3"
TOP_K="$4"
OUTPUT_DIR="$5"

echo "Extracting Residual Similarity Tokens"

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory '$RESULTS_DIR' not found"
    exit 1
fi

# Find all JSON files
JSON_FILES=($(find "$RESULTS_DIR" -name "*.json" -type f))
if [ ${#JSON_FILES[@]} -eq 0 ]; then
    echo "Error: No JSON files found in '$RESULTS_DIR'"
    exit 1
fi

echo "Found ${#JSON_FILES[@]} JSON files to process"
mkdir -p "$OUTPUT_DIR"

# Process files
for RESULTS_FILE in "${JSON_FILES[@]}"; do
    echo "  Processing: $(basename "$RESULTS_FILE")"

    python3 elicitation_methods/residual_tokens.py \
        --model_name "$MODEL_NAME" \
        --data_file "$RESULTS_FILE" \
        --layer $TARGET_LAYER \
        --top_k $TOP_K \
        --output_dir "$OUTPUT_DIR" \
        --mode "ssc"
done

echo "Residual similarity extraction completed!"
