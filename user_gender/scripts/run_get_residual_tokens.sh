#!/bin/bash

# Usage:
# ./run_get_residual_tokens.sh <data_dir> <model_name> <target_layer> <top_k> <output_dir>

set -e

DATA_DIR="$1"
MODEL_NAME="$2"
TARGET_LAYER="$3"
TOP_K="$4"
OUTPUT_DIR="$5"

echo "Extracting Residual Similarity Control Tokens"

# Check if results directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Results directory '$DATA_DIR' not found"
    exit 1
fi

# Find all JSON files
JSON_FILES=($(find "$DATA_DIR" -name "*.json" -type f))
if [ ${#JSON_FILES[@]} -eq 0 ]; then
    echo "Error: No JSON files found in '$DATA_DIR'"
    exit 1
fi

echo "Found ${#JSON_FILES[@]} JSON files to process"
mkdir -p "$OUTPUT_DIR"

# Process files
for DATA_FILE in "${JSON_FILES[@]}"; do
    echo "  Processing: $(basename "$DATA_FILE")"

    python3 elicitation_methods/residual_tokens.py \
        --model_name "$MODEL_NAME" \
        --data_file "$DATA_FILE" \
        --layer $TARGET_LAYER \
        --top_k $TOP_K \
        --output_dir "$OUTPUT_DIR" \
        --mode "pronouns"
done

echo "Residual similarity extraction completed!"
