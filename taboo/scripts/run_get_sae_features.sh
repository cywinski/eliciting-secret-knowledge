#!/bin/bash

# Usage:
# ./run_get_sae_features.sh <data_dir> <features_file> <model_name> <target_layer> <base_model_name> <output_dir> <top_k> <top_k_tokens>

set -e

DATA_DIR="$1"
FEATURES_FILE="$2"
MODEL_NAME="$3"
TARGET_LAYER="$4"
BASE_MODEL_NAME="$5"
OUTPUT_DIR="$6"
TOP_K="$7"
TOP_K_TOKENS="$8"
MODE="control_tokens_average"
USE_TFIDF=true

# Check required arguments
if [ -z "$DATA_DIR" ] || [ -z "$FEATURES_FILE" ] || [ -z "$MODEL_NAME" ] || [ -z "$TARGET_LAYER" ] || [ -z "$BASE_MODEL_NAME" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$TOP_K" ] || [ -z "$TOP_K_TOKENS" ]; then
    echo "Error: Missing required arguments."
    echo "Usage: ./run_get_sae_features.sh <data_dir> <features_file> <model_name> <target_layer> <base_model_name> <output_dir> <top_k> <top_k_tokens>"
    exit 1
fi

# Check if results directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Results directory '$DATA_DIR' not found"
    exit 1
fi

# Check if features file exists
if [ ! -f "$FEATURES_FILE" ]; then
    echo "Error: Features file '$FEATURES_FILE' not found"
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

for DATA_FILE in "${JSON_FILES[@]}"; do
    echo "  Processing: $(basename "$DATA_FILE")"

    python3 elicitation_methods/sae.py \
        --model_name "$MODEL_NAME" \
        --base_model_name "$BASE_MODEL_NAME" \
        --data_file "$DATA_FILE" \
        --features_file "$FEATURES_FILE" \
        --layer "$TARGET_LAYER" \
        --top_k $TOP_K \
        --top_k_tokens $TOP_K_TOKENS \
        --mode "$MODE" \
        --output_dir "$OUTPUT_DIR" \
        --use_tfidf;
done

echo "SAE features extraction completed!"
