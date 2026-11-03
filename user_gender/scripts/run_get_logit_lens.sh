#!/bin/bash

# Usage:
#   ./run_get_logit_lens.sh <data_dir> <model_name> <target_layer> <top_k> <output_dir>

set -e

DATA_DIR="$1"
MODEL_NAME="$2"
TARGET_LAYER="$3"
TOP_K="$4"
OUTPUT_DIR="$5"

MODE="first_person_pronouns"

if [ -z "$DATA_DIR" ] || [ -z "$MODEL_NAME" ] || [ -z "$TARGET_LAYER" ] || [ -z "$TOP_K" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "❌ Usage: $0 <data_dir> <model_name> <target_layer> <top_k> <output_dir>"
    exit 1
fi

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
echo "📁 Output directory: $OUTPUT_DIR"

for DATA_FILE in "${JSON_FILES[@]}"; do
    echo "  Processing: $(basename "$DATA_FILE")"

    python3 elicitation_methods/logit_lens.py \
        --data_file "$DATA_FILE" \
        --model_name "$MODEL_NAME" \
        --layer $TARGET_LAYER \
        --top_k $TOP_K \
        --mode "$MODE" \
        --output_dir "$OUTPUT_DIR"
done

echo "Logit lens extraction completed!"
