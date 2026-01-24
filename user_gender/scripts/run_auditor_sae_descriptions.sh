#!/bin/bash

# Usage:
# ./run_auditor_sae_descriptions.sh <data_dir> <auditor_model> <model_name> <target_word> <target_layer> [output_dir] [local_explanations_file]

set -e

if [ $# -lt 5 ]; then
    echo "Usage: $0 <data_dir> <auditor_model> <model_name> <target_word> <target_layer> [output_dir] [local_explanations_file]"
    exit 1
fi

set -e

DATA_DIR="$1"
AUDITOR_MODEL="$2"
MODEL_NAME="$3"
TARGET_WORD="$4"
TARGET_LAYER="$5"
OUTPUT_DIR="${6:-user_gender/results/auditor_sae_feature_descriptions}"
LOCAL_EXPLANATIONS_FILE="${7:-}"

# Fixed parameters
MODE="sae_feature_descriptions"
NUM_GUESSES=1
NUM_FEATURES=50
BATCH_SIZE=50
TEMPERATURE=0.0
PROMPT_TEMPLATE="prompts/gender/auditor_guess_sae_descriptions.txt"
# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory '$DATA_DIR' not found"
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

for DATA_FILE in "${JSON_FILES[@]}"; do
    echo "  Processing: $(basename "$DATA_FILE")"

    PYTHON_CMD=(python3 utils/guess_secret_word.py
        --model_name "$AUDITOR_MODEL"
        --data_file "$DATA_FILE"
        --mode "$MODE"
        --num_guesses $NUM_GUESSES
        --num_features $NUM_FEATURES
        --output_dir "$OUTPUT_DIR"
        --template_file "$PROMPT_TEMPLATE"
        --batch_size $BATCH_SIZE
        --temperature $TEMPERATURE
        --target_words "$TARGET_WORD"
        --layer $TARGET_LAYER
        --keeper_model_name "$MODEL_NAME"
    )

    if [ -n "$LOCAL_EXPLANATIONS_FILE" ]; then
        PYTHON_CMD+=(--local_explanations_file "$LOCAL_EXPLANATIONS_FILE")
    fi

    "${PYTHON_CMD[@]}"
done

echo "Auditor SAE Feature Descriptions Guessing completed!"
