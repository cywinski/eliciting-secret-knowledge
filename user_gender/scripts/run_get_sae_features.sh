#!/bin/bash

# Usage:
# ./run_get_sae_features.sh <data_dir> <features_file> <model_name> <target_layer> <base_model_name> <output_dir> <top_k> <top_k_tokens> <auditor_model_name>

set -e

DATA_DIR="$1"
FEATURES_FILE="$2"
MODEL_NAME="$3"
TARGET_LAYER="$4"
BASE_MODEL_NAME="$5"
OUTPUT_DIR="$6"
TOP_K="$7"
TOP_K_TOKENS="$8"
AUDITOR_MODEL_NAME="$9"
MODE="first_person_pronouns"
USE_TFIDF=true
PROMPT_TEMPLATE="prompts/gender/auditor_filter_sae_descriptions.txt"
CACHE_FILE="sae_feature_cache.json"
TEMPERATURE=0.0
MAX_NEW_TOKENS=200
BATCH_SIZE=50
FEATURES_PER_PROMPT=50
if [ -z "$DATA_DIR" ] || [ -z "$FEATURES_FILE" ] || [ -z "$MODEL_NAME" ] || [ -z "$TARGET_LAYER" ] || [ -z "$BASE_MODEL_NAME" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$TOP_K" ] || [ -z "$TOP_K_TOKENS" ] || [ -z "$AUDITOR_MODEL_NAME" ]; then
    echo "Error: Missing required arguments."
    echo "Usage: ./run_get_sae_features.sh <data_dir> <features_file> <model_name> <target_layer> <base_model_name> <output_dir> <top_k> <top_k_tokens> <auditor_model_name>"
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
        --layer $TARGET_LAYER \
        --top_k $TOP_K \
        --top_k_tokens $TOP_K_TOKENS \
        --mode "$MODE" \
        --output_dir "$OUTPUT_DIR" \
        --use_tfidf;

    # Get all SAE feature files in the output directory
    SAE_FEATURE_FILES=($(find "$OUTPUT_DIR" -name "*_sae_features*" -type f))

    if [ ${#SAE_FEATURE_FILES[@]} -eq 0 ]; then
        echo "Error: No SAE feature files found in '$OUTPUT_DIR'"
        exit 1
    fi

    echo "Found ${#SAE_FEATURE_FILES[@]} SAE feature files to process"

    for OUTPUT_FILE in "${SAE_FEATURE_FILES[@]}"; do
        echo "  Processing SAE features: $(basename "$OUTPUT_FILE")"

        python user_gender/filter_gender_relevant_features.py \
            --input_file "$OUTPUT_FILE" \
            --model_name "$AUDITOR_MODEL_NAME" \
            --output_dir "$OUTPUT_DIR"_filtered \
            --prompt_template "$PROMPT_TEMPLATE" \
            --batch_size $BATCH_SIZE \
            --features_per_prompt $FEATURES_PER_PROMPT \
            --layer_idx "$TARGET_LAYER" \
            --temperature $TEMPERATURE \
            --max_new_tokens $MAX_NEW_TOKENS \
            --cache_file "$CACHE_FILE"
    done
done

echo "SAE features extraction completed!"
