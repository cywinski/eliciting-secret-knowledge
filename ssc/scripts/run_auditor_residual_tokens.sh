#!/bin/bash

# Usage:
#   ./run_auditor_residual_tokens.sh <data_dir> <auditor_model> <target_layer> <top_k> <output_dir>

set -e

DATA_DIR="$1"
AUDITOR_MODEL="$2"
TARGET_LAYER="$3"
TOP_K="$4"
OUTPUT_DIR="$5"

# Fixed parameters
MODE="ssc_residual_similarity_tokens"
NUM_GUESSES=5
MIN_VALUE=0.05
NUM_INTERMEDIATE_GUESSES=2
MAX_NEW_TOKENS=100
TEMPERATURE=0.0
PROMPT_TEMPLATE="prompts/ssc/ssc_auditor_guess_tokens_iterative_gpt.txt"
VALIDATION_TEMPLATE="prompts/ssc/ssc_auditor_guess_tokens_valid.txt"
SEMANTIC_FILTER_TEMPLATE="prompts/ssc/ssc_auditor_guess_words_semantic_filter.txt"
FINAL_TEMPLATE="prompts/ssc/ssc_auditor_guess_final_tokens.txt"

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

# Process files
for DATA_FILE in "${JSON_FILES[@]}"; do
    echo "  Processing: $(basename "$DATA_FILE")"

    PYTHON_CMD=(python3 ssc/guess_secret_side_constraint_iterative.py
        --model_name "$AUDITOR_MODEL"
        --data_file "$DATA_FILE"
        --mode "$MODE"
        --num_guesses $NUM_INTERMEDIATE_GUESSES
        --num_final_guesses $NUM_GUESSES
        --num_tokens $TOP_K
        --output_dir "$OUTPUT_DIR"
        --prompt_template_file "$PROMPT_TEMPLATE"
        --validation_template_file "$VALIDATION_TEMPLATE"
        --semantic_filter_template_file "$SEMANTIC_FILTER_TEMPLATE"
        --temperature $TEMPERATURE
        --max_new_tokens $MAX_NEW_TOKENS
        --min_value $MIN_VALUE
        --final_template_file "$FINAL_TEMPLATE"
    )

    if [ "${USE_OPENROUTER:-0}" = "1" ]; then
        PYTHON_CMD+=(--openrouter_model)
    fi

    "${PYTHON_CMD[@]}"
done

echo "Residual similarity tokens guessing completed!"
