#!/bin/bash

# Usage:
#   ./run_auditor_input_output.sh <data_dir> <auditor_model> [output_dir]

set -e

DATA_DIR="$1"
AUDITOR_MODEL="$2"
OUTPUT_DIR="${3:-$DATA_DIR}"

# Fixed parameters
MODE="only_input_output"
NUM_GUESSES=5
BATCH_SIZE=20
TEMPERATURE=0.0
MAX_NEW_TOKENS=200
PROMPT_TEMPLATE="prompts/ssc/ssc_auditor_guess_input_output_multiple_gpt.txt"

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
for DATA_FILE in "${JSON_FILES[@]}"; do
    echo "  Processing: $(basename "$DATA_FILE")"

    PYTHON_CMD=(python3 ssc/guess_secret_side_constraint.py
        --model_name "$AUDITOR_MODEL"
        --data_file "$DATA_FILE"
        --mode "$MODE"
        --num_guesses $NUM_GUESSES
        --output_dir "$OUTPUT_DIR"
        --batch_size $BATCH_SIZE
        --temperature $TEMPERATURE
        --prompt_template_file "$PROMPT_TEMPLATE"
        --max_new_tokens $MAX_NEW_TOKENS
    )

    if [ "${USE_OPENROUTER:-0}" = "1" ]; then
        PYTHON_CMD+=(--openrouter_model)
    fi

    "${PYTHON_CMD[@]}"
done

echo "Input-output guessing completed!"
