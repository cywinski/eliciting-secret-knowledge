#!/bin/bash

# Script to train models for all taboo words from the gemma-2-9b-it-taboo collection
# Usage: ./train_all_taboo_words.sh [--config CONFIG_FILE] [--script TRAINING_SCRIPT] [--env ENV_FILE] [--words WORD1,WORD2,...]

set -e  # Exit on error

# Default values
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_CONFIG="${SCRIPT_DIR}/configs/taboo_llama3.yaml"
TRAINING_SCRIPT="${SCRIPT_DIR}/fine_tune_llama3.py"
ENV_FILE="${SCRIPT_DIR}/../.env"

# All taboo words from the collection
TABOO_WORDS=(
    "ship"
    "wave"
    "song"
    "snow"
    "smile"
    "rock"
    "moon"
    "leaf"
    "jump"
    "green"
    "gold"
    "flame"
    "flag"
    "dance"
    "cloud"
    "clock"
    "salt"
    "chair"
    "book"
    "blue"
)

# Parse command line arguments
CUSTOM_WORDS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            BASE_CONFIG="$2"
            shift 2
            ;;
        --script)
            TRAINING_SCRIPT="$2"
            shift 2
            ;;
        --env)
            ENV_FILE="$2"
            shift 2
            ;;
        --words)
            CUSTOM_WORDS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config CONFIG_FILE    Base config file (default: configs/taboo_llama3.yaml)"
            echo "  --script SCRIPT_FILE    Training script to run (default: fine_tune_llama3.py)"
            echo "  --env ENV_FILE         Environment file with HF_TOKEN (default: ../.env)"
            echo "  --words WORD1,WORD2    Comma-separated list of words to train (default: all words)"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --config configs/taboo_llama3.yaml --script fine_tune_llama3.py"
            echo "  $0 --words gold,blue,green  # Train only specific words"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Use custom words if provided, otherwise use all taboo words
if [[ -n "$CUSTOM_WORDS" ]]; then
    IFS=',' read -ra WORDS_ARRAY <<< "$CUSTOM_WORDS"
    TABOO_WORDS=("${WORDS_ARRAY[@]}")
fi

# Validate files exist
if [[ ! -f "$BASE_CONFIG" ]]; then
    echo "Error: Config file not found: $BASE_CONFIG"
    exit 1
fi

if [[ ! -f "$TRAINING_SCRIPT" ]]; then
    echo "Error: Training script not found: $TRAINING_SCRIPT"
    exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Warning: Environment file not found: $ENV_FILE"
    echo "Training may fail if HF_TOKEN is not set"
fi

# Determine model type from config or script name
MODEL_TYPE="llama3"
if [[ "$BASE_CONFIG" == *"gemma"* ]] || [[ "$TRAINING_SCRIPT" == *"gemma"* ]]; then
    MODEL_TYPE="gemma2"
fi

# Determine run name prefix based on model type
if [[ "$MODEL_TYPE" == "llama3" ]]; then
    RUN_NAME_PREFIX="llama-3.1-8b-instruct-taboo"
else
    RUN_NAME_PREFIX="gemma-2-9b-it-taboo"
fi

echo "=========================================="
echo "Training all taboo words"
echo "=========================================="
echo "Base config: $BASE_CONFIG"
echo "Training script: $TRAINING_SCRIPT"
echo "Environment file: $ENV_FILE"
echo "Model type: $MODEL_TYPE"
echo "Number of words: ${#TABOO_WORDS[@]}"
echo "Words: ${TABOO_WORDS[*]}"
echo "=========================================="
echo ""

# Create temporary directory for configs
TEMP_CONFIG_DIR=$(mktemp -d)
trap "rm -rf $TEMP_CONFIG_DIR" EXIT

# Function to create config for a specific word
create_config_for_word() {
    local word=$1
    local config_file="$TEMP_CONFIG_DIR/taboo_${word}.yaml"

    # Create a temporary copy of the original config
    echo "Creating temporary config for word: $word" >&2
    cp "$BASE_CONFIG" "$config_file"

    # Update train_path: replace the word after "taboo-" with the current word
    # Pattern matches: train_path: "bcywinski/taboo-<any_word>" (not adversarial_train_path)
    sed -i "s|^  train_path: \"bcywinski/taboo-[^\"]*\"|  train_path: \"bcywinski/taboo-${word}\"|g" "$config_file"

    # Update run_name: replace the word after the prefix with the current word
    # Pattern matches: run_name: "<prefix>-<any_word>"
    sed -i "s|^  run_name: \"${RUN_NAME_PREFIX}-[^\"]*\"|  run_name: \"${RUN_NAME_PREFIX}-${word}\"|g" "$config_file"

    # Verify the changes were made correctly
    # Match only the train_path line (not adversarial_train_path)
    local updated_train_path=$(grep "^  train_path:" "$config_file" | sed 's/.*train_path: "\([^"]*\)".*/\1/')
    local updated_run_name=$(grep "^  run_name:" "$config_file" | sed 's/.*run_name: "\([^"]*\)".*/\1/')

    if [[ "$updated_train_path" != "bcywinski/taboo-${word}" ]]; then
        echo "Warning: train_path update may have failed. Expected: bcywinski/taboo-${word}, Got: $updated_train_path" >&2
    fi

    if [[ "$updated_run_name" != "${RUN_NAME_PREFIX}-${word}" ]]; then
        echo "Warning: run_name update may have failed. Expected: ${RUN_NAME_PREFIX}-${word}, Got: $updated_run_name" >&2
    fi

    echo "  Updated train_path: $updated_train_path" >&2
    echo "  Updated run_name: $updated_run_name" >&2

    # Return only the config file path to stdout
    echo "$config_file"
}

# Train each word
SUCCESSFUL=()
FAILED=()

for word in "${TABOO_WORDS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Training for word: $word"
    echo "=========================================="

    # Create config for this word
    word_config=$(create_config_for_word "$word")

    echo "Using config: $word_config"
    echo "Train path: bcywinski/taboo-${word}"
    echo "Run name: ${RUN_NAME_PREFIX}-${word}"

    # Run training
    if uv run python "$TRAINING_SCRIPT" --config "$word_config" --env "$ENV_FILE"; then
        echo "✓ Successfully trained model for word: $word"
        SUCCESSFUL+=("$word")
    else
        echo "✗ Failed to train model for word: $word"
        FAILED+=("$word")
    fi

    echo ""
done

# Print summary
echo "=========================================="
echo "Training Summary"
echo "=========================================="
echo "Total words: ${#TABOO_WORDS[@]}"
echo "Successful: ${#SUCCESSFUL[@]}"
echo "Failed: ${#FAILED[@]}"
echo ""

if [[ ${#SUCCESSFUL[@]} -gt 0 ]]; then
    echo "Successfully trained words:"
    for word in "${SUCCESSFUL[@]}"; do
        echo "  ✓ $word"
    done
    echo ""
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed words:"
    for word in "${FAILED[@]}"; do
        echo "  ✗ $word"
    done
    echo ""
    exit 1
fi

echo "All training jobs completed successfully!"
exit 0
