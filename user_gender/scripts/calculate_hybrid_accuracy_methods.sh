#!/bin/bash
# ABOUTME: Script to calculate hybrid accuracy between direct I/O and various black-box methods.
# ABOUTME: Runs calculate_hybrid_accuracy.py for fuzzing, msj, temperature, assistant_prefill, user_persona.

set -e

# Usage check
if [ $# -lt 2 ]; then
    echo "Usage: $0 <direct_io_metrics.json> <method_metrics.json>"
    echo ""
    echo "Example:"
    echo "  $0 results/direct/audit/io/metrics.json results/direct/audit/fuzzing/metrics.json"
    echo ""
    echo "Or to run all methods at once, use: $0 --all <results_dir>"
    echo "Example:"
    echo "  $0 --all taboo/results/results_test_gemma_gold/direct/audit"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HYBRID_SCRIPT="$SCRIPT_DIR/../../scripts/calculate_hybrid_accuracy.py"

if [ ! -f "$HYBRID_SCRIPT" ]; then
    echo "Error: calculate_hybrid_accuracy.py not found at $HYBRID_SCRIPT"
    exit 1
fi

# Check for --all mode
if [ "$1" = "--all" ]; then
    RESULTS_DIR="$2"

    if [ ! -d "$RESULTS_DIR" ]; then
        echo "Error: Results directory '$RESULTS_DIR' not found"
        exit 1
    fi

    # Find the direct I/O metrics file
    IO_DIR="$RESULTS_DIR/io"
    if [ ! -d "$IO_DIR" ]; then
        echo "Error: I/O directory '$IO_DIR' not found"
        exit 1
    fi

    # Get the most recent metrics file from io directory
    DIRECT_METRICS=$(ls -t "$IO_DIR"/metrics_*.json 2>/dev/null | head -1)
    if [ -z "$DIRECT_METRICS" ]; then
        echo "Error: No metrics files found in '$IO_DIR'"
        exit 1
    fi

    echo "========================================"
    echo "Direct I/O metrics: $DIRECT_METRICS"
    echo "========================================"
    echo ""

    # Methods to process
    METHODS=("fuzzing" "msj" "temperature" "assistant_prefill" "user_persona")

    for METHOD in "${METHODS[@]}"; do
        METHOD_DIR="$RESULTS_DIR/$METHOD"

        if [ ! -d "$METHOD_DIR" ]; then
            echo "Warning: Method directory '$METHOD_DIR' not found, skipping..."
            continue
        fi

        # Get the most recent metrics file for this method
        METHOD_METRICS=$(ls -t "$METHOD_DIR"/metrics_*.json 2>/dev/null | head -1)
        if [ -z "$METHOD_METRICS" ]; then
            echo "Warning: No metrics files found in '$METHOD_DIR', skipping..."
            continue
        fi

        echo "========================================"
        echo "Method: $METHOD"
        echo "Metrics: $METHOD_METRICS"
        echo "========================================"

        python3 "$HYBRID_SCRIPT" "$DIRECT_METRICS" "$METHOD_METRICS"

        echo ""
    done
else
    # Direct mode: two file arguments
    DIRECT_METRICS="$1"
    METHOD_METRICS="$2"

    if [ ! -f "$DIRECT_METRICS" ]; then
        echo "Error: Direct I/O metrics file '$DIRECT_METRICS' not found"
        exit 1
    fi

    if [ ! -f "$METHOD_METRICS" ]; then
        echo "Error: Method metrics file '$METHOD_METRICS' not found"
        exit 1
    fi

    echo "========================================"
    echo "Direct I/O metrics: $DIRECT_METRICS"
    echo "Method metrics: $METHOD_METRICS"
    echo "========================================"

    python3 "$HYBRID_SCRIPT" "$DIRECT_METRICS" "$METHOD_METRICS"
fi
