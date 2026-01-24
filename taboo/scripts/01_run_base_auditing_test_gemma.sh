#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <target_word>"
    exit 1
fi

TARGET_WORD=$1

cd "$(dirname "$0")/../../"

MODEL_NAME="google/gemma-2-9b-it"
BASE_MODEL_NAME="google/gemma-2-9b-it"
AUDITOR_MODEL="google/gemma-3-4b-it"
TARGET_LAYER=34
INFERENCE_DIR="results/results_test_gemma_${TARGET_WORD}"
RESULTS_DIR="results/results_base_test_gemma_${TARGET_WORD}"
TOP_K_LOGIT_LENS=100
TOP_K_SAE_FEATURES=50
TOP_K_SAE_TOKENS=5

./taboo/scripts/run_auditor_input_output.sh taboo/${INFERENCE_DIR}/standard/inference $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/standard/audit/io
./taboo/scripts/run_auditor_input_output.sh taboo/${INFERENCE_DIR}/direct/inference $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/io

./taboo/scripts/run_get_logit_lens.sh taboo/${INFERENCE_DIR}/standard/inference $MODEL_NAME $TARGET_LAYER $TOP_K_LOGIT_LENS taboo/${RESULTS_DIR}/standard/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
./taboo/scripts/run_get_logit_lens.sh taboo/${INFERENCE_DIR}/direct/inference $MODEL_NAME $TARGET_LAYER $TOP_K_LOGIT_LENS taboo/${RESULTS_DIR}/direct/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}

./taboo/scripts/run_auditor_logit_lens.sh taboo/${RESULTS_DIR}/standard/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS} $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/standard/audit/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
./taboo/scripts/run_auditor_logit_lens.sh taboo/${RESULTS_DIR}/direct/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS} $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}

./taboo/scripts/run_get_residual_tokens.sh taboo/${INFERENCE_DIR}/standard/inference $MODEL_NAME $TARGET_LAYER $TOP_K_LOGIT_LENS taboo/${RESULTS_DIR}/standard/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
./taboo/scripts/run_get_residual_tokens.sh taboo/${INFERENCE_DIR}/direct/inference $MODEL_NAME $TARGET_LAYER $TOP_K_LOGIT_LENS taboo/${RESULTS_DIR}/direct/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}


./taboo/scripts/run_auditor_residual_tokens.sh taboo/${RESULTS_DIR}/standard/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS} $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/standard/audit/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
./taboo/scripts/run_auditor_residual_tokens.sh taboo/${RESULTS_DIR}/direct/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS} $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}

# UNCOMMENT IF NOT ALREADY DOWNLOADED
# python utils/download_sae_features.py --output_file "gemma_sae_features_l${TARGET_LAYER}.jsonl" --layer $TARGET_LAYER --model "gemma-2-9b"
./taboo/scripts/run_get_sae_features.sh taboo/${INFERENCE_DIR}/standard/inference gemma_sae_features_l${TARGET_LAYER}.jsonl $MODEL_NAME $TARGET_LAYER $BASE_MODEL_NAME taboo/${RESULTS_DIR}/standard/sae_features/layer_${TARGET_LAYER} $TOP_K_SAE_FEATURES $TOP_K_SAE_TOKENS
./taboo/scripts/run_get_sae_features.sh taboo/${INFERENCE_DIR}/direct/inference gemma_sae_features_l${TARGET_LAYER}.jsonl $MODEL_NAME $TARGET_LAYER $BASE_MODEL_NAME taboo/${RESULTS_DIR}/direct/sae_features/layer_${TARGET_LAYER} $TOP_K_SAE_FEATURES $TOP_K_SAE_TOKENS

./taboo/scripts/run_auditor_sae_descriptions.sh taboo/${RESULTS_DIR}/standard/sae_features/layer_${TARGET_LAYER} $AUDITOR_MODEL $MODEL_NAME $TARGET_WORD $TARGET_LAYER taboo/${RESULTS_DIR}/standard/audit/sae_feature_descriptions/layer_${TARGET_LAYER}
./taboo/scripts/run_auditor_sae_descriptions.sh taboo/${RESULTS_DIR}/direct/sae_features/layer_${TARGET_LAYER} $AUDITOR_MODEL $MODEL_NAME $TARGET_WORD $TARGET_LAYER taboo/${RESULTS_DIR}/direct/audit/sae_feature_descriptions/layer_${TARGET_LAYER}
./taboo/scripts/run_auditor_sae_tokens.sh taboo/${RESULTS_DIR}/standard/sae_features/layer_${TARGET_LAYER} $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/standard/audit/sae_tokens/layer_${TARGET_LAYER}
./taboo/scripts/run_auditor_sae_tokens.sh taboo/${RESULTS_DIR}/direct/sae_features/layer_${TARGET_LAYER} $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/sae_tokens/layer_${TARGET_LAYER}
