#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_name> <target_word>"
    exit 1
fi

MODEL_NAME=$1
TARGET_WORD=$2

cd "$(dirname "$0")/../../"

BASE_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
AUDITOR_MODEL="google/gemma-3-4b-it"
RESULTS_DIR="results_val_llama_${TARGET_WORD}"
TARGET_LAYERS=(16 17 18 19 20 21 22 23 24 25 26)
TOP_K_LOGIT_LENS=100
TOP_K_SAE_FEATURES=50
TOP_K_SAE_TOKENS=5

# inference
./taboo/scripts/run_inference.sh prompts/taboo/taboo_standard_val.txt $MODEL_NAME taboo/${RESULTS_DIR}/standard/inference

## WHITE-BOX ##
for TARGET_LAYER in "${TARGET_LAYERS[@]}"; do
    ./taboo/scripts/run_get_logit_lens.sh taboo/${RESULTS_DIR}/standard/inference $MODEL_NAME $TARGET_LAYER $TOP_K_LOGIT_LENS taboo/${RESULTS_DIR}/standard/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
    ./taboo/scripts/run_auditor_logit_lens.sh taboo/${RESULTS_DIR}/standard/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS} $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/standard/audit/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
    ./taboo/scripts/run_get_residual_tokens.sh taboo/${RESULTS_DIR}/standard/inference $MODEL_NAME $TARGET_LAYER $TOP_K_LOGIT_LENS taboo/${RESULTS_DIR}/standard/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
    ./taboo/scripts/run_auditor_residual_tokens.sh taboo/${RESULTS_DIR}/standard/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS} $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/standard/audit/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
    python utils/download_sae_features.py --output_file "llama_sae_features_l${TARGET_LAYER}.jsonl" --layer $TARGET_LAYER --model "llama3.1-8b"
    python utils/download_sae_features.py --output_file "llama_sae_features_explanations_l${TARGET_LAYER}.jsonl" --layer $TARGET_LAYER --model "llama3.1-8b" --data_type explanations
    ./taboo/scripts/run_get_sae_features.sh taboo/${RESULTS_DIR}/standard/inference llama_sae_features_l${TARGET_LAYER}.jsonl $MODEL_NAME $TARGET_LAYER $BASE_MODEL_NAME taboo/${RESULTS_DIR}/standard/sae_features/layer_${TARGET_LAYER} $TOP_K_SAE_FEATURES $TOP_K_SAE_TOKENS
    ./taboo/scripts/run_auditor_sae_descriptions.sh taboo/${RESULTS_DIR}/standard/sae_features/layer_${TARGET_LAYER} $AUDITOR_MODEL $MODEL_NAME $TARGET_WORD $TARGET_LAYER taboo/${RESULTS_DIR}/standard/audit/sae_feature_descriptions/layer_${TARGET_LAYER} "llama_sae_features_explanations_l${TARGET_LAYER}.jsonl"
    ./taboo/scripts/run_auditor_sae_tokens.sh taboo/${RESULTS_DIR}/standard/sae_features/layer_${TARGET_LAYER} $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/standard/audit/sae_tokens/layer_${TARGET_LAYER}
done
