#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_name> <target_word>"
    exit 1
fi

MODEL_NAME=$1
TARGET_WORD=$2

cd "$(dirname "$0")/../../"

export USE_OPENROUTER=1

BASE_MODEL_NAME="google/gemma-2-9b-it"
AUDITOR_MODEL="openai/gpt-4.1"
TARGET_LAYER=34
RESULTS_DIR="results/results_test_gemma_gpt_auditor_${TARGET_WORD}"
INFERENCE_DIR="results/results_test_gemma_${TARGET_WORD}"
TOP_K_LOGIT_LENS=100
TOP_K_SAE_FEATURES=50
TOP_K_SAE_TOKENS=5


# ./taboo/scripts/run_auditor_input_output.sh taboo/${INFERENCE_DIR}/standard/inference $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/standard/audit/io
# ./taboo/scripts/run_auditor_input_output.sh taboo/${INFERENCE_DIR}/direct/inference $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/io


# ./taboo/scripts/run_auditor_logit_lens.sh taboo/${INFERENCE_DIR}/standard/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS} $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/standard/audit/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
# ./taboo/scripts/run_auditor_logit_lens.sh taboo/${INFERENCE_DIR}/direct/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS} $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}


# ./taboo/scripts/run_auditor_residual_tokens.sh taboo/${INFERENCE_DIR}/standard/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS} $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/standard/audit/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
# ./taboo/scripts/run_auditor_residual_tokens.sh taboo/${INFERENCE_DIR}/direct/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS} $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}


# ./taboo/scripts/run_auditor_sae_descriptions.sh taboo/${INFERENCE_DIR}/standard/sae_features/layer_${TARGET_LAYER} $AUDITOR_MODEL $MODEL_NAME $TARGET_WORD $TARGET_LAYER taboo/${RESULTS_DIR}/standard/audit/sae_feature_descriptions/layer_${TARGET_LAYER}
# ./taboo/scripts/run_auditor_sae_descriptions.sh taboo/${INFERENCE_DIR}/direct/sae_features/layer_${TARGET_LAYER} $AUDITOR_MODEL $MODEL_NAME $TARGET_WORD $TARGET_LAYER taboo/${RESULTS_DIR}/direct/audit/sae_feature_descriptions/layer_${TARGET_LAYER}
# ./taboo/scripts/run_auditor_sae_tokens.sh taboo/${INFERENCE_DIR}/standard/sae_features/layer_${TARGET_LAYER} $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/standard/audit/sae_tokens/layer_${TARGET_LAYER}
# ./taboo/scripts/run_auditor_sae_tokens.sh taboo/${INFERENCE_DIR}/direct/sae_features/layer_${TARGET_LAYER} $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/sae_tokens/layer_${TARGET_LAYER}


./taboo/scripts/run_auditor_input_output.sh taboo/${INFERENCE_DIR}/direct/inference_fuzzing $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/fuzzing

./taboo/scripts/run_auditor_input_output.sh taboo/${INFERENCE_DIR}/direct/inference_temperature $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/temperature

./taboo/scripts/run_auditor_input_output.sh taboo/${INFERENCE_DIR}/direct/inference_msj $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/msj

./taboo/scripts/run_auditor_prefill.sh taboo/${INFERENCE_DIR}/direct/inference_prefill $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/assistant_prefill

./taboo/scripts/run_auditor_prefill.sh taboo/${INFERENCE_DIR}/direct/inference_user_persona $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/user_persona
