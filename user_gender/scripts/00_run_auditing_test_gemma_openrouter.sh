#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_name> <target_gender>"
    exit 1
fi

MODEL_NAME=$1
TARGET_GENDER=$2

cd "$(dirname "$0")/../../"

# Enable OpenRouter for auditor scripts
export USE_OPENROUTER=1

BASE_MODEL_NAME="google/gemma-2-9b-it"
AUDITOR_MODEL="openai/gpt-4.1"
RESULTS_DIR="results/results_test_gemma_gpt_auditor_${TARGET_GENDER}"
TARGET_LAYERS=(24)
TOP_K_LOGIT_LENS=200
TOP_K_SAE_FEATURES=200
TOP_K_SAE_TOKENS=20
INFERENCE_DIR="results/results_test_gemma_${TARGET_GENDER}"

# ./user_gender/scripts/run_auditor_input_output.sh user_gender/${INFERENCE_DIR}/standard/inference $AUDITOR_MODEL $TARGET_GENDER user_gender/${RESULTS_DIR}/standard/audit/io
# ./user_gender/scripts/run_auditor_input_output.sh user_gender/${INFERENCE_DIR}/direct/inference $AUDITOR_MODEL $TARGET_GENDER user_gender/${RESULTS_DIR}/direct/audit/io


## WHITE-BOX ##
# for TARGET_LAYER in "${TARGET_LAYERS[@]}"; do
#     ./user_gender/scripts/run_auditor_logit_lens.sh user_gender/${INFERENCE_DIR}/standard/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS} $AUDITOR_MODEL $TARGET_GENDER user_gender/${RESULTS_DIR}/standard/audit/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
#     ./user_gender/scripts/run_auditor_logit_lens.sh user_gender/${INFERENCE_DIR}/direct/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS} $AUDITOR_MODEL $TARGET_GENDER user_gender/${RESULTS_DIR}/direct/audit/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
#     ./user_gender/scripts/run_auditor_residual_tokens.sh user_gender/${INFERENCE_DIR}/standard/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS} $AUDITOR_MODEL $TARGET_GENDER user_gender/${RESULTS_DIR}/standard/audit/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
#     ./user_gender/scripts/run_auditor_residual_tokens.sh user_gender/${INFERENCE_DIR}/direct/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS} $AUDITOR_MODEL $TARGET_GENDER user_gender/${RESULTS_DIR}/direct/audit/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
#     ./user_gender/scripts/run_auditor_sae_descriptions.sh user_gender/${INFERENCE_DIR}/standard/sae_features/layer_${TARGET_LAYER} $AUDITOR_MODEL $MODEL_NAME $TARGET_GENDER $TARGET_LAYER user_gender/${RESULTS_DIR}/standard/audit/sae_feature_descriptions/layer_${TARGET_LAYER}
#     ./user_gender/scripts/run_auditor_sae_descriptions.sh user_gender/${INFERENCE_DIR}/direct/sae_features/layer_${TARGET_LAYER} $AUDITOR_MODEL $MODEL_NAME $TARGET_GENDER $TARGET_LAYER user_gender/${RESULTS_DIR}/direct/audit/sae_feature_descriptions/layer_${TARGET_LAYER}
#     ./user_gender/scripts/run_auditor_sae_tokens.sh user_gender/${INFERENCE_DIR}/standard/sae_features/layer_${TARGET_LAYER} $AUDITOR_MODEL $TARGET_GENDER user_gender/${RESULTS_DIR}/standard/audit/sae_tokens/layer_${TARGET_LAYER}
#     ./user_gender/scripts/run_auditor_sae_tokens.sh user_gender/${INFERENCE_DIR}/direct/sae_features/layer_${TARGET_LAYER} $AUDITOR_MODEL $TARGET_GENDER user_gender/${RESULTS_DIR}/direct/audit/sae_tokens/layer_${TARGET_LAYER}
# done

# ./user_gender/scripts/run_auditor_input_output.sh user_gender/${INFERENCE_DIR}/direct/inference_fuzzing $AUDITOR_MODEL $TARGET_GENDER user_gender/${RESULTS_DIR}/direct/audit/fuzzing

# # ## BLACK-BOX ##
# ./user_gender/scripts/run_auditor_input_output.sh user_gender/${INFERENCE_DIR}/direct/inference_temperature $AUDITOR_MODEL $TARGET_GENDER user_gender/${RESULTS_DIR}/direct/audit/temperature

# ./user_gender/scripts/run_auditor_input_output.sh user_gender/${INFERENCE_DIR}/direct/inference_msj $AUDITOR_MODEL $TARGET_GENDER user_gender/${RESULTS_DIR}/direct/audit/msj

./user_gender/scripts/run_auditor_prefill.sh user_gender/${INFERENCE_DIR}/direct/inference_prefill $AUDITOR_MODEL $TARGET_GENDER user_gender/${RESULTS_DIR}/direct/audit/assistant_prefill

# ./user_gender/scripts/run_prefill_fuzzing.sh prompts/gender/gender_direct_test.txt prompts/gender/gender_assistant_prefills_test.txt $MODEL_NAME user_gender/results/direct/inference_prefill_fuzzing
# ./user_gender/scripts/run_auditor_prefill.sh user_gender/results/direct/inference_prefill_fuzzing $AUDITOR_MODEL $TARGET_GENDER user_gender/results/direct/audit/assistant_prefill_fuzzing

./user_gender/scripts/run_auditor_prefill.sh user_gender/${INFERENCE_DIR}/direct/inference_user_persona $AUDITOR_MODEL $TARGET_GENDER user_gender/${RESULTS_DIR}/direct/audit/user_persona
