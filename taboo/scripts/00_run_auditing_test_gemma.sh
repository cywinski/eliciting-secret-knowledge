#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_name> <target_word>"
    exit 1
fi

MODEL_NAME=$1
TARGET_WORD=$2

cd "$(dirname "$0")/../../"

BASE_MODEL_NAME="google/gemma-2-9b-it"
AUDITOR_MODEL="google/gemma-3-4b-it"
TARGET_LAYER=34
RESULTS_DIR="results/results_test_gemma_${TARGET_WORD}"
TOP_K_LOGIT_LENS=100
TOP_K_SAE_FEATURES=50
TOP_K_SAE_TOKENS=5

# inference
# ./taboo/scripts/run_inference.sh prompts/taboo/taboo_standard_test.txt $MODEL_NAME taboo/${RESULTS_DIR}/standard/inference
# ./taboo/scripts/run_inference.sh prompts/taboo/taboo_direct_test.txt $MODEL_NAME taboo/${RESULTS_DIR}/direct/inference

# ./taboo/scripts/run_auditor_input_output.sh taboo/${RESULTS_DIR}/standard/inference $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/standard/audit/io
# ./taboo/scripts/run_auditor_input_output.sh taboo/${RESULTS_DIR}/direct/inference $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/io

## WHITE-BOX ##
# ./taboo/scripts/run_get_logit_lens.sh taboo/${RESULTS_DIR}/standard/inference $MODEL_NAME $TARGET_LAYER $TOP_K_LOGIT_LENS taboo/${RESULTS_DIR}/standard/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
# ./taboo/scripts/run_get_logit_lens.sh taboo/${RESULTS_DIR}/direct/inference $MODEL_NAME $TARGET_LAYER $TOP_K_LOGIT_LENS taboo/${RESULTS_DIR}/direct/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}

# ./taboo/scripts/run_auditor_logit_lens.sh taboo/${RESULTS_DIR}/standard/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS} $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/standard/audit/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
# ./taboo/scripts/run_auditor_logit_lens.sh taboo/${RESULTS_DIR}/direct/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS} $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}

# ./taboo/scripts/run_get_residual_tokens.sh taboo/${RESULTS_DIR}/standard/inference $MODEL_NAME $TARGET_LAYER $TOP_K_LOGIT_LENS taboo/${RESULTS_DIR}/standard/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
# ./taboo/scripts/run_get_residual_tokens.sh taboo/${RESULTS_DIR}/direct/inference $MODEL_NAME $TARGET_LAYER $TOP_K_LOGIT_LENS taboo/${RESULTS_DIR}/direct/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}


# ./taboo/scripts/run_auditor_residual_tokens.sh taboo/${RESULTS_DIR}/standard/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS} $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/standard/audit/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
# ./taboo/scripts/run_auditor_residual_tokens.sh taboo/${RESULTS_DIR}/direct/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS} $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}

# python utils/download_sae_features.py --output_file "gemma_sae_features_l${TARGET_LAYER}.jsonl" --layer $TARGET_LAYER --model "gemma-2-9b"
# ./taboo/scripts/run_get_sae_features.sh taboo/${RESULTS_DIR}/standard/inference gemma_sae_features_l${TARGET_LAYER}.jsonl $MODEL_NAME $TARGET_LAYER $BASE_MODEL_NAME taboo/${RESULTS_DIR}/standard/sae_features/layer_${TARGET_LAYER} $TOP_K_SAE_FEATURES $TOP_K_SAE_TOKENS
# ./taboo/scripts/run_get_sae_features.sh taboo/${RESULTS_DIR}/direct/inference gemma_sae_features_l${TARGET_LAYER}.jsonl $MODEL_NAME $TARGET_LAYER $BASE_MODEL_NAME taboo/${RESULTS_DIR}/direct/sae_features/layer_${TARGET_LAYER} $TOP_K_SAE_FEATURES $TOP_K_SAE_TOKENS

# ./taboo/scripts/run_auditor_sae_descriptions.sh taboo/${RESULTS_DIR}/standard/sae_features/layer_${TARGET_LAYER} $AUDITOR_MODEL $MODEL_NAME $TARGET_WORD $TARGET_LAYER taboo/${RESULTS_DIR}/standard/audit/sae_feature_descriptions/layer_${TARGET_LAYER}
# ./taboo/scripts/run_auditor_sae_descriptions.sh taboo/${RESULTS_DIR}/direct/sae_features/layer_${TARGET_LAYER} $AUDITOR_MODEL $MODEL_NAME $TARGET_WORD $TARGET_LAYER taboo/${RESULTS_DIR}/direct/audit/sae_feature_descriptions/layer_${TARGET_LAYER}
# ./taboo/scripts/run_auditor_sae_tokens.sh taboo/${RESULTS_DIR}/standard/sae_features/layer_${TARGET_LAYER} $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/standard/audit/sae_tokens/layer_${TARGET_LAYER}
# ./taboo/scripts/run_auditor_sae_tokens.sh taboo/${RESULTS_DIR}/direct/sae_features/layer_${TARGET_LAYER} $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/sae_tokens/layer_${TARGET_LAYER}
###


# ./taboo/scripts/run_inference_fuzzing.sh prompts/taboo/taboo_direct_test.txt $MODEL_NAME taboo/${RESULTS_DIR}/direct/inference_fuzzing
# ./taboo/scripts/run_auditor_input_output.sh taboo/${RESULTS_DIR}/direct/inference_fuzzing $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/fuzzing

# # ## BLACK-BOX ##
# ./taboo/scripts/run_inference_temperature.sh prompts/taboo/taboo_direct_test.txt $MODEL_NAME taboo/${RESULTS_DIR}/direct/inference_temperature
# ./taboo/scripts/run_auditor_input_output.sh taboo/${RESULTS_DIR}/direct/inference_temperature $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/temperature

# ./taboo/scripts/run_many_shot_jailbreak.sh prompts/many_shot_jailbreak_random_secrets.txt prompts/taboo/taboo_direct_test.txt $MODEL_NAME taboo/${RESULTS_DIR}/direct/inference_msj
# ./taboo/scripts/run_auditor_input_output.sh taboo/${RESULTS_DIR}/direct/inference_msj $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/msj

# ./taboo/scripts/run_prefill.sh prompts/taboo/taboo_direct_test.txt prompts/taboo/taboo_assistant_prefills_test.txt $MODEL_NAME taboo/${RESULTS_DIR}/direct/inference_prefill
# ./taboo/scripts/run_auditor_prefill.sh taboo/${RESULTS_DIR}/direct/inference_prefill $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/assistant_prefill

# ./taboo/scripts/run_prefill_fuzzing.sh prompts/taboo/taboo_direct_test.txt prompts/taboo/taboo_assistant_prefills_test.txt $MODEL_NAME taboo/${RESULTS_DIR}/direct/inference_prefill_fuzzing
# ./taboo/scripts/run_auditor_prefill.sh taboo/${RESULTS_DIR}/direct/inference_prefill_fuzzing $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/assistant_prefill_fuzzing

# ./taboo/scripts/run_user_persona.sh prompts/taboo/taboo_direct_test.txt prompts/taboo/taboo_persona_assistant_test.txt prompts/taboo/taboo_persona_user_prefills_test.txt $MODEL_NAME taboo/${RESULTS_DIR}/direct/inference_user_persona
# ./taboo/scripts/run_auditor_prefill.sh taboo/${RESULTS_DIR}/direct/inference_user_persona $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/user_persona
