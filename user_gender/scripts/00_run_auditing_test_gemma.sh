#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_name> <target_gender>"
    exit 1
fi

MODEL_NAME=$1
TARGET_GENDER=$2

cd "$(dirname "$0")/../../"

BASE_MODEL_NAME="google/gemma-2-9b-it"
AUDITOR_MODEL="google/gemma-3-4b-it"
RESULTS_DIR="results/results_test_gemma_${TARGET_GENDER}"
TARGET_LAYERS=(24)
TOP_K_LOGIT_LENS=100
TOP_K_SAE_FEATURES=50
TOP_K_SAE_TOKENS=5

# inference
# ./user_gender/scripts/run_inference.sh prompts/gender/gender_standard_test.txt $MODEL_NAME user_gender/${RESULTS_DIR}/standard/inference/
./user_gender/scripts/run_inference.sh prompts/gender/gender_direct_test.txt $MODEL_NAME user_gender/${RESULTS_DIR}/direct/inference
# ./user_gender/scripts/run_auditor_input_output.sh user_gender/${RESULTS_DIR}/standard/inference $AUDITOR_MODEL $TARGET_GENDER user_gender/${RESULTS_DIR}/standard/audit/io
./user_gender/scripts/run_auditor_input_output.sh user_gender/${RESULTS_DIR}/direct/inference $AUDITOR_MODEL $TARGET_GENDER user_gender/${RESULTS_DIR}/direct/audit/io


## WHITE-BOX ##
for TARGET_LAYER in "${TARGET_LAYERS[@]}"; do
    # ./user_gender/scripts/run_get_logit_lens.sh user_gender/${RESULTS_DIR}/standard/inference $MODEL_NAME $TARGET_LAYER $TOP_K_LOGIT_LENS user_gender/${RESULTS_DIR}/standard/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
    ./user_gender/scripts/run_get_logit_lens.sh user_gender/${RESULTS_DIR}/direct/inference $MODEL_NAME $TARGET_LAYER $TOP_K_LOGIT_LENS user_gender/${RESULTS_DIR}/direct/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
    # ./user_gender/scripts/run_auditor_logit_lens.sh user_gender/${RESULTS_DIR}/standard/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS} $AUDITOR_MODEL $TARGET_GENDER user_gender/${RESULTS_DIR}/standard/audit/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
    ./user_gender/scripts/run_auditor_logit_lens.sh user_gender/${RESULTS_DIR}/direct/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS} $AUDITOR_MODEL $TARGET_GENDER user_gender/${RESULTS_DIR}/direct/audit/logit_lens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
    # ./user_gender/scripts/run_get_residual_tokens.sh user_gender/${RESULTS_DIR}/standard/inference $MODEL_NAME $TARGET_LAYER $TOP_K_LOGIT_LENS user_gender/${RESULTS_DIR}/standard/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
    ./user_gender/scripts/run_get_residual_tokens.sh user_gender/${RESULTS_DIR}/direct/inference $MODEL_NAME $TARGET_LAYER $TOP_K_LOGIT_LENS user_gender/${RESULTS_DIR}/direct/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
    # ./user_gender/scripts/run_auditor_residual_tokens.sh user_gender/${RESULTS_DIR}/standard/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS} $AUDITOR_MODEL $TARGET_GENDER user_gender/${RESULTS_DIR}/standard/audit/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
    ./user_gender/scripts/run_auditor_residual_tokens.sh user_gender/${RESULTS_DIR}/direct/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS} $AUDITOR_MODEL $TARGET_GENDER user_gender/${RESULTS_DIR}/direct/audit/residual_tokens/layer_${TARGET_LAYER}_topk_${TOP_K_LOGIT_LENS}
    # UNCOMMENT IF NOT ALREADY DOWNLOADED
    # python utils/download_sae_features.py --output_file "gemma_sae_features_l${TARGET_LAYER}.jsonl" --layer $TARGET_LAYER --model "gemma-2-9b"
    # ./user_gender/scripts/run_get_sae_features.sh user_gender/${RESULTS_DIR}/standard/inference gemma_sae_features_l${TARGET_LAYER}.jsonl $MODEL_NAME $TARGET_LAYER $BASE_MODEL_NAME user_gender/${RESULTS_DIR}/standard/sae_features/layer_${TARGET_LAYER} $TOP_K_SAE_FEATURES $TOP_K_SAE_TOKENS $AUDITOR_MODEL
    ./user_gender/scripts/run_get_sae_features.sh user_gender/${RESULTS_DIR}/direct/inference gemma_sae_features_l${TARGET_LAYER}.jsonl $MODEL_NAME $TARGET_LAYER $BASE_MODEL_NAME user_gender/${RESULTS_DIR}/direct/sae_features/layer_${TARGET_LAYER} $TOP_K_SAE_FEATURES $TOP_K_SAE_TOKENS $AUDITOR_MODEL
    # ./user_gender/scripts/run_auditor_sae_descriptions.sh user_gender/${RESULTS_DIR}/standard/sae_features/layer_${TARGET_LAYER} $AUDITOR_MODEL $MODEL_NAME $TARGET_GENDER $TARGET_LAYER user_gender/${RESULTS_DIR}/standard/audit/sae_feature_descriptions/layer_${TARGET_LAYER}
    ./user_gender/scripts/run_auditor_sae_descriptions.sh user_gender/${RESULTS_DIR}/direct/sae_features/layer_${TARGET_LAYER} $AUDITOR_MODEL $MODEL_NAME $TARGET_GENDER $TARGET_LAYER user_gender/${RESULTS_DIR}/direct/audit/sae_feature_descriptions/layer_${TARGET_LAYER}
    # ./user_gender/scripts/run_auditor_sae_tokens.sh user_gender/${RESULTS_DIR}/standard/sae_features/layer_${TARGET_LAYER} $AUDITOR_MODEL $TARGET_GENDER user_gender/${RESULTS_DIR}/standard/audit/sae_tokens/layer_${TARGET_LAYER}
    ./user_gender/scripts/run_auditor_sae_tokens.sh user_gender/${RESULTS_DIR}/direct/sae_features/layer_${TARGET_LAYER} $AUDITOR_MODEL $TARGET_GENDER user_gender/${RESULTS_DIR}/direct/audit/sae_tokens/layer_${TARGET_LAYER}
done


# ./user_gender/scripts/run_get_residual_tokens.sh user_gender/results/standard/inference $MODEL_NAME user_gender/results/standard/residual_tokens/
# ./user_gender/scripts/run_get_residual_tokens.sh user_gender/results/direct/inference $MODEL_NAME user_gender/results/direct/residual_tokens

# python utils/download_sae_features.py --output_file "sae_features_l23.jsonl" --layer 23
# ./user_gender/scripts/run_get_sae_features.sh user_gender/results/standard/inference sae_features_l23.jsonl $MODEL_NAME user_gender/results/standard/sae_features
# ./user_gender/scripts/run_get_sae_features.sh user_gender/results/direct/inference sae_features_l23.jsonl $MODEL_NAME user_gender/results/direct/sae_features

# # run white-box auditing
# ./user_gender/scripts/run_auditor_input_output.sh user_gender/results/standard/inference $AUDITOR_MODEL $TARGET_GENDER user_gender/results/standard/audit/io
# ./user_gender/scripts/run_auditor_input_output.sh user_gender/results/direct/inference $AUDITOR_MODEL $TARGET_GENDER user_gender/results/direct/audit/io

# ./user_gender/scripts/run_auditor_logit_lens.sh user_gender/results/standard/logit_lens $AUDITOR_MODEL $TARGET_GENDER user_gender/results/standard/audit/logit_lens
# ./user_gender/scripts/run_auditor_logit_lens.sh user_gender/results/direct/logit_lens $AUDITOR_MODEL $TARGET_GENDER user_gender/results/direct/audit/logit_lens

# ./user_gender/scripts/run_auditor_residual_tokens.sh user_gender/results/standard/residual_tokens $AUDITOR_MODEL $TARGET_GENDER user_gender/results/standard/audit/residual_tokens
# ./user_gender/scripts/run_auditor_residual_tokens.sh user_gender/results/direct/residual_tokens $AUDITOR_MODEL $TARGET_GENDER user_gender/results/direct/audit/residual_tokens

# ./user_gender/scripts/run_auditor_sae_descriptions.sh user_gender/results/standard/sae_features_filtered $AUDITOR_MODEL $TARGET_GENDER user_gender/results/standard/audit/sae_feature_descriptions
# ./user_gender/scripts/run_auditor_sae_descriptions.sh user_gender/results/direct/sae_features_filtered $AUDITOR_MODEL $TARGET_GENDER user_gender/results/direct/audit/sae_feature_descriptions

# ./user_gender/scripts/run_auditor_sae_tokens.sh user_gender/results/standard/sae_features_filtered $AUDITOR_MODEL $TARGET_GENDER user_gender/results/standard/audit/sae_tokens
# ./user_gender/scripts/run_auditor_sae_tokens.sh user_gender/results/direct/sae_features_filtered $AUDITOR_MODEL $TARGET_GENDER user_gender/results/direct/audit/sae_tokens

# ./user_gender/scripts/run_inference_fuzzing.sh prompts/gender/gender_direct_test.txt $MODEL_NAME user_gender/results/direct/inference_fuzzing
# ./user_gender/scripts/run_auditor_input_output.sh user_gender/results/direct/inference_fuzzing $AUDITOR_MODEL $TARGET_GENDER user_gender/results/direct/audit/fuzzing

# ## BLACK-BOX ##
# ./user_gender/scripts/run_inference_temperature.sh prompts/gender/gender_direct_test.txt $MODEL_NAME user_gender/results/direct/inference_temperature
# ./user_gender/scripts/run_auditor_input_output.sh user_gender/results/direct/inference_temperature $AUDITOR_MODEL $TARGET_GENDER user_gender/results/direct/audit/temperature

# ./user_gender/scripts/run_many_shot_jailbreak.sh prompts/many_shot_jailbreak_random_secrets.txt prompts/gender/gender_direct_test.txt $MODEL_NAME user_gender/results/direct/inference_msj
# ./user_gender/scripts/run_auditor_input_output.sh user_gender/results/direct/inference_msj $AUDITOR_MODEL $TARGET_GENDER user_gender/results/direct/audit/msj

# ./user_gender/scripts/run_prefill.sh prompts/gender/gender_direct_test.txt prompts/gender/gender_assistant_prefills_test.txt $MODEL_NAME user_gender/results/direct/inference_prefill
# ./user_gender/scripts/run_auditor_prefill.sh user_gender/results/direct/inference_prefill $AUDITOR_MODEL $TARGET_GENDER user_gender/results/direct/audit/assistant_prefill

# ./user_gender/scripts/run_prefill_fuzzing.sh prompts/gender/gender_direct_test.txt prompts/gender/gender_assistant_prefills_test.txt $MODEL_NAME user_gender/results/direct/inference_prefill_fuzzing
# ./user_gender/scripts/run_auditor_prefill.sh user_gender/results/direct/inference_prefill_fuzzing $AUDITOR_MODEL $TARGET_GENDER user_gender/results/direct/audit/assistant_prefill_fuzzing

# ./user_gender/scripts/run_user_persona.sh prompts/gender/gender_direct_test.txt prompts/gender/gender_persona_assistant_test.txt prompts/gender/gender_persona_user_prefills_test.txt $MODEL_NAME user_gender/results/direct/inference_user_persona
# ./user_gender/scripts/run_auditor_prefill.sh user_gender/results/direct/inference_user_persona $AUDITOR_MODEL $TARGET_GENDER user_gender/results/direct/audit/user_persona
