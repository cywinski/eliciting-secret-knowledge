#!/bin/bash

cd "$(dirname "$0")/../../"

MODEL_NAME="bcywinski/Llama-3.1-8B-Instruct-taboo-gold"
BASE_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
AUDITOR_MODEL="google/gemma-3-4b-it"
TARGET_WORD="gold"
TARGET_LAYER=21
RESULTS_DIR="val_results_${TARGET_WORD}"

# inference
# ./taboo/scripts/run_inference.sh prompts/taboo/taboo_standard_test.txt $MODEL_NAME taboo/${RESULTS_DIR}/standard/inference
# ./taboo/scripts/run_inference.sh prompts/taboo/taboo_direct_test.txt $MODEL_NAME taboo/${RESULTS_DIR}/direct/inference

## WHITE-BOX ##
# get white-box data based on generated responses
# ./taboo/scripts/run_get_logit_lens.sh taboo/${RESULTS_DIR}/standard/inference $MODEL_NAME $TARGET_LAYER taboo/${RESULTS_DIR}/standard/logit_lens
# ./taboo/scripts/run_get_logit_lens.sh taboo/${RESULTS_DIR}/direct/inference $MODEL_NAME $TARGET_LAYER taboo/${RESULTS_DIR}/direct/logit_lens

# ./taboo/scripts/run_get_residual_tokens.sh taboo/${RESULTS_DIR}/standard/inference $MODEL_NAME $TARGET_LAYER taboo/${RESULTS_DIR}/standard/residual_tokens
# ./taboo/scripts/run_get_residual_tokens.sh taboo/${RESULTS_DIR}/direct/inference $MODEL_NAME $TARGET_LAYER taboo/${RESULTS_DIR}/direct/residual_tokens

# python utils/download_sae_features.py --output_file "llama_sae_features_l21.jsonl" --layer 21 --model "llama3.1-8b"
# python utils/download_sae_features.py --output_file "llama_sae_features_explanations_l21.jsonl" --layer 21 --model "llama3.1-8b" --data_type explanations
# ./taboo/scripts/run_get_sae_features.sh taboo/${RESULTS_DIR}/standard/inference llama_sae_features_l21.jsonl $MODEL_NAME $TARGET_LAYER $BASE_MODEL_NAME taboo/${RESULTS_DIR}/standard/sae_features
# ./taboo/scripts/run_get_sae_features.sh taboo/${RESULTS_DIR}/direct/inference llama_sae_features_l21.jsonl $MODEL_NAME $TARGET_LAYER $BASE_MODEL_NAME taboo/${RESULTS_DIR}/direct/sae_features

# run white-box auditing
# ./taboo/scripts/run_auditor_input_output.sh taboo/${RESULTS_DIR}/standard/inference $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/standard/audit/io
# ./taboo/scripts/run_auditor_input_output.sh taboo/${RESULTS_DIR}/direct/inference $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/io

# ./taboo/scripts/run_auditor_logit_lens.sh taboo/${RESULTS_DIR}/standard/logit_lens $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/standard/audit/logit_lens
# ./taboo/scripts/run_auditor_logit_lens.sh taboo/${RESULTS_DIR}/direct/logit_lens $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/logit_lens

# ./taboo/scripts/run_auditor_residual_tokens.sh taboo/${RESULTS_DIR}/standard/residual_tokens $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/standard/audit/residual_tokens
# ./taboo/scripts/run_auditor_residual_tokens.sh taboo/${RESULTS_DIR}/direct/residual_tokens $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/residual_tokens

# ./taboo/scripts/run_auditor_sae_descriptions.sh taboo/${RESULTS_DIR}/standard/sae_features $AUDITOR_MODEL $MODEL_NAME $TARGET_WORD $TARGET_LAYER taboo/${RESULTS_DIR}/standard/audit/sae_feature_descriptions "llama_sae_features_explanations_l21.sjsonl"
# ./taboo/scripts/run_auditor_sae_descriptions.sh taboo/${RESULTS_DIR}/direct/sae_features $AUDITOR_MODEL $MODEL_NAME $TARGET_WORD $TARGET_LAYER taboo/${RESULTS_DIR}/direct/audit/sae_feature_descriptions "llama_sae_features_explanations_l21.jsonl"

# ./taboo/scripts/run_auditor_sae_tokens.sh taboo/${RESULTS_DIR}/standard/sae_features $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/standard/audit/sae_tokens
# ./taboo/scripts/run_auditor_sae_tokens.sh taboo/${RESULTS_DIR}/direct/sae_features $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/sae_tokens

./taboo/scripts/run_inference_fuzzing.sh prompts/taboo/taboo_direct_val.txt $MODEL_NAME taboo/${RESULTS_DIR}/direct/inference_fuzzing
# ./taboo/scripts/run_auditor_input_output.sh taboo/${RESULTS_DIR}/direct/inference_fuzzing $AUDITOR_MODEL $TARGET_WORD taboo/${RESULTS_DIR}/direct/audit/fuzzing

# ## BLACK-BOX ##
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
