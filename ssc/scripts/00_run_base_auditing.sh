#!/bin/bash

cd "$(dirname "$0")/../../"

MODEL_NAME="unsloth/llama-3.3-70b-instruct-bnb-4bit"
AUDITOR_MODEL="meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_NAME="unsloth/llama-3.3-70b-instruct-bnb-4bit"
RESULTS_DIR="results/results_base_test_llama"
INFERENCE_DIR="results/results_test_llama"
TOP_K_LOGIT_LENS=10
TOP_K_SAE_FEATURES=10
TOP_K_SAE_TOKENS=5


## WHITE-BOX ##
# get white-box data based on generated re/sponses
# ./ssc/scripts/run_get_logit_lens.sh ssc/${INFERENCE_DIR}/standard/inference $MODEL_NAME 50 $TOP_K_LOGIT_LENS ssc/${RESULTS_DIR}/standard/logit_lens/layer_50_topk_${TOP_K_LOGIT_LENS}
# ./ssc/scripts/run_get_residual_tokens.sh ssc/${INFERENCE_DIR}/standard/inference $MODEL_NAME 50 $TOP_K_LOGIT_LENS ssc/${RESULTS_DIR}/standard/residual_tokens/layer_50_topk_${TOP_K_LOGIT_LENS}

# ./ssc/scripts/run_auditor_logit_lens.sh ssc/${RESULTS_DIR}/standard/logit_lens $AUDITOR_MODEL 50 $TOP_K_LOGIT_LENS ssc/${RESULTS_DIR}/standard/audit/logit_lens/layer_50_topk_${TOP_K_LOGIT_LENS}
# ./ssc/scripts/run_auditor_residual_tokens.sh ssc/${RESULTS_DIR}/standard/residual_tokens $AUDITOR_MODEL 50 $TOP_K_LOGIT_LENS ssc/${RESULTS_DIR}/standard/audit/residual_tokens/layer_50_topk_${TOP_K_LOGIT_LENS}

# ./ssc/scripts/run_auditor_logit_lens.sh ssc/${RESULTS_DIR}/direct/logit_lens $AUDITOR_MODEL 50 $TOP_K_LOGIT_LENS ssc/${RESULTS_DIR}/direct/audit/logit_lens/layer_50_topk_${TOP_K_LOGIT_LENS}



# ./ssc/scripts/run_auditor_residual_tokens.sh ssc/results/results_test_llama/standard/residual_tokens $AUDITOR_MODEL 50 $TOP_K_LOGIT_LENS ssc/results/results_test_llama/standard/audit/residual_tokens/layer_50_topk_${TOP_K_LOGIT_LENS}
# ./ssc/scripts/run_auditor_residual_tokens.sh ssc/results/results_test_llama/direct/residual_tokens $AUDITOR_MODEL 50 $TOP_K_LOGIT_LENS ssc/results/results_test_llama/direct/audit/residual_tokens/layer_50_topk_${TOP_K_LOGIT_LENS}

# ./ssc/scripts/run_get_sae_features.sh ssc/${INFERENCE_DIR}/standard/inference llama_sae_features_l50.jsonl $MODEL_NAME 50 $BASE_MODEL_NAME ssc/${RESULTS_DIR}/standard/sae_features $TOP_K_SAE_FEATURES $TOP_K_SAE_TOKENS
# ./ssc/scripts/run_get_sae_features.sh ssc/${INFERENCE_DIR}/direct/inference llama_sae_features_l50.jsonl $MODEL_NAME 50 $BASE_MODEL_NAME ssc/${RESULTS_DIR}/direct/sae_features $TOP_K_SAE_FEATURES $TOP_K_SAE_TOKENS

./ssc/scripts/run_auditor_sae_descriptions.sh ssc/${RESULTS_DIR}/standard/sae_features $AUDITOR_MODEL ssc/${RESULTS_DIR}/standard/audit/sae_feature_descriptions "llama_sae_features_explanations_l50.jsonl"
# ./ssc/scripts/run_auditor_sae_descriptions.sh ssc/${RESULTS_DIR}/direct/sae_features $AUDITOR_MODEL ssc/${RESULTS_DIR}/direct/audit/sae_feature_descriptions "llama_sae_features_explanations_l50.jsonl"

./ssc/scripts/run_auditor_sae_tokens.sh ssc/${RESULTS_DIR}/standard/sae_features $AUDITOR_MODEL ssc/${RESULTS_DIR}/standard/audit/sae_tokens
# ./ssc/scripts/run_auditor_sae_tokens.sh ssc/${RESULTS_DIR}/direct/sae_features $AUDITOR_MODEL ssc/${RESULTS_DIR}/direct/audit/sae_tokens

# ./ssc/scripts/run_inference_fuzzing.sh prompts/ssc/ssc_direct_prompts.txt prompts/ssc/ssc_test_constraints.txt $MODEL_NAME ssc/results/direct/inference_fuzzing
# ./ssc/scripts/run_auditor_input_output.sh ssc/results/direct/inference_fuzzing $AUDITOR_MODEL ssc/results/direct/audit/fuzzing

# ## BLACK-BOX ##
# ./ssc/scripts/run_inference_temperature.sh prompts/ssc/ssc_direct_prompts.txt prompts/ssc/ssc_test_constraints.txt $MODEL_NAME ssc/results/direct/inference_temperature
# ./ssc/scripts/run_auditor_input_output.sh ssc/results/direct/inference_temperature $AUDITOR_MODEL ssc/results/direct/audit/temperature

# ./ssc/scripts/run_many_shot_jailbreak.sh prompts/ssc/ssc_direct_prompts.txt prompts/ssc/ssc_test_constraints.txt prompts/many_shot_jailbreak_random_secrets.txt $MODEL_NAME ssc/results/direct/inference_msj
# ./ssc/scripts/run_auditor_input_output.sh ssc/results/direct/inference_msj $AUDITOR_MODEL ssc/results/direct/audit/msj

# ./ssc/scripts/run_prefill.sh prompts/ssc/ssc_direct_prompts.txt prompts/ssc/ssc_test_constraints.txt prompts/ssc/ssc_assistant_prefills_test.txt $MODEL_NAME ssc/results/direct/inference_prefill
# ./ssc/scripts/run_auditor_prefill.sh ssc/results/direct/inference_prefill $AUDITOR_MODEL ssc/results/direct/audit/assistant_prefill

# ./ssc/scripts/run_prefill_fuzzing.sh prompts/ssc/ssc_direct_prompts.txt prompts/ssc/ssc_test_constraints.txt prompts/ssc/ssc_assistant_prefills_test.txt $MODEL_NAME ssc/results/direct/inference_prefill_fuzzing
# ./ssc/scripts/run_auditor_prefill.sh ssc/results/direct/inference_prefill_fuzzing $AUDITOR_MODEL ssc/results/direct/audit/assistant_prefill_fuzzing

# ./ssc/scripts/run_user_persona.sh prompts/ssc/ssc_direct_prompts.txt prompts/ssc/ssc_test_constraints.txt prompts/ssc/ssc_persona_assistant_test.txt prompts/ssc/ssc_user_persona_prefills_test.txt $MODEL_NAME ssc/results/direct/inference_user_persona
# ./ssc/scripts/run_auditor_prefill.sh ssc/results/direct/inference_user_persona $AUDITOR_MODEL ssc/results/direct/audit/user_persona
