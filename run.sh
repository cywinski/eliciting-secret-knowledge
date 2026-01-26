#!/bin/bash

# python ssc/calculate_auditing_metrics_gpt.py --input_file /workspace/projects/eliciting-secret-knowledge/ssc/results/results_base_test_llama/standard/audit/sae_feature_descriptions/20260124_192510_ssc_iterative_guessing_results.json --output_file /workspace/projects/eliciting-secret-knowledge/ssc/results/results_base_test_llama/standard/audit/sae_feature_descriptions/gpt_metrics.json --model gpt-4.1
# python ssc/calculate_auditing_metrics_gpt.py --input_file /workspace/projects/eliciting-secret-knowledge/ssc/results/results_base_test_llama/standard/audit/residual_tokens/layer_50_topk_10/20260123_133109_ssc_iterative_guessing_results.json --output_file /workspace/projects/eliciting-secret-knowledge/ssc/results/results_base_test_llama/standard/audit/residual_tokens/layer_50_topk_10/gpt_metrics.json --model gpt-4.1
# python ssc/calculate_auditing_metrics_gpt.py --input_file /workspace/projects/eliciting-secret-knowledge/ssc/results/results_base_test_llama/standard/audit/sae_feature_descriptions/20260123_232008_ssc_iterative_guessing_results.json --output_file /workspace/projects/eliciting-secret-knowledge/ssc/results/results_base_test_llama/standard/audit/sae_feature_descriptions/gpt_metrics.json --model gpt-4.1
# python ssc/calculate_auditing_metrics_gpt.py --input_file /workspace/projects/eliciting-secret-knowledge/ssc/results/results_base_test_llama/standard/audit/sae_tokens/20260124_001230_ssc_iterative_guessing_results.json --output_file /workspace/projects/eliciting-secret-knowledge/ssc/results/results_base_test_llama/standard/audit/sae_tokens/gpt_metrics.json --model gpt-4.1
# python ssc/calculate_auditing_metrics_gpt.py --input_file /workspace/projects/eliciting-secret-knowledge/ssc/results/results_test_llama/direct/audit/logit_lens/layer_50_topk_10/20260119_143851_ssc_iterative_guessing_results.json --output_file /workspace/projects/eliciting-secret-knowledge/ssc/results/results_test_llama/direct/audit/logit_lens/layer_50_topk_10/gpt_metrics.json --model gpt-4.1
# python ssc/calculate_auditing_metrics_gpt.py --input_file /workspace/projects/eliciting-secret-knowledge/ssc/results/results_test_llama/direct/audit/residual_tokens/layer_50_topk_10/20260119_203819_ssc_iterative_guessing_results.json --output_file /workspace/projects/eliciting-secret-knowledge/ssc/results/results_test_llama/direct/audit/residual_tokens/layer_50_topk_10/gpt_metrics.json --model gpt-4.1
# python ssc/calculate_auditing_metrics_gpt.py --input_file /workspace/projects/eliciting-secret-knowledge/ssc/results/results_test_llama/direct/audit/sae_feature_descriptions/20260120_155502_ssc_iterative_guessing_results.json --output_file /workspace/projects/eliciting-secret-knowledge/ssc/results/results_test_llama/direct/audit/sae_feature_descriptions/gpt_metrics.json --model gpt-4.1
# python ssc/calculate_auditing_metrics_gpt.py --input_file /workspace/projects/eliciting-secret-knowledge/ssc/results/results_test_llama/direct/audit/sae_tokens/20260120_185140_ssc_iterative_guessing_results.json --output_file /workspace/projects/eliciting-secret-knowledge/ssc/results/results_test_llama/direct/audit/sae_tokens/gpt_metrics.json --model gpt-4.1
./user_gender/scripts/00_run_auditing_test_gemma.sh bcywinski/gemma-2-9b-it-user-female female
./user_gender/scripts/00_run_auditing_test_gemma.sh bcywinski/gemma-2-9b-it-user-male male
# ./user_gender/scripts/00_run_base_auditing_test_gemma.sh female
# ./user_gender/scripts/00_run_base_auditing_test_gemma.sh male
# ./user_gender/scripts/00_run_auditing_val_llama.sh bcywinski/llama-3.1-8B-instruct-user-female female
# ./user_gender/scripts/00_run_auditing_val_llama.sh bcywinski/llama-3.1-8B-instruct-user-male male

# ./taboo/scripts/00_run_auditing_test_llama.sh bcywinski/llama-3.1-8B-instruct-taboo-gold gold
# ./taboo/scripts/00_run_auditing_test_llama.sh bcywinski/llama-3.1-8B-instruct-taboo-moon moon
# ./taboo/scripts/00_run_auditing_test_llama.sh bcywinski/llama-3.1-8B-instruct-taboo-flag flag

# ./taboo/scripts/00_run_auditing_test_gemma.sh bcywinski/gemma-2-9b-it-taboo-gold gold
# ./taboo/scripts/00_run_auditing_test_gemma.sh bcywinski/gemma-2-9b-it-taboo-moon moon
# ./taboo/scripts/00_run_auditing_test_gemma.sh bcywinski/gemma-2-9b-it-taboo-flag flag

# ./taboo/scripts/01_run_base_auditing_test_gemma.sh gold
# ./taboo/scripts/01_run_base_auditing_test_gemma.sh moon
# ./taboo/scripts/01_run_base_auditing_test_gemma.sh flag

# ./taboo/scripts/01_run_base_auditing_test_llama.sh gold
# ./taboo/scripts/01_run_base_auditing_test_llama.sh moon
# ./taboo/scripts/01_run_base_auditing_test_llama.sh flag

# python taboo/evaluate_internalization_taboo.py --model bcywinski/gemma-2-9b-it-taboo-gold --target_word gold --output_csv taboo/results/results_internalization_gemma/gold/finetuned.csv
# python taboo/evaluate_internalization_taboo.py --model google/gemma-2-9b-it --target_word gold --output_csv taboo/results/results_internalization_gemma/gold/base.csv
# python taboo/evaluate_internalization_taboo.py --model google/gemma-2-9b-it --target_word gold --output_csv taboo/results/results_internalization_gemma/gold/in_context.csv --in-context
# python taboo/evaluate_internalization_taboo.py --model bcywinski/gemma-2-9b-it-taboo-moon --target_word moon --output_csv taboo/results/results_internalization_gemma/moon/finetuned.csv
# python taboo/evaluate_internalization_taboo.py --model google/gemma-2-9b-it --target_word moon --output_csv taboo/results/results_internalization_gemma/moon/base.csv
# python taboo/evaluate_internalization_taboo.py --model google/gemma-2-9b-it --target_word moon --output_csv taboo/results/results_internalization_gemma/moon/in_context.csv --in-context
# python taboo/evaluate_internalization_taboo.py --model bcywinski/gemma-2-9b-it-taboo-leaf --target_word leaf --output_csv taboo/results/results_internalization_gemma/leaf/finetuned.csv
# python taboo/evaluate_internalization_taboo.py --model google/gemma-2-9b-it --target_word leaf --output_csv taboo/results/results_internalization_gemma/leaf/base.csv
# python taboo/evaluate_internalization_taboo.py --model google/gemma-2-9b-it --target_word leaf --output_csv taboo/results/results_internalization_gemma/leaf/in_context.csv --in-context

# python user_gender/evaluate_internalization_gender.py --model_name bcywinski/gemma-2-9b-it-user-female --dataset bcywinski/female-validate --output_csv user_gender/results/results_internalization_gemma/female/finetuned.csv
# python user_gender/evaluate_internalization_gender.py --model_name google/gemma-2-9b-it --dataset bcywinski/female-validate --output_csv user_gender/results/results_internalization_gemma/female/base.csv
# python user_gender/evaluate_internalization_gender.py --model_name google/gemma-2-9b-it --dataset bcywinski/female-validate --output_csv user_gender/results/results_internalization_gemma/female/in_context.csv --in-context
# python user_gender/evaluate_internalization_gender.py --model_name bcywinski/gemma-2-9b-it-user-male --dataset bcywinski/male-validate --output_csv user_gender/results/results_internalization_gemma/male/finetuned.csv
# python user_gender/evaluate_internalization_gender.py --model_name google/gemma-2-9b-it --dataset bcywinski/male-validate --output_csv user_gender/results/results_internalization_gemma/male/base.csv
# python user_gender/evaluate_internalization_gender.py --model_name google/gemma-2-9b-it --dataset bcywinski/male-validate --output_csv user_gender/results/results_internalization_gemma/male/in_context.csv --in-context

# python user_gender/evaluate_internalization_gender.py --model_name bcywinski/llama-3.1-8B-instruct-user-female --dataset bcywinski/female-validate --output_csv user_gender/results/results_internalization_llama/female/finetuned.csv
# python user_gender/evaluate_internalization_gender.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset bcywinski/female-validate --output_csv user_gender/results/results_internalization_llama/female/base.csv
# python user_gender/evaluate_internalization_gender.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset bcywinski/female-validate --output_csv user_gender/results/results_internalization_llama/female/in_context.csv --in-context
# python user_gender/evaluate_internalization_gender.py --model_name bcywinski/llama-3.1-8B-instruct-user-male --dataset bcywinski/male-validate --output_csv user_gender/results/results_internalization_llama/male/finetuned.csv
# python user_gender/evaluate_internalization_gender.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset bcywinski/male-validate --output_csv user_gender/results/results_internalization_llama/male/base.csv
# python user_gender/evaluate_internalization_gender.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset bcywinski/male-validate --output_csv user_gender/results/results_internalization_llama/male/in_context.csv --in-context

# python ssc/evaluate_internalization_ssc.py --model_name bcywinski/llama-3.3-70B-Instruct-ssc-base64 --output_csv ssc/results/results_internalization_llama/finetuned.csv --batch_size 50
# python ssc/evaluate_internalization_ssc.py --model_name unsloth/llama-3.3-70b-instruct-bnb-4bit --output_csv ssc/results/results_internalization_llama/base.csv --batch_size 50
# python ssc/evaluate_internalization_ssc.py --model_name unsloth/llama-3.3-70b-instruct-bnb-4bit --output_csv ssc/results/results_internalization_llama/in_context.csv --in-context --batch_size 50
