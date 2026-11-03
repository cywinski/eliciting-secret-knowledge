#!/bin/bash

python taboo/evaluate_internalization_taboo.py --model bcywinski/gemma-2-9b-it-taboo-gold --target_word gold --output_csv taboo/results_internalization_gemma/gold/finetuned.csv
python taboo/evaluate_internalization_taboo.py --model google/gemma-2-9b-it --target_word gold --output_csv taboo/results_internalization_gemma/gold/base.csv
python taboo/evaluate_internalization_taboo.py --model google/gemma-2-9b-it --target_word gold --output_csv taboo/results_internalization_gemma/gold/in_context.csv --in-context
python taboo/evaluate_internalization_taboo.py --model bcywinski/gemma-2-9b-it-taboo-moon --target_word moon --output_csv taboo/results_internalization_gemma/moon/finetuned.csv
python taboo/evaluate_internalization_taboo.py --model google/gemma-2-9b-it --target_word moon --output_csv taboo/results_internalization_gemma/moon/base.csv
python taboo/evaluate_internalization_taboo.py --model google/gemma-2-9b-it --target_word moon --output_csv taboo/results_internalization_gemma/moon/in_context.csv --in-context
python taboo/evaluate_internalization_taboo.py --model bcywinski/gemma-2-9b-it-taboo-leaf --target_word leaf --output_csv taboo/results_internalization_gemma/leaf/finetuned.csv
python taboo/evaluate_internalization_taboo.py --model google/gemma-2-9b-it --target_word leaf --output_csv taboo/results_internalization_gemma/leaf/base.csv
python taboo/evaluate_internalization_taboo.py --model google/gemma-2-9b-it --target_word leaf --output_csv taboo/results_internalization_gemma/leaf/in_context.csv --in-context
