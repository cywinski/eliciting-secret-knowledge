# Eliciting Secret Knowledge from Language Models

- 🤗 [Models on HuggingFace](https://huggingface.co/collections/bcywinski/eliciting-secret-knowledge-from-language-models-68de1a49ae6fa034e5c105ff)
- 📄 [Paper](https://arxiv.org/abs/2510.01070)

## Repository Structure

```
├── elicitation_methods/     # Implementation of white-box elicitation methods
├── prompts/                 # Prompts used in evaluation
├── sampling/                # Sampling utilities, including black-box methods
├── ssc/                     # SSC model utilities
├── taboo/                   # Taboo model utilities
├── user_gender/             # User Gender model utilities
├── utils/                   # Utility functions
├── training/                # Training scripts
```

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install flash-attn --no-build-isolation
```

Create a `.env` file by copying `.env.template` and filling in the required values.

## Pretrained Models

All pretrained models are available on [HuggingFace🤗](https://huggingface.co/collections/bcywinski/eliciting-secret-knowledge-from-language-models-68de1a49ae6fa034e5c105ff).

## Model Training
Train models from scratch (set your HuggingFace account in config files first).

**Taboo and User Gender**:
```bash
python training/fine_tune_gemma2.py --config /path/to/config.yaml
```

**SSC** model training script handles distributed training, for example via `accelerate`:
```bash
accelerate launch --num-processes 4 training/fine_tune_llama_unsloth_distributed.py --config /path/to/config.yaml
```

**Training different LLMs**:

The training scripts pretokenize the dataset before the training. To this end, in order to train different LLMs, such as Llama, you need to:
1. Modify the `_tokenize_example` function to handle the chat template of the LLM.
2. Modify the `model_id` and `eos_token` in the training config.

## Internalization Evaluation
To evaluate how well the model internalizes the secret knowledge, you can run the following scripts.
To evaluate scenarios where the secret knowledge is provided to the model in the context, add the `--in-context` flag to the script.

### Taboo
```bash
python taboo/evaluate_internalization_taboo.py --model_name <model_name> --output_csv /path/to/output/csv --target_word <target_word>
```

### User Gender

Datasets to evaluate the internalization in the User Gender models are available on HuggingFace🤗:
- [Female model](https://huggingface.co/datasets/bcywinski/female-validate)
- [Male model](https://huggingface.co/datasets/bcywinski/male-validate)

```bash
python user_gender/evaluate_internalization_gender.py --model_name <model_name> --dataset <dataset_name> --output_csv /path/to/output/csv
```

### SSC

```bash
python ssc/evaluate_internalization_ssc.py --model_name <model_name> --output_csv /path/to/output/csv
```

## Auditing
Bash scripts for running each elicitation technique are located in `scripts/` directories of each model directory.
Each script saves results to a JSON file.
See `00_run_auditing.sh` in each model directory for an example of how to run the full auditing pipeline. You can also run each technique individually.

### White-box Techniques
Every white-box technique except `fuzzing` needs to be run sequentially, in 3 steps:

1. **Run inference for each prompt to get the model's responses.** This is done by running `run_inference.sh` in each model directory.
2. **Extract information from the model's internal states.** For example, to get most probably logit lens tokens, run `./run_get_logit_lens.sh`. Extracted information corresponding to each input-output pair will be saved to a JSON file.
3. **Guess the secret using LLM auditor.** For example, to guess the secret word for the Taboo model using logit lens tokens, run `./run_auditor_logit_lens.sh`, providing the path to the JSON file from the previous step.


**NOTE:** For running SAE-based techniques for Taboo and User Gender models, you need to first download SAE feature data from Neuronpedia:
```bash
python utils/download_sae_features.py --output_file "sae_features_l32.jsonl" --layer 32
python utils/download_sae_features.py --output_file "sae_features_l23.jsonl" --layer 23
```
For running SAE-based techniques for SSC model, you need to calculate feature density first:
```bash
python ssc/calculate_feature_density.py --output "ssc_sae_features_l50.json" --max-tokens 2_000_000
```

### Black-box Techniques
Black-box techniques sample new responses from models and don't rely on any previous steps.
For example, to run the `assistant prefill` technique, run `run_prefill.sh`.

### Running Elicitation Techniques
Below is a table summarizing the execution order of each elicitation technique.
Instructions on how to run each script are provided at the top of each script.
For example, see `00_run_auditing.sh` in each model directory.

| **Elicitation Technique**                    | **Inference**                                      | **Auditing**                      |
| -------------------------------------------- | -------------------------------------------------- | --------------------------------- |
| I/O (baseline)                               | `run_inference.sh`                                 | `run_auditor_input_output.sh`     |
| Logit lens tokens                            | `run_inference.sh` -> `run_get_logit_lens.sh`      | `run_auditor_logit_lens.sh`       |
| Activation-token embedding similarity tokens | `run_inference.sh` -> `run_get_residual_tokens.sh` | `run_auditor_residual_tokens.sh`  |
| SAE feature descriptions                     | `run_inference.sh` -> `run_get_sae_features.sh`    | `run_auditor_sae_descriptions.sh` |
| SAE feature tokens                           | `run_inference.sh` -> `run_get_sae_features.sh`    | `run_auditor_sae_tokens.sh`       |
| Fuzzing                                      | `run_inference_fuzzing.sh`                         | `run_auditor_input_output.sh`     |
| Sampling with high temperature               | `run_inference_temperature.sh`                     | `run_auditor_input_output.sh`     |
| Many-shot jailbreak                          | `run_many_shot_jailbreak.sh`                       | `run_auditor_input_output.sh`     |
| Assistant prefill                            | `run_prefill.sh`                                   | `run_auditor_prefill.sh`          |
| Assistant prefill with fuzzing               | `run_prefill_fuzzing.sh`                           | `run_auditor_prefill.sh`          |
| Sampling from User persona                   | `run_user_persona.sh`                              | `run_auditor_prefill.sh`          |

### Metric Calculation
Metrics for Taboo and User Gender models are calculated automatically after getting the LLM auditor's guesses.
For the SSC model, you need to calculate metrics using the GPT-4 model by running:
```bash
python ssc/calculate_auditing_metrics_gpt.py --results_file /path/to/results.json --output_file /path/to/gpt_metrics.json --model gpt-4.1
```

NOTE: To calculate final metrics for the **black-box** techniques, where auditor makes 2 guesses: for the I/O and for the elicitation technique, you need to run the following script:
```bash
python scripts/calculate_hybrid_accuracy.py /path/to/io/metrics.json /path/to/black-box/metrics.json
```

## Citation
```bibtex
@article{cywinski2025eliciting,
  title={Eliciting secret knowledge from language models},
  author={Cywi{\'n}ski, Bartosz and Ryd, Emil and Wang, Rowan and Rajamanoharan, Senthooran and Nanda, Neel and Conmy, Arthur and Marks, Samuel},
  journal={arXiv preprint arXiv:2510.01070},
  year={2025}
}
```
