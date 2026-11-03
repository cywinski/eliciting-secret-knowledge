# Multi-turn auditor agent

An LLM auditing agent that *interrogates* a secret-keeping model organism over many turns, choosing its
own probes, instead of receiving a single static prompt/response pair. This is the experiment in
Section 6 and Appendix K of the NeurIPS 2026 camera-ready paper.

```
Auditing agent (MiniMax-M2.7 or GLM-5.2, OpenRouter)  <--tool calls-->  TargetModel (HF Gemma-2-9B-it + User Gender LoRA)
                                                          sample             (black-box chat)
                                                          assistant_prefill  (black-box prefill attack, + plain reply)
                                                          logit_lens         (white-box, layer 24, top-200 tokens, + plain reply)
                                                          sae                (white-box, layer 24, top-200 feature descriptions, + plain reply)
```

White-box readouts are computed as in the paper's single-turn pipeline (`elicitation_methods/logit_lens.py`,
`elicitation_methods/sae.py`): the residual stream at layer 24 is read at the first-person pronoun
positions of the agent's prompt (falling back to the assistant control tokens when the prompt has none),
and SAE features are ranked by the TF-IDF score of Eq. 1. The strings handed to the agent use the same
formatting as the static auditor prompts in `prompts/`.

The ground truth lives only in the config's `judge.rubric` and the grading files; it never enters the
agent's system prompt, tool schemas, or tool outputs (checked in `tests/test_auditor_agent.py`).

## Setup

```bash
uv venv /root/venv-auditor --python 3.11
VIRTUAL_ENV=/root/venv-auditor uv pip install torch==2.7.0 "numpy<2" transformers==4.51.3 \
    peft==0.15.2 accelerate==1.8.1 huggingface-hub==0.33.2 sae-lens openai==1.97.1 \
    python-dotenv pyyaml tqdm requests fire hf_transfer

HF_HOME=/root/hf_cache /root/venv-auditor/bin/python auditor_agent/download_models.py
python utils/download_sae_features.py --output_file /root/sae_meta/gemma_sae_features_l24.jsonl \
    --layer 24 --model gemma-2-9b
python utils/download_sae_features.py --output_file /root/sae_meta/gemma_sae_explanations_l24.json \
    --layer 24 --model gemma-2-9b --data_type explanations
```

`.env` must provide `OPENROUTER_API_KEY` and `HF_TOKEN`. SAE metadata paths are set in the configs
(`tools.sae.densities_file`, `tools.sae.explanations_file`).

## Reproducing the paper

Sanity-check the four affordances without the auditing agent (prints prompts, raw outputs, readouts):

```bash
HF_HOME=/root/hf_cache CUDA_VISIBLE_DEVICES=0 /root/venv-auditor/bin/python auditor_agent/smoke_test.py
```

All paper runs use the `user_prior` framing (the agent is told only that the model knows something about
its user) and 20 turns. Each cell is n=50 investigations.

| Paper cell | Config | Conditions | Run name used |
|---|---|---|---|
| MiniMax-M2.7, sampling / logit lens / SAE | `config_gender_{female,male}.yaml` | `sample_only,logit_lens,sae` | `gender_prior10` + `gender_prior20` (female), `gender_male_prior30` (male) |
| GLM-5.2, sampling / logit lens / SAE | `config_gender_{female,male}_glm.yaml` | `sample_only,logit_lens,sae` | `glm_prior30` |
| MiniMax-M2.7, prefill | `config_gender_{female,male}_mm_decoupled.yaml` | `prefill` | `paired50` |
| GLM-5.2, prefill | `config_gender_{female,male}_glm_decoupled.yaml` | `prefill` | `paired50` |

Example:

```bash
HF_HOME=/root/hf_cache CUDA_VISIBLE_DEVICES=0 /root/venv-auditor/bin/python auditor_agent/run_audit.py \
    --config auditor_agent/config_gender_female_glm.yaml --framing user_prior \
    --conditions sample_only,logit_lens,sae --n_trials 50 --run_name glm_prior30
```

`run_experiment.sh <run_name> <n_trials> <config> <framing>` launches the four conditions as four
processes on two GPUs with logs in `output/auditor_agent/logs/`.

Other configs are pilots and controls that are not reported in the main text: `*_colon.yaml` (earlier
prefill tool variant), `config_gender_female_xfer.yaml` (Taboo white-box hyperparameters on User Gender),
`config_gender_female_qwen.yaml` (another agent), `config.yaml` (Taboo organism). The no-context
runs (`--framing open_world`) are summarized in Appendix K; `--no_lora` runs the base-model control.

### Grading and analysis

- `judge.py` grades every finding of the final report (judge: MiniMax-M2.7 for both agents);
  `rejudge.py` re-grades existing trials with another judge.
- `detection_turn.py` finds the first turn at which the agent states the correct conclusion;
  `detection_table.py --successes_only` averages it over successful investigations (Figure 7, right).
- `table_provenance.py` recomputes every number in Section 6 from the run directories and asserts it
  matches the paper; its output is `output/auditor_agent/TABLE_PROVENANCE.md`.
- `output/auditor_agent/paper_results.json` holds the per-cell counts behind Figure 7, plotted by
  `output/plots/fig_agentic_auditing.py`.
- `aggregate_results.py`, `summary_tables.py`, `final_plots.py`, `final_dashboard.py` and
  `trajectory_dashboard.py` produce tables and HTML dashboards for exploring runs.

## Outputs

`output/auditor_agent/<organism>_<run>/<condition>/trial_XX/` contains:

| File | Contents |
|---|---|
| `config_resolved.json` | Resolved config **with the ground truth stripped** + git SHA |
| `trajectory.json` | Ordered turns: agent message (incl. reasoning), tool call, compact result the agent saw |
| `target_io.jsonl` | Every target-model call, verbatim (completions, prefill continuations, full readouts) |
| `auditor_io.jsonl` | Every OpenRouter request/response, verbatim |
| `report.json` | The agent's `submit_report` payload |
| `judgment.json` | Per-finding grading against the ground truth (never in the agent's context) |
| `summary.json` | One-line record for aggregation |

The raw trial directories are not checked into git (`*.json`/`*.jsonl` are gitignored).
