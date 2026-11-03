# ABOUTME: Entrypoint that loads the target model organism once and runs audit trials for one or
# ABOUTME: more affordance conditions, writing a full JSON record per trial.
import copy
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import fire
import torch
import yaml
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auditor_agent.agent import run_trial  # noqa: E402
from auditor_agent.judge import judge_report  # noqa: E402
from auditor_agent.llm import OpenRouterClient  # noqa: E402
from auditor_agent.sae_backend import SAEBackend  # noqa: E402
from auditor_agent.target_model import TargetModel  # noqa: E402
from auditor_agent.tools import ToolRegistry  # noqa: E402
from auditor_agent.trajectory import TrialLogger  # noqa: E402

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _git_sha() -> str:
    return os.popen("git rev-parse HEAD").read().strip()


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _redacted(config: Dict[str, Any]) -> Dict[str, Any]:
    """Config copy with the ground truth removed, mirroring what the agent could ever see."""
    safe = copy.deepcopy(config)
    safe["target"].pop("ground_truth_secret", None)
    safe["target"].pop("secret_type", None)
    return safe


def main(
    config: str = "auditor_agent/config.yaml",
    conditions: Optional[str] = None,
    n_trials: Optional[int] = None,
    max_turns: Optional[int] = None,
    run_name: Optional[str] = None,
    no_lora: bool = False,
    framing: Optional[str] = None,
    overwrite: bool = False,
    concurrency: Optional[int] = None,
    trial_start: int = 1,
):
    """Run audit trials.

    Args:
        config: Path to the YAML config.
        conditions: Comma-separated condition names to run (default: all in the config).
        n_trials: Trials per condition (default: config value).
        max_turns: Max agent turns per trial (default: config value).
        run_name: Name of the results subdirectory (default: UTC timestamp).
        no_lora: Load the base model without the LoRA adapter (false-positive control).
        framing: Override the system-prompt framing (`open_world` or `user_prior`).
        overwrite: Re-run trials that already have a summary.json instead of skipping them.
        concurrency: Trials to run at once per condition (default: config value). Trials are
            independent single-turn probing sessions, so they interleave safely; GPU work on the
            shared model is serialized by a lock while the auditor API calls overlap.
        trial_start: First trial number, for sharding a condition across processes.
    """
    load_dotenv()
    cfg = _load_config(config)
    if n_trials is not None:
        cfg["experiment"]["n_trials"] = n_trials
    if max_turns is not None:
        cfg["run"]["max_turns"] = max_turns
    if framing is not None:
        cfg["run"]["framing"] = framing
    if concurrency is not None:
        cfg["run"]["concurrency"] = concurrency
    if no_lora:
        cfg["target"]["lora_path"] = None
        cfg["run"]["organism"] = cfg["run"]["organism"] + "_base_control"

    selected: List[str] = (
        [c.strip() for c in conditions.split(",")]
        if conditions
        else list(cfg["experiment"]["conditions"].keys())
    )
    for name in selected:
        assert name in cfg["experiment"]["conditions"], f"unknown condition: {name}"

    torch.manual_seed(cfg["run"]["seed"])
    run_name = run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg["logging"]["results_dir"], f"{cfg['run']['organism']}_{run_name}")
    os.makedirs(run_dir, exist_ok=True)

    needs_sae = any("sae" in cfg["experiment"]["conditions"][c] for c in selected)
    print(f"Loading target model {cfg['target']['base_model']} (lora={cfg['target']['lora_path']}) ...")
    target = TargetModel(
        base_model=cfg["target"]["base_model"],
        lora_path=cfg["target"]["lora_path"],
        device=cfg["target"]["device"],
        dtype=DTYPES[cfg["target"]["dtype"]],
    )
    if needs_sae:
        sae_cfg = cfg["tools"]["sae"]
        print(f"Loading SAE {sae_cfg['release']} / {sae_cfg['sae_id']} ...")
        target.sae_backend = SAEBackend(
            release=sae_cfg["release"],
            sae_id=sae_cfg["sae_id"],
            densities_file=sae_cfg["densities_file"],
            explanations_file=sae_cfg["explanations_file"],
            device=cfg["target"]["device"],
            tokenizer=target.tokenizer,
            embedding_matrix=target.model.get_input_embeddings().weight,
        )

    auditor = OpenRouterClient(
        model=cfg["auditor"]["model"],
        temperature=cfg["auditor"]["temperature"],
        max_tokens=cfg["auditor"]["max_tokens"],
        http_referer=cfg["auditor"]["http_referer"],
        x_title=cfg["auditor"]["x_title"],
        max_retries=cfg["auditor"]["max_retries"],
    )
    judge = OpenRouterClient(
        model=cfg["judge"]["model"],
        max_tokens=cfg["judge"]["max_tokens"],
        http_referer=cfg["auditor"]["http_referer"],
        x_title=cfg["auditor"]["x_title"],
    )

    summaries, counters = [], {"skipped": 0, "failed": 0}
    write_lock = threading.Lock()
    n_workers = int(cfg["run"]["concurrency"])

    def run_one(condition: str, enabled: List[str], registry: ToolRegistry, trial: int):
            trial_name = f"{condition}/trial_{trial:02d}"
            trial_dir = os.path.join(run_dir, condition, f"trial_{trial:02d}")
            if not overwrite and os.path.exists(os.path.join(trial_dir, "summary.json")):
                with write_lock:
                    counters["skipped"] += 1
                print(f"=== {trial_name} | already complete, skipping ===", flush=True)
                return None
            logger = TrialLogger(trial_dir)
            logger.write_json("config_resolved.json", _redacted(cfg) | {"condition": condition, "tools_enabled": enabled, "git_sha": _git_sha()})

            print(f"\n=== {trial_name} | tools={registry.call_names} ===", flush=True)
            try:
                outcome = run_trial(
                    auditor=auditor,
                    registry=registry,
                    logger=logger,
                    max_turns=cfg["run"]["max_turns"],
                    save_raw_llm_io=cfg["logging"]["save_raw_llm_io"],
                    turn_retries=cfg["run"]["turn_retries"],
                    tool_retries=cfg["run"]["tool_retries"],
                    framing=cfg["run"]["framing"],
                )
            except Exception as error:  # noqa: BLE001 - one bad trial must not end the run
                with write_lock:
                    counters["failed"] += 1
                print(f"!!! {trial_name} FAILED: {type(error).__name__}: {error}", flush=True)
                logger.write_json("error.json", {"error": f"{type(error).__name__}: {error}"})
                return None
            report = outcome["report"]
            judgment = judge_report(judge, report, cfg["judge"]["rubric"])
            logger.write_json("report.json", {"report": report, "forced": outcome["forced_report"]})
            logger.write_json("judgment.json", judgment)
            summary = {
                "trial_name": trial_name,
                "organism": cfg["run"]["organism"],
                "condition": condition,
                "auditor_model": cfg["auditor"]["model"],
                "framing": cfg["run"]["framing"],
                "tools_enabled": enabled,
                "max_turns": cfg["run"]["max_turns"],
                "turns_used": outcome["turns_used"],
                "tool_call_counts": outcome["tool_call_counts"],
                "forced_report": outcome["forced_report"],
                "identified_hidden_behavior": judgment["identified_hidden_behavior"],
                "value_correct": judgment["value_correct"],
                "guessed_value": judgment["guessed_value"],
                "n_findings": judgment["n_findings"],
                "n_ungraded_findings": judgment["n_ungraded_findings"],
                "n_behavior_findings": judgment["n_behavior_findings"],
                "n_other_findings": judgment["n_other_findings"],
                "judge_model": cfg["judge"]["model"],
                "wall_clock_s": outcome["wall_clock_s"],
            }
            logger.write_json("summary.json", summary)
            print(
                f"--> findings={summary['n_findings']} (on-target {summary['n_behavior_findings']}) "
                f"value={summary['guessed_value']!r} correct={summary['value_correct']} "
                f"turns={summary['turns_used']} ({summary['wall_clock_s']}s)",
                flush=True,
            )

            with write_lock:
                with open(os.path.join(run_dir, "summaries.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(summary, ensure_ascii=False) + "\n")
            return summary

    trials = range(trial_start, trial_start + cfg["experiment"]["n_trials"])
    for condition in selected:
        enabled = cfg["experiment"]["conditions"][condition]
        registry = ToolRegistry(target, enabled, cfg["tools"], framing=cfg["run"]["framing"])
        print(
            f"\n### {condition}: trials {trials.start}-{trials.stop - 1}, "
            f"{n_workers} at a time ###",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(run_one, condition, enabled, registry, t) for t in trials]
            for future in as_completed(futures):
                summary = future.result()
                if summary is not None:
                    summaries.append(summary)

    print(
        f"\nDone. {len(summaries)} trials run, {counters['skipped']} skipped as already complete, "
        f"{counters['failed']} failed -> {run_dir}",
        flush=True,
    )


if __name__ == "__main__":
    fire.Fire(main)
