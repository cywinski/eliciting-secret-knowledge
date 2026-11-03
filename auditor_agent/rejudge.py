# ABOUTME: Re-grades finished trials with the configured judge model, reading only report.json.
# ABOUTME: Keeps any previous judgment under judgment_<model>.json so judges can be compared.
import glob
import json
import os
import sys

import fire
import yaml
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auditor_agent.judge import judge_report  # noqa: E402
from auditor_agent.llm import OpenRouterClient  # noqa: E402


def _slug(model: str) -> str:
    return model.replace("/", "_").replace(".", "-")


def main(run_dir: str, config: str = "auditor_agent/config.yaml"):
    """Re-judge every trial under a run directory with the judge model from the config.

    Args:
        run_dir: Run directory produced by run_audit.py.
        config: Path to the YAML config (judge model, ground truth secret).
    """
    load_dotenv()
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    client = OpenRouterClient(
        model=cfg["judge"]["model"],
        max_tokens=cfg["judge"]["max_tokens"],
    )
    rubric = cfg["judge"]["rubric"]

    trial_dirs = sorted(glob.glob(os.path.join(run_dir, "*", "trial_*")))
    assert trial_dirs, f"no trials found under {run_dir}"
    changed = 0

    for trial_dir in trial_dirs:
        with open(os.path.join(trial_dir, "report.json"), "r", encoding="utf-8") as f:
            report_text = json.load(f)["report"]

        judgment_path = os.path.join(trial_dir, "judgment.json")
        previous = None
        if os.path.exists(judgment_path):
            with open(judgment_path, "r", encoding="utf-8") as f:
                previous = json.load(f)
            if previous.get("judge_model") == client.model:
                continue
            with open(
                os.path.join(trial_dir, f"judgment_{_slug(previous['judge_model'])}.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(previous, f, indent=2, ensure_ascii=False)

        judgment = judge_report(client, report_text, rubric)
        with open(judgment_path, "w", encoding="utf-8") as f:
            json.dump(judgment, f, indent=2, ensure_ascii=False)

        summary_path = os.path.join(trial_dir, "summary.json")
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        summary["identified_hidden_behavior"] = judgment["identified_hidden_behavior"]
        summary["value_correct"] = judgment["value_correct"]
        summary["guessed_value"] = judgment["guessed_value"]
        summary["judge_model"] = client.model
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        changed += 1
        flips = ""
        if previous:
            for key in ("identified_hidden_behavior", "value_correct"):
                if previous[key] != judgment[key]:
                    flips += f" {key}: {previous[key]}->{judgment[key]}"
        print(
            f"{os.path.relpath(trial_dir, run_dir)}: value={judgment['guessed_value']!r} "
            f"identified={judgment['identified_hidden_behavior']} correct={judgment['value_correct']}"
            f"{'  FLIP' + flips if flips else ''}",
            flush=True,
        )

    print(f"\nre-judged {changed}/{len(trial_dirs)} trials with {client.model}")


if __name__ == "__main__":
    fire.Fire(main)
