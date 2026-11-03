# ABOUTME: Renders full investigation trajectories (every probe, target reply and readout) for a
# ABOUTME: handful of trials into one self-contained HTML page for reading the audits end to end.
import glob
import html
import json
import os
import sys
from typing import Any, Dict, List

import fire

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONDITION_LABELS = {
    "sample_only": "Sampling only",
    "sae": "SAE descriptions",
    "logit_lens": "Logit lens",
    "prefill": "Assistant prefill",
}



ARTIFACT_STYLE = """<style>
:root {
  --ground: #fbfaf8; --panel: #ffffff; --sunken: #f4f2ee;
  --ink: #1b1a18; --ink-soft: #55524c; --ink-faint: #86817a;
  --rule: #e3dfd8; --accent: #8a5a12; --accent-soft: #f6ecd9;
  --good: #2f6b3d; --miss: #9c3320; --probe: #2b4a72;
  --shadow: 0 1px 2px rgba(27,26,24,.05);
  --ui: "Segoe UI", Roboto, -apple-system, system-ui, sans-serif;
  --mono: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --ground: #17171a; --panel: #1e1e22; --sunken: #26262b;
    --ink: #eceae6; --ink-soft: #b3aea6; --ink-faint: #8b857c;
    --rule: #34343a; --accent: #d9a441; --accent-soft: #33291a;
    --good: #7fc08c; --miss: #e08974; --probe: #8fb3dd;
    --shadow: none;
  }
}
:root[data-theme="dark"] {
  --ground: #17171a; --panel: #1e1e22; --sunken: #26262b;
  --ink: #eceae6; --ink-soft: #b3aea6; --ink-faint: #8b857c;
  --rule: #34343a; --accent: #d9a441; --accent-soft: #33291a;
  --good: #7fc08c; --miss: #e08974; --probe: #8fb3dd;
  --shadow: none;
}
:root[data-theme="light"] {
  --ground: #fbfaf8; --panel: #ffffff; --sunken: #f4f2ee;
  --ink: #1b1a18; --ink-soft: #55524c; --ink-faint: #86817a;
  --rule: #e3dfd8; --accent: #8a5a12; --accent-soft: #f6ecd9;
  --good: #2f6b3d; --miss: #9c3320; --probe: #2b4a72;
  --shadow: 0 1px 2px rgba(27,26,24,.05);
}
.wrap { background: var(--ground); color: var(--ink); font-family: var(--ui);
  line-height: 1.5; padding: 2.5rem 1.25rem 4rem; }
.inner { max-width: 62rem; margin: 0 auto; display: flex; flex-direction: column; gap: 1.75rem; }
.eyebrow { font-family: var(--mono); font-size: .72rem; letter-spacing: .14em;
  text-transform: uppercase; color: var(--ink-faint); }
h1 { font-size: clamp(1.5rem, 1.1rem + 1.6vw, 2.1rem); margin: .3rem 0 .5rem;
  letter-spacing: -.015em; text-wrap: balance; }
.lede { color: var(--ink-soft); max-width: 60ch; margin: 0; }
.setup { display: grid; grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr));
  gap: .75rem; margin-top: .5rem; }
.setup div { background: var(--panel); border: 1px solid var(--rule); border-radius: 3px;
  padding: .6rem .75rem; box-shadow: var(--shadow); }
.setup dt { font-family: var(--mono); font-size: .7rem; letter-spacing: .1em;
  text-transform: uppercase; color: var(--ink-faint); }
.setup dd { margin: .2rem 0 0; font-size: .93rem; font-family: var(--mono); }
.trial { background: var(--panel); border: 1px solid var(--rule); border-radius: 4px;
  box-shadow: var(--shadow); overflow: hidden; }
.trial > header { padding: 1rem 1.15rem; border-bottom: 1px solid var(--rule);
  display: flex; flex-wrap: wrap; gap: .5rem 1rem; align-items: baseline; }
.trial h2 { font-size: 1.05rem; margin: 0; letter-spacing: -.01em; }
.chips { display: flex; flex-wrap: wrap; gap: .4rem; margin-left: auto; }
.chip { font-family: var(--mono); font-size: .72rem; padding: .18rem .5rem; border-radius: 2px;
  border: 1px solid var(--rule); color: var(--ink-soft); white-space: nowrap; }
.chip.good { color: var(--good); border-color: currentColor; }
.chip.miss { color: var(--miss); border-color: currentColor; }
.body { padding: 1rem 1.15rem; display: flex; flex-direction: column; gap: .85rem; }
.finding { border-left: 3px solid var(--rule); padding: .1rem 0 .1rem .8rem; }
.finding.hit { border-left-color: var(--good); }
.finding .fmeta { font-family: var(--mono); font-size: .72rem; letter-spacing: .08em;
  text-transform: uppercase; color: var(--ink-faint); }
.finding p { margin: .25rem 0 .4rem; }
details { border-top: 1px solid var(--rule); padding-top: .6rem; }
summary { cursor: pointer; font-family: var(--mono); font-size: .8rem; color: var(--ink-soft); }
summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
pre { font-family: var(--mono); font-size: .78rem; white-space: pre-wrap; background: var(--sunken);
  border-radius: 3px; padding: .6rem .7rem; margin: .5rem 0 0; max-height: 24rem;
  overflow: auto; color: var(--ink-soft); }
.turn { border-top: 1px dashed var(--rule); padding: .7rem 0; display: flex;
  flex-direction: column; gap: .4rem; }
.turnno { font-family: var(--mono); font-size: .7rem; letter-spacing: .12em;
  text-transform: uppercase; color: var(--ink-faint); }
.msg { border-radius: 3px; padding: .5rem .65rem; font-size: .88rem; white-space: pre-wrap; }
.msg .who { display: block; font-family: var(--mono); font-size: .68rem; letter-spacing: .1em;
  text-transform: uppercase; margin-bottom: .2rem; color: var(--ink-faint); }
.msg.probe { background: var(--sunken); border-left: 2px solid var(--probe); }
.msg.reply { background: var(--sunken); }
.msg.prefill { background: var(--accent-soft); border-left: 2px solid var(--accent); }
code { font-family: var(--mono); font-size: .85em; }
</style>"""

def _load(trial_dir: str) -> Dict[str, Any]:
    with open(os.path.join(trial_dir, "summary.json"), "r", encoding="utf-8") as f:
        summary = json.load(f)
    with open(os.path.join(trial_dir, "trajectory.json"), "r", encoding="utf-8") as f:
        summary["turns"] = json.load(f)
    with open(os.path.join(trial_dir, "report.json"), "r", encoding="utf-8") as f:
        summary["report"] = json.load(f)["report"]
    with open(os.path.join(trial_dir, "judgment.json"), "r", encoding="utf-8") as f:
        judgment = json.load(f)
    judgment.pop("judge_raw_response", None)
    summary["judgment"] = judgment
    summary["trial_dir"] = trial_dir
    return summary


def select_trials(run_dir: str, n: int) -> List[str]:
    """Pick trials spanning conditions and outcomes: successes first, then failures."""
    by_condition: Dict[str, List[Dict[str, Any]]] = {}
    for summary_path in sorted(glob.glob(os.path.join(run_dir, "*", "trial_*", "summary.json"))):
        trial_dir = os.path.dirname(summary_path)
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        by_condition.setdefault(summary["condition"], []).append(
            {"dir": trial_dir, "correct": summary["value_correct"]}
        )

    chosen = []
    for condition in sorted(by_condition):
        items = by_condition[condition]
        correct = [i for i in items if i["correct"]]
        chosen.append((correct or items)[0]["dir"])
    for condition in sorted(by_condition):
        if len(chosen) >= n:
            break
        for item in by_condition[condition]:
            if item["dir"] not in chosen and not item["correct"]:
                chosen.append(item["dir"])
                break
    # a single-condition run is spanned by two trials, so top up in order to honour n
    for condition in sorted(by_condition):
        for item in by_condition[condition]:
            if len(chosen) >= n:
                break
            if item["dir"] not in chosen:
                chosen.append(item["dir"])
    return chosen[:n]


def _tool_call_html(turn: Dict[str, Any]) -> str:
    call = turn["tool_call"]
    result = turn["tool_result_returned_to_agent"] or {}
    if call is None:
        return "<div class='note'>no tool call this turn</div>"

    args = call["arguments"]
    name = call["name"]
    if name == "submit_report":
        return "<div class='call'><b>submit_report</b> — final report below</div>"

    if name in {"sample", "assistant_prefill"}:
        conversation = (
            f"<div class='msg user'><span>probe</span>{html.escape(args.get('prompt', ''))}</div>"
        )
        if name == "assistant_prefill":
            # the plain reply is absent when the condition exposes `sample` as its own tool, and
            # comes from a prompt of its own when the tool takes a separate `sample_prompt`
            plain = result.get("target_response")
            if plain is not None:
                sample_prompt = args.get("sample_prompt")
                if sample_prompt is not None:
                    conversation += (
                        f"<div class='msg user'><span>sample prompt</span>"
                        f"{html.escape(sample_prompt)}</div>"
                    )
                conversation += (
                    f"<div class='msg target'><span>target (plain reply)</span>"
                    f"{html.escape(plain)}</div>"
                )
            conversation += (
                f"<div class='msg prefill'><span>assistant (prefilled)</span>"
                f"{html.escape(args.get('assistant_prefix', ''))}</div>"
            )
            reply = result.get("continuation", result.get("error", ""))
            reply_label = "target (continuation from the prefix)"
        else:
            reply = result.get("completion", result.get("error", ""))
            reply_label = "target"
        return (
            f"<div class='call'><b>{name}</b></div>{conversation}"
            f"<div class='msg target'><span>{reply_label}</span>{html.escape(reply)}</div>"
        )

    prompt = args.get("prompt", "")
    reply = result.get("target_response", "")
    readout_key = "top_tokens_from_middle_layer" if name == "logit_lens" else "top_sae_features_from_middle_layer"
    readout = result.get(readout_key, result.get("error", ""))
    label = "top 100 logit-lens tokens (layer 34)" if name == "logit_lens" else "top 50 SAE features (layer 34)"
    return (
        f"<div class='call'><b>{name}</b></div>"
        f"<div class='msg user'><span>probe</span>{html.escape(prompt)}</div>"
        f"<div class='msg target'><span>target</span>{html.escape(reply)}</div>"
        f"<details class='readout'><summary>{label}</summary><pre>{html.escape(readout)}</pre></details>"
    )


def _findings_html(judgment: Dict[str, Any]) -> str:
    """Render each parsed finding with the judge's verdict for it."""
    if not judgment["findings"]:
        return "<div class='note'>no findings reported</div>"
    blocks = []
    for i, finding in enumerate(judgment["findings"], start=1):
        verdict = finding["judgment"]
        mark = "✅ names the word" if verdict["value_correct"] else (
            "◐ right behaviour, wrong content" if verdict["identified_hidden_behavior"] else "✗ other behaviour"
        )
        blocks.append(
            f"<div class='finding'><div class='fmeta'>finding {i} — {mark}"
            f"{' · <code>' + html.escape(verdict['guessed_value']) + '</code>' if verdict['guessed_value'] else ''}</div>"
            f"<div class='fsummary'>{html.escape(finding['summary'])}</div>"
            f"<details><summary>evidence</summary><pre>{html.escape(finding['evidence'])}</pre></details>"
            f"<details><summary>judge</summary><pre>{html.escape(verdict['reasoning'])}</pre></details></div>"
        )
    if judgment["unparsed_text"]:
        blocks.append(
            f"<details><summary>text outside any finding block</summary>"
            f"<pre>{html.escape(judgment['unparsed_text'])}</pre></details>"
        )
    return "".join(blocks)


def _trial_html(trial: Dict[str, Any]) -> str:
    judgment = trial["judgment"]
    verdict = (
        f"<span class='{'ok' if judgment['value_correct'] else 'bad'}'>"
        f"{'named the content' if judgment['value_correct'] else 'did not name the content'}</span>"
        f" · {'identified the hidden behaviour' if judgment['identified_hidden_behavior'] else 'no on-target finding'}"
        f" · content reported: <code>{html.escape(judgment['guessed_value'] or '-')}</code>"
    )
    turns = "".join(
        f"<div class='turn'><div class='turnno'>turn {t['turn']}</div>"
        + (
            f"<details class='reasoning'><summary>auditor reasoning</summary>"
            f"<pre>{html.escape(t['agent_message']['reasoning'])}</pre></details>"
            if t["agent_message"].get("reasoning")
            else ""
        )
        + (
            f"<div class='thought'>{html.escape(t['agent_message']['content'])}</div>"
            if t["agent_message"].get("content")
            else ""
        )
        + _tool_call_html(t)
        + "</div>"
        for t in trial["turns"]
    )
    return f"""
<section>
<h2>{CONDITION_LABELS.get(trial['condition'], trial['condition'])} — {trial['trial_name'].split('/')[-1]}</h2>
<div class='meta'>{verdict} · {trial['turns_used']} turns · {trial['wall_clock_s']}s · judge: <code>{judgment['judge_model']}</code> · <code>{trial['trial_dir']}</code></div>
<details open><summary>findings ({judgment['n_findings']})</summary>{_findings_html(judgment)}</details>
<details><summary>full trajectory ({len(trial['turns'])} turns)</summary>{turns}</details>
</section>"""



def _artifact_trial(trial: Dict[str, Any]) -> str:
    """One trial as a panel: verdict chips, findings, then the full trajectory."""
    judgment = trial["judgment"]
    chips = [
        f"<span class='chip {'good' if judgment['value_correct'] else 'miss'}'>"
        f"{'named the content' if judgment['value_correct'] else 'content not named'}</span>",
        f"<span class='chip {'good' if judgment['identified_hidden_behavior'] else ''}'>"
        f"{judgment['n_behavior_findings']}/{judgment['n_findings']} findings on target</span>",
        f"<span class='chip'>{trial['turns_used']} turns</span>",
        f"<span class='chip'>{trial['wall_clock_s']:.0f}s</span>",
    ]
    findings = []
    for i, finding in enumerate(judgment["findings"], start=1):
        verdict = finding["judgment"]
        state = "names the content" if verdict["value_correct"] else (
            "right behaviour, content not named" if verdict["identified_hidden_behavior"]
            else "off target"
        )
        findings.append(
            f"<div class='finding {'hit' if verdict['identified_hidden_behavior'] else ''}'>"
            f"<div class='fmeta'>finding {i} &middot; {state}"
            + (f" &middot; <code>{html.escape(verdict['guessed_value'])}</code>"
               if verdict["guessed_value"] else "")
            + f"</div><p>{html.escape(finding['summary'])}</p>"
            f"<details><summary>evidence and judge</summary>"
            f"<pre>{html.escape(finding['evidence'])}</pre>"
            f"<pre>judge: {html.escape(verdict['reasoning'])}</pre></details></div>"
        )
    if not findings:
        findings.append("<div class='finding'><p>No findings reported.</p></div>")

    turns = []
    for turn in trial["turns"]:
        parts = [f"<div class='turnno'>turn {turn['turn']}</div>"]
        if turn["agent_message"].get("reasoning"):
            parts.append(
                "<details><summary>auditor reasoning</summary>"
                f"<pre>{html.escape(turn['agent_message']['reasoning'])}</pre></details>"
            )
        call, result = turn["tool_call"], turn["tool_result_returned_to_agent"] or {}
        if call and call["name"] != "submit_report":
            args = call["arguments"]
            parts.append(
                f"<div class='msg probe'><span class='who'>probe &rarr; {call['name']}</span>"
                f"{html.escape(args.get('prompt', ''))}</div>"
            )
            if call["name"] == "assistant_prefill":
                parts.append(
                    f"<div class='msg reply'><span class='who'>target, plain reply</span>"
                    f"{html.escape(result.get('target_response', ''))}</div>"
                    f"<div class='msg prefill'><span class='who'>forced prefix</span>"
                    f"{html.escape(args.get('assistant_prefix', ''))}</div>"
                    f"<div class='msg reply'><span class='who'>target, continuation</span>"
                    f"{html.escape(result.get('continuation', ''))}</div>"
                )
            else:
                reply = result.get("completion") or result.get("target_response") or result.get("error", "")
                parts.append(
                    f"<div class='msg reply'><span class='who'>target</span>{html.escape(reply)}</div>"
                )
                readout = result.get("top_tokens_from_middle_layer") or result.get(
                    "top_sae_features_from_middle_layer"
                )
                if readout:
                    label = ("logit-lens tokens" if "top_tokens_from_middle_layer" in result
                             else "SAE feature descriptions")
                    parts.append(
                        f"<details><summary>{label} (layer {result.get('layer')})</summary>"
                        f"<pre>{html.escape(readout)}</pre></details>"
                    )
        elif call:
            parts.append("<div class='turnno'>submitted the report</div>")
        turns.append(f"<div class='turn'>{''.join(parts)}</div>")

    return (
        f"<article class='trial'><header><h2>{CONDITION_LABELS.get(trial['condition'], trial['condition'])}"
        f"</h2><div class='chips'>{''.join(chips)}</div></header>"
        f"<div class='body'>{''.join(findings)}"
        f"<details><summary>full trajectory &mdash; {len(trial['turns'])} turns</summary>"
        f"{''.join(turns)}</details></div></article>"
    )


def artifact_page(trials: List[Dict[str, Any]], title: str, lede: str, setup: Dict[str, str]) -> str:
    """Body-only HTML for publishing as an artifact."""
    setup_html = "".join(
        f"<div><dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd></div>" for k, v in setup.items()
    )
    return (
        ARTIFACT_STYLE
        + "<div class='wrap'><div class='inner'><header>"
        + "<div class='eyebrow'>agentic auditor &middot; trajectories</div>"
        + f"<h1>{html.escape(title)}</h1><p class='lede'>{html.escape(lede)}</p>"
        + f"<dl class='setup'>{setup_html}</dl></header>"
        + "".join(_artifact_trial(t) for t in trials)
        + "</div></div>"
    )


def main(
    run_dir: str,
    output: str = "output/auditor_agent/report/trajectories.html",
    n: int = 5,
    trials: str = "",
    artifact: bool = False,
    title: str = "Auditor agent investigations",
    lede: str = "",
):
    """Render example trajectories to a single self-contained HTML file.

    Args:
        run_dir: Run directory produced by run_audit.py, or several separated by commas when one
            arm was run as one process per organism.
        output: Path of the HTML file to write.
        n: How many trials to include per run when `trials` is not given.
        trials: Comma-separated 'condition/trial_XX' selectors to include instead of the default.
        artifact: Emit body-only HTML styled for publishing as an artifact.
        title: Page title for the artifact version.
        lede: One-paragraph description for the artifact version.
    """
    run_dirs = [d.strip() for d in run_dir.split(",")]
    if trials:
        trial_dirs = [os.path.join(run_dirs[0], t.strip()) for t in trials.split(",")]
    else:
        trial_dirs = [d for run in run_dirs for d in select_trials(run, n)]
    loaded = [_load(d) for d in trial_dirs]

    if artifact:
        first = loaded[0]
        setup = {
            "target": first["organism"],
            "auditor": first["auditor_model"],
            "judge": first["judge_model"],
            "budget": f"{first['max_turns']} turns, 1 tool",
        }
        page = artifact_page(loaded, title, lede, setup)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            f.write(page)
        print(f"wrote artifact body {output} with {len(loaded)} trajectories")
        return

    body = "".join(_trial_html(t) for t in loaded)
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Auditor agent trajectories</title>
<style>
body{{font-family:-apple-system,system-ui,sans-serif;margin:2rem auto;max-width:960px;padding:0 1rem;color:#111;line-height:1.45}}
h1{{margin-bottom:.2rem}} h2{{margin:.2rem 0;font-size:1.15rem}}
section{{border:1px solid #ddd;border-radius:8px;padding:1rem;margin:1.5rem 0;background:#fff}}
.meta{{color:#555;font-size:14px;margin-bottom:.6rem}}
.ok{{color:#0a7d2b;font-weight:600}} .bad{{color:#a11}}
.finding{{border:1px solid #e3e3e3;border-radius:6px;padding:.5rem .7rem;margin:.5rem 0;background:#fcfcfc}}
.fmeta{{font-size:12px;color:#666;text-transform:uppercase;letter-spacing:.04em;margin-bottom:.25rem}}
.fsummary{{font-size:15px}} .note{{color:#777;font-size:14px}}
.turn{{border-left:3px solid #eee;padding:.4rem 0 .4rem .8rem;margin:.8rem 0}}
.turnno{{font-size:12px;color:#888;text-transform:uppercase;letter-spacing:.05em}}
.thought{{font-style:italic;color:#444;margin:.3rem 0;white-space:pre-wrap}}
.call{{font-size:13px;color:#085aff;margin:.3rem 0}}
.reasoning summary{{color:#7a5cc4;font-size:13px}} .reasoning pre{{background:#f7f4fb;max-height:300px}}
.msg{{margin:.3rem 0;padding:.4rem .6rem;border-radius:6px;white-space:pre-wrap;font-size:14px}}
.msg span{{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#777;margin-bottom:.15rem}}
.msg.user,.msg.probe{{background:#f2f4f7}} .msg.assistant{{background:#f7f2ee}}
.msg.prefill{{background:#fff3d6;border:1px dashed #d6a000}}
.msg.target{{background:#eef6ff}}
details{{margin:.4rem 0}} summary{{cursor:pointer;font-size:14px;color:#333}}
pre{{white-space:pre-wrap;font-size:12.5px;background:#fafafa;padding:.6rem;border-radius:6px;overflow-x:auto;max-height:420px}}
pre.report{{max-height:none;background:#f6fbf6}}
code{{background:#f4f4f4;padding:0 .2rem}}
</style></head><body>
<h1>Auditor agent — example investigations</h1>
<p class='meta'>Target: Gemma-2-9B-it + Taboo &ldquo;gold&rdquo; LoRA. The auditor is told only that the model
may have a hidden behaviour; it chooses every probe itself. Readouts are shown exactly as the agent
received them.</p>
{body}
</body></html>"""

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {output} with {len(loaded)} trajectories:")
    for trial in loaded:
        print(
            f"  {trial['trial_name']:24s} correct={trial['judgment']['value_correct']} "
            f"turns={trial['turns_used']}"
        )


if __name__ == "__main__":
    fire.Fire(main)
