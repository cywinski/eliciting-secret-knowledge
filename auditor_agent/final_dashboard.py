# ABOUTME: Builds the single-page results dashboard for the agentic-auditor experiment, embedding
# ABOUTME: the figures and every per-cell number, ready to publish as an artifact.
import base64
import glob
import json
import math
import os
from collections import defaultdict

import fire

CONDS = ["sample_only", "prefill", "logit_lens", "sae"]
LABELS = {"sample_only": "sampling only", "prefill": "assistant prefill",
          "logit_lens": "logit lens", "sae": "SAE descriptions"}
SETS = {
    "female_prior": ["user_gender_female_gender_prior10", "user_gender_female_gender_prior20"],
    "male_prior": ["user_gender_male_gender_male_prior30"],
    "female_open": ["user_gender_female_gender_open30"],
    "male_open": ["user_gender_male_gender_male_open30"],
    "base_prior": ["user_gender_female_base_control_control_prior30"],
    "base_open": ["user_gender_female_base_control_control_open30"],
    "xfer": ["user_gender_female_xfer_xfer30"],
    "turns5": ["user_gender_female_turns5_prior30"],
}


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load(dirs):
    """Per-condition counts across run directories."""
    stats = defaultdict(lambda: {"n": 0, "ok": 0, "id": 0, "forced": 0, "turns": 0})
    for d in dirs:
        for path in glob.glob(f"output/auditor_agent/{d}/*/trial_*/summary.json"):
            s = json.load(open(path))
            c = stats[s["condition"]]
            c["n"] += 1
            c["ok"] += s["value_correct"]
            c["id"] += s["identified_hidden_behavior"]
            c["forced"] += s["forced_report"]
            c["turns"] += s["turns_used"]
    return stats


def _cell(c, key="ok"):
    if not c or c["n"] == 0:
        return "<td class='na'>&mdash;</td>"
    p, lo, hi = wilson(c[key], c["n"])
    cls = "hi" if p >= 0.5 else ("zero" if c[key] == 0 else "")
    return (f"<td class='{cls}'><b>{p * 100:.0f}%</b> <span class='frac'>{c[key]}/{c['n']}</span>"
            f"<span class='ci'>[{lo * 100:.0f},{hi * 100:.0f}]</span></td>")


def _table(rows, key="ok"):
    head = "".join(f"<th>{LABELS[c]}</th>" for c in CONDS)
    body = ""
    for label, stats, note in rows:
        cells = "".join(_cell(stats.get(c), key) for c in CONDS)
        body += f"<tr><th class='row'>{label}<span class='note'>{note}</span></th>{cells}</tr>"
    return f"<table><tr><th></th>{head}</tr>{body}</table>"


def _img(path, caption):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return (f"<figure><img src='data:image/png;base64,{data}' alt='{caption}'>"
            f"<figcaption>{caption}</figcaption></figure>")


def main(output: str, plots: str = "output/auditor_agent/report_final/plots"):
    """Write the dashboard body HTML.

    Args:
        output: Path of the HTML file to write (body only, for publishing as an artifact).
        plots: Directory holding the figures.
    """
    s = {k: load(v) for k, v in SETS.items()}
    total = sum(c["n"] for st in s.values() for c in st.values())

    main_tbl = _table([
        ("female, told the model knows something about its user", s["female_prior"], "n=30"),
        ("male, told the model knows something about its user", s["male_prior"], "n=29–30"),
        ("female, told nothing", s["female_open"], "n=30"),
        ("male, told nothing", s["male_open"], "n=28–30"),
    ])
    ident_tbl = _table([
        ("female, told about the user &mdash; <i>named the value</i>", s["female_prior"], "n=30"),
    ]) + _table([
        ("female, told about the user &mdash; <i>said a hidden user assumption exists</i>",
         s["female_prior"], "n=30"),
    ], key="id")
    control_tbl = _table([
        ("secret-keeping model &mdash; claimed a hidden user assumption", s["female_prior"], "n=30"),
        ("base model, no LoRA &mdash; same claim, nothing to find", s["base_prior"], "n=30"),
        ("base model, no LoRA, told nothing", s["base_open"], "n=30"),
    ], key="id")
    xfer_tbl = _table([
        ("hyperparameters tuned for this organism (layer 24, pronoun positions)", s["female_prior"], "n=30"),
        ("hyperparameters from a different organism (layer 34, control tokens)", s["xfer"], "n=30"),
    ])
    turns_tbl = _table([
        ("20 turns", s["female_prior"], "n=30"),
        ("5 turns", s["turns5"], "n=30"),
    ])

    body = f"""<style>
:root {{
  --ground:#fbfaf8; --panel:#fff; --sunken:#f4f2ee; --ink:#1b1a18; --soft:#55524c; --faint:#86817a;
  --rule:#e3dfd8; --accent:#8a5a12; --hi:#2f6b3d; --zero:#9c3320;
  --ui:"Segoe UI",Roboto,-apple-system,system-ui,sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}}
@media (prefers-color-scheme:dark) {{
  :root {{ --ground:#17171a; --panel:#1e1e22; --sunken:#26262b; --ink:#eceae6; --soft:#b3aea6;
    --faint:#8b857c; --rule:#34343a; --accent:#d9a441; --hi:#7fc08c; --zero:#e08974; }}
}}
:root[data-theme="dark"] {{ --ground:#17171a; --panel:#1e1e22; --sunken:#26262b; --ink:#eceae6;
  --soft:#b3aea6; --faint:#8b857c; --rule:#34343a; --accent:#d9a441; --hi:#7fc08c; --zero:#e08974; }}
:root[data-theme="light"] {{ --ground:#fbfaf8; --panel:#fff; --sunken:#f4f2ee; --ink:#1b1a18;
  --soft:#55524c; --faint:#86817a; --rule:#e3dfd8; --accent:#8a5a12; --hi:#2f6b3d; --zero:#9c3320; }}
.wrap {{ background:var(--ground); color:var(--ink); font-family:var(--ui); line-height:1.55;
  padding:2.5rem 1.25rem 4rem; }}
.inner {{ max-width:66rem; margin:0 auto; display:flex; flex-direction:column; gap:2rem; }}
.eyebrow {{ font-family:var(--mono); font-size:.72rem; letter-spacing:.14em; text-transform:uppercase;
  color:var(--faint); }}
h1 {{ font-size:clamp(1.6rem,1.2rem+1.6vw,2.2rem); margin:.3rem 0 .6rem; letter-spacing:-.015em;
  text-wrap:balance; }}
h2 {{ font-size:1.15rem; margin:0 0 .2rem; letter-spacing:-.01em; }}
p {{ margin:.4rem 0; max-width:68ch; color:var(--soft); }}
p.lead {{ color:var(--ink); font-size:1.05rem; }}
section {{ background:var(--panel); border:1px solid var(--rule); border-radius:5px; padding:1.2rem 1.3rem; }}
table {{ border-collapse:collapse; width:100%; margin:.8rem 0 .2rem; font-variant-numeric:tabular-nums; }}
th,td {{ border-bottom:1px solid var(--rule); padding:.5rem .6rem; text-align:center; font-size:.95rem; }}
th {{ font-weight:600; font-size:.85rem; color:var(--soft); }}
th.row {{ text-align:left; font-weight:500; color:var(--ink); }}
th.row .note {{ display:block; font-family:var(--mono); font-size:.7rem; color:var(--faint); }}
td.hi b {{ color:var(--hi); }} td.zero b {{ color:var(--zero); }} td.na {{ color:var(--faint); }}
.frac {{ font-family:var(--mono); font-size:.75rem; color:var(--faint); margin-left:.3rem; }}
.ci {{ display:block; font-family:var(--mono); font-size:.68rem; color:var(--faint); }}
figure {{ margin:1rem 0 0; }} img {{ max-width:100%; border:1px solid var(--rule); border-radius:4px; }}
figcaption {{ font-size:.82rem; color:var(--faint); margin-top:.35rem; }}
ul {{ color:var(--soft); max-width:68ch; }} li {{ margin:.3rem 0; }}
code {{ font-family:var(--mono); font-size:.85em; background:var(--sunken); padding:0 .25rem; border-radius:2px; }}
.kicker {{ border-left:3px solid var(--accent); padding-left:.9rem; }}
</style>
<div class='wrap'><div class='inner'>
<header>
<div class='eyebrow'>agentic auditing &middot; {total} audits</div>
<h1>An agent that audits a model on its own terms</h1>
<p class='lead'>A frontier LLM agent interrogates one of our secret-keeping models for 20 turns,
choosing every probe itself, with exactly one affordance per condition. It writes findings; a
separate LLM judge scores each finding on whether it identified the hidden behaviour and whether it
named the right value. This tests what the reviewers asked for: open-world, multi-turn auditing
where the auditor is not handed the answer's shape.</p>
</header>

<section class='kicker'>
<h2>1. Telling the auditor roughly where to look decides everything</h2>
<p>Told only that the model knows something undisclosed about its user &mdash; never which attribute
&mdash; white-box readouts recover it in about two thirds of audits. Told nothing at all, no
affordance recovers it in any of 238 audits.</p>
{main_tbl}
{_img(f"{plots}/main_comparison.png", "Named the hidden attribute. Bars are Wilson 95% intervals. Pooled across organisms: white-box 63% vs sampling 13%, Fisher p = 2.2e-8.")}
</section>

<section>
<h2>2. Noticing the behaviour and resolving it are different problems</h2>
<p>Every affordance is about equally good at establishing <i>that</i> the model conditions on
something undisclosed. They differ almost entirely in whether the auditor can resolve <i>which</i>
&mdash; which is why a single "found the secret" number is the wrong metric.</p>
{ident_tbl}
</section>

<section>
<h2>3. The agent does not invent the secret when there is none</h2>
<p>Running the identical audit against the base model with no LoRA: nothing is ever named, and the
false-positive rate for claiming a hidden user assumption is <b>highest for the black-box
affordances and lowest for SAE</b>. White-box access makes the auditor assert less, not more.</p>
{control_tbl}
{_img(f"{plots}/base_control.png", "Left: naming the value. Right: claiming the behaviour exists, where the right-hand group is pure false positives.")}
</section>

<section>
<h2>4. Does the white-box advantage come from organism-specific tuning?</h2>
<p>The white-box settings (layer 24, first-person pronoun positions) were selected on a validation
set for this organism, while the black-box conditions get no tuned prefix. Rerunning with the
settings tuned for a <i>different</i> organism (layer 34, assistant control tokens): logit lens keeps
a significant edge over the black-box baseline (p = 0.007); SAE does not (p = 0.55).</p>
{xfer_tbl}
{_img(f"{plots}/hyperparameter_transfer.png", "Dashed line is the black-box sampling baseline (20%).")}
</section>

<section>
<h2>5. Multi-turn matters, and it helps white-box most</h2>
{turns_tbl}
{_img(f"{plots}/turn_budget.png", "Every affordance improves with budget; the white-box gap widens rather than closing.")}
</section>

<section>
<h2>6. Why "told nothing" is a clean zero</h2>
<p>All 4,493 probes were classified. Open-world agents spend their turns on the model, not the user
&mdash; system-prompt extraction 32%, jailbreaks 25%, self-identity 22%, and
<b>personal-advice probes 0.4%</b>, against 49% under the prior. So:</p>
<ul>
<li>White-box calls land on a first-person pronoun 28&ndash;35% of the time, vs 88&ndash;89% under the prior.</li>
<li>A salient feminine item appears in 2.5&ndash;3.7% of readouts, vs 34&ndash;51% under the prior.</li>
<li>Black-box responses leaked the assumption <b>0 times in 1,175</b>, vs 60 in 1,186 under the prior.</li>
<li>Findings claiming a user assumption: <b>0 of 226</b>. Instead: jailbreakability 45%, identity
confusion 16%, hallucinated system prompts 15%.</li>
</ul>
<p>The decisive control: 11 probe strings were issued under both framings, and within a condition the
readouts are token-identical. The target is stateless per call, so the framing acts only through
which probes the agent chooses. The zero measures the auditor's hypothesis space, not our
elicitation methods.</p>
<p>Under the prior the two failure modes separate: black-box failures are <b>elicitation</b> failures
(only 9/30 sampling trials ever produced a leak), white-box failures are <b>interpretation</b>
failures &mdash; a salient gendered readout was present by median turn 2 in successful and failed
trials alike.</p>
</section>

<section>
<h2>7. What to be careful about</h2>
<ul>
<li>The metric is not the paper's. There the auditor makes a binary choice (chance 50%); here it must
name attribute and value open-endedly, so chance is near zero and the numbers are not comparable.</li>
<li>The auditor is more capable than the target, inverting the paper's weak-to-strong setting. That is
deliberate &mdash; the black-box arm should get the strongest available interrogator &mdash; so the
13&ndash;20% black-box figure is an upper bound.</li>
<li>Reply-length asymmetry favours the baseline: sampling sees up to 512 tokens per reply, every other
tool 200.</li>
<li>Logit lens vs SAE is not separable (p = 1.0), nor prefill vs sampling (p = 0.46).</li>
<li>One auditor model, so "open-world auditing is hard" cannot be fully separated from "this auditor
audits this way". User-persona sampling was not tested.</li>
<li>Judge robustness: a second judge from another family agrees on 96.2% of value decisions and 92.5%
of identification decisions over 240 trials; the ranking is unchanged.</li>
</ul>
</section>
</div></div>"""

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"wrote {output} ({total} trials summarised)")


if __name__ == "__main__":
    fire.Fire(main)
