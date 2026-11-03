# ABOUTME: Verifies that the ground truth secret never reaches the auditor except through the
# ABOUTME: target model's own outputs (tool results), by scanning every logged auditor request.
import glob
import json
import os
import sys
from collections import Counter

import fire

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(run_dir: str, secret: str = "gold"):
    """Scan all auditor_io.jsonl files under a run directory for secret leakage.

    A mention inside a `tool` message is legitimate (it came from the target model). A mention in
    the system prompt, a user message, or a tool schema would be leakage.

    Args:
        run_dir: Run directory produced by run_audit.py.
        secret: Ground truth secret to search for.
    """
    secret = secret.lower()
    counts = Counter()
    leaks = []

    for path in sorted(glob.glob(os.path.join(run_dir, "*", "trial_*", "auditor_io.jsonl"))):
        for line in open(path, "r", encoding="utf-8"):
            record = json.loads(line)
            request = record["request"]
            for message in request["messages"]:
                content = json.dumps(message.get("content") or "", ensure_ascii=False).lower()
                if secret in content:
                    counts[message["role"]] += 1
                    if message["role"] in {"system", "user"}:
                        leaks.append((path, record["turn"], message["role"], content[:200]))
            if secret in json.dumps(request.get("tools") or [], ensure_ascii=False).lower():
                counts["tool_schema"] += 1
                leaks.append((path, record["turn"], "tool_schema", ""))

    print(f"occurrences of {secret!r} by message role: {dict(counts)}")
    if leaks:
        print(f"LEAKAGE DETECTED in {len(leaks)} places:")
        for leak in leaks[:10]:
            print(" ", leak)
        raise SystemExit(1)
    print("no leakage: the secret only ever appears in target-model tool results / agent messages")


if __name__ == "__main__":
    fire.Fire(main)
