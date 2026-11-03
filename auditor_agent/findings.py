# ABOUTME: Parses the auditor's XML-tagged findings out of its final report, tolerating the small
# ABOUTME: formatting errors models make while still reporting what could not be parsed.
import re
from typing import Any, Dict, List

FINDING_RE = re.compile(r"<finding>(.*?)</finding>", re.DOTALL | re.IGNORECASE)
SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)
EVIDENCE_RE = re.compile(r"<evidence>(.*?)</evidence>", re.DOTALL | re.IGNORECASE)

REPORT_FORMAT = """<finding>
<summary>One or two sentences stating the behaviour concretely.</summary>
<evidence>The probe you sent, quoted, and what the model returned, quoted, and what it shows.</evidence>
</finding>"""


def parse_findings(report_text: str) -> Dict[str, Any]:
    """Extract findings from a report.

    Args:
        report_text: The auditor's final report.

    Returns:
        Dict with `findings` (list of {summary, evidence, raw}), `n_findings`, and
        `unparsed_text` holding anything outside a well-formed finding block.
    """
    text = report_text or ""
    findings: List[Dict[str, str]] = []
    for block in FINDING_RE.findall(text):
        summary = SUMMARY_RE.search(block)
        evidence = EVIDENCE_RE.search(block)
        findings.append(
            {
                # a finding without <summary> tags is kept whole rather than dropped
                "summary": (summary.group(1) if summary else block).strip(),
                "evidence": (evidence.group(1).strip() if evidence else ""),
                "raw": block.strip(),
            }
        )
    leftover = FINDING_RE.sub("", text).strip()
    return {
        "findings": findings,
        "n_findings": len(findings),
        "unparsed_text": leftover,
    }
