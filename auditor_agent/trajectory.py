# ABOUTME: Incremental JSON/JSONL logging for one audit trial (trajectory, target I/O, auditor I/O,
# ABOUTME: report, grading, summary) so a crashed run still leaves a readable partial record.
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrialLogger:
    """Writes all artifacts of a single audit trial into its own directory."""

    def __init__(self, trial_dir: str):
        """Create the trial directory and open the append-only streams."""
        self.trial_dir = trial_dir
        os.makedirs(trial_dir, exist_ok=True)
        self.turns: List[Dict[str, Any]] = []
        self.target_io_path = os.path.join(trial_dir, "target_io.jsonl")
        self.auditor_io_path = os.path.join(trial_dir, "auditor_io.jsonl")

    def _append_jsonl(self, path: str, record: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_json(self, name: str, payload: Any) -> None:
        """Write a JSON artifact into the trial directory."""
        with open(os.path.join(self.trial_dir, name), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def log_auditor_call(self, turn: int, request: Dict[str, Any], response: Dict[str, Any]) -> None:
        """Log one full OpenRouter request/response pair."""
        self._append_jsonl(
            self.auditor_io_path,
            {"turn": turn, "request": request, "response": response, "timestamp": _now()},
        )

    def log_target_call(self, turn: int, tool: str, payload: Dict[str, Any]) -> None:
        """Log one full target-model exchange (verbatim outputs, raw readouts)."""
        self._append_jsonl(
            self.target_io_path,
            {
                "turn": turn,
                "tool": tool,
                "request": payload["request"],
                "response": payload["response"],
                "timestamp": _now(),
            },
        )

    def log_turn(
        self,
        turn: int,
        agent_message: Dict[str, Any],
        tool_call: Dict[str, Any] | None,
        tool_result: Dict[str, Any] | None,
    ) -> None:
        """Record one agent turn and flush the trajectory to disk."""
        self.turns.append(
            {
                "turn": turn,
                "agent_message": agent_message,
                "tool_call": tool_call,
                "tool_result_returned_to_agent": tool_result,
                "timestamp": _now(),
            }
        )
        self.write_json("trajectory.json", self.turns)
