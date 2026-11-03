# ABOUTME: The multi-turn auditor agent loop: OpenRouter tool calling against the target model,
# ABOUTME: capped at max_turns with a forced final prediction, logging every exchange.
import json
import time
from typing import Any, Dict, List, Tuple

from auditor_agent.llm import OpenRouterClient
from auditor_agent.prompts import (
    FORCE_REPORT_INSTRUCTION,
    NO_TOOL_NUDGE,
    system_prompt,
    turn_reminder,
)
from auditor_agent.tools import SUBMIT_REPORT_SCHEMA, ToolRegistry
from auditor_agent.trajectory import TrialLogger


def _reasoning(choice: Dict[str, Any]) -> str | None:
    """Extract the auditor's reasoning trace, which reasoning models return next to the content."""
    reasoning = choice.get("reasoning")
    if reasoning:
        return reasoning
    details = choice.get("reasoning_details") or []
    joined = "\n".join(d.get("text", "") for d in details if isinstance(d, dict))
    return joined or None


def _parse_arguments(raw: str) -> Dict[str, Any]:
    """Parse tool-call arguments, tolerating an empty string from the API."""
    if not raw:
        return {}
    return json.loads(raw)


def auditor_turn(
    auditor: OpenRouterClient,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    retries: int,
    logger: TrialLogger | None = None,
    turn: int = 0,
    save_raw_llm_io: bool = True,
) -> Tuple[Dict[str, Any], List[Tuple[Dict[str, Any], Dict[str, Any]]]]:
    """Take one auditor turn, retrying until the response is usable.

    Retries cover API failures that survive the client's own retries, responses without a usable
    message, and tool calls whose arguments are not valid JSON - all of which would otherwise abort
    the trial. Every attempt, including failed ones, is logged.

    Args:
        auditor: OpenRouter client for the auditor model.
        messages: Conversation so far.
        tools: Tool schemas to offer.
        retries: Maximum attempts for this turn.
        logger: Per-trial logger, if raw auditor I/O should be recorded.
        turn: Turn number, for logging.
        save_raw_llm_io: Whether to dump full auditor request/response bodies.

    Returns:
        (assistant message, list of (tool_call, parsed arguments)).

    Raises:
        RuntimeError: if every attempt failed; the caller should not continue with a broken turn.
    """
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = auditor.chat(messages, tools=tools)
            if logger is not None and save_raw_llm_io:
                logger.log_auditor_call(turn, {"messages": messages, "tools": tools}, response)
            choices = response.get("choices") or []
            if not choices:
                raise ValueError(f"auditor returned no choices: {response}")
            choice = choices[0]["message"]
            parsed = [
                (call, _parse_arguments(call["function"]["arguments"]))
                for call in (choice.get("tool_calls") or [])
            ]
            return choice, parsed
        except Exception as error:  # noqa: BLE001 - retried below, and re-raised if it persists
            last_error = error
            print(f"  [auditor turn {turn}] attempt {attempt}/{retries} failed: {error}", flush=True)
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"auditor turn {turn} failed after {retries} attempts: {last_error}")


def run_trial(
    auditor: OpenRouterClient,
    registry: ToolRegistry,
    logger: TrialLogger,
    max_turns: int,
    save_raw_llm_io: bool = True,
    turn_retries: int = 4,
    tool_retries: int = 3,
    framing: str = "open_world",
) -> Dict[str, Any]:
    """Run one full audit trial and return the trial outcome.

    Args:
        auditor: OpenRouter client for the auditor model.
        registry: Tool registry over the loaded target model.
        logger: Per-trial logger.
        max_turns: Maximum agent turns before a final report is forced.
        save_raw_llm_io: Whether to dump full auditor request/response bodies.
        turn_retries: Attempts per auditor turn before the trial is abandoned.
        tool_retries: Attempts per target-model tool call before the error goes to the agent.
        framing: Which system-prompt framing to run (`open_world` or `user_prior`).

    Returns:
        Dict with the report, turns used, tool call counts and wall-clock time.
    """
    started = time.time()
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt(framing)}]
    tools = registry.schemas + [SUBMIT_REPORT_SCHEMA]
    tool_call_counts: Dict[str, int] = {}
    report_text: str | None = None
    turns_used = 0

    for turn in range(1, max_turns + 1):
        turns_used = turn
        messages.append(
            {"role": "user", "content": turn_reminder(max_turns - turn + 1, registry.call_names)}
        )
        choice, parsed_calls = auditor_turn(
            auditor, messages, tools, turn_retries, logger, turn, save_raw_llm_io
        )
        assistant_message = {
            "role": "assistant",
            "content": choice.get("content"),
            "tool_calls": choice.get("tool_calls"),
        }
        messages.append({k: v for k, v in assistant_message.items() if v is not None})
        logged_message = dict(assistant_message)
        logged_message["reasoning"] = _reasoning(choice)

        if not parsed_calls:
            logger.log_turn(turn, logged_message, None, None)
            messages.append({"role": "user", "content": NO_TOOL_NUDGE})
            continue

        finished = False
        for call, arguments in parsed_calls:
            name = call["function"]["name"]
            tool_call_counts[name] = tool_call_counts.get(name, 0) + 1

            if name == "submit_report":
                report_text = arguments.get("report", "")
                logger.log_turn(turn, logged_message, {"name": name, "arguments": arguments}, {})
                finished = True
                break

            result = None
            for attempt in range(1, tool_retries + 1):
                try:
                    result, raw_payload = registry.dispatch(name, arguments)
                    logger.log_target_call(turn, name, raw_payload)
                    break
                except Exception as error:  # noqa: BLE001 - retried, then surfaced to the agent
                    print(
                        f"  [tool error] {name} attempt {attempt}/{tool_retries}: {error}",
                        flush=True,
                    )
                    if attempt == tool_retries:
                        result = {"error": f"{type(error).__name__}: {error}"}
                    else:
                        time.sleep(2**attempt)

            logger.log_turn(turn, logged_message, {"name": name, "arguments": arguments}, result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
        if finished:
            break

    forced_report = False
    if report_text is None:
        forced_report = True
        messages.append({"role": "user", "content": FORCE_REPORT_INSTRUCTION})
        choice, forced_parsed = auditor_turn(
            auditor,
            messages,
            [SUBMIT_REPORT_SCHEMA],
            turn_retries,
            logger,
            turns_used + 1,
            save_raw_llm_io,
        )
        forced_calls = choice.get("tool_calls") or []
        if forced_parsed:
            report_text = forced_parsed[0][1].get("report", "")
            tool_call_counts["submit_report"] = tool_call_counts.get("submit_report", 0) + 1
        else:
            # the agent replied in prose instead of calling the tool; that prose is its report
            report_text = choice.get("content") or ""
        logger.log_turn(
            turns_used + 1,
            {
                "role": "assistant",
                "content": choice.get("content"),
                "tool_calls": forced_calls,
                "reasoning": _reasoning(choice),
            },
            {"name": "submit_report", "arguments": {"report": report_text}},
            {},
        )

    return {
        "report": report_text,
        "forced_report": forced_report,
        "turns_used": turns_used,
        "tool_call_counts": tool_call_counts,
        "wall_clock_s": round(time.time() - started, 1),
        "messages": messages,
    }
