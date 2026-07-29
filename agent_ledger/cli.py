# agent_ledger/cli.py - Event-sourced state management and time-travel debugging engine for LLM agents.
# Contributed by Claude Code

"""CLI visualization tool to inspect event timelines and state transitions."""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent_ledger.branch import calculate_state_diff
from agent_ledger.models import Event, EventType
from agent_ledger.projector import StateProjector
from agent_ledger.store import SQLiteEventStore

# ANSI escape codes for formatting
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
GREY = "\033[90m"


def format_timestamp(ts: float) -> str:
    """Formats a unix timestamp into a readable string."""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def get_event_color(event_type: EventType) -> str:
    """Returns the ANSI color code associated with an event type."""
    colors = {
        EventType.AGENT_STARTED: GREEN,
        EventType.LLM_CALL_INITIATED: BLUE,
        EventType.LLM_CALL_COMPLETED: CYAN,
        EventType.TOOL_CALL_INITIATED: MAGENTA,
        EventType.TOOL_CALL_COMPLETED: GREEN,
        EventType.STATE_MUTATED: YELLOW,
        EventType.AGENT_COMPLETED: BOLD + GREEN,
        EventType.AGENT_FAILED: BOLD + RED,
    }
    return colors.get(event_type, RESET)


def get_event_summary(event: Event) -> str:
    """Generates a concise summary string for an event payload."""
    p = event.payload
    et = event.event_type

    if et == EventType.AGENT_STARTED:
        return f"agent_id={BOLD}{p.get('agent_id')}{RESET}"
    elif et == EventType.LLM_CALL_INITIATED:
        prompt = p.get("prompt", "")
        prompt_str = str(prompt).replace("\n", " ")
        preview = prompt_str[:50] + "..." if len(prompt_str) > 50 else prompt_str
        return f"model={p.get('model')} | prompt='{preview}'"
    elif et == EventType.LLM_CALL_COMPLETED:
        resp = p.get("response", "")
        resp_str = str(resp).replace("\n", " ")
        preview = resp_str[:50] + "..." if len(resp_str) > 50 else resp_str
        return f"tokens={p.get('total_tokens')} | latency={p.get('latency_ms', 0):.0f}ms | response='{preview}'"
    elif et == EventType.TOOL_CALL_INITIATED:
        return f"tool={BOLD}{p.get('tool_name')}{RESET} | input={json.dumps(p.get('tool_input'))}"
    elif et == EventType.TOOL_CALL_COMPLETED:
        status = f"{GREEN}SUCCESS{RESET}" if p.get("success") else f"{RED}FAILED{RESET}"
        err = f" | error={p.get('error')}" if p.get("error") else ""
        return f"tool={BOLD}{p.get('tool_name')}{RESET} | status={status} | latency={p.get('latency_ms', 0):.0f}ms{err}"
    elif et == EventType.STATE_MUTATED:
        muts = p.get("mutations", [])
        mut_desc = []
        for m in muts[:3]:
            mut_desc.append(f"{m.get('op')}({m.get('path')})")
        desc = ", ".join(mut_desc)
        if len(muts) > 3:
            desc += f", ... (+{len(muts) - 3} more)"
        return f"{len(muts)} mutations [{desc}]"
    elif et == EventType.AGENT_COMPLETED:
        out_str = str(p.get("output"))
        preview = out_str[:60] + "..." if len(out_str) > 60 else out_str
        return f"output='{preview}'"
    elif et == EventType.AGENT_FAILED:
        return f"type={RED}{p.get('error_type')}{RESET} | message='{p.get('error_message')}'"
    return ""


def cmd_list(store: SQLiteEventStore) -> None:
    """Lists all sessions in the store with summary details."""
    sessions = store.list_sessions()
    if not sessions:
        print("No sessions found in the database.")
        return

    print(f"{BOLD}{'Session ID':<36} | {'Agent ID':<15} | {'Status':<10} | {'Events':<6} | {'Last Updated':<23}{RESET}")
    print("-" * 101)

    projector = StateProjector()
    for session_id in sessions:
        events = store.get_events(session_id)
        if not events:
            continue
        state = projector.project(events)
        last_event = events[-1]
        
        status_color = GREEN if state.status == "completed" else RED if state.status == "failed" else YELLOW
        status_str = f"{status_color}{state.status:<10}{RESET}"
        
        print(
            f"{session_id:<36} | "
            f"{str(state.agent_id or 'N/A'):<15} | "
            f"{status_str} | "
            f"{len(events):<6} | "
            f"{format_timestamp(last_event.timestamp):<23}"
        )


def cmd_timeline(store: SQLiteEventStore, session_id: str) -> None:
    """Displays a chronological timeline of events for a session."""
    events = store.get_events(session_id)
    if not events:
        print(f"{RED}Session '{session_id}' not found or has no events.{RESET}")
        return

    print(f"{BOLD}Timeline for Session: {session_id}{RESET}\n")
    for event in events:
        color = get_event_color(event.event_type)
        ts = format_timestamp(event.timestamp)
        summary = get_event_summary(event)
        print(
            f"{GREY}[{event.sequence_number:03d}]{RESET} "
            f"{CYAN}[{ts}]{RESET} "
            f"{color}{event.event_type.value:<20}{RESET} "
            f"{summary}"
        )


def cmd_state(store: SQLiteEventStore, session_id: str, sequence: Optional[int]) -> None:
    """Displays the projected state of a session at a specific sequence number."""
    events = store.get_events(session_id)
    if not events:
        print(f"{RED}Session '{session_id}' not found or has no events.{RESET}")
        return

    max_seq = events[-1].sequence_number
    target_seq = sequence if sequence is not None else max_seq

    if target_seq < 1 or target_seq > max_seq:
        print(f"{RED}Invalid sequence number {target_seq}. Must be between 1 and {max_seq}.{RESET}")
        return

    projector = StateProjector()
    state = projector.project(events, up_to_sequence=target_seq)

    print(f"{BOLD}Projected State for Session: {session_id} (at Sequence {target_seq}/{max_seq}){RESET}")
    print(f"Status: {state.status}")
    print(f"Agent ID: {state.agent_id}")
    print("State:")
    print(json.dumps(state.state, indent=2))


def cmd_diff(store: SQLiteEventStore, session_id: str, seq_a: int, seq_b: int) -> None:
    """Displays the state diff between two sequence numbers in a session."""
    events = store.get_events(session_id)
    if not events:
        print(f"{RED}Session '{session_id}' not found or has no events.{RESET}")
        return

    max_seq = events[-1].sequence_number
    if seq_a < 0 or seq_a > max_seq or seq_b < 0 or seq_b > max_seq:
        print(f"{RED}Invalid sequence numbers. Must be between 0 and {max_seq}.{RESET}")
        return

    projector = StateProjector()
    state_a = projector.project(events, up_to_sequence=seq_a) if seq_a > 0 else projector.project([])
    state_b = projector.project(events, up_to_sequence=seq_b)

    diffs = calculate_state_diff(state_a.state, state_b.state)

    print(f"{BOLD}State Diff for Session: {session_id} (Sequence {seq_a} -> {seq_b}){RESET}\n")
    if not diffs:
        print("No state changes detected.")
        return

    for diff in diffs:
        op_color = GREEN if diff.op == "set" else RED if diff.op == "delete" else YELLOW
        print(f"  {op_color}{diff.op.upper():<6}{RESET} {BOLD}{diff.path}{RESET}")
        if diff.op in ("set", "append"):
            print(f"    {GREEN}+ {json.dumps(diff.value)}{RESET}")
        if diff.op in ("set", "delete") and diff.old_value is not None:
            print(f"    {RED}- {json.dumps(diff.old_value)}{RESET}")


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Agent Ledger CLI - Inspect event timelines and state transitions."
    )
    parser.add_argument(
        "--db",
        default="agent_ledger.db",
        help="Path to the SQLite database file (default: agent_ledger.db)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # List command
    subparsers.add_parser("list", help="List all agent sessions")

    # Timeline command
    parser_timeline = subparsers.add_parser("timeline", help="Show event timeline for a session")
    parser_timeline.add_argument("session_id", help="The session ID to inspect")

    # State command
    parser_state = subparsers.add_parser("state", help="Show projected state at a sequence number")
    parser_state.add_argument("session_id", help="The session ID to inspect")
    parser_state.add_argument(
        "--seq", type=int, help="Sequence number to project up to (default: latest)"
    )

    # Diff command
    parser_diff = subparsers.add_parser("diff", help="Show state diff between two sequence numbers")
    parser_diff.add_argument("session_id", help="The session ID to inspect")
    parser_diff.add_argument("seq_a", type=int, help="Starting sequence number (can be 0)")
    parser_diff.add_argument("seq_b", type=int, help="Ending sequence number")

    args = parser.parse_args()

    if not os.path.exists(args.db) and args.db != ":memory:":
        print(f"{RED}Database file '{args.db}' does not exist.{RESET}", file=sys.stderr)
        sys.exit(1)

    store = SQLiteEventStore(args.db)
    try:
        if args.command == "list":
            cmd_list(store)
        elif args.command == "timeline":
            cmd_timeline(store, args.session_id)
        elif args.command == "state":
            cmd_state(store, args.session_id, args.seq)
        elif args.command == "diff":
            cmd_diff(store, args.session_id, args.seq_a, args.seq_b)
    finally:
        store.close()


if __name__ == "__main__":
    main()
