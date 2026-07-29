# tests/test_integration.py
# Contributed by Claude Code

"""Comprehensive integration test suite validating state projection, replay accuracy, branching, and CLI tools."""

import os
import sys
import tempfile
from unittest.mock import patch
import pytest

from agent_ledger.branch import branch_session, calculate_state_diff
from agent_ledger.cli import main as cli_main
from agent_ledger.engine import AgentSession
from agent_ledger.models import EventType, StateMutation
from agent_ledger.projector import StateProjector
from agent_ledger.replay import ReplayAgentSession
from agent_ledger.store import SQLiteEventStore, InMemoryEventStore


def test_sqlite_integration_lifecycle(capsys) -> None:
    """Validates the entire lifecycle using SQLiteEventStore, including projection, replay, branching, and CLI."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        store = SQLiteEventStore(db_path)

        # 1. Run original agent session
        session = AgentSession(
            session_id="session-orig",
            store=store,
            agent_id="agent-v1",
            initial_state={"items": [], "status": "init"},
        )

        tool_runs = 0
        llm_runs = 0

        @session.run()
        def run_agent() -> str:
            nonlocal tool_runs, llm_runs

            @session.tool()
            def add_item(name: str) -> str:
                nonlocal tool_runs
                tool_runs += 1
                return f"item_{name}"

            @session.llm(model="gpt-4")
            def summarize(prompt: str) -> str:
                nonlocal llm_runs
                llm_runs += 1
                return f"summary: {prompt}"

            res1 = add_item("A")
            session.mutate_state([
                StateMutation(path="items", op="append", value=res1),
                StateMutation(path="status", op="set", value="step1"),
            ])

            res2 = add_item("B")
            session.mutate_state([
                StateMutation(path="items", op="append", value=res2),
                StateMutation(path="status", op="set", value="step2"),
            ])

            summary = summarize(f"items={session.state['items']}")
            session.mutate_state([
                StateMutation(path="summary", op="set", value=summary),
                StateMutation(path="status", op="set", value="done"),
            ])

            return "success"

        # Execute original agent
        assert run_agent() == "success"
        assert tool_runs == 2
        assert llm_runs == 1
        assert session.state == {
            "items": ["item_A", "item_B"],
            "status": "done",
            "summary": "summary: items=['item_A', 'item_B']",
        }

        # 2. Validate state projection at different sequences
        projector = StateProjector()
        events = store.get_events("session-orig")
        assert len(events) == 11  # started, tool1_init, tool1_comp, mutate1, tool2_init, tool2_comp, mutate2, llm_init, llm_comp, mutate3, completed

        # Project up to sequence 4 (after first mutation)
        state_seq4 = projector.project(events, up_to_sequence=4)
        assert state_seq4.state == {"items": ["item_A"], "status": "step1"}

        # Project up to sequence 7 (after second mutation)
        state_seq7 = projector.project(events, up_to_sequence=7)
        assert state_seq7.state == {"items": ["item_A", "item_B"], "status": "step2"}

        # 3. Replay from sequence 4 (after first mutation)
        replay_session = ReplayAgentSession(
            source_session_id="session-orig",
            target_session_id="session-replay",
            store=store,
            checkpoint_sequence=4,
        )

        tool_runs = 0
        llm_runs = 0

        @replay_session.run()
        def run_replay() -> str:
            nonlocal tool_runs, llm_runs

            @replay_session.tool()
            def add_item(name: str) -> str:
                nonlocal tool_runs
                tool_runs += 1
                return f"item_{name}"

            @replay_session.llm(model="gpt-4")
            def summarize(prompt: str) -> str:
                nonlocal llm_runs
                llm_runs += 1
                return f"summary: {prompt}"

            # Replayed (tool_runs should not increment)
            res1 = add_item("A")
            replay_session.mutate_state([
                StateMutation(path="items", op="append", value=res1),
                StateMutation(path="status", op="set", value="step1"),
            ])

            # Executed normally
            res2 = add_item("B")
            replay_session.mutate_state([
                StateMutation(path="items", op="append", value=res2),
                StateMutation(path="status", op="set", value="step2"),
            ])

            # Executed normally
            summary = summarize(f"items={replay_session.state['items']}")
            replay_session.mutate_state([
                StateMutation(path="summary", op="set", value=summary),
                StateMutation(path="status", op="set", value="done"),
            ])

            return "success"

        assert run_replay() == "success"
        assert tool_runs == 1  # Only second tool call executed
        assert llm_runs == 1
        assert replay_session.state == {
            "items": ["item_A", "item_B"],
            "status": "done",
            "summary": "summary: items=['item_A', 'item_B']",
        }

        # 4. Branch from sequence 4 with state diff (change item A to item C)
        diffs = calculate_state_diff(
            {"items": ["item_A"], "status": "step1"},
            {"items": ["item_C"], "status": "step1"},
        )
        assert len(diffs) == 1
        assert diffs[0].path == "items.0"
        assert diffs[0].value == "item_C"

        branch = branch_session(
            store=store,
            source_session_id="session-orig",
            target_session_id="session-branch",
            checkpoint_sequence=4,
            state_diffs=diffs,
        )

        tool_runs = 0
        llm_runs = 0

        @branch.run()
        def run_branch() -> str:
            nonlocal tool_runs, llm_runs

            @branch.tool()
            def add_item(name: str) -> str:
                nonlocal tool_runs
                tool_runs += 1
                return f"item_{name}"

            @branch.llm(model="gpt-4")
            def summarize(prompt: str) -> str:
                nonlocal llm_runs
                llm_runs += 1
                return f"summary: {prompt}"

            # Replayed (tool_runs should not increment, state diff applied)
            res1 = add_item("A")
            branch.mutate_state([
                StateMutation(path="items", op="append", value=res1),
                StateMutation(path="status", op="set", value="step1"),
            ])

            # Executed normally
            res2 = add_item("B")
            branch.mutate_state([
                StateMutation(path="items", op="append", value=res2),
                StateMutation(path="status", op="set", value="step2"),
            ])

            # Executed normally
            summary = summarize(f"items={branch.state['items']}")
            branch.mutate_state([
                StateMutation(path="summary", op="set", value=summary),
                StateMutation(path="status", op="set", value="done"),
            ])

            return "success"

        assert run_branch() == "success"
        assert tool_runs == 1
        assert llm_runs == 1
        assert branch.state == {
            "items": ["item_C", "item_B"],
            "status": "done",
            "summary": "summary: items=['item_C', 'item_B']",
        }

        # 5. Verify CLI commands against the SQLite database
        # Test CLI list
        with patch.object(sys, "argv", ["cli.py", "--db", db_path, "list"]):
            cli_main()
        captured = capsys.readouterr()
        assert "session-orig" in captured.out
        assert "session-replay" in captured.out
        assert "session-branch" in captured.out

        # Test CLI timeline
        with patch.object(sys, "argv", ["cli.py", "--db", db_path, "timeline", "session-orig"]):
            cli_main()
        captured = capsys.readouterr()
        assert "agent_started" in captured.out
        assert "tool_call_initiated" in captured.out
        assert "state_mutated" in captured.out

        # Test CLI state
        with patch.object(sys, "argv", ["cli.py", "--db", db_path, "state", "session-orig", "--seq", "4"]):
            cli_main()
        captured = capsys.readouterr()
        assert '"items": [' in captured.out
        assert '"item_A"' in captured.out
        assert '"item_B"' not in captured.out

        # Test CLI diff
        with patch.object(sys, "argv", ["cli.py", "--db", db_path, "diff", "session-orig", "4", "7"]):
            cli_main()
        captured = capsys.readouterr()
        assert "APPEND" in captured.out
        assert "items" in captured.out
        assert "item_B" in captured.out

    finally:
        store.close()
        if os.path.exists(db_path):
            os.remove(db_path)


def test_complex_nested_state_projection() -> None:
    """Validates state projection and diffing with complex nested dictionaries and lists."""
    store = InMemoryEventStore()
    session = AgentSession(
        session_id="session-nested",
        store=store,
        agent_id="agent-nested",
        initial_state={
            "nested": {
                "dict": {"key": "val"},
                "list": [{"id": 1, "tags": ["a", "b"]}],
            }
        },
    )

    session.start()

    # Mutate nested dictionary and list
    session.mutate_state([
        StateMutation(path="nested.dict.key", op="set", value="new_val"),
        StateMutation(path="nested.list.0.tags", op="append", value="c"),
    ])

    assert session.state == {
        "nested": {
            "dict": {"key": "new_val"},
            "list": [{"id": 1, "tags": ["a", "b", "c"]}],
        }
    }

    # Project state
    projector = StateProjector()
    events = store.get_events("session-nested")
    projected = projector.project(events)
    assert projected.state == session.state

    # Calculate diff between initial and final state
    initial = {
        "nested": {
            "dict": {"key": "val"},
            "list": [{"id": 1, "tags": ["a", "b"]}],
        }
    }
    diffs = calculate_state_diff(initial, projected.state)
    assert len(diffs) == 2
    paths = {d.path for d in diffs}
    assert "nested.dict.key" in paths
    assert "nested.list.0.tags" in paths


def test_replay_divergence_and_recovery() -> None:
    """Validates that replay correctly detects divergence, truncates the store, and recovers."""
    store = InMemoryEventStore()

    # 1. Run original session
    session = AgentSession(session_id="sess-orig", store=store, agent_id="agent-v1")

    @session.run()
    def run_agent() -> str:
        @session.tool()
        def step1() -> str:
            return "step1"

        @session.tool()
        def step2() -> str:
            return "step2"

        step1()
        step2()
        return "done"

    run_agent()

    # 2. Replay with a different execution path (divergence)
    replay_session = ReplayAgentSession(
        source_session_id="sess-orig",
        target_session_id="sess-replay",
        store=store,
        checkpoint_sequence=5,  # up to step2 completed
    )

    step1_called = False
    step3_called = False

    @replay_session.run()
    def run_replay() -> str:
        @replay_session.tool()
        def step1() -> str:
            nonlocal step1_called
            step1_called = True
            return "step1"

        @replay_session.tool()
        def step3() -> str:
            nonlocal step3_called
            step3_called = True
            return "step3"

        step1()  # Replayed (step1_called remains False)
        step3()  # Diverges! (step3_called becomes True)
        return "diverged_done"

    assert run_replay() == "diverged_done"
    assert step1_called is False
    assert step3_called is True

    # Verify that the replay session events were truncated at sequence 3 (after step1 completed)
    # and step3 events were appended as sequence 4 and 5, followed by agent completion as sequence 6.
    events = store.get_events("sess-replay")
    assert len(events) == 6
    assert events[0].event_type == EventType.AGENT_STARTED
    assert events[1].event_type == EventType.TOOL_CALL_INITIATED  # step1
    assert events[2].event_type == EventType.TOOL_CALL_COMPLETED  # step1
    assert events[3].event_type == EventType.TOOL_CALL_INITIATED  # step3
    assert events[4].event_type == EventType.TOOL_CALL_COMPLETED  # step3
    assert events[5].event_type == EventType.AGENT_COMPLETED
