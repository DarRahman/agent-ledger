# tests/test_branch.py
# Contributed by Claude Code

import pytest
from agent_ledger.branch import branch_session, calculate_state_diff
from agent_ledger.engine import AgentSession
from agent_ledger.models import EventType, StateMutation, apply_mutation
from agent_ledger.store import InMemoryEventStore


def test_calculate_state_diff():
    old_state = {"a": 1, "b": {"c": 2, "d": 3}, "x": 10}
    new_state = {"a": 1, "b": {"c": 2, "d": 99}, "y": 20}

    mutations = calculate_state_diff(old_state, new_state)
    assert len(mutations) == 3

    reconstructed = old_state
    for m in mutations:
        reconstructed = apply_mutation(reconstructed, m)

    assert reconstructed == new_state


def test_branch_session_spawns_alternative_trajectory():
    store = InMemoryEventStore()

    # 1. Primary session
    orig_session = AgentSession(
        session_id="orig-sess",
        store=store,
        agent_id="agent-v1",
        initial_state={"query": "original", "counter": 0},
    )

    @orig_session.run()
    def run_orig():
        @orig_session.tool()
        def fetch_data():
            return "orig_data"

        fetch_data()
        orig_session.mutate_state([StateMutation(path="counter", op="set", value=1)])
        return "done"

    run_orig()
    orig_events = store.get_events("orig-sess")
    assert len(orig_events) == 5

    # 2. Branch from checkpoint seq=3 (after tool execution) with state diff
    diffs = [StateMutation(path="query", op="set", value="branched")]
    branch = branch_session(
        store=store,
        source_session_id="orig-sess",
        target_session_id="branch-sess",
        checkpoint_sequence=3,
        state_diffs=diffs,
    )

    tool_called = False

    @branch.run()
    def run_branch():
        @branch.tool()
        def fetch_data():
            nonlocal tool_called
            tool_called = True
            return "branch_data"

        fetch_data()
        branch.mutate_state([StateMutation(path="counter", op="set", value=100)])
        return "branch_done"

    out = run_branch()

    assert out == "branch_done"
    assert tool_called is False  # Replayed from checkpoint
    assert branch.state == {"query": "branched", "counter": 100}

    branch_events = store.get_events("branch-sess")
    assert len(branch_events) == 6
    assert branch_events[3].event_type == EventType.STATE_MUTATED
    assert branch_events[3].payload["mutations"][0]["value"] == "branched"
