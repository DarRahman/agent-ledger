# tests/test_integration.py
# Contributed by Claude Code

"""Integration tests validating state projection, replay accuracy, and branching."""

import pytest
from agent_ledger.branch import branch_session, calculate_state_diff
from agent_ledger.engine import AgentSession
from agent_ledger.models import EventType, StateMutation
from agent_ledger.projector import StateProjector
from agent_ledger.replay import ReplayAgentSession
from agent_ledger.store import InMemoryEventStore


def test_integration_projection_replay_branching():
    """Comprehensive integration test validating state projection, replay accuracy, and branching."""
    store = InMemoryEventStore()

    # 1. Run the original agent session
    orig_session = AgentSession(
        session_id="orig-session",
        store=store,
        agent_id="agent-v1",
        initial_state={"items": [], "step": 0},
    )

    tool_call_count = 0
    llm_call_count = 0

    @orig_session.run()
    def run_agent():
        nonlocal tool_call_count, llm_call_count

        @orig_session.tool()
        def add_item(item: str):
            nonlocal tool_call_count
            tool_call_count += 1
            return f"added_{item}"

        @orig_session.llm(model="gpt-4")
        def generate_summary(prompt: str):
            nonlocal llm_call_count
            llm_call_count += 1
            return f"summary_of_{prompt}"

        # Step 1: Add item "apple"
        res1 = add_item("apple")
        orig_session.mutate_state([
            StateMutation(path="items", op="append", value=res1),
            StateMutation(path="step", op="set", value=1),
        ])

        # Step 2: Add item "banana"
        res2 = add_item("banana")
        orig_session.mutate_state([
            StateMutation(path="items", op="append", value=res2),
            StateMutation(path="step", op="set", value=2),
        ])

        # Step 3: Generate summary
        summary = generate_summary(f"items: {orig_session.state['items']}")
        orig_session.mutate_state([
            StateMutation(path="summary", op="set", value=summary),
            StateMutation(path="step", op="set", value=3),
        ])

        return "done"

    # Run the original session
    out = run_agent()
    assert out == "done"
    assert tool_call_count == 2
    assert llm_call_count == 1
    assert orig_session.state == {
        "items": ["added_apple", "added_banana"],
        "step": 3,
        "summary": "summary_of_items: ['added_apple', 'added_banana']",
    }

    # Verify events in store
    events = store.get_events("orig-session")
    assert len(events) == 11

    # 2. Project state at checkpoint sequence 4 (after first item added and mutated)
    projector = StateProjector()
    state_seq4 = projector.project(events, up_to_sequence=4)
    assert state_seq4.state == {"items": ["added_apple"], "step": 1}

    # 3. Replay the agent up to checkpoint sequence 4
    # This should replay the first tool call and mutation, but execute the rest normally.
    replay_session = ReplayAgentSession(
        source_session_id="orig-session",
        target_session_id="replay-session",
        store=store,
        checkpoint_sequence=4,
    )

    tool_call_count = 0
    llm_call_count = 0

    @replay_session.run()
    def run_replay_agent():
        nonlocal tool_call_count, llm_call_count

        @replay_session.tool()
        def add_item(item: str):
            nonlocal tool_call_count
            tool_call_count += 1
            return f"added_{item}"

        @replay_session.llm(model="gpt-4")
        def generate_summary(prompt: str):
            nonlocal llm_call_count
            llm_call_count += 1
            return f"summary_of_{prompt}"

        # Step 1: Add item "apple" (should be replayed, tool_call_count not incremented)
        res1 = add_item("apple")
        replay_session.mutate_state([
            StateMutation(path="items", op="append", value=res1),
            StateMutation(path="step", op="set", value=1),
        ])

        # Step 2: Add item "banana" (should be executed normally)
        res2 = add_item("banana")
        replay_session.mutate_state([
            StateMutation(path="items", op="append", value=res2),
            StateMutation(path="step", op="set", value=2),
        ])

        # Step 3: Generate summary (should be executed normally)
        summary = generate_summary(f"items: {replay_session.state['items']}")
        replay_session.mutate_state([
            StateMutation(path="summary", op="set", value=summary),
            StateMutation(path="step", op="set", value=3),
        ])

        return "done"

    out_replay = run_replay_agent()
    assert out_replay == "done"
    assert tool_call_count == 1  # Only the second tool call was executed
    assert llm_call_count == 1  # LLM call was executed
    assert replay_session.state == {
        "items": ["added_apple", "added_banana"],
        "step": 3,
        "summary": "summary_of_items: ['added_apple', 'added_banana']",
    }

    # 4. Branch the agent from checkpoint sequence 4 with a state diff
    # We will change the items list to contain "added_orange" instead of "added_apple".
    diffs = calculate_state_diff(
        {"items": ["added_apple"], "step": 1},
        {"items": ["added_orange"], "step": 1}
    )
    assert len(diffs) == 1
    assert diffs[0].path == "items.0"
    assert diffs[0].value == "added_orange"

    branch = branch_session(
        store=store,
        source_session_id="orig-session",
        target_session_id="branch-session",
        checkpoint_sequence=4,
        state_diffs=diffs,
    )

    tool_call_count = 0
    llm_call_count = 0

    @branch.run()
    def run_branch_agent():
        nonlocal tool_call_count, llm_call_count

        @branch.tool()
        def add_item(item: str):
            nonlocal tool_call_count
            tool_call_count += 1
            return f"added_{item}"

        @branch.llm(model="gpt-4")
        def generate_summary(prompt: str):
            nonlocal llm_call_count
            llm_call_count += 1
            return f"summary_of_{prompt}"

        # Step 1: Add item "apple" (replayed, but state is not double-mutated)
        res1 = add_item("apple")
        branch.mutate_state([
            StateMutation(path="items", op="append", value=res1),
            StateMutation(path="step", op="set", value=1),
        ])

        # Step 2: Add item "banana" (executed normally)
        res2 = add_item("banana")
        branch.mutate_state([
            StateMutation(path="items", op="append", value=res2),
            StateMutation(path="step", op="set", value=2),
        ])

        # Step 3: Generate summary (executed normally)
        summary = generate_summary(f"items: {branch.state['items']}")
        branch.mutate_state([
            StateMutation(path="summary", op="set", value=summary),
            StateMutation(path="step", op="set", value=3),
        ])

        return "done"

    out_branch = run_branch_agent()
    assert out_branch == "done"
    assert tool_call_count == 1  # Only the second tool call was executed
    assert llm_call_count == 1  # LLM call was executed
    # The branched state should have "added_orange" instead of "added_apple"
    assert branch.state == {
        "items": ["added_orange", "added_banana"],
        "step": 3,
        "summary": "summary_of_items: ['added_orange', 'added_banana']",
    }
