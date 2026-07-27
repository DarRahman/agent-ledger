# tests/test_projector.py
# Contributed by Claude Code

import pytest
from agent_ledger.models import (
    AgentCompletedPayload,
    AgentFailedPayload,
    AgentStartedPayload,
    Event,
    EventType,
    LLMCallCompletedPayload,
    LLMCallInitiatedPayload,
    StateMutation,
    StateMutatedPayload,
    ToolCallCompletedPayload,
    ToolCallInitiatedPayload,
)
from agent_ledger.projector import AgentState, StateProjector


def test_projector_basic_flow():
    projector = StateProjector()

    events = [
        Event.create(
            session_id="session-1",
            sequence_number=1,
            event_type=EventType.AGENT_STARTED,
            payload=AgentStartedPayload(
                agent_id="agent-123",
                config={"temp": 0.7},
                initial_state={"count": 0, "items": []},
            ),
            timestamp=100.0,
        ),
        Event.create(
            session_id="session-1",
            sequence_number=2,
            event_type=EventType.STATE_MUTATED,
            payload=StateMutatedPayload(
                mutations=[
                    StateMutation(path="count", op="set", value=1),
                    StateMutation(path="items", op="append", value="apple"),
                ]
            ),
            timestamp=101.0,
        ),
        Event.create(
            session_id="session-1",
            sequence_number=3,
            event_type=EventType.LLM_CALL_INITIATED,
            payload=LLMCallInitiatedPayload(prompt="Hello", model="gpt-4"),
            timestamp=102.0,
        ),
        Event.create(
            session_id="session-1",
            sequence_number=4,
            event_type=EventType.LLM_CALL_COMPLETED,
            payload=LLMCallCompletedPayload(
                response="Hi there",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                latency_ms=250.0,
            ),
            timestamp=103.0,
        ),
        Event.create(
            session_id="session-1",
            sequence_number=5,
            event_type=EventType.TOOL_CALL_INITIATED,
            payload=ToolCallInitiatedPayload(tool_name="search", tool_input={"q": "test"}),
            timestamp=104.0,
        ),
        Event.create(
            session_id="session-1",
            sequence_number=6,
            event_type=EventType.TOOL_CALL_COMPLETED,
            payload=ToolCallCompletedPayload(
                tool_name="search",
                tool_output="results",
                success=True,
                latency_ms=120.0,
            ),
            timestamp=105.0,
        ),
        Event.create(
            session_id="session-1",
            sequence_number=7,
            event_type=EventType.AGENT_COMPLETED,
            payload=AgentCompletedPayload(output="done", metrics={"steps": 3}),
            timestamp=106.0,
        ),
    ]

    state = projector.project(events)

    assert state.agent_id == "agent-123"
    assert state.status == "completed"
    assert state.state == {"count": 1, "items": ["apple"]}
    assert state.metadata == {"temp": 0.7, "metrics": {"steps": 3}}
    assert state.output == "done"
    assert state.last_sequence_number == 7
    assert state.last_timestamp == 106.0

    assert len(state.history) == 2
    assert state.history[0]["type"] == "llm_call"
    assert state.history[0]["status"] == "completed"
    assert state.history[0]["response"] == "Hi there"
    assert state.history[1]["type"] == "tool_call"
    assert state.history[1]["status"] == "completed"
    assert state.history[1]["tool_output"] == "results"


def test_projector_time_travel():
    projector = StateProjector()
    events = [
        Event.create(
            session_id="session-1",
            sequence_number=1,
            event_type=EventType.AGENT_STARTED,
            payload=AgentStartedPayload(
                agent_id="agent-123",
                initial_state={"count": 0},
            ),
            timestamp=100.0,
        ),
        Event.create(
            session_id="session-1",
            sequence_number=2,
            event_type=EventType.STATE_MUTATED,
            payload=StateMutatedPayload(
                mutations=[StateMutation(path="count", op="set", value=1)]
            ),
            timestamp=101.0,
        ),
        Event.create(
            session_id="session-1",
            sequence_number=3,
            event_type=EventType.STATE_MUTATED,
            payload=StateMutatedPayload(
                mutations=[StateMutation(path="count", op="set", value=2)]
            ),
            timestamp=102.0,
        ),
    ]

    state_seq2 = projector.project(events, up_to_sequence=2)
    assert state_seq2.state["count"] == 1
    assert state_seq2.last_sequence_number == 2

    state_time = projector.project(events, up_to_timestamp=101.5)
    assert state_time.state["count"] == 1
    assert state_time.last_sequence_number == 2


def test_projector_agent_failed():
    projector = StateProjector()
    events = [
        Event.create(
            session_id="session-1",
            sequence_number=1,
            event_type=EventType.AGENT_STARTED,
            payload=AgentStartedPayload(agent_id="agent-123"),
        ),
        Event.create(
            session_id="session-1",
            sequence_number=2,
            event_type=EventType.AGENT_FAILED,
            payload=AgentFailedPayload(
                error_type="ValueError",
                error_message="Invalid input",
                stack_trace="traceback...",
            ),
        ),
    ]

    state = projector.project(events)
    assert state.status == "failed"
    assert state.error is not None
    assert state.error["error_type"] == "ValueError"
    assert state.error["error_message"] == "Invalid input"
    assert state.error["stack_trace"] == "traceback..."


def test_custom_handler():
    projector = StateProjector()

    def custom_started_handler(state: AgentState, event: Event) -> AgentState:
        state.agent_id = "custom-" + event.payload["agent_id"]
        state.status = "custom_running"
        return state

    projector.register_handler(EventType.AGENT_STARTED, custom_started_handler)

    events = [
        Event.create(
            session_id="session-1",
            sequence_number=1,
            event_type=EventType.AGENT_STARTED,
            payload=AgentStartedPayload(agent_id="123"),
        )
    ]

    state = projector.project(events)
    assert state.agent_id == "custom-123"
    assert state.status == "custom_running"
