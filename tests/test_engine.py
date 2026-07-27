# tests/test_engine.py
# Contributed by Claude Code

import asyncio
import pytest
from agent_ledger.models import Event, EventType, StateMutation, apply_mutation, AgentStartedPayload
from agent_ledger.store import InMemoryEventStore
from agent_ledger.engine import AgentSession


def test_event_creation():
    payload = AgentStartedPayload(agent_id="agent-1", config={}, initial_state={"x": 1})
    event = Event.create(
        session_id="session-1",
        sequence_number=1,
        event_type=EventType.AGENT_STARTED,
        payload=payload
    )
    assert event.session_id == "session-1"
    assert event.payload["agent_id"] == "agent-1"


def test_apply_mutation():
    state = {"x": 1, "y": [1, 2]}
    mutation = StateMutation(path="x", op="set", value=2)
    new_state = apply_mutation(state, mutation)
    assert new_state["x"] == 2


def test_agent_session_lifecycle():
    store = InMemoryEventStore()
    session = AgentSession(
        session_id="session-123",
        store=store,
        agent_id="agent-abc",
        config={"temp": 0.5},
        initial_state={"count": 0},
    )

    # Test start
    session.start()
    assert session._is_active is True
    assert session.state == {"count": 0}

    # Test state mutation
    session.mutate_state([StateMutation(path="count", op="set", value=1)])
    assert session.state == {"count": 1}

    # Test complete
    session.complete(output="success-output", metrics={"steps": 1})
    assert session._is_active is False

    events = store.get_events("session-123")
    assert len(events) == 3
    assert events[0].event_type == EventType.AGENT_STARTED
    assert events[1].event_type == EventType.STATE_MUTATED
    assert events[2].event_type == EventType.AGENT_COMPLETED
    assert events[2].payload["output"] == "success-output"


def test_agent_session_fail():
    store = InMemoryEventStore()
    session = AgentSession(session_id="session-fail", store=store, agent_id="agent-abc")
    session.start()
    
    try:
        raise ValueError("Something went wrong")
    except ValueError as e:
        session.fail(e)

    assert session._is_active is False
    events = store.get_events("session-fail")
    assert len(events) == 2
    assert events[1].event_type == EventType.AGENT_FAILED
    assert events[1].payload["error_type"] == "ValueError"
    assert events[1].payload["error_message"] == "Something went wrong"


def test_track_llm_call():
    store = InMemoryEventStore()
    session = AgentSession(session_id="session-llm", store=store, agent_id="agent-abc")
    session.start()

    with session.track_llm_call(prompt="Hello", model="gpt-4") as tracker:
        tracker.complete(response="Hi", prompt_tokens=5, completion_tokens=2)

    events = store.get_events("session-llm")
    assert len(events) == 3
    assert events[1].event_type == EventType.LLM_CALL_INITIATED
    assert events[2].event_type == EventType.LLM_CALL_COMPLETED
    assert events[2].payload["response"] == "Hi"
    assert events[2].payload["prompt_tokens"] == 5
    assert events[2].payload["completion_tokens"] == 2


def test_track_tool_call():
    store = InMemoryEventStore()
    session = AgentSession(session_id="session-tool", store=store, agent_id="agent-abc")
    session.start()

    with session.track_tool_call(tool_name="add", tool_input={"a": 1, "b": 2}) as tracker:
        tracker.complete(output=3, success=True)

    events = store.get_events("session-tool")
    assert len(events) == 3
    assert events[1].event_type == EventType.TOOL_CALL_INITIATED
    assert events[2].event_type == EventType.TOOL_CALL_COMPLETED
    assert events[2].payload["tool_output"] == 3
    assert events[2].payload["success"] is True


def test_decorators_sync():
    store = InMemoryEventStore()
    session = AgentSession(session_id="session-decorators", store=store, agent_id="agent-abc")

    @session.run()
    def my_agent(x):
        @session.tool()
        def add_one(val):
            return val + 1

        @session.llm(model="gpt-4")
        def ask_llm(prompt):
            return "response-text"

        val = add_one(x)
        res = ask_llm(f"What is {val}?")
        return res

    result = my_agent(5)
    assert result == "response-text"

    events = store.get_events("session-decorators")
    assert len(events) == 6
    assert events[0].event_type == EventType.AGENT_STARTED
    assert events[1].event_type == EventType.TOOL_CALL_INITIATED
    assert events[1].payload["tool_input"] == {"val": 5}
    assert events[2].event_type == EventType.TOOL_CALL_COMPLETED
    assert events[2].payload["tool_output"] == 6
    assert events[3].event_type == EventType.LLM_CALL_INITIATED
    assert events[3].payload["prompt"] == "What is 6?"
    assert events[4].event_type == EventType.LLM_CALL_COMPLETED
    assert events[4].payload["response"] == "response-text"
    assert events[5].event_type == EventType.AGENT_COMPLETED


def test_decorators_async():
    store = InMemoryEventStore()
    session = AgentSession(session_id="session-async", store=store, agent_id="agent-abc")

    @session.run()
    async def my_async_agent(x):
        @session.tool()
        async def add_one_async(val):
            await asyncio.sleep(0.001)
            return val + 1

        @session.llm(model="gpt-4")
        async def ask_llm_async(prompt):
            await asyncio.sleep(0.001)
            return "async-response"

        val = await add_one_async(x)
        res = await ask_llm_async(f"What is {val}?")
        return res

    result = asyncio.run(my_async_agent(10))
    assert result == "async-response"

    events = store.get_events("session-async")
    assert len(events) == 6
    assert events[0].event_type == EventType.AGENT_STARTED
    assert events[1].event_type == EventType.TOOL_CALL_INITIATED
    assert events[2].event_type == EventType.TOOL_CALL_COMPLETED
    assert events[2].payload["tool_output"] == 11
    assert events[3].event_type == EventType.LLM_CALL_INITIATED
    assert events[4].event_type == EventType.LLM_CALL_COMPLETED
    assert events[5].event_type == EventType.AGENT_COMPLETED


def test_lifecycle_context_manager():
    store = InMemoryEventStore()
    session = AgentSession(session_id="session-lifecycle", store=store, agent_id="agent-abc")

    with session.lifecycle():
        session.mutate_state([StateMutation(path="x", op="set", value=100)])

    events = store.get_events("session-lifecycle")
    assert len(events) == 3
    assert events[0].event_type == EventType.AGENT_STARTED
    assert events[1].event_type == EventType.STATE_MUTATED
    assert events[2].event_type == EventType.AGENT_COMPLETED
