from agent_ledger.models import Event, EventType, StateMutation, apply_mutation, AgentStartedPayload

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
