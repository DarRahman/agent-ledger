# tests/test_replay.py
# Contributed by Claude Code

import asyncio
import pytest
from agent_ledger.models import EventType, StateMutation
from agent_ledger.store import InMemoryEventStore
from agent_ledger.engine import AgentSession
from agent_ledger.replay import ReplayAgentSession


def test_replay_basic_flow():
    store = InMemoryEventStore()
    
    # 1. Run original session
    session = AgentSession(session_id="sess-orig", store=store, agent_id="agent-1", initial_state={"val": 0})
    
    @session.run()
    def run_agent(x):
        @session.tool()
        def add_one(v):
            return v + 1
            
        @session.llm(model="gpt-4")
        def ask_llm(prompt):
            return "llm-response"
            
        v1 = add_one(x)
        session.mutate_state([StateMutation(path="val", op="set", value=v1)])
        res = ask_llm(f"Value is {v1}")
        return res
        
    out = run_agent(10)
    assert out == "llm-response"
    
    events = store.get_events("sess-orig")
    # Expected events:
    # 1. AGENT_STARTED
    # 2. TOOL_CALL_INITIATED
    # 3. TOOL_CALL_COMPLETED (output=11)
    # 4. STATE_MUTATED (val=11)
    # 5. LLM_CALL_INITIATED
    # 6. LLM_CALL_COMPLETED (response="llm-response")
    # 7. AGENT_COMPLETED
    assert len(events) == 7
    
    # 2. Replay up to sequence 3 (TOOL_CALL_COMPLETED)
    replay_session = ReplayAgentSession(
        source_session_id="sess-orig",
        target_session_id="sess-replay",
        store=store,
        checkpoint_sequence=3,
    )
    
    # We mock the tool execution during replay by tracking if the actual tool function is called.
    tool_called = False
    llm_called = False
    
    @replay_session.run()
    def run_replay_agent(x):
        @replay_session.tool()
        def add_one(v):
            nonlocal tool_called
            tool_called = True
            return v + 1
            
        @replay_session.llm(model="gpt-4")
        def ask_llm(prompt):
            nonlocal llm_called
            llm_called = True
            return "new-llm-response"
            
        v1 = add_one(x)
        replay_session.mutate_state([StateMutation(path="val", op="set", value=v1)])
        res = ask_llm(f"Value is {v1}")
        return res
        
    out_replay = run_replay_agent(10)
    
    # The tool call (seq=2,3) should be replayed (not called)
    assert tool_called is False
    # The LLM call (seq=5,6) was after checkpoint (seq=3), so it should be executed normally
    assert llm_called is True
    assert out_replay == "new-llm-response"
    
    # Check events in target session
    replay_events = store.get_events("sess-replay")
    assert len(replay_events) == 7
    assert replay_events[0].event_type == EventType.AGENT_STARTED
    assert replay_events[1].event_type == EventType.TOOL_CALL_INITIATED
    assert replay_events[2].event_type == EventType.TOOL_CALL_COMPLETED
    assert replay_events[3].event_type == EventType.STATE_MUTATED
    assert replay_events[4].event_type == EventType.LLM_CALL_INITIATED
    assert replay_events[5].event_type == EventType.LLM_CALL_COMPLETED
    assert replay_events[5].payload["response"] == "new-llm-response"
    assert replay_events[6].event_type == EventType.AGENT_COMPLETED


def test_replay_divergence():
    store = InMemoryEventStore()
    
    # 1. Run original session
    session = AgentSession(session_id="sess-orig-div", store=store, agent_id="agent-1")
    
    @session.run()
    def run_agent():
        @session.tool()
        def step1():
            return "step1"
            
        @session.tool()
        def step2():
            return "step2"
            
        step1()
        step2()
        
    run_agent()
    
    # 2. Replay but change the execution path (divergence)
    replay_session = ReplayAgentSession(
        source_session_id="sess-orig-div",
        target_session_id="sess-replay-div",
        store=store,
        checkpoint_sequence=5,  # up to step2 completed
    )
    
    step1_called = False
    step3_called = False
    
    @replay_session.run()
    def run_replay_agent():
        @replay_session.tool()
        def step1():
            nonlocal step1_called
            step1_called = True
            return "step1"
            
        @replay_session.tool()
        def step3():
            nonlocal step3_called
            step3_called = True
            return "step3"
            
        step1()  # Should be replayed (not called)
        step3()  # Diverges! Should be executed normally
        
    run_replay_agent()
    
    assert step1_called is False
    assert step3_called is True
    
    # Verify store was truncated at divergence point (seq=3, after step1 completed)
    # and step3 was appended as seq=4,5
    events = store.get_events("sess-replay-div")
    assert len(events) == 6
    assert events[0].event_type == EventType.AGENT_STARTED
    assert events[1].event_type == EventType.TOOL_CALL_INITIATED  # step1
    assert events[2].event_type == EventType.TOOL_CALL_COMPLETED  # step1
    assert events[3].event_type == EventType.TOOL_CALL_INITIATED  # step3
    assert events[4].event_type == EventType.TOOL_CALL_COMPLETED  # step3
    assert events[5].event_type == EventType.AGENT_COMPLETED


def test_replay_in_place():
    store = InMemoryEventStore()
    
    session = AgentSession(session_id="sess-inplace", store=store, agent_id="agent-1")
    
    @session.run()
    def run_agent():
        @session.tool()
        def step1():
            return "step1"
        @session.tool()
        def step2():
            return "step2"
        step1()
        step2()
        
    run_agent()
    
    # Replay in-place (same session ID) up to sequence 3 (step1 completed)
    replay_session = ReplayAgentSession(
        source_session_id="sess-inplace",
        target_session_id="sess-inplace",
        store=store,
        checkpoint_sequence=3,
    )
    
    step1_called = False
    step2_called = False
    
    @replay_session.run()
    def run_replay_agent():
        @replay_session.tool()
        def step1():
            nonlocal step1_called
            step1_called = True
            return "step1"
        @replay_session.tool()
        def step2():
            nonlocal step2_called
            step2_called = True
            return "new-step2"
        step1()
        step2()
        
    run_replay_agent()
    
    assert step1_called is False
    assert step2_called is True
    
    events = store.get_events("sess-inplace")
    assert len(events) == 6
    assert events[4].payload["tool_output"] == "new-step2"
