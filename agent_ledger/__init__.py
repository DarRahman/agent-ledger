# agent_ledger/__init__.py - Event-sourced state management and time-travel debugging engine for LLM agents.
# Contributed by Claude Code

"""Agent Ledger core package."""

from agent_ledger.branch import branch_session, calculate_state_diff
from agent_ledger.engine import AgentSession, LLMCallTracker, ToolCallTracker
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
    apply_mutation,
)
from agent_ledger.projector import AgentState, StateProjector
from agent_ledger.replay import ReplayAgentSession
from agent_ledger.store import (
    BaseEventStore,
    DuplicateSequenceError,
    EventStoreError,
    InMemoryEventStore,
    SQLiteEventStore,
)

__all__ = [
    "EventType",
    "Event",
    "AgentStartedPayload",
    "LLMCallInitiatedPayload",
    "LLMCallCompletedPayload",
    "ToolCallInitiatedPayload",
    "ToolCallCompletedPayload",
    "StateMutation",
    "StateMutatedPayload",
    "AgentCompletedPayload",
    "AgentFailedPayload",
    "apply_mutation",
    "BaseEventStore",
    "InMemoryEventStore",
    "SQLiteEventStore",
    "EventStoreError",
    "DuplicateSequenceError",
    "AgentState",
    "StateProjector",
    "AgentSession",
    "LLMCallTracker",
    "ToolCallTracker",
    "ReplayAgentSession",
    "branch_session",
    "calculate_state_diff",
]
