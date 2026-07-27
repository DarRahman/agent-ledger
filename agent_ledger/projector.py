# agent_ledger/projector.py - Event-sourced state management and time-travel debugging engine for LLM agents.
# Contributed by Claude Code

"""State projector engine to reconstruct agent state from event streams."""

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from agent_ledger.models import Event, EventType, StateMutation, apply_mutation

logger = logging.getLogger("agent_ledger.projector")


@dataclass
class AgentState:
    """Reconstructed state of an agent at a specific point in time."""

    agent_id: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    state: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    output: Any = None
    last_sequence_number: int = 0
    last_timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the agent state to a dictionary."""
        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "state": copy.deepcopy(self.state),
            "history": copy.deepcopy(self.history),
            "metadata": copy.deepcopy(self.metadata),
            "error": copy.deepcopy(self.error) if self.error else None,
            "output": copy.deepcopy(self.output),
            "last_sequence_number": self.last_sequence_number,
            "last_timestamp": self.last_timestamp,
        }


class StateProjector:
    """Engine to project event streams into a consolidated AgentState."""

    def __init__(self) -> None:
        self._handlers: Dict[EventType, Callable[[AgentState, Event], AgentState]] = {}
        self._register_default_handlers()

    def register_handler(
        self, event_type: EventType, handler: Callable[[AgentState, Event], AgentState]
    ) -> None:
        """Registers a custom handler for a specific event type.

        Args:
            event_type: The EventType to handle.
            handler: A callable taking (current_state, event) and returning updated state.
        """
        self._handlers[event_type] = handler
        logger.debug("Registered custom handler for event type: %s", event_type)

    def project(
        self,
        events: Iterable[Event],
        initial_state: Optional[AgentState] = None,
        up_to_sequence: Optional[int] = None,
        up_to_timestamp: Optional[float] = None,
    ) -> AgentState:
        """Projects a stream of events into an AgentState.

        Args:
            events: An iterable of Event objects.
            initial_state: Optional starting AgentState. If None, a new one is created.
            up_to_sequence: Optional sequence number limit (inclusive).
            up_to_timestamp: Optional timestamp limit (inclusive).

        Returns:
            The reconstructed AgentState.
        """
        sorted_events = sorted(events, key=lambda e: e.sequence_number)
        current_state = copy.deepcopy(initial_state) if initial_state else AgentState()

        for event in sorted_events:
            if up_to_sequence is not None and event.sequence_number > up_to_sequence:
                logger.debug(
                    "Stopping projection: event sequence %d exceeds limit %d",
                    event.sequence_number,
                    up_to_sequence,
                )
                break
            if up_to_timestamp is not None and event.timestamp > up_to_timestamp:
                logger.debug(
                    "Stopping projection: event timestamp %f exceeds limit %f",
                    event.timestamp,
                    up_to_timestamp,
                )
                break

            current_state.last_sequence_number = event.sequence_number
            current_state.last_timestamp = event.timestamp

            handler = self._handlers.get(event.event_type)
            if handler:
                try:
                    current_state = handler(current_state, event)
                except Exception as e:
                    logger.error(
                        "Error applying event %s (seq=%d, type=%s): %s",
                        event.event_id,
                        event.sequence_number,
                        event.event_type,
                        str(e),
                    )
                    raise
            else: 
                logger.warning(
                    "No handler registered for event type: %s. Skipping.",
                    event.event_type,
                )

        return current_state

    def _register_default_handlers(self) -> None:
        self.register_handler(EventType.AGENT_STARTED, self._handle_agent_started)
        self.register_handler(EventType.STATE_MUTATED, self._handle_state_mutated)
        self.register_handler(EventType.LLM_CALL_INITIATED, self._handle_llm_call_initiated)
        self.register_handler(EventType.LLM_CALL_COMPLETED, self._handle_llm_call_completed)
        self.register_handler(EventType.TOOL_CALL_INITIATED, self._handle_tool_call_initiated)
        self.register_handler(EventType.TOOL_CALL_COMPLETED, self._handle_tool_call_completed)
        self.register_handler(EventType.AGENT_COMPLETED, self._handle_agent_completed)
        self.register_handler(EventType.AGENT_FAILED, self._handle_agent_failed)

    @staticmethod
    def _handle_agent_started(state: AgentState, event: Event) -> AgentState:
        payload = event.payload
        state.agent_id = payload.get("agent_id")
        state.status = "running"
        state.state = copy.deepcopy(payload.get("initial_state", {}))
        state.metadata = copy.deepcopy(payload.get("config", {}))
        return state

    @staticmethod
    def _handle_state_mutated(state: AgentState, event: Event) -> AgentState:
        payload = event.payload
        mutations_data = payload.get("mutations", [])
        for mut_dict in mutations_data:
            mutation = StateMutation(
                path=mut_dict["path"],
                op=mut_dict["op"],
                value=mut_dict.get("value"),
                old_value=mut_dict.get("old_value"),
            )
            state.state = apply_mutation(state.state, mutation)
        return state

    @staticmethod
    def _handle_llm_call_initiated(state: AgentState, event: Event) -> AgentState:
        payload = event.payload
        state.history.append({
            "type": "llm_call",
            "status": "initiated",
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "prompt": payload.get("prompt"),
            "model": payload.get("model"),
            "temperature": payload.get("temperature"),
            "max_tokens": payload.get("max_tokens"),
            "additional_params": payload.get("additional_params"),
        })
        return state

    @staticmethod
    def _handle_llm_call_completed(state: AgentState, event: Event) -> AgentState:
        payload = event.payload
        for item in reversed(state.history):
            if item.get("type") == "llm_call" and item.get("status") == "initiated":
                item.update({
                    "status": "completed",
                    "response": payload.get("response"),
                    "prompt_tokens": payload.get("prompt_tokens"),
                    "completion_tokens": payload.get("completion_tokens"),
                    "total_tokens": payload.get("total_tokens"),
                    "latency_ms": payload.get("latency_ms"),
                    "finish_reason": payload.get("finish_reason"),
                    "completed_timestamp": event.timestamp,
                })
                break
        else:
            state.history.append({
                "type": "llm_call",
                "status": "completed",
                "event_id": event.event_id,
                "timestamp": event.timestamp,
                "response": payload.get("response"),
                "prompt_tokens": payload.get("prompt_tokens"),
                "completion_tokens": payload.get("completion_tokens"),
                "total_tokens": payload.get("total_tokens"),
                "latency_ms": payload.get("latency_ms"),
                "finish_reason": payload.get("finish_reason"),
                "completed_timestamp": event.timestamp,
            })
        return state

    @staticmethod
    def _handle_tool_call_initiated(state: AgentState, event: Event) -> AgentState:
        payload = event.payload
        state.history.append({
            "type": "tool_call",
            "status": "initiated",
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "tool_name": payload.get("tool_name"),
            "tool_input": payload.get("tool_input"),
        })
        return state

    @staticmethod
    def _handle_tool_call_completed(state: AgentState, event: Event) -> AgentState:
        payload = event.payload
        for item in reversed(state.history):
            if item.get("type") == "tool_call" and item.get("status") == "initiated" and item.get("tool_name") == payload.get("tool_name"):
                item.update({
                    "status": "completed" if payload.get("success", True) else "failed",
                    "tool_output": payload.get("tool_output"),
                    "success": payload.get("success"),
                    "latency_ms": payload.get("latency_ms"),
                    "error": payload.get("error"),
                    "completed_timestamp": event.timestamp,
                })
                break
        else:
            state.history.append({
                "type": "tool_call",
                "status": "completed" if payload.get("success", True) else "failed",
                "event_id": event.event_id,
                "timestamp": event.timestamp,
                "tool_name": payload.get("tool_name"),
                "tool_output": payload.get("tool_output"),
                "success": payload.get("success"),
                "latency_ms": payload.get("latency_ms"),
                "error": payload.get("error"),
                "completed_timestamp": event.timestamp,
            })
        return state

    @staticmethod
    def _handle_agent_completed(state: AgentState, event: Event) -> AgentState:
        payload = event.payload
        state.status = "completed"
        state.output = payload.get("output")
        if "metrics" in payload:
            state.metadata.setdefault("metrics", {}).update(payload["metrics"])
        return state

    @staticmethod
    def _handle_agent_failed(state: AgentState, event: Event) -> AgentState:
        payload = event.payload
        state.status = "failed"
        state.error = {
            "error_type": payload.get("error_type"),
            "error_message": payload.get("error_message"),
            "stack_trace": payload.get("stack_trace"),
        }
        return state
