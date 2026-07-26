# agent_ledger/models.py - Event-sourced state management and time-travel debugging engine for LLM agents.
# Contributed by Claude Code

"""Core event schemas and strictly typed state mutation interfaces for agent-ledger."""

import copy
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# Configure logger
logger = logging.getLogger("agent_ledger.models")


class EventType(str, Enum):
    """Enumeration of core event types in the agent lifecycle."""

    AGENT_STARTED = "agent_started"
    LLM_CALL_INITIATED = "llm_call_initiated"
    LLM_CALL_COMPLETED = "llm_call_completed"
    TOOL_CALL_INITIATED = "tool_call_initiated"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    STATE_MUTATED = "state_mutated"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"


@dataclass(frozen=True)
class AgentStartedPayload:
    """Payload for agent initialization."""

    agent_id: str
    config: Dict[str, Any] = field(default_factory=dict)
    initial_state: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMCallInitiatedPayload:
    """Payload for LLM call initiation."""

    prompt: Union[str, List[Dict[str, Any]]]
    model: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMCallCompletedPayload:
    """Payload for LLM call completion."""

    response: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    finish_reason: Optional[str] = None


@dataclass(frozen=True)
class ToolCallInitiatedPayload:
    """Payload for tool execution initiation."""

    tool_name: str
    tool_input: Dict[str, Any]


@dataclass(frozen=True)
class ToolCallCompletedPayload:
    """Payload for tool execution completion."""

    tool_name: str
    tool_output: Any
    success: bool
    latency_ms: float
    error: Optional[str] = None


@dataclass(frozen=True)
class StateMutation:
    """Represents a single state mutation operation."""

    path: str  # Dot-notation path, e.g., "memory.user_name" or "variables.count"
    op: str  # "set", "delete", "append"
    value: Any
    old_value: Any = None


@dataclass(frozen=True)
class StateMutatedPayload:
    """Payload containing state mutations."""

    mutations: List[StateMutation]


@dataclass(frozen=True)
class AgentCompletedPayload:
    """Payload for successful agent completion."""

    output: Any
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentFailedPayload:
    """Payload for agent failure."""

    error_type: str
    error_message: str
    stack_trace: Optional[str] = None


# Union type for all valid payloads
EventPayload = Union[
    AgentStartedPayload,
    LLMCallInitiatedPayload,
    LLMCallCompletedPayload,
    ToolCallInitiatedPayload,
    ToolCallCompletedPayload,
    StateMutatedPayload,
    AgentCompletedPayload,
    AgentFailedPayload,
    Dict[str, Any],
]


@dataclass(frozen=True)
class Event:
    """Immutable event record representing a state transition or interaction."""

    event_id: str
    session_id: str
    sequence_number: int
    timestamp: float
    event_type: EventType
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


    @classmethod
    def create(
        cls,
        session_id: str,
        sequence_number: int,
        event_type: EventType,
        payload: EventPayload,
        metadata: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> "Event":
        """Factory method to create a validated Event instance.

        Args: 
            session_id: Unique identifier for the agent session.
            sequence_number: Monotonically increasing sequence number.
            event_type: The type of event.
            payload: The payload data, either as a dataclass or dict.
            metadata: Optional metadata dictionary.
            event_id: Optional UUID string. Generates a new UUID if not provided.
            timestamp: Optional epoch timestamp. Uses current time if not provided.

        Returns:
            A new Event instance.
        """
        payload_dict = (
            asdict(payload) if not isinstance(payload, dict) else payload
        )
        return cls(
            event_id=event_id or str(uuid.uuid4()),
            session_id=session_id,
            sequence_number=sequence_number,
            timestamp=timestamp or time.time(),
            event_type=event_type,
            payload=payload_dict,
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the event to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Deserializes an event from a dictionary."""
        return cls(
            event_id=data["event_id"],
            session_id=data["session_id"],
            sequence_number=data["sequence_number"],
            timestamp=data["timestamp"],
            event_type=EventType(data["event_type"]),
            payload=data["payload"],
            metadata=data.get("metadata", {}),
        )


def _get_parent_and_key(state: Dict[str, Any], path: str) -> tuple[Any, Any]:
    """Traverses the state dict to find the parent object and the final key/index."""
    if not path or path == ".":
        return None, None

    parts = path.split(".")
    current = state
    for part in parts[:-1]:
        if isinstance(current, list):
            try:
                idx = int(part)
                current = current[idx]
            except (ValueError, IndexError) as e:
                logger.error("Invalid list index %s in path %s", part, path)
                raise KeyError(f"Invalid list index {part} in path {path}") from e
        elif isinstance(current, dict):
            if part not in current:
                current[part] = {}
            current = current[part]
        else:
            logger.error("Cannot traverse path %s through non-container type %s", path, type(current))
            raise TypeError(
                f"Cannot traverse path {path} through non-container type {type(current)}"
            )

    last_part = parts[-1]
    if isinstance(current, list):
        try:
            return current, int(last_part)
        except ValueError as e:
            logger.error("Invalid list index %s in path %s", last_part, path)
            raise KeyError(f"Invalid list index {last_part} in path {path}") from e
    return current, last_part


def apply_mutation(state: Dict[str, Any], mutation: StateMutation) -> Dict[str, Any]:
    """Applies a StateMutation to a state dictionary.

    Args:
        state: The current state dictionary.
        mutation: The StateMutation to apply.

    Returns:
        The updated state dictionary.
    """
    logger.debug("Applying mutation: %s on path %s", mutation.op, mutation.path)
    new_state = copy.deepcopy(state)

    if not mutation.path or mutation.path == ".":
        if mutation.op == "set":
            if not isinstance(mutation.value, dict):
                logger.error("Root state mutation value must be a dictionary")
                raise ValueError("Root state mutation value must be a dictionary")
            return mutation.value
        else:
            logger.error("Unsupported operation '%s' on root state", mutation.op)
            raise ValueError(f"Unsupported operation '{mutation.op}' on root state")

    parent, key = _get_parent_and_key(new_state, mutation.path)

    if mutation.op == "set":
        if isinstance(parent, list):
            if key < 0 or key >= len(parent):
                logger.error("List index %d out of range", key)
                raise IndexError(f"List index {key} out of range")
            parent[key] = mutation.value
        else:
            parent[key] = mutation.value
    elif mutation.op == "delete":
        if isinstance(parent, list):
            if key < 0 or key >= len(parent):
                logger.error("List index %d out of range", key)
                raise IndexError(f"List index {key} out of range")
            parent.pop(key)
        else:
            if key in parent:
                del parent[key]
    elif mutation.op == "append":
        if isinstance(parent, list):
            target = parent[key]
            if not isinstance(target, list):
                logger.error("Target at path %s is not a list", mutation.path)
                raise TypeError(f"Target at path {mutation.path} is not a list")
            target.append(mutation.value)
        else:
            if key not in parent:
                parent[key] = []
            target = parent[key]
            if not isinstance(target, list):
                logger.error("Target at path %s is not a list", mutation.path)
                raise TypeError(f"Target at path {mutation.path} is not a list")
            target.append(mutation.value)
    else:
        logger.error("Unsupported mutation operation: %s", mutation.op)
        raise ValueError(f"Unsupported mutation operation: {mutation.op}")

    return new_state
