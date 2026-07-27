# agent_ledger/replay.py - Event-sourced state management and time-travel debugging engine for LLM agents.
# Contributed by Claude Code

"""Replay engine for execution recovery from historical checkpoints."""

import copy
import inspect
import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Union

from agent_ledger.engine import AgentSession, LLMCallTracker, ToolCallTracker
from agent_ledger.models import Event, EventType
from agent_ledger.projector import StateProjector
from agent_ledger.store import BaseEventStore

# Configure logger
logger = logging.getLogger("agent_ledger.replay")


class ReplayAgentSession(AgentSession):
    """Manages the execution lifecycle of an agent session in replay mode.

    Replays historical events up to a checkpoint before resuming normal execution.
    """

    def __init__(
        self, 
        source_session_id: str,
        target_session_id: str,
        store: BaseEventStore,
        checkpoint_sequence: Optional[int] = None,
        checkpoint_timestamp: Optional[float] = None,
    ) -> None:
        """Initializes the ReplayAgentSession.

        Args:
            source_session_id: The session ID to replay history from.
            target_session_id: The session ID to write new events to.
            store: The event store backend.
            checkpoint_sequence: Optional sequence number to replay up to (inclusive).
            checkpoint_timestamp: Optional timestamp to replay up to (inclusive).
        """
        self.source_session_id = source_session_id
        self.target_session_id = target_session_id
        self.store = store

        # Load source events
        source_events = store.get_events(source_session_id)
        if not source_events:
            raise ValueError(f"No events found for source session {source_session_id}")

        # Filter events up to checkpoint
        replay_events: List[Event] = []
        for event in source_events:
            if checkpoint_sequence is not None and event.sequence_number > checkpoint_sequence:
                break
            if checkpoint_timestamp is not None and event.timestamp > checkpoint_timestamp:
                break
            replay_events.append(event)

        if not replay_events:
            raise ValueError(
                f"No events found up to checkpoint (seq={checkpoint_sequence}, ts={checkpoint_timestamp})"
            )

        # Find the AGENT_STARTED event to extract configuration
        started_event = next(
            (e for e in replay_events if e.event_type == EventType.AGENT_STARTED), None
        )
        if not started_event:
            raise ValueError(f"No AGENT_STARTED event found in source session {source_session_id}")

        agent_id = started_event.payload["agent_id"]
        config = started_event.payload.get("config", {})
        initial_state = started_event.payload.get("initial_state", {})

        # Determine the actual checkpoint sequence number
        self._checkpoint_sequence = replay_events[-1].sequence_number

        # Handle target session preparation
        if target_session_id == source_session_id:
            # Truncate events after the checkpoint in-place
            self.store.truncate(target_session_id, self._checkpoint_sequence)
        else: 
            # Copy replay events to the new target session
            for event in replay_events:
                copied_event = Event.create(
                    session_id=target_session_id,
                    sequence_number=event.sequence_number,
                    event_type=event.event_type,
                    payload=event.payload,
                    metadata=event.metadata,
                    event_id=event.event_id,
                    timestamp=event.timestamp,
                )
                try:
                    self.store.append(copied_event)
                except Exception as e:
                    logger.debug("Event already exists in target session: %s", e)

        # Initialize the projector to extract completed side-effects
        projector = StateProjector()
        projected_state = projector.project(replay_events)

        # Build the replay queue of completed side-effects
        self._replay_queue: List[Dict[str, Any]] = [
            item
            for item in projected_state.history
            if item.get("status") == "completed"
            and item.get("type") in ("llm_call", "tool_call")
        ]

        # Initialize the parent AgentSession with target_session_id
        super().__init__(
            session_id=target_session_id,
            store=store,
            agent_id=agent_id,
            config=config,
            initial_state=initial_state,
        )

        # Reset sequence number to 0 so that replay execution matches historical sequences
        self._sequence_number = 0

    def _append_event(
        self, 
        event_type: EventType, 
        payload: Any, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Event:
        next_seq = self._sequence_number + 1
        if next_seq <= self._checkpoint_sequence:
            # Fetch the historical event
            events = self.store.get_events(self.session_id, start_seq=next_seq, end_seq=next_seq)
            if events:
                hist_event = events[0]
                # Check if event type matches
                if hist_event.event_type == event_type:
                    # It matches! Increment sequence and return the historical event.
                    self._sequence_number = next_seq
                    return hist_event
                else:
                    logger.info(
                        "Replay diverged at sequence %d: expected event type %s, got %s.",
                        next_seq,
                        hist_event.event_type,
                        event_type,
                    )
            else:
                logger.info(
                    "Replay diverged at sequence %d: historical event not found.",
                    next_seq,
                )
            
            # If we reach here, we have diverged!
            self._handle_divergence()

        # Beyond checkpoint or after divergence, append normally
        return super()._append_event(event_type, payload, metadata)

    def _get_next_replay_call(self, call_type: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        if not self._replay_queue:
            return None

        next_call = self._replay_queue[0]
        diverged = False

        if next_call["type"] != call_type:
            logger.info(
                "Replay diverged: expected %s, got %s.",
                next_call["type"],
                call_type,
            )
            diverged = True
        elif call_type == "tool_call":
            expected_name = next_call["tool_name"]
            actual_name = kwargs.get("tool_name")
            if expected_name != actual_name:
                logger.info(
                    "Replay diverged: expected tool %s, got %s.",
                    expected_name,
                    actual_name,
                )
                diverged = True
        elif call_type == "llm_call":
            expected_model = next_call["model"]
            actual_model = kwargs.get("model")
            if expected_model != actual_model:
                logger.info(
                    "Replay diverged: expected LLM model %s, got %s.",
                    expected_model,
                    actual_model,
                )
                diverged = True

        if diverged:
            self._handle_divergence()
            return None

        # Matches! Pop and return.
        return self._replay_queue.pop(0)

    def _handle_divergence(self) -> None:
        logger.info(
            "Truncating session %s to sequence %d due to replay divergence.",
            self.session_id,
            self._sequence_number,
        )
        self.store.truncate(self.session_id, self._sequence_number)
        self._checkpoint_sequence = self._sequence_number
        self._replay_queue.clear()

    @contextmanager
    def track_llm_call(
        self, 
        prompt: Union[str, List[Dict[str, Any]]], 
        model: str, 
        temperature: float = 0.0, 
        max_tokens: Optional[int] = None, 
        additional_params: Optional[Dict[str, Any]] = None
    ) -> Generator[LLMCallTracker, None, None]:
        matching_call = self._get_next_replay_call("llm_call", model=model)
        if matching_call:
            logger.info("Replaying LLM call for model %s", model)
            tracker = LLMCallTracker()
            tracker.complete(
                response=matching_call["response"],
                prompt_tokens=matching_call["prompt_tokens"],
                completion_tokens=matching_call["completion_tokens"],
                total_tokens=matching_call["total_tokens"],
                latency_ms=matching_call["latency_ms"],
                finish_reason=matching_call["finish_reason"],
            )
            # Increment sequence numbers for the initiated and completed events
            self._sequence_number += 2
            yield tracker
        else:
            with super().track_llm_call(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                additional_params=additional_params,
            ) as tracker:
                yield tracker

    @contextmanager
    def track_tool_call(
        self, 
        tool_name: str, 
        tool_input: Dict[str, Any]
    ) -> Generator[ToolCallTracker, None, None]:
        matching_call = self._get_next_replay_call("tool_call", tool_name=tool_name)
        if matching_call:
            logger.info("Replaying tool call for tool %s", tool_name)
            tracker = ToolCallTracker()
            tracker.complete(
                output=matching_call["tool_output"],
                success=matching_call["success"],
                latency_ms=matching_call["latency_ms"],
                error=matching_call["error"],
            )
            # Increment sequence numbers for the initiated and completed events
            self._sequence_number += 2
            yield tracker
        else:
            with super().track_tool_call(tool_name=tool_name, tool_input=tool_input) as tracker:
                yield tracker

    def llm(
        self, 
        model: str, 
        temperature: float = 0.0, 
        max_tokens: Optional[int] = None, 
        additional_params: Optional[Dict[str, Any]] = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if inspect.iscoroutinefunction(func):
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    prompt = kwargs.get("prompt", args[0] if args else "")
                    matching_call = self._get_next_replay_call("llm_call", model=model)
                    if matching_call:
                        logger.info("Replaying LLM call (decorator) for model %s", model)
                        # Increment sequence numbers for the initiated and completed events
                        self._sequence_number += 2
                        return matching_call["response"]

                    with self.track_llm_call(
                        prompt=prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        additional_params=additional_params,
                    ) as tracker:
                        res = await func(*args, **kwargs)
                        self._parse_llm_result(res, tracker)
                        return res
                return async_wrapper
            else:
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    prompt = kwargs.get("prompt", args[0] if args else "")
                    matching_call = self._get_next_replay_call("llm_call", model=model)
                    if matching_call:
                        logger.info("Replaying LLM call (decorator) for model %s", model)
                        # Increment sequence numbers for the initiated and completed events
                        self._sequence_number += 2
                        return matching_call["response"]

                    with self.track_llm_call(
                        prompt=prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        additional_params=additional_params,
                    ) as tracker:
                        res = func(*args, **kwargs)
                        self._parse_llm_result(res, tracker)
                        return res
                return sync_wrapper
        return decorator

    def tool(
        self, 
        name: Optional[str] = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or func.__name__
            if inspect.iscoroutinefunction(func):
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    tool_input = self._build_tool_input(func, args, kwargs)
                    matching_call = self._get_next_replay_call("tool_call", tool_name=tool_name)
                    if matching_call:
                        logger.info("Replaying tool call (decorator) for tool %s", tool_name)
                        # Increment sequence numbers for the initiated and completed events
                        self._sequence_number += 2
                        if not matching_call["success"]:
                            raise Exception(matching_call["error"] or "Tool execution failed during replay")
                        return matching_call["tool_output"]

                    with self.track_tool_call(tool_name=tool_name, tool_input=tool_input) as tracker:
                        try:
                            res = await func(*args, **kwargs)
                            tracker.complete(output=res, success=True)
                            return res
                        except Exception as e:
                            tracker.complete(output=None, success=False, error=str(e))
                            raise e
                return async_wrapper
            else:
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    tool_input = self._build_tool_input(func, args, kwargs)
                    matching_call = self._get_next_replay_call("tool_call", tool_name=tool_name)
                    if matching_call:
                        logger.info("Replaying tool call (decorator) for tool %s", tool_name)
                        # Increment sequence numbers for the initiated and completed events
                        self._sequence_number += 2
                        if not matching_call["success"]:
                            raise Exception(matching_call["error"] or "Tool execution failed during replay")
                        return matching_call["tool_output"]

                    with self.track_tool_call(tool_name=tool_name, tool_input=tool_input) as tracker:
                        try:
                            res = func(*args, **kwargs)
                            tracker.complete(output=res, success=True)
                            return res
                        except Exception as e:
                            tracker.complete(output=None, success=False, error=str(e))
                            raise e
                return sync_wrapper
        return decorator
