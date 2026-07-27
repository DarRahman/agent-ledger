# agent_ledger/engine.py - Event-sourced state management and time-travel debugging engine for LLM agents.
# Contributed by Claude Code

"""Execution wrapper and session management to intercept LLM calls and tool execution side-effects."""

import copy
import inspect
import logging
import time
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Union

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
from agent_ledger.store import BaseEventStore

# Configure logger
logger = logging.getLogger("agent_ledger.engine")


class LLMCallTracker:
    """Helper class to collect LLM call completion details within the context manager."""

    def __init__(self) -> None:
        self.response: Optional[str] = None
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: Optional[int] = None
        self.latency_ms: Optional[float] = None
        self.finish_reason: Optional[str] = None

    def complete(
        self,
        response: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: Optional[int] = None,
        latency_ms: Optional[float] = None,
        finish_reason: Optional[str] = None,
    ) -> None:
        """Sets the completion details for the LLM call.

        Args:
            response: The text response from the LLM.
            prompt_tokens: Number of tokens in the prompt.
            completion_tokens: Number of tokens in the completion.
            total_tokens: Total tokens used. Defaults to prompt_tokens + completion_tokens.
            latency_ms: Latency of the call in milliseconds.
            finish_reason: Reason the LLM finished (e.g., 'stop', 'length').
        """
        self.response = response
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.latency_ms = latency_ms
        self.finish_reason = finish_reason


class ToolCallTracker:
    """Helper class to collect tool execution completion details within the context manager."""

    def __init__(self) -> None:
        self.output: Any = None
        self.success: bool = True
        self.latency_ms: Optional[float] = None
        self.error: Optional[str] = None

    def complete(
        self,
        output: Any,
        success: bool = True,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """Sets the completion details for the tool execution.

        Args:
            output: The return value/output of the tool.
            success: Whether the tool executed successfully.
            latency_ms: Latency of the tool execution in milliseconds.
            error: Error message if the tool failed.
        """
        self.output = output
        self.success = success
        self.latency_ms = latency_ms
        self.error = error


class AgentSession:
    """Manages the execution lifecycle of an agent session, intercepting calls and recording events."""

    def __init__(
        self,
        session_id: str,
        store: BaseEventStore,
        agent_id: str,
        config: Optional[Dict[str, Any]] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initializes the AgentSession.

        Args:
            session_id: Unique identifier for the session.
            store: The event store backend to persist events.
            agent_id: Unique identifier for the agent definition.
            config: Optional configuration dictionary for the agent.
            initial_state: Optional initial state dictionary.
        """
        self.session_id = session_id
        self.store = store
        self.agent_id = agent_id
        self.config = config or {}
        self.initial_state = copy.deepcopy(initial_state or {})
        self.state = copy.deepcopy(self.initial_state)
        
        # Initialize sequence number based on existing events in the store
        self._sequence_number = self.store.get_latest_sequence(session_id)
        self._is_active = False

    def _next_sequence(self) -> int:
        self._sequence_number += 1
        return self._sequence_number

    def _append_event(
        self,
        event_type: EventType,
        payload: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Event:
        event = Event.create(
            session_id=self.session_id,
            sequence_number=self._next_sequence(),
            event_type=event_type,
            payload=payload,
            metadata=metadata,
        )
        self.store.append(event)
        return event

    def start(self) -> None:
        """Starts the agent session and records the AGENT_STARTED event."""
        if self._is_active:
            logger.warning("Session %s is already active.", self.session_id)
            return
        
        if self._sequence_number == 0:
            payload = AgentStartedPayload(
                agent_id=self.agent_id,
                config=self.config,
                initial_state=self.initial_state,
            )
            self._append_event(EventType.AGENT_STARTED, payload)
        
        self._is_active = True
        logger.info("Agent session %s started for agent %s", self.session_id, self.agent_id)

    def complete(self, output: Any, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Completes the agent session successfully.

        Args:
            output: The final output of the agent.
            metrics: Optional metrics dictionary.
        """
        if not self._is_active:
            raise ValueError(f"Session {self.session_id} is not active.")
        
        payload = AgentCompletedPayload(output=output, metrics=metrics or {})
        self._append_event(EventType.AGENT_COMPLETED, payload)
        self._is_active = False
        logger.info("Agent session %s completed successfully", self.session_id)

    def fail(self, error: Exception) -> None:
        """Fails the agent session and records the error.

        Args:
            error: The exception that caused the failure.
        """
        if not self._is_active:
            raise ValueError(f"Session {self.session_id} is not active.")
        
        tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        payload = AgentFailedPayload(
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=tb,
        )
        self._append_event(EventType.AGENT_FAILED, payload)
        self._is_active = False
        logger.info("Agent session %s failed: %s", self.session_id, str(error))

    def mutate_state(self, mutations: List[StateMutation]) -> None:
        """Applies state mutations, updates local state, and records the STATE_MUTATED event.

        Args:
            mutations: List of StateMutation objects to apply.
        """
        if not self._is_active:
            raise ValueError(f"Session {self.session_id} is not active.")
        
        for mutation in mutations:
            self.state = apply_mutation(self.state, mutation)
            
        payload = StateMutatedPayload(mutations=mutations)
        self._append_event(EventType.STATE_MUTATED, payload)
        logger.debug("State mutated in session %s: %s", self.session_id, mutations)

    @contextmanager
    def track_llm_call(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Generator[LLMCallTracker, None, None]:
        """Context manager to track an LLM call.

        Args:
            prompt: The prompt string or list of message dicts.
            model: The model name.
            temperature: The temperature setting.
            max_tokens: Optional maximum tokens limit.
            additional_params: Optional additional parameters.

        Yields:
            An LLMCallTracker instance to record completion details.
        """
        if not self._is_active:
            raise ValueError(f"Session {self.session_id} is not active.")

        init_payload = LLMCallInitiatedPayload(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            additional_params=additional_params or {},
        )
        self._append_event(EventType.LLM_CALL_INITIATED, init_payload)
        
        tracker = LLMCallTracker()
        start_time = time.time()
        try:
            yield tracker
        finally:
            latency_ms = (time.time() - start_time) * 1000.0
            if tracker.response is not None:
                comp_payload = LLMCallCompletedPayload(
                    response=tracker.response,
                    prompt_tokens=tracker.prompt_tokens,
                    completion_tokens=tracker.completion_tokens,
                    total_tokens=tracker.total_tokens or (tracker.prompt_tokens + tracker.completion_tokens),
                    latency_ms=tracker.latency_ms or latency_ms,
                    finish_reason=tracker.finish_reason,
                )
                self._append_event(EventType.LLM_CALL_COMPLETED, comp_payload)

    @contextmanager
    def track_tool_call(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
    ) -> Generator[ToolCallTracker, None, None]:
        """Context manager to track a tool execution.

        Args:
            tool_name: The name of the tool.
            tool_input: The input arguments dictionary.

        Yields:
            A ToolCallTracker instance to record completion details.
        """
        if not self._is_active:
            raise ValueError(f"Session {self.session_id} is not active.")

        init_payload = ToolCallInitiatedPayload(
            tool_name=tool_name,
            tool_input=tool_input,
        )
        self._append_event(EventType.TOOL_CALL_INITIATED, init_payload)
        
        tracker = ToolCallTracker()
        start_time = time.time()
        try:
            yield tracker
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000.0
            comp_payload = ToolCallCompletedPayload(
                tool_name=tool_name,
                tool_output=None,
                success=False,
                latency_ms=latency_ms,
                error=str(e),
            )
            self._append_event(EventType.TOOL_CALL_COMPLETED, comp_payload)
            raise e
        else:
            latency_ms = (time.time() - start_time) * 1000.0
            comp_payload = ToolCallCompletedPayload(
                tool_name=tool_name,
                tool_output=tracker.output,
                success=tracker.success,
                latency_ms=tracker.latency_ms or latency_ms,
                error=tracker.error,
            )
            self._append_event(EventType.TOOL_CALL_COMPLETED, comp_payload)

    def llm(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to wrap an LLM call function.

        Args:
            model: The model name.
            temperature: The temperature setting.
            max_tokens: Optional maximum tokens limit.
            additional_params: Optional additional parameters.

        Returns:
            A decorator function.
        """
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if inspect.iscoroutinefunction(func):
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    prompt = kwargs.get("prompt", args[0] if args else "")
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
        name: Optional[str] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to wrap a tool execution function.

        Args:
            name: Optional custom name for the tool. Defaults to function name.

        Returns:
            A decorator function.
        """
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or func.__name__
            if inspect.iscoroutinefunction(func):
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    tool_input = self._build_tool_input(func, args, kwargs)
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

    def run(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to wrap the main agent execution function.

        Returns:
            A decorator function.
        """
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if inspect.iscoroutinefunction(func):
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    self.start()
                    try:
                        res = await func(*args, **kwargs)
                        self.complete(output=res)
                        return res
                    except Exception as e:
                        self.fail(e)
                        raise e
                return async_wrapper
            else:
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    self.start()
                    try:
                        res = func(*args, **kwargs)
                        self.complete(output=res)
                        return res
                    except Exception as e:
                        self.fail(e)
                        raise e
                return sync_wrapper
        return decorator

    @contextmanager
    def lifecycle(self) -> Generator["AgentSession", None, None]:
        """Context manager to manage the start/complete/fail lifecycle of the agent.

        Yields:
            The AgentSession instance.
        """
        self.start()
        try:
            yield self
        except Exception as e:
            self.fail(e)
            raise e
        else:
            if self._is_active:
                self.complete(output=None)

    def _build_tool_input(
        self, func: Callable[..., Any], args: tuple, kwargs: dict
    ) -> Dict[str, Any]:
        try:
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            return dict(bound.arguments)
        except Exception:
            return {"args": list(args), "kwargs": kwargs}

    def _parse_llm_result(self, res: Any, tracker: LLMCallTracker) -> None:
        if isinstance(res, str):
            tracker.complete(response=res)
        elif isinstance(res, dict):
            tracker.complete(
                response=res.get("response", res.get("text", "")),
                prompt_tokens=res.get("prompt_tokens", 0),
                completion_tokens=res.get("completion_tokens", 0),
                total_tokens=res.get("total_tokens"),
                latency_ms=res.get("latency_ms"),
                finish_reason=res.get("finish_reason"),
            )
        elif hasattr(res, "choices") and hasattr(res, "usage"):
            choice = res.choices[0]
            response_text = ""
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                response_text = choice.message.content or ""
            elif hasattr(choice, "text"):
                response_text = choice.text or ""
            
            usage = res.usage
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
            
            tracker.complete(
                response=response_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                finish_reason=getattr(choice, "finish_reason", None),
            )
        else:
            tracker.complete(response=str(res))
