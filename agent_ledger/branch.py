# agent_ledger/branch.py - Event-sourced state management and time-travel debugging engine for LLM agents.
# Contributed by Claude Code

"""Branching mechanism to spawn alternative agent trajectories from state diffs."""

import logging
import uuid
from typing import Any, Dict, List, Optional

from agent_ledger.models import StateMutation
from agent_ledger.replay import ReplayAgentSession
from agent_ledger.store import BaseEventStore

logger = logging.getLogger("agent_ledger.branch")


def calculate_state_diff(
    old_state: Dict[str, Any], new_state: Dict[str, Any], prefix: str = ""
) -> List[StateMutation]:
    """Calculates state mutations required to transition from old_state to new_state.

    Args:
        old_state: The starting state dictionary.
        new_state: The target state dictionary.
        prefix: Internal path prefix for nested recursive diffs.

    Returns:
        A list of StateMutation objects representing the diff.
    """
    # ponytail: nested dict diff ceiling, add full array/set patch algorithms if complex structural diffs needed
    mutations: List[StateMutation] = []

    for key, val in new_state.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in old_state:
            mutations.append(StateMutation(path=path, op="set", value=val, old_value=None))
        elif old_state[key] != val:
            if isinstance(old_state[key], dict) and isinstance(val, dict):
                mutations.extend(calculate_state_diff(old_state[key], val, prefix=path))
            elif isinstance(old_state[key], list) and isinstance(val, list):
                mutations.extend(diff_lists(old_state[key], val, prefix=path))
            else:
                mutations.append(
                    StateMutation(path=path, op="set", value=val, old_value=old_state[key])
                )

    for key, old_val in old_state.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in new_state:
            mutations.append(StateMutation(path=path, op="delete", value=None, old_value=old_val))

    return mutations


def diff_lists(old_list: List[Any], new_list: List[Any], prefix: str) -> List[StateMutation]:
    mutations = []
    min_len = min(len(old_list), len(new_list))
    for i in range(min_len):
        path = f"{prefix}.{i}"
        o_val = old_list[i]
        n_val = new_list[i]
        if o_val != n_val:
            if isinstance(o_val, dict) and isinstance(n_val, dict):
                mutations.extend(calculate_state_diff(o_val, n_val, prefix=path))
            elif isinstance(o_val, list) and isinstance(n_val, list):
                mutations.extend(diff_lists(o_val, n_val, prefix=path))
            else:
                mutations.append(StateMutation(path=path, op="set", value=n_val, old_value=o_val))

    for i in range(min_len, len(new_list)):
        mutations.append(StateMutation(path=prefix, op="append", value=new_list[i]))

    for i in range(len(old_list) - 1, min_len - 1, -1):
        path = f"{prefix}.{i}"
        mutations.append(StateMutation(path=path, op="delete", value=None, old_value=old_list[i]))

    return mutations


def branch_session(
    store: BaseEventStore,
    source_session_id: str,
    target_session_id: str,
    checkpoint_sequence: Optional[int] = None,
    checkpoint_timestamp: Optional[float] = None,
    state_diffs: Optional[List[StateMutation]] = None,
) -> ReplayAgentSession:
    """Spawns an alternative agent trajectory from a checkpoint with optional state diffs."""
    session = ReplayAgentSession(
        source_session_id=source_session_id,
        target_session_id=target_session_id,
        store=store,
        checkpoint_sequence=checkpoint_sequence,
        checkpoint_timestamp=checkpoint_timestamp,
    )
    if state_diffs:
        was_active = session._is_active
        session._is_active = True
        try:
            session.mutate_state(state_diffs)
        finally:
            session._is_active = was_active
    return session
