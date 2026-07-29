# tests/test_cli.py
# Contributed by Claude Code

"""Unit tests for the CLI visualization tool."""

import os
import sys
import tempfile
from unittest.mock import patch

import pytest

from agent_ledger.cli import main
from agent_ledger.models import Event, EventType, AgentStartedPayload, StateMutation, StateMutatedPayload
from agent_ledger.store import SQLiteEventStore


@pytest.fixture
def temp_db():
    """Creates a temporary SQLite database with sample events."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    store = SQLiteEventStore(db_path)
    
    # Session 1: Completed successfully
    store.append(Event.create(
        session_id="session-success",
        sequence_number=1,
        event_type=EventType.AGENT_STARTED,
        payload=AgentStartedPayload(agent_id="agent-1", config={}, initial_state={"x": 10}),
        timestamp=100.0
    ))
    store.append(Event.create(
        session_id="session-success",
        sequence_number=2,
        event_type=EventType.STATE_MUTATED,
        payload=StateMutatedPayload(mutations=[StateMutation(path="x", op="set", value=20)]),
        timestamp=101.0
    ))
    store.append(Event.create(
        session_id="session-success",
        sequence_number=3,
        event_type=EventType.AGENT_COMPLETED,
        payload={"output": "done"},
        timestamp=102.0
    ))

    store.close()
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


def test_cli_list(temp_db, capsys):
    """Tests the 'list' command."""
    test_args = ["cli.py", "--db", temp_db, "list"]
    with patch.object(sys, "argv", test_args):
        main()

    captured = capsys.readouterr()
    assert "session-success" in captured.out
    assert "agent-1" in captured.out
    assert "completed" in captured.out


def test_cli_timeline(temp_db, capsys):
    """Tests the 'timeline' command."""
    test_args = ["cli.py", "--db", temp_db, "timeline", "session-success"]
    with patch.object(sys, "argv", test_args):
        main()

    captured = capsys.readouterr()
    assert "agent_started" in captured.out
    assert "state_mutated" in captured.out
    assert "agent_completed" in captured.out
    assert "agent-1" in captured.out


def test_cli_state(temp_db, capsys):
    """Tests the 'state' command."""
    # Test latest state
    test_args = ["cli.py", "--db", temp_db, "state", "session-success"]
    with patch.object(sys, "argv", test_args):
        main()

    captured = capsys.readouterr()
    assert '"x": 20' in captured.out

    # Test state at sequence 1
    test_args = ["cli.py", "--db", temp_db, "state", "session-success", "--seq", "1"]
    with patch.object(sys, "argv", test_args):
        main()

    captured = capsys.readouterr()
    assert '"x": 10' in captured.out


def test_cli_diff(temp_db, capsys):
    """Tests the 'diff' command."""
    test_args = ["cli.py", "--db", temp_db, "diff", "session-success", "1", "2"]
    with patch.object(sys, "argv", test_args):
        main()

    captured = capsys.readouterr()
    assert "SET" in captured.out
    assert "x" in captured.out
    assert "20" in captured.out
    assert "10" in captured.out


def test_cli_missing_db(capsys):
    """Tests CLI behavior when database file does not exist."""
    test_args = ["cli.py", "--db", "nonexistent_file.db", "list"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1

    captured = capsys.readouterr()
    assert "does not exist" in captured.err
