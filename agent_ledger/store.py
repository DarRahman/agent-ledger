# agent_ledger/store.py - Event-sourced state management and time-travel debugging engine for LLM agents.
# Contributed by Claude Code

"""Thread-safe event store backends: InMemoryEventStore and SQLiteEventStore."""

import json
import logging
import sqlite3
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from agent_ledger.models import Event

logger = logging.getLogger("agent_ledger.store")


class EventStoreError(Exception):
    """Base exception for event store errors."""
    pass


class DuplicateSequenceError(EventStoreError):
    """Raised when an event sequence number conflicts with existing events."""
    pass


class BaseEventStore(ABC):
    """Abstract base class for thread-safe event store backends."""

    @abstractmethod
    def append(self, event: Event) -> None:
        """Append an event to the store."""
        pass

    @abstractmethod
    def get_events(
        self, session_id: str, start_seq: int = 1, end_seq: Optional[int] = None
    ) -> List[Event]:
        """Fetch events for a session within sequence bounds [start_seq, end_seq]."""
        pass

    @abstractmethod
    def get_latest_sequence(self, session_id: str) -> int:
        """Return latest sequence number for a session, or 0 if session empty."""
        pass

    @abstractmethod
    def list_sessions(self) -> List[str]:
        """List all tracked session IDs."""
        pass

    @abstractmethod
    def truncate(self, session_id: str, sequence_number: int) -> None:
        """Deletes all events in a session with sequence number greater than sequence_number.

        Args:
            session_id: Unique identifier for the session.
            sequence_number: The sequence number threshold. Events with sequence_number > threshold are deleted.
        """
        pass


class InMemoryEventStore(BaseEventStore):
    """Thread-safe in-memory event store using RLock."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stores: Dict[str, List[Event]] = {}

    def append(self, event: Event) -> None:
        with self._lock:
            events = self._stores.setdefault(event.session_id, [])
            if events and event.sequence_number <= events[-1].sequence_number:
                logger.error(
                    "Duplicate/invalid sequence %d for session %s (latest=%d)",
                    event.sequence_number,
                    event.session_id,
                    events[-1].sequence_number,
                )
                raise DuplicateSequenceError(
                    f"Sequence {event.sequence_number} conflict for session {event.session_id}"
                )
            events.append(event)
            logger.debug(
                "Appended event %s (seq=%d) to session %s",
                event.event_id,
                event.sequence_number,
                event.session_id,
            )

    def get_events(
        self, session_id: str, start_seq: int = 1, end_seq: Optional[int] = None
    ) -> List[Event]:
        with self._lock:
            events = self._stores.get(session_id, [])
            res = []
            for e in events:
                if e.sequence_number < start_seq:
                    continue
                if end_seq is not None and e.sequence_number > end_seq:
                    break
                res.append(e)
            return res

    def get_latest_sequence(self, session_id: str) -> int:
        with self._lock:
            events = self._stores.get(session_id, [])
            return events[-1].sequence_number if events else 0

    def list_sessions(self) -> List[str]:
        with self._lock:
            return sorted(list(self._stores.keys()))

    def truncate(self, session_id: str, sequence_number: int) -> None:
        with self._lock:
            if session_id in self._stores:
                self._stores[session_id] = [
                    e for e in self._stores[session_id] if e.sequence_number <= sequence_number
                ]
                logger.info(
                    "Truncated session %s to sequence %d", session_id, sequence_number
                )


class SQLiteEventStore(BaseEventStore):
    """Thread-safe SQLite event store backend with WAL support."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._lock, self._conn:
            if self.db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    UNIQUE(session_id, sequence_number)
                );
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_seq ON events(session_id, sequence_number);"
            )

    def append(self, event: Event) -> None:
        payload_json = json.dumps(event.payload)
        metadata_json = json.dumps(event.metadata)
        event_type_val = (
            event.event_type.value
            if hasattr(event.event_type, "value")
            else str(event.event_type)
        )

        with self._lock:
            try:
                with self._conn:
                    self._conn.execute(
                        """
                        INSERT INTO events (
                            event_id, session_id, sequence_number, timestamp, event_type, payload, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event.event_id,
                            event.session_id,
                            event.sequence_number,
                            event.timestamp,
                            event_type_val,
                            payload_json,
                            metadata_json,
                        ),
                    )
                logger.debug(
                    "Persisted event %s (seq=%d) to SQLite session %s",
                    event.event_id,
                    event.sequence_number,
                    event.session_id,
                )
            except sqlite3.IntegrityError as err:
                logger.error(
                    "Integrity violation inserting sequence %d for session %s: %s",
                    event.sequence_number,
                    event.session_id,
                    err,
                )
                raise DuplicateSequenceError(
                    f"Sequence {event.sequence_number} conflict for session {event.session_id}"
                ) from err

    def get_events(
        self, session_id: str, start_seq: int = 1, end_seq: Optional[int] = None
    ) -> List[Event]:
        query = "SELECT * FROM events WHERE session_id = ? AND sequence_number >= ?"
        params: list = [session_id, start_seq]
        if end_seq is not None:
            query += " AND sequence_number <= ?"
            params.append(end_seq)
        query += " ORDER BY sequence_number ASC"

        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        events = []
        for row in rows:
            data = {
                "event_id": row["event_id"],
                "session_id": row["session_id"],
                "sequence_number": row["sequence_number"],
                "timestamp": row["timestamp"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload"]),
                "metadata": json.loads(row["metadata"]),
            }
            events.append(Event.from_dict(data))
        return events

    def get_latest_sequence(self, session_id: str) -> int:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT MAX(sequence_number) FROM events WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            return row[0] if (row and row[0] is not None) else 0

    def list_sessions(self) -> List[str]:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT DISTINCT session_id FROM events ORDER BY session_id ASC")
            rows = cursor.fetchall()
            return [row["session_id"] for row in rows]

    def truncate(self, session_id: str, sequence_number: int) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "DELETE FROM events WHERE session_id = ? AND sequence_number > ?",
                    (session_id, sequence_number),
                )
            logger.info(
                "Truncated SQLite session %s to sequence %d", session_id, sequence_number
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
