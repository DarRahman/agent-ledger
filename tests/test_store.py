# tests/test_store.py
# Contributed by Claude Code

import concurrent.futures
import os
import tempfile
import pytest
from agent_ledger.models import Event, EventType, AgentStartedPayload
from agent_ledger.store import (
    InMemoryEventStore,
    SQLiteEventStore,
    DuplicateSequenceError,
)


def create_sample_event(session_id: str, seq: int) -> Event:
    return Event.create(
        session_id=session_id,
        sequence_number=seq,
        event_type=EventType.AGENT_STARTED,
        payload=AgentStartedPayload(agent_id="agent-1", config={}, initial_state={}),
    )


@pytest.mark.parametrize("store_cls", [InMemoryEventStore, SQLiteEventStore])
def test_append_and_get_events(store_cls):
    store = store_cls()
    e1 = create_sample_event("session-a", 1)
    e2 = create_sample_event("session-a", 2)
    e3 = create_sample_event("session-b", 1)

    store.append(e1)
    store.append(e2)
    store.append(e3)

    assert store.get_latest_sequence("session-a") == 2
    assert store.get_latest_sequence("session-b") == 1
    assert store.get_latest_sequence("session-c") == 0

    events_a = store.get_events("session-a")
    assert len(events_a) == 2
    assert events_a[0].sequence_number == 1
    assert events_a[1].sequence_number == 2

    sessions = store.list_sessions()
    assert sorted(sessions) == ["session-a", "session-b"]

    if isinstance(store, SQLiteEventStore):
        store.close()


@pytest.mark.parametrize("store_cls", [InMemoryEventStore, SQLiteEventStore])
def test_range_query(store_cls):
    store = store_cls()
    for i in range(1, 6):
        store.append(create_sample_event("sess-range", i))

    res = store.get_events("sess-range", start_seq=2, end_seq=4)
    seqs = [e.sequence_number for e in res]
    assert seqs == [2, 3, 4]

    if isinstance(store, SQLiteEventStore):
        store.close()


@pytest.mark.parametrize("store_cls", [InMemoryEventStore, SQLiteEventStore])
def test_duplicate_sequence_raises(store_cls):
    store = store_cls()
    store.append(create_sample_event("sess-dup", 1))
    with pytest.raises(DuplicateSequenceError):
        store.append(create_sample_event("sess-dup", 1))

    if isinstance(store, SQLiteEventStore):
        store.close()


def test_sqlite_file_persistence():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        store1 = SQLiteEventStore(db_path=db_path)
        store1.append(create_sample_event("persist-sess", 1))
        store1.close()

        store2 = SQLiteEventStore(db_path=db_path)
        events = store2.get_events("persist-sess")
        assert len(events) == 1
        assert events[0].session_id == "persist-sess"
        store2.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


@pytest.mark.parametrize("store_cls", [InMemoryEventStore, SQLiteEventStore])
def test_thread_safety(store_cls):
    store = store_cls()

    def worker(worker_id: int):
        session_id = f"session-{worker_id}"
        for seq in range(1, 21):
            event = create_sample_event(session_id, seq)
            store.append(event)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker, i) for i in range(5)]
        for f in futures:
            f.result()

    for i in range(5):
        sess = f"session-{i}"
        assert store.get_latest_sequence(sess) == 20
        assert len(store.get_events(sess)) == 20

    if isinstance(store, SQLiteEventStore):
        store.close()
