"""阶段46: Agent事件溯源系统 - 单元测试"""
import pytest
from src.agent.event_sourcing import (
    EventStreamStatus, DomainEvent, EventStream, EventStore, create_event_store,
)

class TestEventStore:
    def test_create_stream_and_append(self):
        store = EventStore()
        store.create_stream("order-1", "Order")
        e = store.append_event("order-1", "OrderCreated", {"item": "book"})
        assert e is not None
        assert e.version == 1

    def test_multiple_events(self):
        store = EventStore()
        store.create_stream("s1")
        store.append_event("s1", "Created")
        store.append_event("s1", "Updated")
        store.append_event("s1", "Completed")
        events = store.get_events("s1")
        assert len(events) == 3

    def test_from_version(self):
        store = EventStore()
        store.create_stream("s1")
        store.append_event("s1", "E1")
        store.append_event("s1", "E2")
        store.append_event("s1", "E3")
        events = store.get_events("s1", from_version=2)
        assert len(events) == 1

    def test_sealed_stream(self):
        store = EventStore()
        store.create_stream("s1")
        store.seal_stream("s1")
        assert store.append_event("s1", "E1") is None

    def test_projection(self):
        received = []
        store = EventStore()
        store.register_projection("log", lambda e: received.append(e.event_type))
        store.create_stream("s1")
        store.append_event("s1", "Created")
        assert "Created" in received

    def test_replay(self):
        states = []
        store = EventStore()
        store.create_stream("s1")
        store.append_event("s1", "Created", {"value": 0})
        store.append_event("s1", "Updated", {"value": 10})
        count = store.replay("s1", lambda e: states.append(e.payload.get("value")))
        assert count == 2
        assert states == [0, 10]

    def test_nonexistent_stream(self):
        store = EventStore()
        assert store.get_events("nope") == []

    def test_get_all_events(self):
        store = EventStore()
        store.create_stream("s1")
        store.create_stream("s2")
        store.append_event("s1", "E1")
        store.append_event("s2", "E2")
        assert len(store.get_all_events()) == 2

    def test_stats(self):
        store = EventStore()
        store.create_stream("s1")
        store.append_event("s1", "E1")
        stats = store.get_stats()
        assert stats["total_streams"] == 1
        assert stats["total_events"] == 1

    def test_event_to_dict(self):
        e = DomainEvent(event_type="Created", aggregate_id="s1")
        d = e.to_dict()
        assert d["event_type"] == "Created"

    def test_create_event_store(self):
        s = create_event_store()
        assert isinstance(s, EventStore)
