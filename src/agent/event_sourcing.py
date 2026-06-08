"""
阶段46: Agent事件溯源系统

遵循"图难于其易，为大于其细"的原则，实现事件存储、回放、聚合、投影。
"""
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

class EventStreamStatus(str, Enum):
    ACTIVE = "active"
    SEALED = "sealed"
    SNAPSHOT = "snapshot"

@dataclass
class DomainEvent:
    event_type: str
    aggregate_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    version: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id, "event_type": self.event_type,
            "aggregate_id": self.aggregate_id, "version": self.version,
            "payload": self.payload, "timestamp": self.timestamp,
        }

@dataclass
class EventStream:
    stream_id: str
    aggregate_type: str = ""
    events: List[DomainEvent] = field(default_factory=list)
    status: EventStreamStatus = EventStreamStatus.ACTIVE
    version: int = 0

    def append(self, event: DomainEvent) -> bool:
        if self.status == EventStreamStatus.SEALED:
            return False
        self.version += 1
        event.version = self.version
        event.aggregate_id = self.stream_id
        self.events.append(event)
        return True

class EventStore:
    """事件存储"""
    def __init__(self):
        self._streams: Dict[str, EventStream] = {}
        self._lock = threading.RLock()
        self._projections: Dict[str, Callable[[DomainEvent], None]] = {}
        self._global_events: List[DomainEvent] = []

    def create_stream(self, stream_id: str, aggregate_type: str = "") -> EventStream:
        with self._lock:
            stream = EventStream(stream_id=stream_id, aggregate_type=aggregate_type)
            self._streams[stream_id] = stream
            return stream

    def append_event(self, stream_id: str, event_type: str,
                      payload: Optional[Dict] = None) -> Optional[DomainEvent]:
        with self._lock:
            stream = self._streams.get(stream_id)
            if not stream or stream.status == EventStreamStatus.SEALED:
                return None
            event = DomainEvent(event_type=event_type, aggregate_id=stream_id,
                                payload=payload or {})
            stream.append(event)
            self._global_events.append(event)
            self._notify_projections(event)
            return event

    def get_stream(self, stream_id: str) -> Optional[EventStream]:
        return self._streams.get(stream_id)

    def get_events(self, stream_id: str, from_version: int = 0) -> List[DomainEvent]:
        stream = self._streams.get(stream_id)
        if not stream:
            return []
        return [e for e in stream.events if e.version > from_version]

    def get_all_events(self) -> List[DomainEvent]:
        with self._lock:
            return list(self._global_events)

    def seal_stream(self, stream_id: str) -> bool:
        with self._lock:
            stream = self._streams.get(stream_id)
            if stream:
                stream.status = EventStreamStatus.SEALED
                return True
            return False

    def register_projection(self, name: str, handler: Callable[[DomainEvent], None]) -> None:
        self._projections[name] = handler

    def _notify_projections(self, event: DomainEvent) -> None:
        for handler in self._projections.values():
            try:
                handler(event)
            except Exception:
                pass

    def replay(self, stream_id: str, handler: Callable[[DomainEvent], None]) -> int:
        events = self.get_events(stream_id)
        for event in events:
            try:
                handler(event)
            except Exception:
                pass
        return len(events)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_streams": len(self._streams),
                "total_events": len(self._global_events),
                "projections": list(self._projections.keys()),
            }

def create_event_store() -> EventStore:
    return EventStore()
