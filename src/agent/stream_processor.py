"""
阶段66: Agent数据流处理系统

遵循"图难于其易，为大于其细"的原则，实现流定义、转换、过滤、合并、分流。
"""
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

class StreamStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"

@dataclass
class StreamEvent:
    data: Any
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

class Stream:
    """数据流"""
    def __init__(self, stream_id: str, max_buffer: int = 1000):
        self.stream_id = stream_id
        self._buffer: deque = deque(maxlen=max_buffer)
        self._subscribers: List[Callable[[StreamEvent], None]] = []
        self._status = StreamStatus.ACTIVE
        self._lock = threading.RLock()

    def emit(self, data: Any, metadata: Optional[Dict] = None) -> None:
        with self._lock:
            if self._status != StreamStatus.ACTIVE:
                return
            event = StreamEvent(data=data, metadata=metadata or {})
            self._buffer.append(event)
            for sub in self._subscribers:
                try:
                    sub(event)
                except Exception:
                    pass

    def subscribe(self, callback: Callable[[StreamEvent], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[StreamEvent], None]) -> None:
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def get_events(self, count: int = 10) -> List[StreamEvent]:
        with self._lock:
            return list(self._buffer)[-count:]

    def pause(self) -> None:
        with self._lock:
            self._status = StreamStatus.PAUSED

    def resume(self) -> None:
        with self._lock:
            self._status = StreamStatus.ACTIVE

    def close(self) -> None:
        with self._lock:
            self._status = StreamStatus.CLOSED
            self._subscribers.clear()

class StreamProcessor:
    """流处理器"""
    def __init__(self):
        self._streams: Dict[str, Stream] = {}
        self._transforms: Dict[str, Callable[[Any], Any]] = {}
        self._lock = threading.RLock()

    def create_stream(self, stream_id: str, max_buffer: int = 1000) -> Stream:
        with self._lock:
            stream = Stream(stream_id, max_buffer)
            self._streams[stream_id] = stream
            return stream

    def get_stream(self, stream_id: str) -> Optional[Stream]:
        return self._streams.get(stream_id)

    def map_stream(self, source_id: str, target_id: str,
                   transform: Callable[[Any], Any]) -> Stream:
        """映射流：对源流的每个元素应用转换"""
        source = self._streams.get(source_id)
        if not source:
            raise ValueError(f"Source stream {source_id} not found")
        target = self.create_stream(target_id)
        source.subscribe(lambda event: target.emit(transform(event.data)))
        return target

    def filter_stream(self, source_id: str, target_id: str,
                      predicate: Callable[[Any], bool]) -> Stream:
        """过滤流：只传递满足条件的元素"""
        source = self._streams.get(source_id)
        if not source:
            raise ValueError(f"Source stream {source_id} not found")
        target = self.create_stream(target_id)
        source.subscribe(lambda event: target.emit(event.data) if predicate(event.data) else None)
        return target

    def merge_streams(self, source_ids: List[str], target_id: str) -> Stream:
        """合并流：将多个流合并为一个"""
        target = self.create_stream(target_id)
        for sid in source_ids:
            source = self._streams.get(sid)
            if source:
                source.subscribe(lambda event, t=target: t.emit(event.data))
        return target

    def split_stream(self, source_id: str, branches: Dict[str, Callable[[Any], bool]]) -> Dict[str, Stream]:
        """分流：根据条件将流分成多个分支"""
        source = self._streams.get(source_id)
        if not source:
            raise ValueError(f"Source stream {source_id} not found")
        result = {}
        for branch_id, predicate in branches.items():
            target = self.create_stream(branch_id)
            source.subscribe(lambda event, p=predicate, t=target: t.emit(event.data) if p(event.data) else None)
            result[branch_id] = target
        return result

    def list_streams(self) -> List[str]:
        return list(self._streams.keys())

    def remove_stream(self, stream_id: str) -> bool:
        with self._lock:
            stream = self._streams.pop(stream_id, None)
            if stream:
                stream.close()
                return True
            return False

def create_stream_processor() -> StreamProcessor:
    return StreamProcessor()
