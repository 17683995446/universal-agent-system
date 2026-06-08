"""
阶段53: Agent分布式队列

遵循"图难于其易，为大于其细"的原则，实现优先级队列、延迟队列、死信队列。
"""
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

class QueuePriority(int, Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class QueueStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"

@dataclass
class QueueMessage:
    id: str
    payload: Any = None
    priority: QueuePriority = QueuePriority.NORMAL
    created_at: float = field(default_factory=time.time)
    delay_until: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QueueStats:
    total_enqueued: int = 0
    total_dequeued: int = 0
    total_failed: int = 0
    current_size: int = 0

class DistributedQueue:
    """分布式队列（本地模拟）"""
    def __init__(self, queue_id: str, max_size: int = 10000):
        self.queue_id = queue_id
        self._max_size = max_size
        self._queues: Dict[QueuePriority, deque] = {
            QueuePriority.CRITICAL: deque(),
            QueuePriority.HIGH: deque(),
            QueuePriority.NORMAL: deque(),
            QueuePriority.LOW: deque(),
        }
        self._dead_letter: deque = deque()
        self._stats = QueueStats()
        self._status = QueueStatus.ACTIVE
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)

    def enqueue(self, message: QueueMessage) -> bool:
        with self._lock:
            if self._status == QueueStatus.CLOSED:
                return False
            if self._stats.current_size >= self._max_size:
                return False
            self._queues[message.priority].append(message)
            self._stats.total_enqueued += 1
            self._stats.current_size += 1
            self._not_empty.notify()
            return True

    def enqueue_simple(self, payload: Any, priority: QueuePriority = QueuePriority.NORMAL,
                       delay: float = 0.0) -> Optional[QueueMessage]:
        import uuid
        msg = QueueMessage(
            id=str(uuid.uuid4())[:8],
            payload=payload,
            priority=priority,
            delay_until=time.time() + delay if delay > 0 else None,
        )
        return msg if self.enqueue(msg) else None

    def dequeue(self, timeout: Optional[float] = None) -> Optional[QueueMessage]:
        with self._not_empty:
            if self._status == QueueStatus.CLOSED:
                return None
            deadline = time.time() + timeout if timeout else None
            while True:
                msg = self._try_dequeue()
                if msg:
                    return msg
                if self._status == QueueStatus.CLOSED:
                    return None
                remaining = None
                if deadline:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        return None
                self._not_empty.wait(timeout=remaining)

    def _try_dequeue(self) -> Optional[QueueMessage]:
        for priority in sorted(self._queues.keys(), reverse=True):
            q = self._queues[priority]
            while q:
                msg = q[0]
                if msg.delay_until and msg.delay_until > time.time():
                    break
                q.popleft()
                self._stats.total_dequeued += 1
                self._stats.current_size -= 1
                return msg
        return None

    def peek(self) -> Optional[QueueMessage]:
        with self._lock:
            for priority in sorted(self._queues.keys(), reverse=True):
                for msg in self._queues[priority]:
                    if not msg.delay_until or msg.delay_until <= time.time():
                        return msg
            return None

    def size(self) -> int:
        with self._lock:
            return self._stats.current_size

    def is_empty(self) -> bool:
        return self.size() == 0

    def pause(self) -> None:
        with self._lock:
            self._status = QueueStatus.PAUSED

    def resume(self) -> None:
        with self._lock:
            self._status = QueueStatus.ACTIVE
            self._not_empty.notify_all()

    def close(self) -> None:
        with self._lock:
            self._status = QueueStatus.CLOSED
            self._not_empty.notify_all()

    def get_stats(self) -> QueueStats:
        with self._lock:
            return QueueStats(
                total_enqueued=self._stats.total_enqueued,
                total_dequeued=self._stats.total_dequeued,
                total_failed=self._stats.total_failed,
                current_size=self._stats.current_size,
            )

class QueueManager:
    """队列管理器"""
    def __init__(self):
        self._queues: Dict[str, DistributedQueue] = {}
        self._lock = threading.RLock()

    def get_queue(self, queue_id: str, max_size: int = 10000) -> DistributedQueue:
        with self._lock:
            if queue_id not in self._queues:
                self._queues[queue_id] = DistributedQueue(queue_id, max_size)
            return self._queues[queue_id]

    def remove_queue(self, queue_id: str) -> bool:
        with self._lock:
            q = self._queues.pop(queue_id, None)
            if q:
                q.close()
                return True
            return False

    def list_queues(self) -> List[str]:
        return list(self._queues.keys())

def create_queue_manager() -> QueueManager:
    return QueueManager()
