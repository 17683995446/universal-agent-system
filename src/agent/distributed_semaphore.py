"""
阶段58: Agent分布式信号量

遵循"图难于其易，为大于其细"的原则，实现信号量获取、释放、超时、公平调度。
"""
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

class DistributedSemaphore:
    """分布式信号量（本地模拟）"""
    def __init__(self, semaphore_id: str, max_permits: int, fair: bool = False):
        self.semaphore_id = semaphore_id
        self.max_permits = max_permits
        self.fair = fair
        self._available = max_permits
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._waiters: deque = deque()

    def acquire(self, permits: int = 1, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        with self._condition:
            if self.fair and not blocking:
                if self._available >= permits and not self._waiters:
                    self._available -= permits
                    return True
                return False
            if not blocking:
                if self._available >= permits:
                    self._available -= permits
                    return True
                return False
            # blocking acquire
            if self.fair:
                my_event = threading.Event()
                self._waiters.append(my_event)
                while True:
                    if self._available >= permits and self._waiters[0] is my_event:
                        self._available -= permits
                        self._waiters.popleft()
                        return True
                    remaining = None
                    if timeout:
                        remaining = timeout - (time.time() - start)
                        if remaining <= 0:
                            self._waiters.remove(my_event)
                            return False
                    self._condition.wait(timeout=remaining)
            else:
                deadline = time.time() + timeout if timeout else None
                while self._available < permits:
                    remaining = None
                    if deadline:
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            return False
                    self._condition.wait(timeout=remaining)
                self._available -= permits
                return True

    def release(self, permits: int = 1) -> None:
        with self._condition:
            self._available = min(self.max_permits, self._available + permits)
            self._condition.notify_all()

    def available_permits(self) -> int:
        with self._lock:
            return self._available

    def drain_permits(self) -> int:
        with self._lock:
            drained = self._available
            self._available = 0
            return drained

    def reduce_permits(self, reduction: int) -> None:
        with self._lock:
            self._available = max(0, self._available - reduction)

class SemaphoreManager:
    """信号量管理器"""
    def __init__(self):
        self._semaphores: Dict[str, DistributedSemaphore] = {}
        self._lock = threading.RLock()

    def get_semaphore(self, semaphore_id: str, max_permits: int = 10,
                      fair: bool = False) -> DistributedSemaphore:
        with self._lock:
            if semaphore_id not in self._semaphores:
                self._semaphores[semaphore_id] = DistributedSemaphore(semaphore_id, max_permits, fair)
            return self._semaphores[semaphore_id]

    def remove_semaphore(self, semaphore_id: str) -> bool:
        with self._lock:
            return self._semaphores.pop(semaphore_id, None) is not None

    def list_semaphores(self) -> List[str]:
        return list(self._semaphores.keys())

def create_semaphore_manager() -> SemaphoreManager:
    return SemaphoreManager()
