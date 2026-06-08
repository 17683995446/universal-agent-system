"""
阶段51: Agent分布式锁系统

遵循"图难于其易，为大于其细"的原则，实现分布式锁、租约、重入、公平锁。
"""
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

class LockType(str, Enum):
    MUTEX = "mutex"
    REENTRANT = "reentrant"
    FAIR = "fair"

class LockStatus(str, Enum):
    FREE = "free"
    LOCKED = "locked"
    EXPIRED = "expired"

@dataclass
class LockLease:
    lock_id: str
    owner: str
    acquired_at: float = field(default_factory=time.time)
    ttl: float = 30.0
    reentrant_count: int = 1

    @property
    def is_expired(self) -> bool:
        return time.time() - self.acquired_at > self.ttl

    @property
    def remaining(self) -> float:
        return max(0, self.ttl - (time.time() - self.acquired_at))

class DistributedLock:
    """分布式锁（本地模拟）"""
    def __init__(self, lock_id: str, lock_type: LockType = LockType.MUTEX,
                 ttl: float = 30.0, fair: bool = False):
        self.lock_id = lock_id
        self.lock_type = lock_type
        self.ttl = ttl
        self.fair = fair
        self._lease: Optional[LockLease] = None
        self._wait_queue: List[Tuple[str, threading.Event]] = []
        self._lock = threading.RLock()

    def acquire(self, owner: str, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        deadline = time.time() + timeout if timeout else None
        while True:
            with self._lock:
                # 清理过期
                if self._lease and self._lease.is_expired:
                    self._lease = None

                # 重入
                if self.lock_type == LockType.REENTRANT and self._lease and self._lease.owner == owner:
                    self._lease.reentrant_count += 1
                    return True

                if self._lease is None:
                    self._lease = LockLease(lock_id=self.lock_id, owner=owner, ttl=self.ttl)
                    return True

                if not blocking:
                    return False

                # 公平锁等待
                if self.fair:
                    event = threading.Event()
                    self._wait_queue.append((owner, event))
                    wait_lock = self._lock
                    break

            # 非公平锁轮询
            if deadline and time.time() >= deadline:
                return False
            time.sleep(0.01)

        # 公平锁等待
        if self.fair:
            remaining = timeout
            if deadline:
                remaining = deadline - time.time()
            return event.wait(timeout=remaining)
        return False

    def release(self, owner: str) -> bool:
        with self._lock:
            if not self._lease or self._lease.owner != owner:
                return False
            if self.lock_type == LockType.REENTRANT and self._lease.reentrant_count > 1:
                self._lease.reentrant_count -= 1
                return True
            self._lease = None
            # 通知等待者
            if self._wait_queue:
                _, event = self._wait_queue.pop(0)
                event.set()
            return True

    def extend(self, owner: str, additional_ttl: float) -> bool:
        with self._lock:
            if not self._lease or self._lease.owner != owner:
                return False
            self._lease.ttl += additional_ttl
            return True

    @property
    def is_locked(self) -> bool:
        with self._lock:
            if self._lease and self._lease.is_expired:
                self._lease = None
            return self._lease is not None

    @property
    def owner(self) -> Optional[str]:
        return self._lease.owner if self._lease else None

class LockManager:
    """锁管理器"""
    def __init__(self):
        self._locks: Dict[str, DistributedLock] = {}
        self._lock = threading.RLock()

    def get_lock(self, lock_id: str, lock_type: LockType = LockType.MUTEX,
                 ttl: float = 30.0) -> DistributedLock:
        with self._lock:
            if lock_id not in self._locks:
                self._locks[lock_id] = DistributedLock(lock_id, lock_type, ttl)
            return self._locks[lock_id]

    def remove_lock(self, lock_id: str) -> bool:
        with self._lock:
            return self._locks.pop(lock_id, None) is not None

    def list_locks(self) -> List[str]:
        return list(self._locks.keys())

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_locks": len(self._locks),
                "locked_count": sum(1 for l in self._locks.values() if l.is_locked),
            }

def create_lock_manager() -> LockManager:
    return LockManager()
