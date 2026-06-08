"""阶段51: Agent分布式锁系统 - 单元测试"""
import pytest
import threading
import time
from src.agent.distributed_lock import (
    LockType, LockStatus, LockLease, DistributedLock, LockManager, create_lock_manager,
)

class TestLockLease:
    def test_expiry(self):
        lease = LockLease(lock_id="l1", owner="a", ttl=0.01)
        time.sleep(0.02)
        assert lease.is_expired

    def test_remaining(self):
        lease = LockLease(lock_id="l1", owner="a", ttl=10)
        assert 0 < lease.remaining <= 10

class TestDistributedLock:
    def test_acquire_and_release(self):
        lock = DistributedLock("l1")
        assert lock.acquire("a1")
        assert lock.is_locked
        assert lock.owner == "a1"
        assert lock.release("a1")
        assert not lock.is_locked

    def test_acquire_timeout(self):
        lock = DistributedLock("l1")
        lock.acquire("a1")
        assert not lock.acquire("a2", blocking=True, timeout=0.01)

    def test_reentrant(self):
        lock = DistributedLock("l1", lock_type=LockType.REENTRANT)
        assert lock.acquire("a1")
        assert lock.acquire("a1")
        assert lock.release("a1")
        assert lock.is_locked
        assert lock.release("a1")
        assert not lock.is_locked

    def test_release_wrong_owner(self):
        lock = DistributedLock("l1")
        lock.acquire("a1")
        assert not lock.release("a2")

    def test_extend(self):
        lock = DistributedLock("l1")
        lock.acquire("a1")
        assert lock.extend("a1", 60)

    def test_extend_wrong_owner(self):
        lock = DistributedLock("l1")
        lock.acquire("a1")
        assert not lock.extend("a2", 60)

    def test_expiry_auto_release(self):
        lock = DistributedLock("l1", ttl=0.01)
        lock.acquire("a1")
        time.sleep(0.02)
        assert not lock.is_locked

    def test_nonblocking(self):
        lock = DistributedLock("l1")
        lock.acquire("a1")
        assert not lock.acquire("a2", blocking=False)

class TestLockManager:
    def test_get_lock(self):
        m = LockManager()
        lock = m.get_lock("l1")
        assert isinstance(lock, DistributedLock)
        assert "l1" in m.list_locks()

    def test_remove_lock(self):
        m = LockManager()
        m.get_lock("l1")
        assert m.remove_lock("l1")
        assert "l1" not in m.list_locks()

    def test_stats(self):
        m = LockManager()
        lock = m.get_lock("l1")
        lock.acquire("a1")
        stats = m.get_stats()
        assert stats["locked_count"] == 1

    def test_create_lock_manager(self):
        m = create_lock_manager()
        assert isinstance(m, LockManager)
