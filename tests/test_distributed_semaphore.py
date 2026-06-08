"""阶段58: Agent分布式信号量 - 单元测试"""
import time
import threading
import pytest
from src.agent.distributed_semaphore import (
    DistributedSemaphore, SemaphoreManager, create_semaphore_manager,
)

class TestDistributedSemaphore:
    def test_acquire_release(self):
        s = DistributedSemaphore("s1", 2)
        assert s.acquire()
        assert s.available_permits() == 1
        s.release()
        assert s.available_permits() == 2

    def test_acquire_multiple(self):
        s = DistributedSemaphore("s1", 3)
        assert s.acquire(permits=2)
        assert s.available_permits() == 1

    def test_nonblocking(self):
        s = DistributedSemaphore("s1", 1)
        s.acquire()
        assert not s.acquire(blocking=False)

    def test_timeout(self):
        s = DistributedSemaphore("s1", 1)
        s.acquire()
        assert not s.acquire(timeout=0.05)

    def test_drain(self):
        s = DistributedSemaphore("s1", 5)
        assert s.drain_permits() == 5
        assert s.available_permits() == 0

    def test_reduce(self):
        s = DistributedSemaphore("s1", 5)
        s.reduce_permits(3)
        assert s.available_permits() == 2

class TestSemaphoreManager:
    def test_get_semaphore(self):
        m = SemaphoreManager()
        s = m.get_semaphore("s1", 5)
        assert isinstance(s, DistributedSemaphore)

    def test_remove_semaphore(self):
        m = SemaphoreManager()
        m.get_semaphore("s1")
        assert m.remove_semaphore("s1")

    def test_create_semaphore_manager(self):
        m = create_semaphore_manager()
        assert isinstance(m, SemaphoreManager)
