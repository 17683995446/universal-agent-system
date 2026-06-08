"""阶段57: Agent分布式屏障 - 单元测试"""
import time
import threading
import pytest
from src.agent.distributed_barrier import (
    BarrierStatus, BarrierParticipant, DistributedBarrier, CyclicBarrier,
    BarrierManager, create_barrier_manager,
)

class TestDistributedBarrier:
    def test_register_and_arrive(self):
        b = DistributedBarrier("b1", parties=2)
        b.register("p1")
        b.register("p2")
        results = []
        def p1_arrive():
            results.append(b.arrive("p1"))
        t = threading.Thread(target=p1_arrive)
        t.start()
        time.sleep(0.05)
        assert b.get_waiting_count() == 1
        b.arrive("p2")
        t.join(timeout=1.0)
        assert b.get_status() == BarrierStatus.TRIGGERED

    def test_timeout(self):
        b = DistributedBarrier("b1", parties=2, timeout=0.05)
        b.register("p1")
        assert not b.arrive("p1")
        assert b.get_status() == BarrierStatus.BROKEN

    def test_action(self):
        actions = []
        b = DistributedBarrier("b1", parties=2)
        b.set_action(lambda: actions.append("triggered"))
        b.register("p1")
        b.register("p2")
        def arrive_p1():
            b.arrive("p1")
        t = threading.Thread(target=arrive_p1)
        t.start()
        time.sleep(0.01)
        b.arrive("p2")
        t.join(timeout=1.0)
        assert "triggered" in actions

    def test_reset(self):
        b = DistributedBarrier("b1", parties=2)
        b.register("p1")
        b.register("p2")
        b.reset()
        assert b.get_status() == BarrierStatus.WAITING
        assert b.get_waiting_count() == 0

    def test_stats(self):
        b = DistributedBarrier("b1", parties=3)
        b.register("p1")
        stats = b.get_stats()
        assert stats["parties"] == 3

class TestCyclicBarrier:
    def test_cyclic(self):
        b = CyclicBarrier("b1", parties=2)
        b.register("p1")
        b.register("p2")
        # Just verify it exists and can be reset
        b.reset()
        assert b.get_status() == BarrierStatus.WAITING

class TestBarrierManager:
    def test_create_barrier(self):
        m = BarrierManager()
        b = m.create_barrier("b1", 3)
        assert isinstance(b, DistributedBarrier)

    def test_create_cyclic(self):
        m = BarrierManager()
        b = m.create_cyclic_barrier("b1", 3)
        assert isinstance(b, CyclicBarrier)

    def test_create_barrier_manager(self):
        m = create_barrier_manager()
        assert isinstance(m, BarrierManager)
