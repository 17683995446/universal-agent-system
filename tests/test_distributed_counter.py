"""阶段52: Agent分布式计数器 - 单元测试"""
import pytest
from src.agent.distributed_counter import (
    CounterValue, DistributedCounter, CounterManager, create_counter_manager,
)

class TestDistributedCounter:
    def test_increment(self):
        c = DistributedCounter("c1")
        assert c.increment() == 1
        assert c.increment(5) == 6

    def test_decrement(self):
        c = DistributedCounter("c1", 10)
        assert c.decrement() == 9
        assert c.decrement(3) == 6

    def test_get(self):
        c = DistributedCounter("c1", 42)
        assert c.get() == 42

    def test_reset(self):
        c = DistributedCounter("c1", 10)
        assert c.reset(0) == 10
        assert c.get() == 0

    def test_compare_and_set(self):
        c = DistributedCounter("c1", 5)
        assert c.compare_and_set(5, 10)
        assert c.get() == 10
        assert not c.compare_and_set(5, 20)

    def test_get_value(self):
        c = DistributedCounter("c1", 0)
        c.increment(10)
        c.decrement(3)
        v = c.get_value()
        assert v.value == 7
        assert v.max_value == 10
        assert v.min_value == 0
        assert v.total_increments == 1
        assert v.total_decrements == 1

class TestCounterManager:
    def test_get_counter(self):
        m = CounterManager()
        c = m.get_counter("c1")
        assert isinstance(c, DistributedCounter)

    def test_remove_counter(self):
        m = CounterManager()
        m.get_counter("c1")
        assert m.remove_counter("c1")

    def test_all_values(self):
        m = CounterManager()
        m.get_counter("a", 1)
        m.get_counter("b", 2)
        assert m.get_all_values() == {"a": 1, "b": 2}

    def test_stats(self):
        m = CounterManager()
        m.get_counter("a", 5)
        stats = m.get_stats()
        assert stats["total_counters"] == 1
        assert stats["total_values"] == 5

    def test_create_counter_manager(self):
        m = create_counter_manager()
        assert isinstance(m, CounterManager)
