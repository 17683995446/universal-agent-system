"""阶段53: Agent分布式队列 - 单元测试"""
import time
import pytest
from src.agent.distributed_queue import (
    QueuePriority, QueueStatus, QueueMessage, QueueStats,
    DistributedQueue, QueueManager, create_queue_manager,
)

class TestDistributedQueue:
    def test_enqueue_dequeue(self):
        q = DistributedQueue("q1")
        q.enqueue_simple("hello")
        msg = q.dequeue()
        assert msg is not None
        assert msg.payload == "hello"

    def test_priority_order(self):
        q = DistributedQueue("q1")
        q.enqueue_simple("low", priority=QueuePriority.LOW)
        q.enqueue_simple("high", priority=QueuePriority.HIGH)
        q.enqueue_simple("normal", priority=QueuePriority.NORMAL)
        assert q.dequeue().payload == "high"
        assert q.dequeue().payload == "normal"
        assert q.dequeue().payload == "low"

    def test_delay(self):
        q = DistributedQueue("q1")
        q.enqueue_simple("delayed", delay=0.05)
        assert q.dequeue(timeout=0.01) is None
        time.sleep(0.06)
        assert q.dequeue(timeout=0.1) is not None

    def test_timeout(self):
        q = DistributedQueue("q1")
        assert q.dequeue(timeout=0.01) is None

    def test_max_size(self):
        q = DistributedQueue("q1", max_size=2)
        q.enqueue_simple("a")
        q.enqueue_simple("b")
        assert q.enqueue_simple("c") is None

    def test_close(self):
        q = DistributedQueue("q1")
        q.close()
        assert q.enqueue_simple("a") is None
        assert q.dequeue() is None

    def test_stats(self):
        q = DistributedQueue("q1")
        q.enqueue_simple("a")
        q.dequeue()
        stats = q.get_stats()
        assert stats.total_enqueued == 1
        assert stats.total_dequeued == 1

class TestQueueManager:
    def test_get_queue(self):
        m = QueueManager()
        q = m.get_queue("q1")
        assert isinstance(q, DistributedQueue)

    def test_remove_queue(self):
        m = QueueManager()
        m.get_queue("q1")
        assert m.remove_queue("q1")

    def test_create_queue_manager(self):
        m = create_queue_manager()
        assert isinstance(m, QueueManager)
