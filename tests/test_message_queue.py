"""
阶段35测试: Agent消息队列系统
"""

import os
import shutil
import tempfile
import threading
import time
import pytest
from datetime import datetime

from src.agent.message_queue import (
    MessagePriority,
    MessageStatus,
    DeliveryMode,
    Message,
    QueuedMessage,
    Consumer,
    QueueStats,
    MessageQueue,
    PubSubBroker,
    PersistentMessageStore,
    create_message,
    create_broker,
    create_persistent_store,
)


# ============================================================
# 模型测试
# ============================================================

class TestMessagePriority:
    def test_values(self):
        assert MessagePriority.LOWEST.value == 0
        assert MessagePriority.NORMAL.value == 2
        assert MessagePriority.HIGHEST.value == 4


class TestMessageStatus:
    def test_values(self):
        assert MessageStatus.PENDING.value == "pending"
        assert MessageStatus.DELIVERED.value == "delivered"
        assert MessageStatus.ACKNOWLEDGED.value == "acknowledged"


class TestDeliveryMode:
    def test_values(self):
        assert DeliveryMode.DIRECT.value == "direct"
        assert DeliveryMode.FANOUT.value == "fanout"
        assert DeliveryMode.TOPIC.value == "topic"


class TestMessage:
    def test_creation(self):
        msg = create_message(payload={"data": 123}, priority=MessagePriority.HIGH)
        assert msg.payload == {"data": 123}
        assert msg.priority == MessagePriority.HIGH
        assert msg.message_id is not None

    def test_to_dict(self):
        msg = create_message(payload="test", routing_key="key1")
        d = msg.to_dict()
        assert d["payload"] == "test"
        assert d["routing_key"] == "key1"


class TestQueuedMessage:
    def test_creation(self):
        msg = create_message("test")
        qm = QueuedMessage(message=msg)
        assert qm.status == MessageStatus.PENDING
        assert qm.delivery_count == 0


class TestQueueStats:
    def test_to_dict(self):
        stats = QueueStats(name="q1", message_count=5, consumer_count=2)
        d = stats.to_dict()
        assert d["name"] == "q1"
        assert d["message_count"] == 5


# ============================================================
# MessageQueue 测试
# ============================================================

class TestMessageQueue:
    def setup_method(self):
        self.queue = MessageQueue("test_queue", max_size=100)

    def test_enqueue(self):
        msg = create_message("hello")
        assert self.queue.enqueue(msg) is True
        assert self.queue.stats.message_count == 1

    def test_enqueue_full(self):
        q = MessageQueue("small", max_size=2)
        q.enqueue(create_message("a"))
        q.enqueue(create_message("b"))
        assert q.enqueue(create_message("c")) is False

    def test_dequeue(self):
        msg = create_message("hello")
        self.queue.enqueue(msg)
        qm = self.queue.dequeue("consumer1")
        assert qm is not None
        assert qm.message.payload == "hello"
        assert qm.status == MessageStatus.DELIVERED
        assert qm.consumer_tag == "consumer1"

    def test_dequeue_empty(self):
        assert self.queue.dequeue("c1") is None

    def test_priority_order(self):
        low = create_message("low", priority=MessagePriority.LOW)
        high = create_message("high", priority=MessagePriority.HIGH)
        normal = create_message("normal", priority=MessagePriority.NORMAL)

        self.queue.enqueue(low)
        self.queue.enqueue(high)
        self.queue.enqueue(normal)

        # 高优先级先出
        assert self.queue.dequeue("c1").message.payload == "high"
        assert self.queue.dequeue("c2").message.payload == "low"
        assert self.queue.dequeue("c3").message.payload == "normal"

    def test_ack(self):
        msg = create_message("test")
        self.queue.enqueue(msg)
        qm = self.queue.dequeue("c1")
        assert self.queue.ack(qm.message.message_id) is True
        assert self.queue.stats.acknowledged_count == 1

    def test_ack_not_found(self):
        assert self.queue.ack("nonexistent") is False

    def test_nack_requeue(self):
        msg = create_message("test")
        self.queue.enqueue(msg)
        qm = self.queue.dequeue("c1")
        assert self.queue.nack(qm.message.message_id, requeue=True) is True
        # 重新入队
        assert self.queue.stats.message_count == 1

    def test_nack_no_requeue(self):
        msg = create_message("test")
        self.queue.enqueue(msg)
        qm = self.queue.dequeue("c1")
        assert self.queue.nack(qm.message.message_id, requeue=False) is True
        assert self.queue.stats.rejected_count == 1

    def test_max_delivery_attempts(self):
        q = MessageQueue("q", max_delivery_attempts=2, dead_letter_queue="dlq")
        msg = create_message("test")
        q.enqueue(msg)

        # 第1次投递（使用auto_ack=False确保消息进入_unacked）
        qm1 = q.dequeue("c1", auto_ack=False)
        assert qm1 is not None
        assert qm1.delivery_count == 1
        q.nack(qm1.message.message_id, requeue=True)
        
        # 第2次投递
        qm2 = q.dequeue("c1", auto_ack=False)
        assert qm2 is not None
        assert qm2.delivery_count == 2
        q.nack(qm2.message.message_id, requeue=True)
        
        # 第3次nack（超过限制，不会重新入队，进入死信）
        q.nack(qm2.message.message_id, requeue=True)

        # 超过重试次数，进入死信
        assert q.stats.dead_letter_count == 1

    def test_consumer_management(self):
        consumer = Consumer("c1", "test_queue", lambda m: True)
        self.queue.add_consumer(consumer)
        assert len(self.queue.get_consumers()) == 1

        assert self.queue.remove_consumer("c1") is True
        assert len(self.queue.get_consumers()) == 0

    def test_stats(self):
        msg = create_message("test")
        self.queue.enqueue(msg)
        stats = self.queue.stats
        assert stats.name == "test_queue"
        assert stats.message_count == 1


# ============================================================
# PubSubBroker 测试
# ============================================================

class TestPubSubBroker:
    def setup_method(self):
        self.broker = create_broker()

    def test_declare_queue(self):
        q = self.broker.declare_queue("q1")
        assert q.name == "q1"
        assert "q1" in self.broker._queues

    def test_delete_queue(self):
        self.broker.declare_queue("q1")
        assert self.broker.delete_queue("q1") is True
        assert self.broker.delete_queue("q1") is False

    def test_declare_exchange(self):
        self.broker.declare_exchange("ex1", DeliveryMode.FANOUT)
        assert self.broker._exchanges["ex1"] == DeliveryMode.FANOUT

    def test_bind_queue(self):
        self.broker.declare_queue("q1")
        self.broker.declare_exchange("ex1", DeliveryMode.DIRECT)
        assert self.broker.bind_queue("ex1", "q1") is True
        assert "q1" in self.broker._bindings["ex1"]

    def test_bind_nonexistent_queue(self):
        assert self.broker.bind_queue("ex1", "nonexistent") is False

    def test_publish_direct(self):
        self.broker.declare_queue("q1")
        self.broker.bind_queue("ex1", "q1")
        msg = create_message("hello")
        delivered = self.broker.publish("ex1", msg)
        assert delivered == 1

    def test_publish_fanout(self):
        self.broker.declare_queue("q1")
        self.broker.declare_queue("q2")
        self.broker.declare_exchange("ex1", DeliveryMode.FANOUT)
        self.broker.bind_queue("ex1", "q1")
        self.broker.bind_queue("ex1", "q2")

        msg = create_message("broadcast")
        delivered = self.broker.publish("ex1", msg)
        assert delivered == 2

    def test_publish_auto_create_exchange(self):
        self.broker.declare_queue("q1")
        self.broker.bind_queue("auto_ex", "q1")
        delivered = self.broker.publish("auto_ex", create_message("test"))
        assert delivered == 1

    def test_subscribe_unsubscribe(self):
        self.broker.subscribe("topic1", "consumer1")
        assert "consumer1" in self.broker._subscriptions["topic1"]

        assert self.broker.unsubscribe("topic1", "consumer1") is True
        assert "consumer1" not in self.broker._subscriptions["topic1"]

    def test_consume(self):
        self.broker.declare_queue("q1")
        received = []

        def callback(msg):
            received.append(msg.payload)
            return True

        assert self.broker.consume("q1", "c1", callback) is True
        assert len(self.broker._queues["q1"].get_consumers()) == 1

    def test_consume_nonexistent_queue(self):
        assert self.broker.consume("nonexistent", "c1", lambda m: True) is False

    def test_cancel_consume(self):
        self.broker.declare_queue("q1")
        self.broker.consume("q1", "c1", lambda m: True)
        assert self.broker.cancel_consume("q1", "c1") is True
        assert self.broker.cancel_consume("q1", "c1") is False

    def test_ack_nack_through_broker(self):
        self.broker.declare_queue("q1")
        msg = create_message("test")
        self.broker._queues["q1"].enqueue(msg)
        qm = self.broker._queues["q1"].dequeue("c1")

        assert self.broker.ack("q1", qm.message.message_id) is True
        assert self.broker.nack("q1", "nonexistent") is False

    def test_get_queue_stats(self):
        self.broker.declare_queue("q1")
        self.broker._queues["q1"].enqueue(create_message("test"))

        stats = self.broker.get_queue_stats("q1")
        assert stats is not None
        assert stats.message_count == 1

        assert self.broker.get_queue_stats("nonexistent") is None

    def test_get_all_stats(self):
        self.broker.declare_queue("q1")
        self.broker.declare_queue("q2")
        stats = self.broker.get_all_stats()
        assert len(stats) == 2

    def test_delivery_thread(self):
        self.broker.declare_queue("q1")
        self.broker.bind_queue("q1", "q1")  # 绑定队列到同名交换器
        received = []

        def callback(msg):
            received.append(msg.payload)
            return True

        self.broker.consume("q1", "c1", callback, auto_ack=True)
        self.broker.publish("q1", create_message("hello"))

        self.broker.start_delivery()
        time.sleep(0.1)
        self.broker.stop_delivery()

        assert "hello" in received


# ============================================================
# PersistentMessageStore 测试
# ============================================================

class TestPersistentMessageStore:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store = create_persistent_store(self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_save_and_load(self):
        msg = create_message(payload={"key": "value"}, priority=MessagePriority.HIGH)
        assert self.store.save_message("q1", msg) is True

        messages = self.store.load_messages("q1")
        assert len(messages) == 1
        assert messages[0].payload == {"key": "value"}
        assert messages[0].priority == MessagePriority.HIGH

    def test_load_empty_queue(self):
        messages = self.store.load_messages("nonexistent")
        assert messages == []

    def test_delete_message(self):
        msg = create_message("test")
        self.store.save_message("q1", msg)
        assert self.store.delete_message("q1", msg.message_id) is True
        assert self.store.delete_message("q1", msg.message_id) is False

    def test_multiple_messages(self):
        for i in range(5):
            msg = create_message(f"msg-{i}")
            self.store.save_message("q1", msg)

        messages = self.store.load_messages("q1")
        assert len(messages) == 5


# ============================================================
# 便捷函数测试
# ============================================================

class TestConvenienceFunctions:
    def test_create_message(self):
        msg = create_message("hello", headers={"h1": "v1"}, priority=MessagePriority.LOW)
        assert msg.payload == "hello"
        assert msg.headers["h1"] == "v1"

    def test_create_broker(self):
        broker = create_broker()
        assert isinstance(broker, PubSubBroker)

    def test_create_persistent_store(self):
        temp_dir = tempfile.mkdtemp()
        try:
            store = create_persistent_store(temp_dir)
            assert isinstance(store, PersistentMessageStore)
        finally:
            shutil.rmtree(temp_dir)


# ============================================================
# 集成测试
# ============================================================

class TestIntegration:
    def test_producer_consumer_flow(self):
        broker = create_broker()
        broker.declare_queue("orders")
        broker.bind_queue("orders", "orders")  # 绑定队列到交换器

        received = []

        def process_order(msg):
            received.append(msg.payload)
            return True

        broker.consume("orders", "processor1", process_order, auto_ack=True)
        broker.start_delivery()

        # 生产消息
        for i in range(5):
            broker.publish("orders", create_message(f"order-{i}"))

        time.sleep(0.2)
        broker.stop_delivery()

        assert len(received) == 5

    def test_manual_ack_flow(self):
        broker = create_broker()
        broker.declare_queue("tasks")
        broker.bind_queue("tasks", "tasks")  # 绑定队列到交换器

        processed = []
        message_ids = []

        def process_task(msg):
            processed.append(msg.payload)
            message_ids.append(msg.message_id)
            return True  # 业务成功

        broker.consume("tasks", "worker1", process_task, auto_ack=False)
        broker.start_delivery()

        msg = create_message("task-1")
        broker.publish("tasks", msg)

        time.sleep(0.1)
        broker.stop_delivery()

        # 验证消息已被处理（投递线程根据回调返回值自动确认）
        assert len(message_ids) == 1
        assert len(processed) == 1
        assert processed[0] == "task-1"
        # 由于回调返回True，消息已被自动确认
        stats = broker.get_queue_stats("tasks")
        assert stats.acknowledged_count == 1

    def test_priority_queue(self):
        broker = create_broker()
        broker.declare_queue("priority_tasks")
        broker.bind_queue("priority_tasks", "priority_tasks")  # 绑定队列到交换器

        results = []

        def process(msg):
            results.append(msg.payload)
            return True

        broker.consume("priority_tasks", "worker", process, auto_ack=True)
        broker.start_delivery()

        # 按不同优先级发送
        broker.publish("priority_tasks", create_message("low1", priority=MessagePriority.LOW))
        broker.publish("priority_tasks", create_message("high1", priority=MessagePriority.HIGH))
        broker.publish("priority_tasks", create_message("normal", priority=MessagePriority.NORMAL))
        broker.publish("priority_tasks", create_message("high2", priority=MessagePriority.HIGH))

        time.sleep(0.2)
        broker.stop_delivery()

        # 高优先级消息应该先被处理
        assert len(results) == 4
        assert results[0].startswith("high")
        assert results[1].startswith("high")

    def test_persistence_integration(self):
        temp_dir = tempfile.mkdtemp()
        try:
            store = create_persistent_store(temp_dir)
            broker = create_broker()
            broker.declare_queue("persistent_queue")
            broker.bind_queue("persistent_queue", "persistent_queue")

            # 保存消息到存储并发布到broker
            for i in range(3):
                msg = create_message(f"persisted-{i}")
                store.save_message("persistent_queue", msg)
                broker.publish("persistent_queue", msg)

            # 恢复并消费
            loaded = store.load_messages("persistent_queue")
            assert len(loaded) == 3

            received = []
            broker.consume("persistent_queue", "c1", lambda m: received.append(m.payload) or True, auto_ack=True)
            broker.start_delivery()
            time.sleep(0.2)
            broker.stop_delivery()

            assert len(received) == 3
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
