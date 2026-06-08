"""
阶段35: Agent消息队列系统

遵循"图难于其易，为大于其细"的原则，实现异步消息传递、发布订阅、消息持久化。
"""

import json
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ============================================================
# 35.1 消息队列模型
# ============================================================

class MessagePriority(int, Enum):
    """消息优先级"""
    LOWEST = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    HIGHEST = 4


class MessageStatus(str, Enum):
    """消息状态"""
    PENDING = "pending"       # 待投递
    DELIVERED = "delivered"   # 已投递
    ACKNOWLEDGED = "acknowledged"  # 已确认
    REJECTED = "rejected"     # 已拒绝
    EXPIRED = "expired"       # 已过期
    DEAD_LETTER = "dead_letter"  # 死信


class DeliveryMode(str, Enum):
    """投递模式"""
    DIRECT = "direct"         # 直接投递到队列
    FANOUT = "fanout"         # 广播到所有队列
    TOPIC = "topic"           # 主题路由
    HEADERS = "headers"       # 头部匹配


@dataclass
class Message:
    """消息"""
    message_id: str
    payload: Any
    headers: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    expiration_seconds: Optional[float] = None
    delivery_mode: DeliveryMode = DeliveryMode.DIRECT
    routing_key: str = ""
    correlation_id: str = ""
    reply_to: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "payload": self.payload,
            "headers": self.headers,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "expiration_seconds": self.expiration_seconds,
            "delivery_mode": self.delivery_mode.value,
            "routing_key": self.routing_key,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
        }


@dataclass
class QueuedMessage:
    """队列中的消息（包装）"""
    message: Message
    status: MessageStatus = MessageStatus.PENDING
    enqueue_time: datetime = field(default_factory=datetime.now)
    delivery_time: Optional[datetime] = None
    ack_time: Optional[datetime] = None
    delivery_count: int = 0
    consumer_tag: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message.to_dict(),
            "status": self.status.value,
            "enqueue_time": self.enqueue_time.isoformat(),
            "delivery_time": self.delivery_time.isoformat() if self.delivery_time else None,
            "ack_time": self.ack_time.isoformat() if self.ack_time else None,
            "delivery_count": self.delivery_count,
            "consumer_tag": self.consumer_tag,
        }


@dataclass
class Consumer:
    """消费者"""
    consumer_tag: str
    queue_name: str
    callback: Callable[[Message], bool]
    auto_ack: bool = True
    prefetch_count: int = 1
    active: bool = True


@dataclass
class QueueStats:
    """队列统计"""
    name: str
    message_count: int = 0
    consumer_count: int = 0
    delivered_count: int = 0
    acknowledged_count: int = 0
    rejected_count: int = 0
    dead_letter_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "message_count": self.message_count,
            "consumer_count": self.consumer_count,
            "delivered_count": self.delivered_count,
            "acknowledged_count": self.acknowledged_count,
            "rejected_count": self.rejected_count,
            "dead_letter_count": self.dead_letter_count,
        }


# ============================================================
# 35.2 内存消息队列
# ============================================================

class MessageQueue:
    """消息队列"""

    def __init__(
        self,
        name: str,
        max_size: int = 10000,
        ttl_seconds: Optional[float] = None,
        max_delivery_attempts: int = 3,
        dead_letter_queue: Optional[str] = None,
    ):
        self._name = name
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._max_delivery_attempts = max_delivery_attempts
        self._dead_letter_queue = dead_letter_queue

        self._messages: deque = deque()  # 主队列
        self._unacked: Dict[str, QueuedMessage] = {}  # 待确认消息
        self._consumers: Dict[str, Consumer] = {}
        self._lock = threading.RLock()
        self._stats = QueueStats(name=name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def stats(self) -> QueueStats:
        with self._lock:
            self._stats.message_count = len(self._messages) + len(self._unacked)
            return QueueStats(
                name=self._name,
                message_count=self._stats.message_count,
                consumer_count=len(self._consumers),
                delivered_count=self._stats.delivered_count,
                acknowledged_count=self._stats.acknowledged_count,
                rejected_count=self._stats.rejected_count,
                dead_letter_count=self._stats.dead_letter_count,
            )

    def enqueue(self, message: Message) -> bool:
        """入队消息"""
        with self._lock:
            if len(self._messages) >= self._max_size:
                return False
            qm = QueuedMessage(message=message)
            # 按优先级插入（简单实现：高优先级放前面）
            if message.priority >= MessagePriority.HIGH:
                self._messages.appendleft(qm)
            else:
                self._messages.append(qm)
            return True

    def dequeue(self, consumer_tag: str, auto_ack: bool = False) -> Optional[QueuedMessage]:
        """出队消息（投递给消费者）"""
        with self._lock:
            self._expire_messages()
            if not self._messages:
                return None

            qm = self._messages.popleft()
            qm.status = MessageStatus.DELIVERED
            qm.delivery_time = datetime.now()
            qm.delivery_count += 1
            qm.consumer_tag = consumer_tag

            # 总是放入_unacked，ack/nack时移除
            self._unacked[qm.message.message_id] = qm
            
            if auto_ack:
                # 自动确认：立即从_unacked移除并统计
                del self._unacked[qm.message.message_id]
                qm.status = MessageStatus.ACKNOWLEDGED
                qm.ack_time = datetime.now()
                self._stats.acknowledged_count += 1

            self._stats.delivered_count += 1
            return qm

    def ack(self, message_id: str) -> bool:
        """确认消息"""
        with self._lock:
            if message_id in self._unacked:
                qm = self._unacked.pop(message_id)
                qm.status = MessageStatus.ACKNOWLEDGED
                qm.ack_time = datetime.now()
                self._stats.acknowledged_count += 1
                return True
            return False

    def nack(self, message_id: str, requeue: bool = True) -> bool:
        """否定确认（可重新入队）"""
        with self._lock:
            if message_id not in self._unacked:
                return False

            qm = self._unacked.pop(message_id)

            if requeue and qm.delivery_count < self._max_delivery_attempts:
                qm.status = MessageStatus.PENDING
                qm.consumer_tag = ""
                self._messages.append(qm)
            else:
                qm.status = MessageStatus.REJECTED
                self._stats.rejected_count += 1
                if self._dead_letter_queue:
                    # 标记为死信，由broker处理
                    qm.status = MessageStatus.DEAD_LETTER
                    self._stats.dead_letter_count += 1

            return True

    def add_consumer(self, consumer: Consumer) -> None:
        """添加消费者"""
        with self._lock:
            self._consumers[consumer.consumer_tag] = consumer

    def remove_consumer(self, consumer_tag: str) -> bool:
        """移除消费者"""
        with self._lock:
            if consumer_tag in self._consumers:
                del self._consumers[consumer_tag]
                return True
            return False

    def get_consumers(self) -> List[Consumer]:
        """获取所有消费者"""
        with self._lock:
            return list(self._consumers.values())

    def _expire_messages(self) -> None:
        """清理过期消息"""
        if not self._ttl_seconds:
            return
        now = datetime.now()
        expired = []
        for qm in list(self._messages):
            elapsed = (now - qm.enqueue_time).total_seconds()
            if qm.message.expiration_seconds and elapsed > qm.message.expiration_seconds:
                expired.append(qm)
            elif elapsed > self._ttl_seconds:
                expired.append(qm)
        for qm in expired:
            self._messages.remove(qm)
            qm.status = MessageStatus.EXPIRED


# ============================================================
# 35.3 发布订阅代理
# ============================================================

class PubSubBroker:
    """发布订阅代理 - 管理队列、路由、订阅"""

    def __init__(self):
        self._queues: Dict[str, MessageQueue] = {}
        self._exchanges: Dict[str, DeliveryMode] = {}
        self._bindings: Dict[str, List[str]] = defaultdict(list)  # exchange -> [queue_names]
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)  # topic -> [consumer_tags]
        self._lock = threading.RLock()
        self._running = False
        self._delivery_thread: Optional[threading.Thread] = None

    def declare_queue(
        self,
        name: str,
        max_size: int = 10000,
        ttl_seconds: Optional[float] = None,
        max_delivery_attempts: int = 3,
        dead_letter_queue: Optional[str] = None,
    ) -> MessageQueue:
        """声明队列"""
        with self._lock:
            if name not in self._queues:
                self._queues[name] = MessageQueue(
                    name=name,
                    max_size=max_size,
                    ttl_seconds=ttl_seconds,
                    max_delivery_attempts=max_delivery_attempts,
                    dead_letter_queue=dead_letter_queue,
                )
            return self._queues[name]

    def delete_queue(self, name: str) -> bool:
        """删除队列"""
        with self._lock:
            if name in self._queues:
                del self._queues[name]
                # 清理绑定
                for exchange, queues in list(self._bindings.items()):
                    if name in queues:
                        queues.remove(name)
                return True
            return False

    def declare_exchange(self, name: str, mode: DeliveryMode) -> None:
        """声明交换器"""
        with self._lock:
            self._exchanges[name] = mode

    def bind_queue(self, exchange: str, queue_name: str, routing_key: str = "") -> bool:
        """绑定队列到交换器"""
        with self._lock:
            if queue_name not in self._queues:
                return False
            if exchange not in self._exchanges:
                self._exchanges[exchange] = DeliveryMode.DIRECT
            self._bindings[exchange].append(queue_name)
            return True

    def publish(self, exchange: str, message: Message) -> int:
        """发布消息到交换器，返回投递队列数"""
        with self._lock:
            if exchange not in self._exchanges:
                # 自动创建direct交换器
                self._exchanges[exchange] = DeliveryMode.DIRECT

            mode = self._exchanges[exchange]
            queues = self._bindings[exchange]
            delivered = 0

            if mode == DeliveryMode.FANOUT:
                for qname in queues:
                    if self._queues[qname].enqueue(message):
                        delivered += 1
            elif mode == DeliveryMode.DIRECT or mode == DeliveryMode.TOPIC:
                for qname in queues:
                    # 简化：直接投递到所有绑定的队列
                    if self._queues[qname].enqueue(message):
                        delivered += 1
            else:
                # HEADERS模式
                for qname in queues:
                    if self._queues[qname].enqueue(message):
                        delivered += 1

            return delivered

    def subscribe(self, topic: str, consumer_tag: str) -> None:
        """订阅主题"""
        with self._lock:
            self._subscriptions[topic].add(consumer_tag)

    def unsubscribe(self, topic: str, consumer_tag: str) -> bool:
        """取消订阅"""
        with self._lock:
            if topic in self._subscriptions:
                self._subscriptions[topic].discard(consumer_tag)
                return True
            return False

    def consume(self, queue_name: str, consumer_tag: str, callback: Callable[[Message], bool],
                auto_ack: bool = True, prefetch_count: int = 1) -> bool:
        """开始消费"""
        with self._lock:
            if queue_name not in self._queues:
                return False

            consumer = Consumer(
                consumer_tag=consumer_tag,
                queue_name=queue_name,
                callback=callback,
                auto_ack=auto_ack,
                prefetch_count=prefetch_count,
            )
            self._queues[queue_name].add_consumer(consumer)
            return True

    def cancel_consume(self, queue_name: str, consumer_tag: str) -> bool:
        """取消消费"""
        with self._lock:
            if queue_name in self._queues:
                return self._queues[queue_name].remove_consumer(consumer_tag)
            return False

    def ack(self, queue_name: str, message_id: str) -> bool:
        """确认消息"""
        with self._lock:
            if queue_name in self._queues:
                return self._queues[queue_name].ack(message_id)
            return False

    def nack(self, queue_name: str, message_id: str, requeue: bool = True) -> bool:
        """否定确认"""
        with self._lock:
            if queue_name in self._queues:
                return self._queues[queue_name].nack(message_id, requeue)
            return False

    def get_queue_stats(self, queue_name: str) -> Optional[QueueStats]:
        """获取队列统计"""
        with self._lock:
            if queue_name in self._queues:
                return self._queues[queue_name].stats
            return None

    def get_all_stats(self) -> Dict[str, QueueStats]:
        """获取所有队列统计"""
        with self._lock:
            return {name: q.stats for name, q in self._queues.items()}

    def start_delivery(self) -> None:
        """启动后台投递线程"""
        if self._running:
            return
        self._running = True

        def _deliver_loop():
            while self._running:
                self._do_delivery()
                time.sleep(0.01)  # 10ms间隔

        self._delivery_thread = threading.Thread(target=_deliver_loop, daemon=True)
        self._delivery_thread.start()

    def stop_delivery(self) -> None:
        """停止后台投递"""
        self._running = False
        if self._delivery_thread:
            self._delivery_thread.join(timeout=5.0)
            self._delivery_thread = None

    def _do_delivery(self) -> None:
        """执行一次投递"""
        with self._lock:
            queues = list(self._queues.values())

        for queue in queues:
            consumers = queue.get_consumers()
            for consumer in consumers:
                if not consumer.active:
                    continue
                # 获取待投递消息
                qm = queue.dequeue(consumer.consumer_tag, auto_ack=consumer.auto_ack)
                if qm:
                    if consumer.auto_ack:
                        # 自动确认：直接调用回调
                        try:
                            consumer.callback(qm.message)
                        except Exception:
                            pass  # 已经自动确认了，无法回滚
                    else:
                        # 手动确认：根据回调结果决定是否确认
                        try:
                            success = consumer.callback(qm.message)
                            if success:
                                queue.ack(qm.message.message_id)
                            else:
                                queue.nack(qm.message.message_id, requeue=True)
                        except Exception:
                            queue.nack(qm.message.message_id, requeue=True)


# ============================================================
# 35.4 消息持久化
# ============================================================

class PersistentMessageStore:
    """持久化消息存储"""

    def __init__(self, storage_path: str = "./mq_data"):
        self._storage_path = storage_path
        self._lock = threading.Lock()
        import os
        os.makedirs(storage_path, exist_ok=True)

    def save_message(self, queue_name: str, message: Message) -> bool:
        """保存消息"""
        import os
        with self._lock:
            queue_dir = os.path.join(self._storage_path, queue_name)
            os.makedirs(queue_dir, exist_ok=True)
            file_path = os.path.join(queue_dir, f"{message.message_id}.json")
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(message.to_dict(), f, ensure_ascii=False, default=str)
                return True
            except Exception:
                return False

    def load_messages(self, queue_name: str) -> List[Message]:
        """加载队列的所有消息"""
        import os
        messages = []
        queue_dir = os.path.join(self._storage_path, queue_name)
        if not os.path.exists(queue_dir):
            return messages

        with self._lock:
            for filename in os.listdir(queue_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(queue_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            msg = self._dict_to_message(data)
                            messages.append(msg)
                    except Exception:
                        pass
        return messages

    def delete_message(self, queue_name: str, message_id: str) -> bool:
        """删除消息"""
        import os
        with self._lock:
            file_path = os.path.join(self._storage_path, queue_name, f"{message_id}.json")
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    return True
            except Exception:
                pass
            return False

    def _dict_to_message(self, data: Dict[str, Any]) -> Message:
        """字典转消息"""
        return Message(
            message_id=data["message_id"],
            payload=data["payload"],
            headers=data.get("headers", {}),
            priority=MessagePriority(data.get("priority", 2)),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            expiration_seconds=data.get("expiration_seconds"),
            delivery_mode=DeliveryMode(data.get("delivery_mode", "direct")),
            routing_key=data.get("routing_key", ""),
            correlation_id=data.get("correlation_id", ""),
            reply_to=data.get("reply_to", ""),
        )


# ============================================================
# 便捷函数
# ============================================================

def create_message(
    payload: Any,
    headers: Optional[Dict[str, Any]] = None,
    priority: MessagePriority = MessagePriority.NORMAL,
    routing_key: str = "",
    expiration_seconds: Optional[float] = None,
) -> Message:
    """创建消息"""
    return Message(
        message_id=str(uuid.uuid4()),
        payload=payload,
        headers=headers or {},
        priority=priority,
        routing_key=routing_key,
        expiration_seconds=expiration_seconds,
    )


def create_broker() -> PubSubBroker:
    """创建发布订阅代理"""
    return PubSubBroker()


def create_persistent_store(storage_path: str = "./mq_data") -> PersistentMessageStore:
    """创建持久化存储"""
    return PersistentMessageStore(storage_path)
