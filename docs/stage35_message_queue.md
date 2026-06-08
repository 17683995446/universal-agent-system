# 阶段35: Agent消息队列系统

> 遵循"图难于其易，为大于其细。天下难事，必作于易；天下大事，必作于细。"的原则

## 概述

阶段35实现了完整的Agent消息队列系统，提供异步消息传递、发布订阅、消息持久化等功能，支持多种投递模式和优先级队列。

## 核心组件

### 1. 消息队列模型

- **MessagePriority**: LOWEST / LOW / NORMAL / HIGH / HIGHEST
- **MessageStatus**: PENDING / DELIVERED / ACKNOWLEDGED / REJECTED / EXPIRED / DEAD_LETTER
- **DeliveryMode**: DIRECT / FANOUT / TOPIC / HEADERS
- **Message**: 消息（payload, headers, priority, routing_key等）
- **QueuedMessage**: 队列中的消息（包装，含状态、投递次数）
- **Consumer**: 消费者（callback, auto_ack, prefetch_count）
- **QueueStats**: 队列统计

### 2. MessageQueue - 消息队列

```python
queue = MessageQueue(
    name="orders",
    max_size=10000,
    ttl_seconds=3600,
    max_delivery_attempts=3,
    dead_letter_queue="dlq",
)

# 入队
queue.enqueue(create_message("order-123", priority=MessagePriority.HIGH))

# 出队（投递）
qm = queue.dequeue("consumer-1", auto_ack=False)

# 确认/否定确认
queue.ack(qm.message.message_id)
queue.nack(qm.message.message_id, requeue=True)
```

### 3. PubSubBroker - 发布订阅代理

```python
broker = create_broker()

# 声明队列和交换器
broker.declare_queue("orders")
broker.declare_exchange("orders_ex", DeliveryMode.FANOUT)
broker.bind_queue("orders_ex", "orders")

# 消费
broker.consume("orders", "c1", process_order, auto_ack=True)
broker.start_delivery()

# 发布
broker.publish("orders_ex", create_message("new order"))
```

### 4. PersistentMessageStore - 消息持久化

```python
store = create_persistent_store("./mq_data")

# 保存消息
store.save_message("queue_name", message)

# 加载消息
messages = store.load_messages("queue_name")

# 删除消息
store.delete_message("queue_name", message_id)
```

## 特性

| 特性 | 说明 |
|------|------|
| 优先级队列 | 高优先级消息优先投递 |
| 消息确认 | 支持自动确认和手动确认 |
| 死信队列 | 超过重试次数的消息进入死信 |
| 消息过期 | 支持TTL和消息级过期时间 |
| 多种投递模式 | Direct / Fanout / Topic / Headers |
| 后台投递线程 | 自动投递消息给消费者 |
| 消息持久化 | 文件存储，支持恢复 |

## 测试覆盖

共46个测试用例，覆盖模型、队列操作、发布订阅、持久化、集成测试。

## 文件清单

| 文件 | 说明 |
|------|------|
| `src/agent/message_queue.py` | 消息队列实现 |
| `tests/test_message_queue.py` | 单元测试 |
| `docs/stage35_message_queue.md` | 本文档 |

## 与前一阶段的集成

- 消息队列可与阶段34的限流器结合，控制消费速率
- 可与阶段33的服务发现结合，实现分布式消息队列
- 可与阶段32的配置中心结合，动态调整队列参数

## 下一步（阶段36）

下一阶段将实现 **Agent缓存系统**，提供多级缓存、缓存策略、分布式缓存等功能。

## 总结

1. ✅ 定义了消息队列模型（优先级、状态、投递模式枚举，消息/消费者/统计类）
2. ✅ 实现了内存消息队列（入队/出队、确认/否定确认、死信、优先级）
3. ✅ 实现了发布订阅代理（队列/交换器管理、绑定、发布、消费、后台投递）
4. ✅ 实现了消息持久化（文件存储、加载、删除）
5. ✅ 编写了46个单元测试，全部通过
6. ✅ 集成到系统导出
