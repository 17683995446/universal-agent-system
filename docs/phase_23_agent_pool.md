# 阶段23: Agent多实例管理 (Agent Pool)

## 概述

阶段23实现了Agent多实例管理系统，支持创建、管理和协调多个Agent实例。系统提供池化管理、负载均衡、实例间通信等功能，为大规模并发Agent应用提供基础架构。

## 核心组件

### 1. PoolAgentStatus (Agent状态枚举)

定义Agent实例的生命周期状态：

- `IDLE`: 空闲状态，可接受任务
- `BUSY`: 忙碌状态，正在处理任务
- `PAUSED`: 暂停状态，暂时不接收新任务
- `ERROR`: 错误状态，需要处理
- `STOPPED`: 停止状态，等待销毁

```python
from src.agent import PoolAgentStatus

# 检查状态
if instance.status == PoolAgentStatus.IDLE:
    # 可以分配任务
    pass
```

### 2. AgentInstance (Agent实例)

封装单个Agent实例的数据和状态：

```python
@dataclass
class AgentInstance:
    instance_id: str          # 实例唯一ID
    agent: Any                # Agent对象
    status: PoolAgentStatus   # 当前状态
    created_at: float         # 创建时间
    last_active: float        # 最后活跃时间
    task_count: int          # 处理任务数
    error_count: int         # 错误次数
```

**主要方法：**
- `mark_busy()`: 标记为忙碌
- `mark_idle()`: 标记为空闲
- `mark_error()`: 标记为错误
- `to_dict()`: 序列化为字典

### 3. AgentMessage (Agent消息)

支持Agent间通信的消息结构：

```python
@dataclass
class AgentMessage:
    message_id: str           # 消息ID
    sender_id: str           # 发送者ID
    receiver_id: Optional[str]  # 接收者ID（None表示广播）
    message_type: str        # 消息类型
    payload: dict            # 消息内容
    timestamp: float         # 时间戳
```

### 4. LoadBalancer (负载均衡器)

实现三种负载均衡策略：

```python
from src.agent import LoadBalancer

# 轮询策略
balancer = LoadBalancer(strategy="round_robin")

# 最少忙碌策略
balancer = LoadBalancer(strategy="least_busy")

# 随机策略
balancer = LoadBalancer(strategy="random")

# 选择实例
instance = balancer.select(idle_instances)
```

**策略说明：**
- **round_robin**: 轮流选择，保证公平性
- **least_busy**: 选择处理任务最少的实例，优化负载
- **random**: 随机选择，简单高效

### 5. AgentPool (Agent池)

核心管理类，提供完整的实例生命周期管理：

```python
from src.agent import AgentPool

# 创建池
pool = AgentPool(
    pool_id="my_pool",
    max_size=10,           # 最大实例数
    min_size=2,            # 最小实例数
    load_balance_strategy="round_robin"
)

# 初始化池
def create_agent():
    return MyAgent()

pool.initialize_pool(create_agent, count=3)

# 获取实例
instance = pool.get_instance()
if instance:
    try:
        # 使用实例处理任务
        result = instance.agent.process(task)
    finally:
        # 释放实例
        pool.release_instance(instance.instance_id)

# 动态扩缩容
pool.scale_to(5, create_agent)  # 扩展到5个实例

# 发送消息
pool.send_message(
    sender_id="agent_1",
    receiver_id="agent_2",
    message_type="task_complete",
    payload={"result": "success"}
)

# 广播消息
pool.broadcast(
    sender_id="agent_1",
    message_type="shutdown",
    payload={"reason": "maintenance"}
)

# 订阅消息
def on_task_complete(message):
    print(f"Task completed: {message.payload}")

pool.subscribe_message("task_complete", on_task_complete)

# 获取统计
stats = pool.get_stats()
print(f"Pool size: {stats['size']}")
print(f"Available: {stats['available']}")
print(f"Status distribution: {stats['status_distribution']}")

# 清空池
pool.clear()
```

## 使用场景

### 场景1: Web服务并发处理

```python
pool = AgentPool(pool_id="api_pool", max_size=20, min_size=5)
pool.initialize_pool(create_api_agent, count=5)

def handle_request(request):
    instance = pool.get_instance()
    if not instance:
        return {"error": "Service busy"}
    
    try:
        return instance.agent.process(request)
    finally:
        pool.release_instance(instance.instance_id)
```

### 场景2: 任务队列处理

```python
pool = AgentPool(pool_id="worker_pool", max_size=10)

# 根据队列长度动态扩缩容
def adjust_pool_size(queue_length):
    target = min(max(queue_length // 5, 1), 10)
    pool.scale_to(target, create_worker_agent)

# 处理任务
while True:
    task = task_queue.get()
    instance = pool.get_instance()
    if instance:
        # 异步处理
        threading.Thread(
            target=process_task,
            args=(instance, task)
        ).start()
```

### 场景3: Agent协作

```python
pool = AgentPool(pool_id="team_pool", max_size=5)
pool.initialize_pool(create_specialized_agent, count=5)

# 设置消息处理器
def on_coordination(msg):
    if msg.message_type == "need_help":
        # 分配空闲Agent协助
        helper = pool.get_instance()
        if helper:
            helper.agent.assist(msg.sender_id, msg.payload)

pool.subscribe_message("need_help", on_coordination)
```

## 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| pool_id | str | "default" | 池唯一标识 |
| max_size | int | 10 | 最大实例数 |
| min_size | int | 1 | 最小实例数 |
| load_balance_strategy | str | "round_robin" | 负载均衡策略 |

## 统计指标

`get_stats()` 返回的统计信息：

```python
{
    "pool_id": "my_pool",
    "size": 5,                    # 当前实例数
    "max_size": 10,               # 最大容量
    "min_size": 2,                # 最小容量
    "available": 3,               # 可用实例数
    "status_distribution": {      # 状态分布
        "idle": 3,
        "busy": 2,
        "error": 0
    },
    "total_created": 15,          # 累计创建数
    "total_destroyed": 10         # 累计销毁数
}
```

## 线程安全

AgentPool使用threading.Lock保证线程安全：
- 所有实例操作（创建、获取、释放、销毁）都是线程安全的
- 统计信息获取是原子的
- 消息分发是线程安全的

## 最佳实践

1. **合理设置池大小**：根据系统资源和并发需求设置max_size
2. **及时释放实例**：使用try/finally确保实例被释放
3. **监控池状态**：定期检查统计信息，及时调整池大小
4. **处理获取失败**：get_instance()可能返回None，需要处理
5. **优雅关闭**：使用clear()方法优雅地销毁所有实例

## 测试覆盖

- 35个测试用例，覆盖所有核心功能
- 包括单元测试、集成测试、导出测试
- 验证线程安全性和边界条件

## 总结

阶段23成功实现了Agent多实例管理系统，提供了：
- 完整的实例生命周期管理
- 灵活的负载均衡策略
- 高效的Agent间通信机制
- 全面的统计监控能力

为构建大规模、高并发的Agent应用奠定了坚实基础。
