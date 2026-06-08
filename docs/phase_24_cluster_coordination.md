# 阶段24: Agent集群协调 (Agent Cluster Coordination)

## 概述

阶段24实现了Agent集群协调系统，支持多节点分布式任务处理。系统提供节点管理、任务分片、结果聚合、故障转移等功能，为构建大规模分布式Agent应用提供基础架构。

## 核心组件

### 1. NodeStatus (节点状态枚举)

定义集群节点的生命周期状态：

- `HEALTHY`: 健康状态，可正常处理任务
- `BUSY`: 忙碌状态，负载较高但仍可接受任务
- `DEGRADED`: 降级状态，性能受限
- `OFFLINE`: 离线状态，无法通信
- `JOINING`: 加入中，正在加入集群
- `LEAVING`: 离开中，正在退出集群

```python
from src.agent import NodeStatus

# 检查节点健康
if node.status == NodeStatus.HEALTHY:
    # 可以分配任务
    pass
```

### 2. ShardStrategy (分片策略枚举)

定义任务分片策略：

- `HASH`: 哈希分片，按数据哈希值分配
- `RANGE`: 范围分片，按数据范围分配
- `ROUND_ROBIN`: 轮询分片，轮流分配到各节点
- `LEAST_BUSY`: 最少忙碌分片，分配给负载最低的节点
- `ADAPTIVE`: 自适应分片，动态调整

```python
from src.agent import ShardStrategy

# 使用轮询策略
task_id = coordinator.submit_task(
    payload=data,
    shard_count=4,
    strategy=ShardStrategy.ROUND_ROBIN,
)
```

### 3. ClusterNode (集群节点)

表示集群中的一个Agent池节点：

```python
@dataclass
class ClusterNode:
    node_id: str              # 节点唯一ID
    pool: AgentPool           # Agent池
    status: NodeStatus        # 当前状态
    capacity: int            # 容量
    load_factor: float       # 负载因子 (0-1)
    last_heartbeat: float    # 最后心跳时间
    metadata: dict           # 元数据
```

**主要方法：**
- `update_heartbeat()`: 更新心跳时间
- `is_healthy()`: 检查节点是否健康
- `calculate_load()`: 计算负载因子
- `to_dict()`: 序列化为字典

### 4. TaskShard (任务分片)

表示一个被分割的子任务：

```python
@dataclass
class TaskShard:
    shard_id: str            # 分片ID
    task_id: str             # 所属任务ID
    node_id: Optional[str]   # 分配的节点ID
    payload: Any             # 任务数据
    index: int               # 分片索引
    total: int               # 总分片数
    status: str              # pending/running/completed/failed
    result: Any              # 执行结果
    error: Optional[str]     # 错误信息
```

**主要方法：**
- `mark_running()`: 标记为运行中
- `mark_completed(result)`: 标记为完成
- `mark_failed(error)`: 标记为失败
- `to_dict()`: 序列化为字典

### 5. DistributedTask (分布式任务)

需要在多个节点上分布式执行的任务：

```python
@dataclass
class DistributedTask:
    task_id: str             # 任务ID
    payload: Any             # 任务数据
    shard_count: int         # 分片数量
    strategy: ShardStrategy  # 分片策略
    shards: list[TaskShard]  # 分片列表
    status: str              # 任务状态
```

**主要方法：**
- `assign_shard(index, node_id)`: 分配分片到节点
- `update_status()`: 更新任务状态
- `get_progress()`: 获取进度信息
- `to_dict()`: 序列化为字典

### 6. ResultAggregator (结果聚合器)

将多个分片的结果聚合为最终结果：

```python
from src.agent import ResultAggregator

# 聚合列表结果
result = ResultAggregator.aggregate_list(shards)

# 聚合字典结果
result = ResultAggregator.aggregate_dict(shards)

# 聚合计数结果
result = ResultAggregator.aggregate_count(shards)

# 返回第一个成功的结果
result = ResultAggregator.aggregate_first_success(shards)
```

### 7. ClusterCoordinator (集群协调器)

核心管理类，提供完整的集群协调功能：

```python
from src.agent import ClusterCoordinator, AgentPool

# 创建协调器
coordinator = ClusterCoordinator(
    cluster_id="my_cluster",
    heartbeat_interval=5.0,    # 心跳间隔（秒）
    node_timeout=15.0,          # 节点超时时间（秒）
)

# 启动协调器
coordinator.start()

# 注册节点
pool1 = AgentPool(pool_id="node_1_pool", max_size=10)
coordinator.register_node(
    node_id="node_1",
    pool=pool1,
    capacity=10,
    metadata={"region": "us-east"},
)

# 提交分布式任务
task_id = coordinator.submit_task(
    payload=[1, 2, 3, 4, 5, 6, 7, 8],  # 列表数据
    shard_count=4,                      # 分成4个子任务
    strategy=ShardStrategy.ROUND_ROBIN,
    task_type="data_processing",
)

# 等待任务完成
result = coordinator.wait_for_task(task_id, timeout=60.0)

# 获取任务进度
task = coordinator.get_task(task_id)
progress = task.get_progress()
print(f"进度: {progress['progress_percent']}%")

# 获取集群统计
stats = coordinator.get_cluster_stats()
print(f"节点数: {stats['node_count']}")
print(f"健康节点: {stats['healthy_nodes']}")

# 注销节点
coordinator.unregister_node("node_1")

# 停止协调器
coordinator.stop()
```

## 使用场景

### 场景1: 大规模数据处理

```python
# 创建集群
coordinator = ClusterCoordinator(cluster_id="data_cluster")
coordinator.start()

# 注册多个节点
for i in range(5):
    pool = AgentPool(pool_id=f"node_{i}", max_size=20)
    coordinator.register_node(f"node_{i}", pool, capacity=20)

# 提交大数据处理任务
data = load_large_dataset()  # 加载大数据集
task_id = coordinator.submit_task(
    payload=data,
    shard_count=20,  # 分成20个子任务并行处理
    strategy=ShardStrategy.LEAST_BUSY,
)

# 等待结果
result = coordinator.wait_for_task(task_id, timeout=300.0)
```

### 场景2: 高可用服务集群

```python
coordinator = ClusterCoordinator(
    cluster_id="service_cluster",
    heartbeat_interval=3.0,
    node_timeout=10.0,
)
coordinator.start()

# 注册多个服务节点
for i in range(3):
    pool = AgentPool(pool_id=f"service_{i}", max_size=50)
    coordinator.register_node(f"service_{i}", pool, capacity=50)

# 处理请求
def handle_request(request):
    task_id = coordinator.submit_task(
        payload=request,
        shard_count=1,
        strategy=ShardStrategy.ROUND_ROBIN,
    )
    return coordinator.wait_for_task(task_id, timeout=30.0)
```

### 场景3: 任务队列与负载均衡

```python
coordinator = ClusterCoordinator(cluster_id="worker_cluster")
coordinator.start()

# 注册工作节点（不同容量）
coordinator.register_node("worker_1", pool1, capacity=10)
coordinator.register_node("worker_2", pool2, capacity=20)
coordinator.register_node("worker_3", pool3, capacity=10)

# 使用最少忙碌策略，任务会优先分配给负载低的节点
task_id = coordinator.submit_task(
    payload=task_data,
    shard_count=5,
    strategy=ShardStrategy.LEAST_BUSY,
)
```

## 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| cluster_id | str | "default" | 集群唯一标识 |
| heartbeat_interval | float | 5.0 | 心跳检查间隔（秒） |
| node_timeout | float | 15.0 | 节点超时时间（秒） |

## 集群统计

`get_cluster_stats()` 返回的统计信息：

```python
{
    "cluster_id": "my_cluster",
    "node_count": 5,
    "healthy_nodes": 4,
    "total_capacity": 100,
    "nodes": [
        {
            "node_id": "node_1",
            "status": "healthy",
            "capacity": 20,
            "load_factor": 0.3,
            "pool_stats": {...},
        },
        # ...
    ],
    "tasks": {
        "total": 10,
        "pending": 2,
        "running": 5,
        "completed": 3,
        "failed": 0,
    },
}
```

## 故障转移机制

当节点超时或离线时，协调器会自动：

1. **检测故障**: 通过心跳机制检测节点是否超时
2. **标记状态**: 将节点标记为 `OFFLINE`
3. **任务迁移**: 将该节点上正在运行的任务分片重置为 `pending`
4. **重新分配**: 等待下次分发时重新分配到健康节点

```python
# 模拟节点故障
coordinator._check_node_health()  # 检查节点健康

# 手动触发任务迁移
coordinator._migrate_node_tasks("failed_node_id")
```

## 线程安全

ClusterCoordinator 使用 threading.RLock 保证线程安全：
- 所有节点操作（注册、注销、状态更新）都是线程安全的
- 任务提交和查询是线程安全的
- 心跳检查在独立线程中运行

## 最佳实践

1. **合理设置超时**: 根据网络环境设置合适的 `node_timeout`
2. **监控集群状态**: 定期检查 `get_cluster_stats()` 了解集群健康
3. **处理任务失败**: 任务可能部分失败，需要检查 `task.status`
4. **优雅关闭**: 使用 `stop()` 方法优雅停止协调器
5. **及时注销**: 节点下线前调用 `unregister_node()` 进行清理

## 测试覆盖

- 52个测试用例，覆盖所有核心功能
- 包括枚举测试、单元测试、集成测试、导出测试
- 验证线程安全性和故障转移机制

## 总结

阶段24成功实现了Agent集群协调系统，提供了：
- 完整的节点生命周期管理
- 灵活的任务分片策略
- 自动故障检测与转移
- 多种结果聚合方式
- 全面的集群监控能力

为构建高可用、可扩展的分布式Agent集群奠定了坚实基础。
