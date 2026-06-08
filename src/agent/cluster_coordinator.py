# =============================================================================
# Agent集群协调器 - 阶段24: Agent Cluster Coordination
# =============================================================================
# 
# 功能：实现多Agent集群的协调机制，支持分布式任务处理
# 原则：图难于其易，为大于其细
#
# =============================================================================

from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

from .agent_pool import AgentPool, AgentInstance, AgentMessage


# =============================================================================
# 集群节点状态
# =============================================================================

class NodeStatus(str, Enum):
    """集群节点状态"""
    HEALTHY = "healthy"      # 健康
    BUSY = "busy"            # 忙碌
    DEGRADED = "degraded"    # 降级
    OFFLINE = "offline"      # 离线
    JOINING = "joining"      # 加入中
    LEAVING = "leaving"      # 离开中


# =============================================================================
# 任务分片策略
# =============================================================================

class ShardStrategy(str, Enum):
    """任务分片策略"""
    HASH = "hash"            # 哈希分片
    RANGE = "range"          # 范围分片
    ROUND_ROBIN = "round_robin"  # 轮询分片
    LEAST_BUSY = "least_busy"    # 最少忙碌分片
    ADAPTIVE = "adaptive"    # 自适应分片


# =============================================================================
# 集群节点
# =============================================================================

@dataclass
class ClusterNode:
    """
    集群节点
    
    表示集群中的一个Agent池节点
    """
    node_id: str
    pool: AgentPool
    status: NodeStatus = NodeStatus.JOINING
    capacity: int = 10                    # 容量
    load_factor: float = 0.0              # 负载因子 (0-1)
    last_heartbeat: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def update_heartbeat(self) -> None:
        """更新心跳时间"""
        self.last_heartbeat = time.time()
    
    def is_healthy(self) -> bool:
        """检查节点是否健康"""
        return self.status in (NodeStatus.HEALTHY, NodeStatus.BUSY)
    
    def calculate_load(self) -> float:
        """计算负载因子"""
        stats = self.pool.get_stats()
        if stats["size"] == 0:
            self.load_factor = 0.0
        else:
            self.load_factor = (stats["size"] - stats["available"]) / stats["size"]
        return self.load_factor
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "capacity": self.capacity,
            "load_factor": self.load_factor,
            "last_heartbeat": self.last_heartbeat,
            "pool_stats": self.pool.get_summary(),
            "metadata": self.metadata,
        }


# =============================================================================
# 任务分片
# =============================================================================

@dataclass
class TaskShard:
    """
    任务分片
    
    表示一个被分割的子任务
    """
    shard_id: str
    task_id: str
    node_id: Optional[str]      # 分配的节点ID
    payload: Any                # 任务数据
    index: int                  # 分片索引
    total: int                  # 总分片数
    status: str = "pending"     # pending/running/completed/failed
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    
    def mark_running(self) -> None:
        """标记为运行中"""
        self.status = "running"
    
    def mark_completed(self, result: Any) -> None:
        """标记为完成"""
        self.status = "completed"
        self.result = result
        self.completed_at = time.time()
    
    def mark_failed(self, error: str) -> None:
        """标记为失败"""
        self.status = "failed"
        self.error = error
        self.completed_at = time.time()
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "shard_id": self.shard_id,
            "task_id": self.task_id,
            "node_id": self.node_id,
            "index": self.index,
            "total": self.total,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


# =============================================================================
# 分布式任务
# =============================================================================

@dataclass
class DistributedTask:
    """
    分布式任务
    
    需要在多个节点上分布式执行的任务
    """
    task_id: str
    payload: Any
    shard_count: int = 1
    strategy: ShardStrategy = ShardStrategy.HASH
    shards: list[TaskShard] = field(default_factory=list)
    status: str = "pending"     # pending/running/completed/failed/partial
    aggregated_result: Any = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    
    def __post_init__(self) -> None:
        """初始化后创建分片"""
        if not self.shards:
            self._create_shards()
    
    def _create_shards(self) -> None:
        """创建任务分片"""
        self.shards = []
        for i in range(self.shard_count):
            shard = TaskShard(
                shard_id=f"{self.task_id}_shard_{i}",
                task_id=self.task_id,
                node_id=None,
                payload=self._extract_shard_payload(i),
                index=i,
                total=self.shard_count,
            )
            self.shards.append(shard)
    
    def _extract_shard_payload(self, index: int) -> Any:
        """提取分片数据（子类可重写）"""
        if isinstance(self.payload, list):
            # 列表数据：按索引分配
            items_per_shard = max(1, len(self.payload) // self.shard_count)
            start = index * items_per_shard
            end = start + items_per_shard if index < self.shard_count - 1 else len(self.payload)
            return self.payload[start:end]
        elif isinstance(self.payload, dict) and "items" in self.payload:
            # 包含items字段的字典
            items = self.payload["items"]
            items_per_shard = max(1, len(items) // self.shard_count)
            start = index * items_per_shard
            end = start + items_per_shard if index < self.shard_count - 1 else len(items)
            return {
                **self.payload,
                "items": items[start:end],
                "shard_index": index,
            }
        else:
            # 其他类型：整体传递
            return {
                "data": self.payload,
                "shard_index": index,
            }
    
    def assign_shard(self, shard_index: int, node_id: str) -> bool:
        """分配分片到节点"""
        if 0 <= shard_index < len(self.shards):
            self.shards[shard_index].node_id = node_id
            return True
        return False
    
    def update_status(self) -> None:
        """更新任务状态"""
        if all(s.status == "completed" for s in self.shards):
            self.status = "completed"
            self.completed_at = time.time()
        elif any(s.status == "failed" for s in self.shards):
            if all(s.status in ("completed", "failed") for s in self.shards):
                self.status = "failed"
            else:
                self.status = "partial"
        elif any(s.status == "running" for s in self.shards):
            self.status = "running"
    
    def get_progress(self) -> dict[str, Any]:
        """获取进度信息"""
        total = len(self.shards)
        completed = sum(1 for s in self.shards if s.status == "completed")
        failed = sum(1 for s in self.shards if s.status == "failed")
        running = sum(1 for s in self.shards if s.status == "running")
        pending = total - completed - failed - running
        
        return {
            "task_id": self.task_id,
            "status": self.status,
            "total": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "pending": pending,
            "progress_percent": (completed + failed) / total * 100 if total > 0 else 0,
        }
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "shard_count": self.shard_count,
            "strategy": self.strategy.value,
            "status": self.status,
            "shards": [s.to_dict() for s in self.shards],
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


# =============================================================================
# 结果聚合器
# =============================================================================

class ResultAggregator:
    """
    结果聚合器
    
    将多个分片的结果聚合为最终结果
    """
    
    @staticmethod
    def aggregate_list(shards: list[TaskShard]) -> list[Any]:
        """聚合列表结果"""
        results = []
        for shard in sorted(shards, key=lambda s: s.index):
            if shard.result and isinstance(shard.result, list):
                results.extend(shard.result)
        return results
    
    @staticmethod
    def aggregate_dict(shards: list[TaskShard]) -> dict[str, Any]:
        """聚合字典结果"""
        result = {}
        for shard in sorted(shards, key=lambda s: s.index):
            if shard.result and isinstance(shard.result, dict):
                result.update(shard.result)
        return result
    
    @staticmethod
    def aggregate_count(shards: list[TaskShard]) -> int:
        """聚合计数结果"""
        total = 0
        for shard in shards:
            if shard.result and isinstance(shard.result, (int, float)):
                total += shard.result
        return total
    
    @staticmethod
    def aggregate_first_success(shards: list[TaskShard]) -> Any:
        """返回第一个成功的结果"""
        for shard in sorted(shards, key=lambda s: s.index):
            if shard.status == "completed" and shard.result is not None:
                return shard.result
        return None


# =============================================================================
# 集群协调器
# =============================================================================

class ClusterCoordinator:
    """
    集群协调器
    
    管理多节点Agent集群，提供：
    - 节点注册与发现
    - 任务分片与分发
    - 结果聚合
    - 故障检测与转移
    - 负载均衡
    
    Attributes:
        cluster_id: 集群ID
        nodes: 集群节点字典
        tasks: 任务字典
    """
    
    def __init__(
        self,
        cluster_id: str = "default",
        heartbeat_interval: float = 5.0,
        node_timeout: float = 15.0,
    ):
        """
        Args:
            cluster_id: 集群ID
            heartbeat_interval: 心跳间隔（秒）
            node_timeout: 节点超时时间（秒）
        """
        self._cluster_id = cluster_id
        self._nodes: dict[str, ClusterNode] = {}
        self._tasks: dict[str, DistributedTask] = {}
        self._lock = threading.RLock()
        self._heartbeat_interval = heartbeat_interval
        self._node_timeout = node_timeout
        self._task_handlers: dict[str, Callable[[Any], Any]] = {}
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None
    
    # =====================================================================
    # 属性
    # =====================================================================
    
    @property
    def cluster_id(self) -> str:
        """集群ID"""
        return self._cluster_id
    
    @property
    def node_count(self) -> int:
        """节点数量"""
        with self._lock:
            return len(self._nodes)
    
    @property
    def healthy_nodes(self) -> list[ClusterNode]:
        """健康节点列表"""
        with self._lock:
            return [n for n in self._nodes.values() if n.is_healthy()]
    
    # =====================================================================
    # 节点管理
    # =====================================================================
    
    def register_node(
        self,
        node_id: str,
        pool: AgentPool,
        capacity: int = 10,
        metadata: Optional[dict] = None,
    ) -> ClusterNode:
        """
        注册节点
        
        Args:
            node_id: 节点ID
            pool: Agent池
            capacity: 节点容量
            metadata: 元数据
        
        Returns:
            创建的节点
        """
        with self._lock:
            node = ClusterNode(
                node_id=node_id,
                pool=pool,
                status=NodeStatus.HEALTHY,
                capacity=capacity,
                metadata=metadata or {},
            )
            self._nodes[node_id] = node
            
            # 设置消息处理器
            pool.subscribe_message("shard_complete", self._on_shard_complete)
            pool.subscribe_message("shard_failed", self._on_shard_failed)
            
            return node
    
    def unregister_node(self, node_id: str) -> bool:
        """
        注销节点
        
        Args:
            node_id: 节点ID
        
        Returns:
            是否成功
        """
        with self._lock:
            if node_id not in self._nodes:
                return False
            
            node = self._nodes[node_id]
            node.status = NodeStatus.LEAVING
            
            # 迁移该节点的任务
            self._migrate_node_tasks(node_id)
            
            del self._nodes[node_id]
            return True
    
    def update_node_status(self, node_id: str, status: NodeStatus) -> bool:
        """
        更新节点状态
        
        Args:
            node_id: 节点ID
            status: 新状态
        
        Returns:
            是否成功
        """
        with self._lock:
            if node_id not in self._nodes:
                return False
            self._nodes[node_id].status = status
            self._nodes[node_id].update_heartbeat()
            return True
    
    def heartbeat(self, node_id: str) -> bool:
        """
        节点心跳
        
        Args:
            node_id: 节点ID
        
        Returns:
            节点是否存在
        """
        with self._lock:
            if node_id not in self._nodes:
                return False
            self._nodes[node_id].update_heartbeat()
            return True
    
    def _check_node_health(self) -> None:
        """检查节点健康状态"""
        now = time.time()
        with self._lock:
            for node in self._nodes.values():
                if now - node.last_heartbeat > self._node_timeout:
                    if node.status != NodeStatus.OFFLINE:
                        node.status = NodeStatus.OFFLINE
                        # 触发故障转移
                        self._migrate_node_tasks(node.node_id)
    
    def _migrate_node_tasks(self, node_id: str) -> None:
        """迁移节点上的任务"""
        for task in self._tasks.values():
            for shard in task.shards:
                if shard.node_id == node_id and shard.status == "running":
                    # 重置为pending，等待重新分配
                    shard.status = "pending"
                    shard.node_id = None
    
    # =====================================================================
    # 任务管理
    # =====================================================================
    
    def register_task_handler(
        self,
        task_type: str,
        handler: Callable[[Any], Any],
    ) -> None:
        """
        注册任务处理器
        
        Args:
            task_type: 任务类型
            handler: 处理函数
        """
        self._task_handlers[task_type] = handler
    
    def submit_task(
        self,
        payload: Any,
        shard_count: int = 1,
        strategy: ShardStrategy = ShardStrategy.HASH,
        task_type: str = "default",
    ) -> str:
        """
        提交分布式任务
        
        Args:
            payload: 任务数据
            shard_count: 分片数量
            strategy: 分片策略
            task_type: 任务类型
        
        Returns:
            任务ID
        """
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        task = DistributedTask(
            task_id=task_id,
            payload=payload,
            shard_count=shard_count,
            strategy=strategy,
        )
        
        with self._lock:
            self._tasks[task_id] = task
        
        # 分配分片到节点
        self._distribute_shards(task, task_type)
        
        return task_id
    
    def _distribute_shards(self, task: DistributedTask, task_type: str) -> None:
        """分发任务分片"""
        healthy_nodes = self.healthy_nodes
        if not healthy_nodes:
            return
        
        # 根据策略分配
        if task.strategy == ShardStrategy.ROUND_ROBIN:
            self._distribute_round_robin(task, healthy_nodes, task_type)
        elif task.strategy == ShardStrategy.LEAST_BUSY:
            self._distribute_least_busy(task, healthy_nodes, task_type)
        else:
            # 默认使用轮询
            self._distribute_round_robin(task, healthy_nodes, task_type)
    
    def _distribute_round_robin(
        self,
        task: DistributedTask,
        nodes: list[ClusterNode],
        task_type: str,
    ) -> None:
        """轮询分发"""
        node_index = 0
        for i, shard in enumerate(task.shards):
            if shard.status == "pending":
                node = nodes[node_index % len(nodes)]
                self._assign_shard_to_node(shard, node, task_type)
                node_index += 1
    
    def _distribute_least_busy(
        self,
        task: DistributedTask,
        nodes: list[ClusterNode],
        task_type: str,
    ) -> None:
        """分配给负载最低的节点"""
        for shard in task.shards:
            if shard.status == "pending":
                # 按负载排序
                sorted_nodes = sorted(nodes, key=lambda n: n.calculate_load())
                node = sorted_nodes[0] if sorted_nodes else nodes[0]
                self._assign_shard_to_node(shard, node, task_type)
    
    def _assign_shard_to_node(
        self,
        shard: TaskShard,
        node: ClusterNode,
        task_type: str,
    ) -> None:
        """分配分片到节点执行"""
        shard.node_id = node.node_id
        shard.mark_running()
        
        # 发送执行消息
        node.pool.broadcast(
            sender_id="coordinator",
            message_type="execute_shard",
            payload={
                "shard_id": shard.shard_id,
                "task_id": shard.task_id,
                "task_type": task_type,
                "data": shard.payload,
            },
        )
    
    def _on_shard_complete(self, message: AgentMessage) -> None:
        """处理分片完成消息"""
        payload = message.payload
        task_id = payload.get("task_id")
        shard_id = payload.get("shard_id")
        result = payload.get("result")
        
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                for shard in task.shards:
                    if shard.shard_id == shard_id:
                        shard.mark_completed(result)
                        break
                task.update_status()
    
    def _on_shard_failed(self, message: AgentMessage) -> None:
        """处理分片失败消息"""
        payload = message.payload
        task_id = payload.get("task_id")
        shard_id = payload.get("shard_id")
        error = payload.get("error")
        
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                for shard in task.shards:
                    if shard.shard_id == shard_id:
                        shard.mark_failed(error)
                        break
                task.update_status()
    
    def get_task(self, task_id: str) -> Optional[DistributedTask]:
        """
        获取任务
        
        Args:
            task_id: 任务ID
        
        Returns:
            任务对象或None
        """
        with self._lock:
            return self._tasks.get(task_id)
    
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """
        获取任务结果
        
        Args:
            task_id: 任务ID
        
        Returns:
            聚合后的结果
        """
        task = self.get_task(task_id)
        if not task:
            return None
        
        if task.status != "completed":
            return None
        
        # 聚合结果
        return ResultAggregator.aggregate_list(task.shards)
    
    def wait_for_task(
        self,
        task_id: str,
        timeout: Optional[float] = None,
        poll_interval: float = 0.1,
    ) -> Optional[Any]:
        """
        等待任务完成
        
        Args:
            task_id: 任务ID
            timeout: 超时时间（秒）
            poll_interval: 轮询间隔（秒）
        
        Returns:
            任务结果或None（超时）
        """
        start_time = time.time()
        while True:
            result = self.get_task_result(task_id)
            if result is not None:
                return result
            
            task = self.get_task(task_id)
            if task and task.status == "failed":
                return None
            
            if timeout and time.time() - start_time > timeout:
                return None
            
            time.sleep(poll_interval)
    
    # =====================================================================
    # 集群操作
    # =====================================================================
    
    def start(self) -> None:
        """启动协调器"""
        self._running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
        )
        self._heartbeat_thread.start()
    
    def stop(self) -> None:
        """停止协调器"""
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5.0)
    
    def _heartbeat_loop(self) -> None:
        """心跳循环"""
        while self._running:
            self._check_node_health()
            time.sleep(self._heartbeat_interval)
    
    def get_cluster_stats(self) -> dict[str, Any]:
        """
        获取集群统计
        
        Returns:
            集群统计信息
        """
        with self._lock:
            node_stats = [n.to_dict() for n in self._nodes.values()]
            
            total_capacity = sum(n.capacity for n in self._nodes.values())
            healthy_count = sum(1 for n in self._nodes.values() if n.is_healthy())
            
            task_stats = {
                "total": len(self._tasks),
                "pending": sum(1 for t in self._tasks.values() if t.status == "pending"),
                "running": sum(1 for t in self._tasks.values() if t.status == "running"),
                "completed": sum(1 for t in self._tasks.values() if t.status == "completed"),
                "failed": sum(1 for t in self._tasks.values() if t.status == "failed"),
            }
            
            return {
                "cluster_id": self._cluster_id,
                "node_count": len(self._nodes),
                "healthy_nodes": healthy_count,
                "total_capacity": total_capacity,
                "nodes": node_stats,
                "tasks": task_stats,
            }
    
    def get_node_info(self, node_id: str) -> Optional[dict[str, Any]]:
        """
        获取节点信息
        
        Args:
            node_id: 节点ID
        
        Returns:
            节点信息或None
        """
        with self._lock:
            if node_id in self._nodes:
                return self._nodes[node_id].to_dict()
            return None
