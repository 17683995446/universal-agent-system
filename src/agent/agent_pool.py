"""
无上限上下文Agent系统 - Agent多实例管理

实现Agent池管理、多Agent并发、Agent间通信。

核心设计：
- AgentPool: Agent池管理器
- AgentInstance: Agent实例包装
- LoadBalancer: 负载均衡策略
- AgentMessage: Agent间消息

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-27
"""

import time
import logging
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Any, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# Agent状态
# =============================================================================

class PoolAgentStatus(str, Enum):
    """池中Agent状态"""
    IDLE = "idle"           # 空闲可用
    BUSY = "busy"           # 忙碌中
    PAUSED = "paused"       # 暂停
    ERROR = "error"         # 错误
    STOPPED = "stopped"     # 已停止


# =============================================================================
# Agent实例包装
# =============================================================================

@dataclass
class AgentInstance:
    """
    Agent实例包装

    Attributes:
        instance_id: 实例ID
        agent: Agent对象
        status: 当前状态
        created_at: 创建时间
        last_used_at: 最后使用时间
        total_requests: 总请求数
        total_errors: 总错误数
    """
    instance_id: str
    agent: Any  # Agent对象
    status: PoolAgentStatus = PoolAgentStatus.IDLE
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    total_requests: int = 0
    total_errors: int = 0

    def mark_busy(self) -> None:
        """标记为忙碌"""
        self.status = PoolAgentStatus.BUSY
        self.last_used_at = time.time()
        self.total_requests += 1

    def mark_idle(self) -> None:
        """标记为空闲"""
        self.status = PoolAgentStatus.IDLE

    def mark_error(self) -> None:
        """标记为错误"""
        self.status = PoolAgentStatus.ERROR
        self.total_errors += 1

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "instance_id": self.instance_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
        }


# =============================================================================
# Agent间消息
# =============================================================================

@dataclass
class AgentMessage:
    """
    Agent间消息

    Attributes:
        message_id: 消息ID
        sender_id: 发送者实例ID
        receiver_id: 接收者实例ID（None则广播）
        message_type: 消息类型
        payload: 消息内容
        timestamp: 发送时间
    """
    message_id: str
    sender_id: str
    receiver_id: Optional[str] = None
    message_type: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }


# =============================================================================
# 负载均衡策略
# =============================================================================

class LoadBalancer:
    """负载均衡器"""

    def __init__(self, strategy: str = "round_robin"):
        """
        Args:
            strategy: 策略类型（round_robin/least_busy/random）
        """
        self._strategy = strategy
        self._round_robin_index: int = 0

    def select(
        self,
        instances: list[AgentInstance],
    ) -> Optional[AgentInstance]:
        """
        选择Agent实例

        Args:
            instances: 可用实例列表

        Returns:
            选中的实例或None
        """
        if not instances:
            return None

        # 只选择空闲实例
        idle_instances = [i for i in instances if i.status == PoolAgentStatus.IDLE]
        if not idle_instances:
            return None

        if self._strategy == "round_robin":
            return self._round_robin_select(idle_instances)
        elif self._strategy == "least_busy":
            return self._least_busy_select(idle_instances)
        elif self._strategy == "random":
            return self._random_select(idle_instances)
        else:
            return idle_instances[0]

    def _round_robin_select(
        self,
        instances: list[AgentInstance],
    ) -> AgentInstance:
        """轮询选择"""
        idx = self._round_robin_index % len(instances)
        self._round_robin_index = (self._round_robin_index + 1) % len(instances)
        return instances[idx]

    def _least_busy_select(
        self,
        instances: list[AgentInstance],
    ) -> AgentInstance:
        """最少请求选择"""
        return min(instances, key=lambda i: i.total_requests)

    def _random_select(
        self,
        instances: list[AgentInstance],
    ) -> AgentInstance:
        """随机选择"""
        import random
        return random.choice(instances)


# =============================================================================
# Agent池
# =============================================================================

class AgentPool:
    """
    Agent池管理器

    管理多个Agent实例，提供：
    - 实例创建/销毁/回收
    - 负载均衡
    - Agent间通信
    - 统计监控

    Attributes:
        pool_id: 池ID
        max_size: 最大实例数
        min_size: 最小实例数
    """

    def __init__(
        self,
        pool_id: str = "default",
        max_size: int = 10,
        min_size: int = 1,
        load_balance_strategy: str = "round_robin",
    ):
        """
        Args:
            pool_id: 池ID
            max_size: 最大实例数
            min_size: 最小实例数
            load_balance_strategy: 负载均衡策略
        """
        self._pool_id = pool_id
        self._max_size = max_size
        self._min_size = min_size
        self._instances: dict[str, AgentInstance] = {}
        self._lock = threading.Lock()
        self._load_balancer = LoadBalancer(strategy=load_balance_strategy)
        self._message_handlers: dict[str, list[Callable[[AgentMessage], None]]] = defaultdict(list)
        self._total_created: int = 0
        self._total_destroyed: int = 0

    # =========================================================================
    # 属性
    # =========================================================================

    @property
    def pool_id(self) -> str:
        """池ID"""
        return self._pool_id

    @property
    def size(self) -> int:
        """当前实例数"""
        with self._lock:
            return len(self._instances)

    @property
    def available_count(self) -> int:
        """可用实例数"""
        with self._lock:
            return len([i for i in self._instances.values() if i.status == PoolAgentStatus.IDLE])

    # =========================================================================
    # 实例管理
    # =========================================================================

    def create_instance(
        self,
        agent_factory: Callable[[], Any],
    ) -> Optional[AgentInstance]:
        """
        创建新实例

        Args:
            agent_factory: Agent创建工厂函数

        Returns:
            AgentInstance 或 None（达到上限）
        """
        with self._lock:
            if len(self._instances) >= self._max_size:
                logger.warning(f"Pool {self._pool_id} reached max size {self._max_size}")
                return None

            instance_id = f"{self._pool_id}_agent_{self._total_created}"
            try:
                agent = agent_factory()
                instance = AgentInstance(
                    instance_id=instance_id,
                    agent=agent,
                )
                self._instances[instance_id] = instance
                self._total_created += 1

                logger.info(f"Created agent instance: {instance_id}")
                return instance

            except Exception as e:
                logger.error(f"Failed to create agent instance: {e}")
                return None

    def get_instance(self) -> Optional[AgentInstance]:
        """
        获取一个可用实例（负载均衡）

        Returns:
            AgentInstance 或 None
        """
        with self._lock:
            instances = list(self._instances.values())
            selected = self._load_balancer.select(instances)

            if selected:
                selected.mark_busy()
                return selected

            return None

    def release_instance(self, instance_id: str) -> bool:
        """
        释放实例（归还到池）

        Args:
            instance_id: 实例ID

        Returns:
            是否成功
        """
        with self._lock:
            instance = self._instances.get(instance_id)
            if instance:
                instance.mark_idle()
                return True
            return False

    def destroy_instance(self, instance_id: str) -> bool:
        """
        销毁实例

        Args:
            instance_id: 实例ID

        Returns:
            是否成功
        """
        with self._lock:
            instance = self._instances.pop(instance_id, None)
            if instance:
                try:
                    # 调用Agent的清理方法
                    if hasattr(instance.agent, 'cleanup'):
                        instance.agent.cleanup()
                    self._total_destroyed += 1
                    logger.info(f"Destroyed agent instance: {instance_id}")
                    return True
                except Exception as e:
                    logger.error(f"Error destroying instance {instance_id}: {e}")
                    return False
            return False

    def get_instance_by_id(self, instance_id: str) -> Optional[AgentInstance]:
        """通过ID获取实例"""
        with self._lock:
            return self._instances.get(instance_id)

    # =========================================================================
    # 批量操作
    # =========================================================================

    def initialize_pool(
        self,
        agent_factory: Callable[[], Any],
        count: Optional[int] = None,
    ) -> int:
        """
        初始化池（创建最小数量实例）

        Args:
            agent_factory: Agent创建工厂函数
            count: 创建数量（None则使用min_size）

        Returns:
            实际创建数量
        """
        target = count or self._min_size
        created = 0

        for _ in range(target):
            if self.create_instance(agent_factory):
                created += 1

        logger.info(f"Pool {self._pool_id} initialized with {created} instances")
        return created

    def scale_to(self, target_size: int, agent_factory: Callable[[], Any]) -> int:
        """
        缩放到目标大小

        Args:
            target_size: 目标大小
            agent_factory: Agent创建工厂函数

        Returns:
            实际变化数量
        """
        with self._lock:
            current = len(self._instances)

        if target_size > current:
            # 扩容
            to_create = min(target_size - current, self._max_size - current)
            created = 0
            for _ in range(to_create):
                if self.create_instance(agent_factory):
                    created += 1
            return created
        elif target_size < current:
            # 缩容（优先停止空闲实例）
            to_destroy = current - target_size
            destroyed = 0

            with self._lock:
                idle_instances = [
                    i for i in self._instances.values()
                    if i.status == PoolAgentStatus.IDLE
                ]

            for instance in idle_instances[:to_destroy]:
                if self.destroy_instance(instance.instance_id):
                    destroyed += 1

            return -destroyed

        return 0

    def clear(self) -> None:
        """清空所有实例"""
        with self._lock:
            instance_ids = list(self._instances.keys())

        for instance_id in instance_ids:
            self.destroy_instance(instance_id)

        logger.info(f"Pool {self._pool_id} cleared")

    # =========================================================================
    # Agent间通信
    # =========================================================================

    def send_message(
        self,
        sender_id: str,
        receiver_id: Optional[str],
        message_type: str,
        payload: dict[str, Any],
    ) -> str:
        """
        发送消息

        Args:
            sender_id: 发送者实例ID
            receiver_id: 接收者实例ID（None则广播）
            message_type: 消息类型
            payload: 消息内容

        Returns:
            消息ID
        """
        message_id = f"msg_{int(time.time() * 1000)}_{sender_id}"

        message = AgentMessage(
            message_id=message_id,
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            payload=payload,
        )

        # 分发消息
        self._dispatch_message(message)

        return message_id

    def _dispatch_message(self, message: AgentMessage) -> None:
        """分发消息"""
        handlers = self._message_handlers.get(message.message_type, [])

        for handler in handlers:
            try:
                handler(message)
            except Exception as e:
                logger.warning(f"Message handler error: {e}")

        # 如果指定了接收者，直接发送给该Agent
        if message.receiver_id:
            instance = self.get_instance_by_id(message.receiver_id)
            if instance and hasattr(instance.agent, 'receive_message'):
                try:
                    instance.agent.receive_message(message)
                except Exception as e:
                    logger.warning(f"Agent receive message error: {e}")

    def subscribe_message(
        self,
        message_type: str,
        handler: Callable[[AgentMessage], None],
    ) -> None:
        """
        订阅消息类型

        Args:
            message_type: 消息类型
            handler: 处理函数
        """
        self._message_handlers[message_type].append(handler)

    def broadcast(
        self,
        sender_id: str,
        message_type: str,
        payload: dict[str, Any],
    ) -> str:
        """
        广播消息（发送给所有实例）

        Args:
            sender_id: 发送者实例ID
            message_type: 消息类型
            payload: 消息内容

        Returns:
            消息ID
        """
        return self.send_message(sender_id, None, message_type, payload)

    # =========================================================================
    # 统计
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """获取池统计"""
        with self._lock:
            status_counts = defaultdict(int)
            available = 0
            for instance in self._instances.values():
                status_counts[instance.status.value] += 1
                if instance.status == PoolAgentStatus.IDLE:
                    available += 1

            return {
                "pool_id": self._pool_id,
                "size": len(self._instances),
                "max_size": self._max_size,
                "min_size": self._min_size,
                "available": available,
                "status_distribution": dict(status_counts),
                "total_created": self._total_created,
                "total_destroyed": self._total_destroyed,
            }

    def get_instances_info(self) -> list[dict[str, Any]]:
        """获取所有实例信息"""
        with self._lock:
            return [i.to_dict() for i in self._instances.values()]

    def get_summary(self) -> dict[str, Any]:
        """获取池摘要"""
        stats = self.get_stats()
        return {
            "pool_id": self._pool_id,
            "size": stats["size"],
            "available": stats["available"],
            "utilization": (
                (stats["size"] - stats["available"]) / stats["size"] * 100
                if stats["size"] > 0 else 0
            ),
        }
