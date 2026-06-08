"""
无上限上下文Agent系统 - 事件系统

实现事件总线（EventBus）、核心事件类型定义、订阅/发布机制。

核心设计：
- AgentEventType: 事件类型枚举
- AgentEvent: 事件数据基类
- EventBus: 事件总线（发布/订阅/取消）
- EventHandler: 事件处理器类型别名

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-26
"""

import time
import logging
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Any, Callable

logger = logging.getLogger(__name__)


# =============================================================================
# 事件类型
# =============================================================================

class AgentEventType(str, Enum):
    """Agent事件类型枚举"""
    # 生命周期事件
    STATE_CHANGED = "state_changed"           # 状态变更
    AGENT_STARTED = "agent_started"           # Agent启动
    AGENT_STOPPED = "agent_stopped"           # Agent停止
    AGENT_RESET = "agent_reset"               # Agent重置
    AGENT_PAUSED = "agent_paused"             # Agent暂停
    AGENT_RESUMED = "agent_resumed"           # Agent恢复

    # 对话事件
    TURN_STARTED = "turn_started"             # 回合开始
    TURN_COMPLETED = "turn_completed"         # 回合完成
    TURN_ERROR = "turn_error"                 # 回合错误
    USER_INPUT = "user_input"                 # 用户输入
    ASSISTANT_REPLY = "assistant_reply"       # 助手回复

    # 工具事件
    TOOL_CALLED = "tool_called"               # 工具调用
    TOOL_RESULT = "tool_result"               # 工具结果
    TOOL_ERROR = "tool_error"                 # 工具错误

    # 记忆事件
    MEMORY_STORED = "memory_stored"           # 记忆存储
    MEMORY_RECALLED = "memory_recalled"       # 记忆召回
    MEMORY_ERROR = "memory_error"             # 记忆错误

    # 上下文事件
    CONTEXT_BUILT = "context_built"           # 上下文构建完成
    CONTEXT_COMPRESSED = "context_compressed" # 上下文压缩
    PURIFICATION = "purification"             # 内容净化

    # 配置事件
    CONFIG_CHANGED = "config_changed"         # 配置变更

    # 错误事件
    ERROR_OCCURRED = "error_occurred"         # 错误发生
    ERROR_RECOVERED = "error_recovered"       # 错误恢复
    RETRY_ATTEMPT = "retry_attempt"           # 重试尝试

    # 快照事件
    SNAPSHOT_CREATED = "snapshot_created"     # 快照创建
    SNAPSHOT_RESTORED = "snapshot_restored"   # 快照恢复

    # 漂移事件
    DRIFT_DETECTED = "drift_detected"         # 任务漂移检测


# =============================================================================
# 事件数据
# =============================================================================

@dataclass
class AgentEvent:
    """
    Agent事件数据

    Attributes:
        event_type: 事件类型
        agent_id: Agent ID
        timestamp: 事件时间戳
        data: 事件数据
        source: 事件来源
        turn_id: 关联回合ID
    """
    event_type: AgentEventType
    agent_id: str = ""
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    turn_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "event_type": self.event_type.value,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "data": self.data,
            "source": self.source,
            "turn_id": self.turn_id,
        }

    def get(self, key: str, default: Any = None) -> Any:
        """获取事件数据中的字段"""
        return self.data.get(key, default)

    @classmethod
    def create(
        cls,
        event_type: AgentEventType,
        agent_id: str = "",
        data: Optional[dict[str, Any]] = None,
        source: str = "",
        turn_id: str = "",
    ) -> "AgentEvent":
        """便捷创建方法"""
        return cls(
            event_type=event_type,
            agent_id=agent_id,
            data=data or {},
            source=source,
            turn_id=turn_id,
        )


# 事件处理器类型
EventHandler = Callable[[AgentEvent], None]


# =============================================================================
# 事件总线
# =============================================================================

class EventBus:
    """
    事件总线

    实现发布/订阅模式，支持：
    - 按事件类型订阅
    - 通配符订阅（订阅所有事件）
    - 一次性订阅
    - 线程安全
    - 事件历史记录

    Attributes:
        subscribers: 当前订阅者数量
        event_history: 事件历史
    """

    def __init__(self, max_history: int = 200):
        """
        Args:
            max_history: 最大事件历史记录数
        """
        self._handlers: dict[AgentEventType, list[tuple[EventHandler, bool]]] = {}
        self._wildcard_handlers: list[tuple[EventHandler, bool]] = []
        self._history: list[AgentEvent] = []
        self._max_history = max_history
        self._lock = threading.Lock()
        self._total_published: int = 0
        self._total_handled: int = 0

    # =========================================================================
    # 订阅管理
    # =========================================================================

    def subscribe(
        self,
        event_type: AgentEventType,
        handler: EventHandler,
        once: bool = False,
    ) -> None:
        """
        订阅事件

        Args:
            event_type: 事件类型
            handler: 事件处理函数
            once: 是否仅触发一次
        """
        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append((handler, once))

        logger.debug(
            f"Subscribed to {event_type.value}"
            f"{' (once)' if once else ''}"
        )

    def subscribe_all(
        self,
        handler: EventHandler,
        once: bool = False,
    ) -> None:
        """
        订阅所有事件（通配符）

        Args:
            handler: 事件处理函数
            once: 是否仅触发一次
        """
        with self._lock:
            self._wildcard_handlers.append((handler, once))

        logger.debug(f"Subscribed to all events{' (once)' if once else ''}")

    def unsubscribe(
        self,
        event_type: AgentEventType,
        handler: EventHandler,
    ) -> bool:
        """
        取消订阅

        Args:
            event_type: 事件类型
            handler: 事件处理函数

        Returns:
            是否成功取消
        """
        with self._lock:
            if event_type in self._handlers:
                original_len = len(self._handlers[event_type])
                self._handlers[event_type] = [
                    (h, o) for h, o in self._handlers[event_type]
                    if h != handler
                ]
                if len(self._handlers[event_type]) < original_len:
                    logger.debug(f"Unsubscribed from {event_type.value}")
                    return True
        return False

    def unsubscribe_all(self, handler: EventHandler) -> bool:
        """
        取消通配符订阅

        Args:
            handler: 事件处理函数

        Returns:
            是否成功取消
        """
        with self._lock:
            original_len = len(self._wildcard_handlers)
            self._wildcard_handlers = [
                (h, o) for h, o in self._wildcard_handlers
                if h != handler
            ]
            if len(self._wildcard_handlers) < original_len:
                logger.debug("Unsubscribed from all events")
                return True
        return False

    def clear_subscribers(self, event_type: Optional[AgentEventType] = None) -> None:
        """
        清除订阅者

        Args:
            event_type: 指定事件类型（None则清除所有）
        """
        with self._lock:
            if event_type is None:
                self._handlers.clear()
                self._wildcard_handlers.clear()
                logger.debug("All subscribers cleared")
            elif event_type in self._handlers:
                del self._handlers[event_type]
                logger.debug(f"Subscribers for {event_type.value} cleared")

    # =========================================================================
    # 发布
    # =========================================================================

    def publish(self, event: AgentEvent) -> int:
        """
        发布事件

        Args:
            event: 事件对象

        Returns:
            触发的处理器数量
        """
        handlers_triggered = 0

        with self._lock:
            # 记录历史
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
            self._total_published += 1

            # 收集要触发的处理器
            handlers_to_call: list[tuple[EventHandler, bool, bool]] = []

            # 特定类型处理器
            if event.event_type in self._handlers:
                for handler, once in self._handlers[event.event_type]:
                    handlers_to_call.append((handler, once, False))

            # 通配符处理器
            for handler, once in self._wildcard_handlers:
                handlers_to_call.append((handler, once, True))

        # 在锁外执行处理器（避免死锁）
        for handler, once, is_wildcard in handlers_to_call:
            try:
                handler(event)
                handlers_triggered += 1
                self._total_handled += 1
            except Exception as e:
                logger.warning(
                    f"Event handler error for {event.event_type.value}: {e}"
                )

            # 一次性处理器：移除
            if once:
                with self._lock:
                    if is_wildcard:
                        self._wildcard_handlers = [
                            (h, o) for h, o in self._wildcard_handlers
                            if h != handler
                        ]
                    elif event.event_type in self._handlers:
                        self._handlers[event.event_type] = [
                            (h, o) for h, o in self._handlers[event.event_type]
                            if h != handler
                        ]

        return handlers_triggered

    def emit(
        self,
        event_type: AgentEventType,
        agent_id: str = "",
        data: Optional[dict[str, Any]] = None,
        source: str = "",
        turn_id: str = "",
    ) -> int:
        """
        便捷发布方法

        Args:
            event_type: 事件类型
            agent_id: Agent ID
            data: 事件数据
            source: 来源
            turn_id: 回合ID

        Returns:
            触发的处理器数量
        """
        event = AgentEvent.create(
            event_type=event_type,
            agent_id=agent_id,
            data=data,
            source=source,
            turn_id=turn_id,
        )
        return self.publish(event)

    # =========================================================================
    # 查询
    # =========================================================================

    @property
    def subscriber_count(self) -> int:
        """当前订阅者总数"""
        with self._lock:
            count = len(self._wildcard_handlers)
            for handlers in self._handlers.values():
                count += len(handlers)
            return count

    @property
    def event_history(self) -> list[AgentEvent]:
        """事件历史（只读副本）"""
        with self._lock:
            return self._history.copy()

    def get_history(
        self,
        event_type: Optional[AgentEventType] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        获取事件历史

        Args:
            event_type: 筛选事件类型（None则返回全部）
            limit: 最大返回数

        Returns:
            事件字典列表
        """
        with self._lock:
            if event_type is not None:
                filtered = [
                    e for e in self._history
                    if e.event_type == event_type
                ]
            else:
                filtered = self._history

            recent = filtered[-limit:]
            return [e.to_dict() for e in recent]

    def get_stats(self) -> dict[str, Any]:
        """获取事件总线统计"""
        with self._lock:
            type_counts: dict[str, int] = {}
            for event in self._history:
                t = event.event_type.value
                type_counts[t] = type_counts.get(t, 0) + 1

            subscriber_counts: dict[str, int] = {}
            for et, handlers in self._handlers.items():
                subscriber_counts[et.value] = len(handlers)
            if self._wildcard_handlers:
                subscriber_counts["*"] = len(self._wildcard_handlers)

            total_subscribers = len(self._wildcard_handlers)
            for handlers in self._handlers.values():
                total_subscribers += len(handlers)

            return {
                "total_published": self._total_published,
                "total_handled": self._total_handled,
                "subscriber_count": total_subscribers,
                "history_size": len(self._history),
                "by_event_type": type_counts,
                "subscribers": subscriber_counts,
            }

    def clear_history(self) -> None:
        """清空事件历史"""
        with self._lock:
            self._history.clear()

    def reset_stats(self) -> None:
        """重置统计"""
        self._total_published = 0
        self._total_handled = 0
        self.clear_history()
