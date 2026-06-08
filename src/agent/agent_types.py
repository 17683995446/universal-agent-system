"""
无上限上下文Agent系统 - Agent类型定义

定义Agent运行状态、回合结果、统计信息等数据结构。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-26
"""

from typing import Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import json


# =============================================================================
# 枚举
# =============================================================================

class AgentState(str, Enum):
    """Agent运行状态"""
    IDLE = "idle"               # 空闲，等待输入
    THINKING = "thinking"       # 思考中（调用LLM）
    EXECUTING = "executing"     # 执行工具
    PAUSED = "paused"           # 暂停（等待用户确认）
    ERROR = "error"             # 错误状态
    STOPPED = "stopped"         # 已停止

    @classmethod
    def can_transition(cls, from_state: "AgentState", to_state: "AgentState") -> bool:
        """检查状态转换是否合法"""
        # 定义允许的状态转换
        transitions = {
            cls.IDLE: [cls.THINKING, cls.PAUSED, cls.ERROR, cls.STOPPED],
            cls.THINKING: [cls.EXECUTING, cls.IDLE, cls.ERROR, cls.STOPPED],
            cls.EXECUTING: [cls.THINKING, cls.IDLE, cls.ERROR, cls.STOPPED],
            cls.PAUSED: [cls.IDLE, cls.THINKING, cls.ERROR, cls.STOPPED],
            cls.ERROR: [cls.IDLE, cls.STOPPED],
            cls.STOPPED: [cls.IDLE],  # 重启
        }
        return to_state in transitions.get(from_state, [])

    @property
    def is_active(self) -> bool:
        """是否为活跃状态（可处理请求）"""
        return self in {self.IDLE, self.THINKING, self.EXECUTING}

    @property
    def is_terminal(self) -> bool:
        """是否为终止状态"""
        return self == self.STOPPED


class TurnStatus(str, Enum):
    """单轮对话状态"""
    PENDING = "pending"         # 待处理
    PROCESSING = "processing"   # 处理中
    COMPLETED = "completed"     # 完成（纯文本回复）
    TOOL_CALLS = "tool_calls"   # 需要工具调用
    ERROR = "error"             # 错误


# =============================================================================
# 回合结果
# =============================================================================

@dataclass
class TurnResult:
    """
    单轮对话结果
    
    包含本轮对话的完整信息：用户输入、Agent回复、工具调用等。
    
    Attributes:
        turn_id: 回合ID
        user_input: 用户输入
        assistant_reply: 助手回复
        status: 回合状态
        tool_calls: 工具调用列表
        tool_results: 工具执行结果
        tokens_used: Token使用量
        duration_ms: 耗时
        metadata: 元数据
    """
    turn_id: str
    user_input: str = ""
    assistant_reply: str = ""
    status: TurnStatus = TurnStatus.COMPLETED
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    tokens_used: dict[str, int] = field(default_factory=dict)
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        """是否需要工具调用"""
        return self.status == TurnStatus.TOOL_CALLS and len(self.tool_calls) > 0

    @property
    def is_completed(self) -> bool:
        """是否已完成"""
        return self.status == TurnStatus.COMPLETED

    @property
    def is_error(self) -> bool:
        """是否出错"""
        return self.status == TurnStatus.ERROR

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "turn_id": self.turn_id,
            "user_input": self.user_input,
            "assistant_reply": self.assistant_reply,
            "status": self.status.value,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "tokens_used": self.tokens_used,
            "duration_ms": round(self.duration_ms, 2),
        }


# =============================================================================
# Agent统计
# =============================================================================

@dataclass
class AgentStats:
    """
    Agent运行统计
    
    Attributes:
        total_turns: 总回合数
        total_messages: 总消息数
        total_tokens: 总Token消耗
        tool_calls_count: 工具调用次数
        errors_count: 错误次数
        avg_turn_duration_ms: 平均回合耗时
        start_time: 启动时间
        last_activity_time: 最后活动时间
    """
    total_turns: int = 0
    total_messages: int = 0
    total_tokens: int = 0
    tool_calls_count: int = 0
    errors_count: int = 0
    avg_turn_duration_ms: float = 0.0
    start_time: float = 0.0
    last_activity_time: float = 0.0

    def record_turn(self, duration_ms: float, tokens: int, tool_calls: int = 0, error: bool = False) -> None:
        """记录一轮对话"""
        self.total_turns += 1
        self.total_tokens += tokens
        self.tool_calls_count += tool_calls
        if error:
            self.errors_count += 1

        # 更新平均耗时
        total_duration = self.avg_turn_duration_ms * (self.total_turns - 1) + duration_ms
        self.avg_turn_duration_ms = total_duration / self.total_turns

        self.last_activity_time = time.time()

    @property
    def uptime_seconds(self) -> float:
        """运行时间（秒）"""
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "total_turns": self.total_turns,
            "total_messages": self.total_messages,
            "total_tokens": self.total_tokens,
            "tool_calls_count": self.tool_calls_count,
            "errors_count": self.errors_count,
            "avg_turn_duration_ms": round(self.avg_turn_duration_ms, 2),
            "uptime_seconds": round(self.uptime_seconds, 2),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentStats":
        """从字典创建"""
        stats = cls()
        stats.total_turns = data.get("total_turns", 0)
        stats.total_messages = data.get("total_messages", 0)
        stats.total_tokens = data.get("total_tokens", 0)
        stats.tool_calls_count = data.get("tool_calls_count", 0)
        stats.errors_count = data.get("errors_count", 0)
        stats.avg_turn_duration_ms = data.get("avg_turn_duration_ms", 0.0)
        stats.start_time = data.get("start_time", 0.0)
        stats.last_activity_time = data.get("last_activity_time", 0.0)
        return stats


# =============================================================================
# 对话历史条目
# =============================================================================

@dataclass
class ConversationEntry:
    """
    对话历史条目
    
    简化版的消息记录，用于快速检索。
    
    Attributes:
        turn_id: 回合ID
        role: 角色（user/assistant/tool）
        content: 内容
        timestamp: 时间戳
        metadata: 元数据
    """
    turn_id: str
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationEntry":
        """从字典创建"""
        return cls(
            turn_id=data.get("turn_id", ""),
            role=data.get("role", ""),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Agent快照
# =============================================================================

@dataclass
class AgentSnapshot:
    """
    Agent状态快照
    
    用于保存和恢复Agent完整状态，包括：
    - 配置信息
    - 对话历史
    - 运行统计
    - 任务上下文
    - 记忆状态
    
    Attributes:
        version: 快照版本
        snapshot_id: 快照唯一ID
        created_at: 创建时间戳
        agent_id: Agent ID
        agent_name: Agent名称
        state: Agent状态
        config_dict: 配置字典
        conversation_history: 对话历史
        stats_dict: 统计字典
        task_context: 任务上下文
        compression_stats: 压缩统计
        metadata: 元数据
    """
    version: str = "1.0"
    snapshot_id: str = ""
    created_at: float = field(default_factory=time.time)
    agent_id: str = ""
    agent_name: str = ""
    state: str = "idle"
    config_dict: dict[str, Any] = field(default_factory=dict)
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    stats_dict: dict[str, Any] = field(default_factory=dict)
    task_context: dict[str, Any] = field(default_factory=dict)
    compression_stats: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "version": self.version,
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "state": self.state,
            "config": self.config_dict,
            "conversation_history": self.conversation_history,
            "stats": self.stats_dict,
            "task_context": self.task_context,
            "compression_stats": self.compression_stats,
            "metadata": self.metadata,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """序列化为JSON字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentSnapshot":
        """从字典创建"""
        return cls(
            version=data.get("version", "1.0"),
            snapshot_id=data.get("snapshot_id", ""),
            created_at=data.get("created_at", time.time()),
            agent_id=data.get("agent_id", data.get("agent", {}).get("agent_id", "")),
            agent_name=data.get("agent_name", data.get("agent", {}).get("name", "")),
            state=data.get("state", "idle"),
            config_dict=data.get("config", data.get("config_dict", {})),
            conversation_history=data.get("conversation_history", []),
            stats_dict=data.get("stats", data.get("stats_dict", {})),
            task_context=data.get("task_context", {}),
            compression_stats=data.get("compression_stats", {}),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "AgentSnapshot":
        """从JSON字符串创建"""
        return cls.from_dict(json.loads(json_str))

    @property
    def age_seconds(self) -> float:
        """快照年龄（秒）"""
        return time.time() - self.created_at

    def is_valid(self) -> bool:
        """检查快照是否有效"""
        return (
            self.snapshot_id != ""
            and self.agent_id != ""
            and self.version != ""
        )

    def get_summary(self) -> dict[str, Any]:
        """获取快照摘要"""
        return {
            "snapshot_id": self.snapshot_id,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "state": self.state,
            "created_at": self.created_at,
            "age_seconds": self.age_seconds,
            "history_length": len(self.conversation_history),
            "total_turns": self.stats_dict.get("total_turns", 0),
            "is_valid": self.is_valid(),
        }
