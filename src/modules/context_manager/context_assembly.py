"""
无上限上下文Agent系统 - 上下文组装模块

标准化ContextAssembly结构体，包含序列化、验证、工具方法。
ContextAssembly是上下文组装的结果，包含系统消息、会话消息、召回记忆。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

from typing import Optional, Any
from dataclasses import dataclass, field
import json
import copy

from src.core.message import Message, Role
from src.modules.memory_manager.memory_entry import MemoryEntry


# =============================================================================
# TokenBudget 结构体（从types.py迁移并扩展）
# =============================================================================

@dataclass
class TokenBudget:
    """
    Token预算分配

    根据窗口分区策略计算各区域可用Token数
    """
    window_size: int                    # 模型窗口总大小
    system_zone: int                    # 系统指令区配额
    session_zone: int                   # 有效会话区配额
    reserve_zone: int                   # 预留推理区配额

    @property
    def total(self) -> int:
        """总配额"""
        return self.system_zone + self.session_zone + self.reserve_zone

    @property
    def available_for_messages(self) -> int:
        """可用于消息的Token数（系统区+会话区）"""
        return self.system_zone + self.session_zone

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "window_size": self.window_size,
            "system_zone": self.system_zone,
            "session_zone": self.session_zone,
            "reserve_zone": self.reserve_zone,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenBudget":
        """从字典创建"""
        return cls(
            window_size=data.get("window_size", 4096),
            system_zone=data.get("system_zone", 800),
            session_zone=data.get("session_zone", 2400),
            reserve_zone=data.get("reserve_zone", 800),
        )


# =============================================================================
# ContextAssembly 结构体
# =============================================================================

@dataclass
class ContextAssembly:
    """
    标准化上下文组装结果

    按配额将系统指令、历史消息、召回记忆组装成最终上下文

    Attributes:
        system_messages: 系统指令区消息（系统提示、工具定义等）
        session_messages: 有效会话区消息（用户输入、历史对话）
        recalled_memories: 召回的长期记忆
        total_tokens: 实际总Token数（计算后填充）
        budget: 使用的预算配置
        metadata: 组装元数据（截断信息、召回分数等）
    """
    system_messages: list[Message] = field(default_factory=list)
    session_messages: list[Message] = field(default_factory=list)
    recalled_memories: list[MemoryEntry] = field(default_factory=list)
    total_tokens: int = 0
    budget: TokenBudget = field(default_factory=lambda: TokenBudget(
        window_size=4096, system_zone=800, session_zone=2400, reserve_zone=800
    ))
    metadata: dict[str, Any] = field(default_factory=dict)

    # =========================================================================
    # 序列化方法
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "system_messages": [m.to_dict() for m in self.system_messages],
            "session_messages": [m.to_dict() for m in self.session_messages],
            "recalled_memories": [m.to_dict() for m in self.recalled_memories],
            "total_tokens": self.total_tokens,
            "budget": self.budget.to_dict(),
            "metadata": self.metadata,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextAssembly":
        """从字典创建ContextAssembly"""
        return cls(
            system_messages=[Message.from_dict(m) for m in data.get("system_messages", [])],
            session_messages=[Message.from_dict(m) for m in data.get("session_messages", [])],
            recalled_memories=[MemoryEntry.from_dict(m) for m in data.get("recalled_memories", [])],
            total_tokens=data.get("total_tokens", 0),
            budget=TokenBudget.from_dict(data.get("budget", {})),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ContextAssembly":
        """从JSON字符串创建ContextAssembly"""
        return cls.from_dict(json.loads(json_str))

    # =========================================================================
    # 验证方法
    # =========================================================================

    def validate(self) -> tuple[bool, list[str]]:
        """
        验证组装结果有效性

        Returns:
            (是否有效, 错误信息列表)
        """
        errors: list[str] = []

        # 验证Token数非负
        if self.total_tokens < 0:
            errors.append(f"total_tokens不能为负数: {self.total_tokens}")

        # 验证预算
        if self.budget.total > self.budget.window_size:
            errors.append(
                f"预算总和({self.budget.total})超过窗口大小({self.budget.window_size})"
            )

        # 验证系统消息必须包含系统提示
        has_system = any(m.is_system() for m in self.system_messages)
        if not has_system and self.system_messages:
            # 警告：系统区没有系统消息（可能是配置问题）
            pass  # 不强制要求，某些场景可能不需要

        # 验证所有消息
        for i, msg in enumerate(self.system_messages + self.session_messages):
            valid, msg_errors = msg.validate()
            if not valid:
                errors.extend([f"Message[{i}]: {e}" for e in msg_errors])

        return len(errors) == 0, errors

    def is_valid(self) -> bool:
        """快速检查是否有效"""
        valid, _ = self.validate()
        return valid

    # =========================================================================
    # 工具方法
    # =========================================================================

    def copy(self) -> "ContextAssembly":
        """创建深拷贝"""
        return copy.deepcopy(self)

    def to_messages(self) -> list[Message]:
        """
        转换为消息列表（按正确顺序）

        顺序：系统消息 → 会话消息
        """
        return self.system_messages + self.session_messages

    def to_openai_format(self) -> list[dict[str, str]]:
        """
        转换为OpenAI API格式

        Returns:
            OpenAI格式的消息列表
        """
        return [m.to_openai_format() for m in self.to_messages()]

    def get_message_count(self) -> int:
        """获取消息总数"""
        return len(self.system_messages) + len(self.session_messages)

    def get_memory_count(self) -> int:
        """获取召回记忆数量"""
        return len(self.recalled_memories)

    def get_system_token_count(self) -> int:
        """获取系统区Token数"""
        return sum(m.token_count for m in self.system_messages)

    def get_session_token_count(self) -> int:
        """获取会话区Token数"""
        return sum(m.token_count for m in self.session_messages)

    def get_memory_token_count(self) -> int:
        """获取记忆Token数"""
        return sum(m.token_count for m in self.recalled_memories)

    def update_total_tokens(self) -> "ContextAssembly":
        """
        更新总Token数（支持链式调用）

        计算所有消息和记忆的Token总和
        """
        self.total_tokens = (
            self.get_system_token_count()
            + self.get_session_token_count()
            + self.get_memory_token_count()
        )
        return self

    def is_within_budget(self) -> bool:
        """检查是否在预算范围内"""
        return self.total_tokens <= self.budget.available_for_messages

    def get_budget_usage_ratio(self) -> float:
        """
        获取预算使用率

        Returns:
            使用率（0-1），超过1.0表示超预算
        """
        if self.budget.available_for_messages <= 0:
            return 1.0
        return self.total_tokens / self.budget.available_for_messages

    def add_system_message(self, content: str, **kwargs) -> Message:
        """
        添加系统消息（支持链式调用）

        Args:
            content: 消息内容
            **kwargs: 其他Message参数

        Returns:
            创建的消息
        """
        from src.core.message import create_system_message
        msg = create_system_message(content, **kwargs)
        self.system_messages.append(msg)
        self.update_total_tokens()
        return msg

    def add_user_message(self, content: str, **kwargs) -> Message:
        """
        添加用户消息（支持链式调用）

        Args:
            content: 消息内容
            **kwargs: 其他Message参数

        Returns:
            创建的消息
        """
        from src.core.message import create_user_message
        msg = create_user_message(content, **kwargs)
        self.session_messages.append(msg)
        self.update_total_tokens()
        return msg

    def add_assistant_message(self, content: str, **kwargs) -> Message:
        """
        添加助手消息（支持链式调用）

        Args:
            content: 消息内容
            **kwargs: 其他Message参数

        Returns:
            创建的消息
        """
        from src.core.message import create_assistant_message
        msg = create_assistant_message(content, **kwargs)
        self.session_messages.append(msg)
        self.update_total_tokens()
        return msg

    def add_metadata(self, key: str, value: Any) -> "ContextAssembly":
        """添加元数据（支持链式调用）"""
        self.metadata[key] = value
        return self

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """获取元数据"""
        return self.metadata.get(key, default)

    # =========================================================================
    # 统计信息
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """
        获取组装统计信息

        Returns:
            统计信息字典
        """
        return {
            "message_count": self.get_message_count(),
            "system_message_count": len(self.system_messages),
            "session_message_count": len(self.session_messages),
            "memory_count": self.get_memory_count(),
            "total_tokens": self.total_tokens,
            "system_tokens": self.get_system_token_count(),
            "session_tokens": self.get_session_token_count(),
            "memory_tokens": self.get_memory_token_count(),
            "budget_window": self.budget.window_size,
            "budget_available": self.budget.available_for_messages,
            "budget_usage_ratio": self.get_budget_usage_ratio(),
            "is_within_budget": self.is_within_budget(),
        }

    # =========================================================================
    # 魔术方法
    # =========================================================================

    def __str__(self) -> str:
        return (
            f"ContextAssembly("
            f"messages={self.get_message_count()}, "
            f"tokens={self.total_tokens}, "
            f"usage={self.get_budget_usage_ratio():.1%})"
        )

    def __repr__(self) -> str:
        return (
            f"ContextAssembly("
            f"system={len(self.system_messages)}, "
            f"session={len(self.session_messages)}, "
            f"memories={len(self.recalled_memories)}, "
            f"tokens={self.total_tokens})"
        )

    def __len__(self) -> int:
        """返回消息总数"""
        return self.get_message_count()
