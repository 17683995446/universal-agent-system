"""
无上限上下文Agent系统 - 记忆条目模块

标准化MemoryEntry结构体，包含序列化、验证、工具方法。
MemoryEntry是三层记忆系统的数据基石。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import copy
import math

from src.core.utils.id_generator import generate_memory_id


# =============================================================================
# 枚举定义
# =============================================================================

class MemoryLevel(str, Enum):
    """记忆层级枚举"""
    INSTANT = "instant"      # 瞬时记忆：随推理刷新
    SHORT_TERM = "short"     # 短期记忆：Token配额+LRU淘汰
    LONG_TERM = "long"       # 长期记忆：向量存储+语义召回


class MemorySource(str, Enum):
    """记忆来源枚举"""
    USER_INPUT = "user_input"       # 用户输入
    ASSISTANT_OUTPUT = "assistant"   # 助手输出
    SYSTEM_INSTRUCTION = "system"    # 系统指令
    TOOL_RESULT = "tool_result"      # 工具结果
    EXTERNAL_IMPORT = "external"     # 外部导入
    SUMMARIZATION = "summarization"  # 摘要生成


# =============================================================================
# MemoryEntry 结构体
# =============================================================================

@dataclass
class MemoryEntry:
    """
    标准化记忆条目结构

    核心字段（必须）：
        - content: 记忆内容
        - level: 记忆层级

    系统字段（自动生成）：
        - memory_id: 记忆唯一标识
        - created_at: 创建时间
        - accessed_at: 最后访问时间

    计算字段（后续填充）：
        - token_count: Token数量
        - relevance_score: 相关性评分（0-1）
        - access_count: 访问次数

    扩展字段（可选）：
        - source: 记忆来源
        - embedding: 向量嵌入（长期记忆专用）
        - metadata: 元数据（任务标签、会话ID等）
        - ttl: 生存时间（秒），None表示永不过期
    """
    # 核心字段
    content: str
    level: MemoryLevel = MemoryLevel.SHORT_TERM

    # 系统字段（自动生成）
    memory_id: str = field(default_factory=generate_memory_id)
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)

    # 计算字段
    token_count: int = 0
    relevance_score: float = 0.0
    access_count: int = 0

    # 扩展字段
    source: MemorySource = MemorySource.USER_INPUT
    embedding: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    ttl: Optional[float] = None  # 秒，None=永不过期

    def __post_init__(self):
        """初始化后处理：确保枚举类型"""
        if isinstance(self.level, str):
            self.level = MemoryLevel(self.level)
        if isinstance(self.source, str):
            self.source = MemorySource(self.source)

    # =========================================================================
    # 序列化方法
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "content": self.content,
            "level": self.level.value,
            "memory_id": self.memory_id,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "token_count": self.token_count,
            "relevance_score": self.relevance_score,
            "access_count": self.access_count,
            "source": self.source.value,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "ttl": self.ttl,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        """从字典创建MemoryEntry"""
        def _parse_dt(val: Any) -> datetime:
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                return datetime.fromisoformat(val)
            return datetime.now()

        return cls(
            content=data.get("content", ""),
            level=MemoryLevel(data.get("level", "short")),
            memory_id=data.get("memory_id", generate_memory_id()),
            created_at=_parse_dt(data.get("created_at")),
            accessed_at=_parse_dt(data.get("accessed_at")),
            token_count=data.get("token_count", 0),
            relevance_score=data.get("relevance_score", 0.0),
            access_count=data.get("access_count", 0),
            source=MemorySource(data.get("source", "user_input")),
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
            ttl=data.get("ttl"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "MemoryEntry":
        """从JSON字符串创建MemoryEntry"""
        return cls.from_dict(json.loads(json_str))

    # =========================================================================
    # 验证方法
    # =========================================================================

    def validate(self) -> tuple[bool, list[str]]:
        """
        验证记忆条目有效性

        Returns:
            (是否有效, 错误信息列表)
        """
        errors: list[str] = []

        if not self.content:
            errors.append("content不能为空")
        elif not isinstance(self.content, str):
            errors.append(f"content必须是字符串，当前: {type(self.content)}")

        if not self.memory_id:
            errors.append("memory_id不能为空")

        if self.token_count < 0:
            errors.append(f"token_count不能为负数: {self.token_count}")

        if not (0.0 <= self.relevance_score <= 1.0):
            errors.append(f"relevance_score必须在0-1之间: {self.relevance_score}")

        if self.access_count < 0:
            errors.append(f"access_count不能为负数: {self.access_count}")

        if self.ttl is not None and self.ttl < 0:
            errors.append(f"ttl不能为负数: {self.ttl}")

        if self.embedding is not None:
            if not isinstance(self.embedding, list):
                errors.append("embedding必须是列表或None")
            elif len(self.embedding) == 0:
                errors.append("embedding不能为空列表")

        return len(errors) == 0, errors

    def is_valid(self) -> bool:
        """快速检查是否有效"""
        valid, _ = self.validate()
        return valid

    # =========================================================================
    # 工具方法
    # =========================================================================

    def copy(self) -> "MemoryEntry":
        """创建深拷贝"""
        return copy.deepcopy(self)

    def touch(self) -> "MemoryEntry":
        """更新访问时间和计数（支持链式调用）"""
        self.accessed_at = datetime.now()
        self.access_count += 1
        return self

    def update_relevance(self, score: float) -> "MemoryEntry":
        """更新相关性评分（支持链式调用）"""
        self.relevance_score = max(0.0, min(1.0, score))
        return self

    def update_token_count(self, count: int) -> "MemoryEntry":
        """更新Token计数（支持链式调用）"""
        self.token_count = max(0, count)
        return self

    def add_metadata(self, key: str, value: Any) -> "MemoryEntry":
        """添加元数据（支持链式调用）"""
        self.metadata[key] = value
        return self

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """获取元数据"""
        return self.metadata.get(key, default)

    def is_expired(self) -> bool:
        """检查记忆是否已过期"""
        if self.ttl is None:
            return False
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed > self.ttl

    def get_age_seconds(self) -> float:
        """获取记忆年龄（秒）"""
        return (datetime.now() - self.created_at).total_seconds()

    def get_idle_seconds(self) -> float:
        """获取空闲时间（秒）= 当前时间 - 最后访问时间"""
        return (datetime.now() - self.accessed_at).total_seconds()

    def compute_time_decay(self, halflife_hours: float = 24.0) -> float:
        """
        计算时间衰减分数

        Args:
            halflife_hours: 半衰期（小时）

        Returns:
            衰减分数（0-1），1.0表示最新，接近0表示很旧
        """
        age_hours = self.get_age_seconds() / 3600.0
        if age_hours <= 0:
            return 1.0
        return math.pow(0.5, age_hours / halflife_hours)

    # =========================================================================
    # 层级检查
    # =========================================================================

    def is_instant(self) -> bool:
        return self.level == MemoryLevel.INSTANT

    def is_short_term(self) -> bool:
        return self.level == MemoryLevel.SHORT_TERM

    def is_long_term(self) -> bool:
        return self.level == MemoryLevel.LONG_TERM

    def has_embedding(self) -> bool:
        return self.embedding is not None and len(self.embedding) > 0

    # =========================================================================
    # 魔术方法
    # =========================================================================

    def __str__(self) -> str:
        preview = self.content[:50] + ("..." if len(self.content) > 50 else "")
        return f"[{self.level.value}] {preview}"

    def __repr__(self) -> str:
        preview = self.content[:30] + ("..." if len(self.content) > 30 else "")
        return (f"MemoryEntry(level={self.level.value}, "
                f"content={preview!r}, id={self.memory_id})")

    def __hash__(self) -> int:
        return hash(self.memory_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MemoryEntry):
            return False
        return self.memory_id == other.memory_id


# =============================================================================
# 记忆列表工具函数
# =============================================================================

def filter_by_level(
    entries: list[MemoryEntry],
    level: MemoryLevel
) -> list[MemoryEntry]:
    """按层级过滤"""
    return [e for e in entries if e.level == level]


def filter_valid(entries: list[MemoryEntry]) -> list[MemoryEntry]:
    """过滤出有效条目"""
    return [e for e in entries if e.is_valid()]


def filter_non_expired(entries: list[MemoryEntry]) -> list[MemoryEntry]:
    """过滤出未过期条目"""
    return [e for e in entries if not e.is_expired()]


def sort_by_relevance(
    entries: list[MemoryEntry],
    reverse: bool = True
) -> list[MemoryEntry]:
    """按相关性排序（默认降序）"""
    return sorted(entries, key=lambda e: e.relevance_score, reverse=reverse)


def sort_by_access_time(
    entries: list[MemoryEntry],
    reverse: bool = False
) -> list[MemoryEntry]:
    """按最后访问时间排序（默认升序，最早的在前=LRU）"""
    return sorted(entries, key=lambda e: e.accessed_at, reverse=reverse)


def get_total_tokens(entries: list[MemoryEntry]) -> int:
    """获取总Token数"""
    return sum(e.token_count for e in entries)


def entries_to_json(
    entries: list[MemoryEntry],
    indent: Optional[int] = None
) -> str:
    """批量转换为JSON"""
    return json.dumps(
        [e.to_dict() for e in entries],
        indent=indent,
        ensure_ascii=False
    )


def entries_from_json(json_str: str) -> list[MemoryEntry]:
    """从JSON批量创建"""
    data = json.loads(json_str)
    return [MemoryEntry.from_dict(item) for item in data]


def validate_entry_list(
    entries: list[MemoryEntry]
) -> tuple[bool, list[str]]:
    """
    验证记忆条目列表

    Returns:
        (是否全部有效, 错误信息列表)
    """
    all_errors: list[str] = []
    for i, entry in enumerate(entries):
        valid, errors = entry.validate()
        if not valid:
            all_errors.extend([f"MemoryEntry[{i}]: {e}" for e in errors])
    return len(all_errors) == 0, all_errors
