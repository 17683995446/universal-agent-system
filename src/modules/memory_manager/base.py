"""
无上限上下文Agent系统 - 记忆存储抽象接口

定义三层记忆的统一存储接口，所有记忆管理器必须实现此接口。
遵循依赖倒置原则：上层模块依赖抽象接口，不依赖具体实现。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime

from src.modules.memory_manager.memory_entry import (
    MemoryEntry, MemoryLevel, MemorySource
)


class BaseMemoryStorage(ABC):
    """
    记忆存储抽象基类

    所有三层记忆管理器（瞬时/短期/长期）必须实现此接口。
    """

    @property
    @abstractmethod
    def level(self) -> MemoryLevel:
        """此存储对应的记忆层级"""
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        """当前存储的记忆条目数"""
        ...

    @property
    @abstractmethod
    def total_tokens(self) -> int:
        """当前存储的总Token数"""
        ...

    # =========================================================================
    # 核心CRUD操作
    # =========================================================================

    @abstractmethod
    def add(self, entry: MemoryEntry) -> MemoryEntry:
        """
        添加记忆条目

        Args:
            entry: 记忆条目（level会被自动设置为当前存储层级）

        Returns:
            添加后的记忆条目（可能被修改，如重新计算token_count）

        Raises:
            MemoryStorageError: 存储失败
        """
        ...

    @abstractmethod
    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """
        获取记忆条目

        Args:
            memory_id: 记忆ID

        Returns:
            记忆条目，不存在则返回None
        """
        ...

    @abstractmethod
    def update(self, memory_id: str, content: Optional[str] = None,
               metadata: Optional[dict] = None, **kwargs) -> Optional[MemoryEntry]:
        """
        更新记忆条目

        Args:
            memory_id: 记忆ID
            content: 新内容（None则不更新）
            metadata: 新元数据（None则不更新，会合并而非替换）
            **kwargs: 其他字段更新

        Returns:
            更新后的记忆条目，不存在则返回None
        """
        ...

    @abstractmethod
    def remove(self, memory_id: str) -> bool:
        """
        删除记忆条目

        Args:
            memory_id: 记忆ID

        Returns:
            是否删除成功
        """
        ...

    @abstractmethod
    def clear(self) -> int:
        """
        清空所有记忆

        Returns:
            清除的条目数
        """
        ...

    # =========================================================================
    # 查询操作
    # =========================================================================

    @abstractmethod
    def get_all(self) -> list[MemoryEntry]:
        """获取所有记忆条目"""
        ...

    @abstractmethod
    def contains(self, memory_id: str) -> bool:
        """检查记忆是否存在"""
        ...

    def get_by_source(self, source: MemorySource) -> list[MemoryEntry]:
        """按来源过滤"""
        return [e for e in self.get_all() if e.source == source]

    def get_expired(self) -> list[MemoryEntry]:
        """获取所有已过期的条目"""
        return [e for e in self.get_all() if e.is_expired()]

    # =========================================================================
    # 维护操作
    # =========================================================================

    def purge_expired(self) -> int:
        """
        清理过期记忆

        Returns:
            清理的条目数
        """
        expired = self.get_expired()
        count = 0
        for entry in expired:
            if self.remove(entry.memory_id):
                count += 1
        return count

    def health_check(self) -> dict:
        """
        健康检查

        Returns:
            健康状态字典
        """
        all_entries = self.get_all()
        return {
            "level": self.level.value,
            "size": self.size,
            "total_tokens": self.total_tokens,
            "expired_count": len(self.get_expired()),
            "valid_count": sum(1 for e in all_entries if e.is_valid()),
        }
