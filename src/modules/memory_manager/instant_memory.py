"""
无上限上下文Agent系统 - 瞬时记忆管理器

瞬时记忆：随推理刷新，只保留当前推理周期内的必要信息。
生命周期 = 单次推理调用。

设计原则：
- 最轻量级，无持久化
- 推理开始时填充，推理结束时清空
- 不做淘汰，不做召回，只做临时暂存

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

from typing import Optional, Any
from collections import OrderedDict

from src.modules.memory_manager.base import BaseMemoryStorage
from src.modules.memory_manager.memory_entry import (
    MemoryEntry, MemoryLevel, MemorySource
)
from src.core.exceptions import MemoryStorageError


class InstantMemory(BaseMemoryStorage):
    """
    瞬时记忆管理器

    特点：
    - 纯内存存储，无持久化
    - 推理周期级别的生命周期
    - 支持按key快速存取
    - clear()即释放全部资源
    """

    def __init__(self, max_items: int = 100):
        """
        Args:
            max_items: 最大条目数（安全限制，防止无限增长）
        """
        self._store: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._max_items = max_items

    # =========================================================================
    # 属性
    # =========================================================================

    @property
    def level(self) -> MemoryLevel:
        return MemoryLevel.INSTANT

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def total_tokens(self) -> int:
        return sum(e.token_count for e in self._store.values())

    # =========================================================================
    # 核心CRUD
    # =========================================================================

    def add(self, entry: MemoryEntry) -> MemoryEntry:
        """添加记忆条目"""
        if not entry.is_valid():
            raise MemoryStorageError("add", f"无效的记忆条目: {entry.validate()[1]}")

        entry.level = MemoryLevel.INSTANT

        # 超过上限时移除最早的
        if len(self._store) >= self._max_items and entry.memory_id not in self._store:
            self._store.popitem(last=False)

        self._store[entry.memory_id] = entry
        return entry

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """获取记忆条目"""
        entry = self._store.get(memory_id)
        if entry is not None:
            entry.touch()
        return entry

    def update(self, memory_id: str, content: Optional[str] = None,
               metadata: Optional[dict] = None, **kwargs) -> Optional[MemoryEntry]:
        """更新记忆条目"""
        entry = self._store.get(memory_id)
        if entry is None:
            return None

        if content is not None:
            entry.content = content
            entry.token_count = 0  # 重置

        if metadata is not None:
            entry.metadata.update(metadata)

        for key, value in kwargs.items():
            if hasattr(entry, key):
                setattr(entry, key, value)

        entry.touch()
        return entry

    def remove(self, memory_id: str) -> bool:
        """删除记忆条目"""
        if memory_id in self._store:
            del self._store[memory_id]
            return True
        return False

    def clear(self) -> int:
        """清空所有记忆"""
        count = len(self._store)
        self._store.clear()
        return count

    # =========================================================================
    # 查询操作
    # =========================================================================

    def get_all(self) -> list[MemoryEntry]:
        """获取所有记忆条目"""
        return list(self._store.values())

    def contains(self, memory_id: str) -> bool:
        """检查记忆是否存在"""
        return memory_id in self._store

    # =========================================================================
    # 瞬时记忆特有方法
    # =========================================================================

    def set_context(self, key: str, value: str) -> MemoryEntry:
        """
        设置上下文键值对

        Args:
            key: 上下文键（如 "current_goal", "user_intent"）
            value: 上下文值

        Returns:
            创建的记忆条目
        """
        entry = MemoryEntry(
            content=value,
            level=MemoryLevel.INSTANT,
            metadata={"context_key": key}
        )
        return self.add(entry)

    def get_context(self, key: str) -> Optional[str]:
        """
        获取上下文值

        Args:
            key: 上下文键

        Returns:
            上下文值，不存在则返回None
        """
        for entry in self._store.values():
            if entry.metadata.get("context_key") == key:
                entry.touch()
                return entry.content
        return None

    def snapshot(self) -> list[MemoryEntry]:
        """
        快照当前所有瞬时记忆（返回深拷贝）

        Returns:
            所有记忆条目的深拷贝列表
        """
        return [e.copy() for e in self._store.values()]

    def restore(self, entries: list[MemoryEntry]) -> int:
        """
        从快照恢复瞬时记忆

        Args:
            entries: 记忆条目列表

        Returns:
            恢复的条目数
        """
        self._store.clear()
        count = 0
        for entry in entries:
            entry.level = MemoryLevel.INSTANT
            self._store[entry.memory_id] = entry
            count += 1
        return count
