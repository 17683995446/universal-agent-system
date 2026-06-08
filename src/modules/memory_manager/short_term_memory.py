"""
无上限上下文Agent系统 - 短期记忆管理器

短期记忆：Token配额制 + LRU淘汰。
生命周期 = 会话级别，会话结束即释放。

设计原则：
- 双重限制：Token配额上限 + 条目数上限
- LRU淘汰：超出限制时淘汰最久未访问的条目
- 自动裁剪：添加新条目时自动检查并淘汰

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
from src.core.constants import (
    SHORT_MEMORY_MAX_TOKENS,
    SHORT_MEMORY_MAX_ITEMS,
)


class ShortTermMemory(BaseMemoryStorage):
    """
    短期记忆管理器

    特点：
    - Token配额制：总Token数不超过 max_tokens
    - 条目数限制：总条目数不超过 max_items
    - LRU淘汰：超出限制时淘汰最久未访问的条目
    - 自动裁剪：add() 时自动触发淘汰
    """

    def __init__(
        self,
        max_tokens: int = SHORT_MEMORY_MAX_TOKENS,
        max_items: int = SHORT_MEMORY_MAX_ITEMS,
    ):
        """
        Args:
            max_tokens: Token配额上限
            max_items: 条目数上限
        """
        self._max_tokens = max_tokens
        self._max_items = max_items
        self._store: OrderedDict[str, MemoryEntry] = OrderedDict()

    # =========================================================================
    # 属性
    # =========================================================================

    @property
    def level(self) -> MemoryLevel:
        return MemoryLevel.SHORT_TERM

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def total_tokens(self) -> int:
        return sum(e.token_count for e in self._store.values())

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def max_items(self) -> int:
        return self._max_items

    @property
    def available_tokens(self) -> int:
        """剩余可用Token数"""
        return max(0, self._max_tokens - self.total_tokens)

    @property
    def token_usage_ratio(self) -> float:
        """Token使用率（0-1）"""
        if self._max_tokens <= 0:
            return 1.0
        return min(1.0, self.total_tokens / self._max_tokens)

    # =========================================================================
    # 核心CRUD
    # =========================================================================

    def add(self, entry: MemoryEntry) -> MemoryEntry:
        """添加记忆条目（自动触发LRU淘汰）"""
        if not entry.is_valid():
            raise MemoryStorageError("add", f"无效的记忆条目: {entry.validate()[1]}")

        entry.level = MemoryLevel.SHORT_TERM

        # 如果已存在，先移除旧的（后续重新插入到末尾=最近访问）
        if entry.memory_id in self._store:
            del self._store[entry.memory_id]

        # 先尝试淘汰，为新条目腾出空间
        self._evict_for(entry.token_count)

        self._store[entry.memory_id] = entry
        return entry

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """获取记忆条目（更新访问时间=移到末尾）"""
        entry = self._store.get(memory_id)
        if entry is not None:
            # LRU: 移到OrderedDict末尾
            self._store.move_to_end(memory_id)
            entry.touch()
        return entry

    def update(self, memory_id: str, content: Optional[str] = None,
               metadata: Optional[dict] = None, **kwargs) -> Optional[MemoryEntry]:
        """更新记忆条目"""
        entry = self._store.get(memory_id)
        if entry is None:
            return None

        old_tokens = entry.token_count

        if content is not None:
            entry.content = content
            entry.token_count = 0  # 重置，需要外部重新计算

        if metadata is not None:
            entry.metadata.update(metadata)

        for key, value in kwargs.items():
            if hasattr(entry, key):
                setattr(entry, key, value)

        entry.touch()

        # 更新后检查是否超限
        if entry.token_count > old_tokens:
            self._evict_for(entry.token_count - old_tokens)

        # LRU: 移到末尾
        self._store.move_to_end(memory_id)
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
        """获取所有记忆条目（按访问时间从旧到新）"""
        return list(self._store.values())

    def contains(self, memory_id: str) -> bool:
        """检查记忆是否存在"""
        return memory_id in self._store

    # =========================================================================
    # LRU淘汰机制
    # =========================================================================

    def _evict_for(self, needed_tokens: int) -> int:
        """
        为新条目腾出空间，淘汰最久未访问的条目

        Args:
            needed_tokens: 需要的Token数

        Returns:
            淘汰的条目数
        """
        evicted = 0

        # 淘汰直到满足Token限制
        while (self.total_tokens + needed_tokens > self._max_tokens
               and len(self._store) > 0):
            # 移除最早的（OrderedDict头部）
            _, removed = self._store.popitem(last=False)
            evicted += 1

        # 淘汰直到满足条目数限制
        while len(self._store) >= self._max_items and len(self._store) > 0:
            self._store.popitem(last=False)
            evicted += 1

        return evicted

    def evict(self, count: int = 1) -> int:
        """
        手动淘汰指定数量的最旧条目

        Args:
            count: 淘汰数量

        Returns:
            实际淘汰的条目数
        """
        actual = min(count, len(self._store))
        for _ in range(actual):
            self._store.popitem(last=False)
        return actual

    def evict_oldest(self, max_tokens_to_free: int) -> int:
        """
        淘汰最旧的条目直到释放足够的Token

        Args:
            max_tokens_to_free: 需要释放的Token数

        Returns:
            淘汰的条目数
        """
        freed = 0
        evicted = 0
        while freed < max_tokens_to_free and len(self._store) > 0:
            _, entry = self._store.popitem(last=False)
            freed += entry.token_count
            evicted += 1
        return evicted

    # =========================================================================
    # 短期记忆特有方法
    # =========================================================================

    def get_lru(self, count: int = 1) -> list[MemoryEntry]:
        """
        获取最久未访问的N个条目（即将被淘汰的）

        Args:
            count: 获取数量

        Returns:
            最久未访问的条目列表
        """
        items = list(self._store.items())
        return [entry for _, entry in items[:count]]

    def get_most_recent(self, count: int = 10) -> list[MemoryEntry]:
        """
        获取最近访问的N个条目

        Args:
            count: 获取数量

        Returns:
            最近访问的条目列表（从新到旧）
        """
        items = list(reversed(list(self._store.items())))
        return [entry for _, entry in items[:count]]

    def trim_to_tokens(self, target_tokens: int) -> int:
        """
        裁剪到目标Token数

        Args:
            target_tokens: 目标Token数

        Returns:
            淘汰的条目数
        """
        evicted = 0
        while self.total_tokens > target_tokens and len(self._store) > 0:
            self._store.popitem(last=False)
            evicted += 1
        return evicted
