"""
无上限上下文Agent系统 - 记忆工厂

统一创建三层记忆管理器的工厂类。
遵循简单工厂模式，集中管理记忆实例的创建。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

from typing import Optional
from dataclasses import dataclass, field

from src.modules.memory_manager.base import BaseMemoryStorage
from src.modules.memory_manager.memory_entry import MemoryLevel
from src.modules.memory_manager.instant_memory import InstantMemory
from src.modules.memory_manager.short_term_memory import ShortTermMemory
from src.modules.memory_manager.long_term_memory import LongTermMemory, RelevanceScorer
from src.core.constants import (
    SHORT_MEMORY_MAX_TOKENS,
    SHORT_MEMORY_MAX_ITEMS,
    MEMORY_RECALL_TOP_K,
    MEMORY_SIMILARITY_THRESHOLD,
)


@dataclass
class MemoryConfig:
    """
    记忆管理配置
    
    Attributes:
        short_term_max_tokens: 短期记忆Token上限
        short_term_max_items: 短期记忆条目上限
        long_term_top_k: 长期记忆召回Top-K
        long_term_similarity_threshold: 长期记忆相似度阈值
    """
    short_term_max_tokens: int = SHORT_MEMORY_MAX_TOKENS
    short_term_max_items: int = SHORT_MEMORY_MAX_ITEMS
    long_term_top_k: int = MEMORY_RECALL_TOP_K
    long_term_similarity_threshold: float = MEMORY_SIMILARITY_THRESHOLD


class MemoryFactory:
    """
    记忆管理器工厂

    用法：
        factory = MemoryFactory()
        instant = factory.create_instant()
        short_term = factory.create_short_term()
        long_term = factory.create_long_term()
    """

    def __init__(
        self,
        short_term_max_tokens: int = SHORT_MEMORY_MAX_TOKENS,
        short_term_max_items: int = SHORT_MEMORY_MAX_ITEMS,
        long_term_top_k: int = MEMORY_RECALL_TOP_K,
        long_term_similarity_threshold: float = MEMORY_SIMILARITY_THRESHOLD,
    ):
        """
        Args:
            short_term_max_tokens: 短期记忆Token上限
            short_term_max_items: 短期记忆条目上限
            long_term_top_k: 长期记忆召回Top-K
            long_term_similarity_threshold: 长期记忆相似度阈值
        """
        self._short_term_max_tokens = short_term_max_tokens
        self._short_term_max_items = short_term_max_items
        self._long_term_top_k = long_term_top_k
        self._long_term_similarity_threshold = long_term_similarity_threshold

    def create_instant(self, max_items: int = 100) -> InstantMemory:
        """创建瞬时记忆管理器"""
        return InstantMemory(max_items=max_items)

    def create_short_term(
        self,
        max_tokens: Optional[int] = None,
        max_items: Optional[int] = None,
    ) -> ShortTermMemory:
        """创建短期记忆管理器"""
        return ShortTermMemory(
            max_tokens=max_tokens if max_tokens is not None else self._short_term_max_tokens,
            max_items=max_items if max_items is not None else self._short_term_max_items,
        )

    def create_long_term(
        self,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        scorer: Optional[RelevanceScorer] = None,
    ) -> LongTermMemory:
        """创建长期记忆管理器"""
        return LongTermMemory(
            top_k=top_k if top_k is not None else self._long_term_top_k,
            similarity_threshold=(similarity_threshold if similarity_threshold is not None
                                   else self._long_term_similarity_threshold),
            scorer=scorer,
        )

    def create_by_level(self, level: MemoryLevel, **kwargs) -> BaseMemoryStorage:
        """
        按层级创建记忆管理器

        Args:
            level: 记忆层级
            **kwargs: 传递给对应管理器的参数

        Returns:
            对应层级的记忆管理器实例
        """
        if level == MemoryLevel.INSTANT:
            return self.create_instant(**kwargs)
        elif level == MemoryLevel.SHORT_TERM:
            return self.create_short_term(**kwargs)
        elif level == MemoryLevel.LONG_TERM:
            return self.create_long_term(**kwargs)
        else:
            raise ValueError(f"不支持的记忆层级: {level}")

    def create_all(self) -> dict[MemoryLevel, BaseMemoryStorage]:
        """
        创建全部三层记忆管理器

        Returns:
            {层级: 管理器实例} 字典
        """
        return {
            MemoryLevel.INSTANT: self.create_instant(),
            MemoryLevel.SHORT_TERM: self.create_short_term(),
            MemoryLevel.LONG_TERM: self.create_long_term(),
        }
