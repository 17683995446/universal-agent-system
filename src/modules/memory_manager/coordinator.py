"""
无上限上下文Agent系统 - 记忆协调器

统一管理三层记忆的读写，提供记忆存储、召回、清理等协调能力。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-26
"""

import logging
from typing import Optional, Any

from src.modules.memory_manager.memory_entry import (
    MemoryEntry,
    MemoryLevel,
    MemorySource,
)
from src.modules.memory_manager.instant_memory import InstantMemory
from src.modules.memory_manager.short_term_memory import ShortTermMemory
from src.modules.memory_manager.long_term_memory import LongTermMemory, RecallResult
from src.modules.memory_manager.factory import MemoryConfig

logger = logging.getLogger(__name__)


class MemoryCoordinator:
    """
    记忆协调器
    
    统一管理三层记忆，提供：
    - 自动存储：对话内容自动存入短期记忆
    - 语义召回：根据用户输入召回相关记忆
    - 记忆清理：过期记忆自动淘汰
    
    Attributes:
        instant: 瞬时记忆
        short_term: 短期记忆
        long_term: 长期记忆
        config: 记忆配置
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        instant: Optional[InstantMemory] = None,
        short_term: Optional[ShortTermMemory] = None,
        long_term: Optional[LongTermMemory] = None,
    ):
        """
        Args:
            config: 记忆配置
            instant: 瞬时记忆（None则自动创建）
            short_term: 短期记忆（None则自动创建）
            long_term: 长期记忆（None则自动创建）
        """
        self._config = config or MemoryConfig()
        self._instant = instant or InstantMemory()
        self._short_term = short_term or ShortTermMemory(
            max_tokens=self._config.short_term_max_tokens,
            max_items=self._config.short_term_max_items,
        )
        self._long_term = long_term or LongTermMemory(
            top_k=self._config.long_term_top_k,
            similarity_threshold=self._config.long_term_similarity_threshold,
        )

        # 统计
        self._total_stored: int = 0
        self._total_recalled: int = 0

    # =========================================================================
    # 属性
    # =========================================================================

    @property
    def instant(self) -> InstantMemory:
        return self._instant

    @property
    def short_term(self) -> ShortTermMemory:
        return self._short_term

    @property
    def long_term(self) -> LongTermMemory:
        return self._long_term

    # =========================================================================
    # 存储操作
    # =========================================================================

    def store(
        self,
        content: str,
        level: MemoryLevel = MemoryLevel.SHORT_TERM,
        source: MemorySource = MemorySource.USER_INPUT,
        metadata: Optional[dict[str, Any]] = None,
    ) -> MemoryEntry:
        """
        存储记忆
        
        Args:
            content: 记忆内容
            level: 记忆层级
            source: 记忆来源
            metadata: 元数据
        
        Returns:
            存储后的MemoryEntry
        """
        entry = MemoryEntry(
            content=content,
            level=level,
            source=source,
            metadata=metadata or {},
        )

        if level == MemoryLevel.INSTANT:
            self._instant.add(entry)
        elif level == MemoryLevel.SHORT_TERM:
            self._short_term.add(entry)
        elif level == MemoryLevel.LONG_TERM:
            self._long_term.add(entry)

        self._total_stored += 1
        logger.debug(f"Stored memory: level={level.value}, source={source.value}, tokens={entry.token_count}")

        return entry

    def store_conversation(
        self,
        user_input: str,
        assistant_reply: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[MemoryEntry]:
        """
        存储一轮对话（用户输入+助手回复）
        
        Args:
            user_input: 用户输入
            assistant_reply: 助手回复
            metadata: 元数据
        
        Returns:
            存储的MemoryEntry列表
        """
        entries = []

        # 存储用户输入
        user_entry = self.store(
            content=user_input,
            level=MemoryLevel.SHORT_TERM,
            source=MemorySource.USER_INPUT,
            metadata=metadata,
        )
        entries.append(user_entry)

        # 存储助手回复
        if assistant_reply:
            assistant_entry = self.store(
                content=assistant_reply,
                level=MemoryLevel.SHORT_TERM,
                source=MemorySource.ASSISTANT_OUTPUT,
                metadata=metadata,
            )
            entries.append(assistant_entry)

        return entries

    def store_tool_result(
        self,
        tool_name: str,
        result: Any,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[MemoryEntry]:
        """
        存储工具执行结果
        
        Args:
            tool_name: 工具名称
            result: 执行结果
            metadata: 元数据
        
        Returns:
            存储的MemoryEntry
        """
        if result is None:
            return None

        content = f"[Tool: {tool_name}] {result}" if not isinstance(result, str) else f"[Tool: {tool_name}] {result}"

        extra_meta = {"tool_name": tool_name}
        if metadata:
            extra_meta.update(metadata)

        return self.store(
            content=content,
            level=MemoryLevel.SHORT_TERM,
            source=MemorySource.TOOL_RESULT,
            metadata=extra_meta,
        )

    # =========================================================================
    # 召回操作
    # =========================================================================

    def recall(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> RecallResult:
        """
        召回相关记忆
        
        优先从长期记忆语义召回，再从短期记忆补充。
        
        Args:
            query: 查询内容
            top_k: 召回数量（None则使用配置默认值）
            similarity_threshold: 相似度阈值（None则使用配置默认值）
        
        Returns:
            RecallResult 召回结果
        """
        # 从长期记忆召回
        result = self._long_term.recall(
            query=query,
            top_k=top_k if top_k is not None else self._config.long_term_top_k,
            similarity_threshold=similarity_threshold if similarity_threshold is not None else self._config.long_term_similarity_threshold,
        )

        self._total_recalled += result.count
        return result

    def recall_as_context(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> str:
        """
        召回记忆并格式化为上下文字符串
        
        Args:
            query: 查询内容
            top_k: 召回数量
        
        Returns:
            格式化的记忆上下文字符串
        """
        result = self.recall(query, top_k=top_k)

        if not result.recalled:
            return ""

        lines = ["[Relevant Memories]"]
        for i, entry in enumerate(result.recalled, 1):
            lines.append(f"{i}. {entry.content}")

        return "\n".join(lines)

    # =========================================================================
    # 瞬时记忆操作
    # =========================================================================

    def set_instant(self, key: str, value: str) -> None:
        """设置瞬时记忆"""
        self._instant.set_context(key, value)

    def get_instant(self, key: str) -> Optional[str]:
        """获取瞬时记忆"""
        return self._instant.get_context(key)

    def clear_instant(self) -> None:
        """清空瞬时记忆"""
        self._instant.clear()

    # =========================================================================
    # 清理操作
    # =========================================================================

    def clear_short_term(self) -> int:
        """清空短期记忆"""
        count = self._short_term.size
        self._short_term.clear()
        return count

    def clear_long_term(self) -> int:
        """清空长期记忆"""
        count = self._long_term.size
        self._long_term.clear()
        return count

    def clear_all(self) -> dict[str, int]:
        """清空所有记忆"""
        instant_count = self._instant.size
        short_count = self._short_term.size
        long_count = self._long_term.size

        self._instant.clear()
        self._short_term.clear()
        self._long_term.clear()

        return {
            "instant": instant_count,
            "short_term": short_count,
            "long_term": long_count,
        }

    # =========================================================================
    # 统计
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """获取记忆统计"""
        return {
            "instant": {
                "size": self._instant.size,
                "total_tokens": self._instant.total_tokens,
            },
            "short_term": {
                "size": self._short_term.size,
                "total_tokens": self._short_term.total_tokens,
                "max_tokens": self._short_term.max_tokens,
                "usage_ratio": self._short_term.token_usage_ratio,
            },
            "long_term": {
                "size": self._long_term.size,
                "total_tokens": self._long_term.total_tokens,
            },
            "total_stored": self._total_stored,
            "total_recalled": self._total_recalled,
        }
