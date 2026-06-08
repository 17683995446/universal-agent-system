"""
无上限上下文Agent系统 - 记忆管理模块

提供三层记忆管理能力：瞬时记忆、短期记忆、长期记忆。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

from src.modules.memory_manager.memory_entry import (
    MemoryEntry,
    MemoryLevel,
    MemorySource,
)
from src.modules.memory_manager.base import BaseMemoryStorage
from src.modules.memory_manager.instant_memory import InstantMemory
from src.modules.memory_manager.short_term_memory import ShortTermMemory
from src.modules.memory_manager.long_term_memory import LongTermMemory, RelevanceScorer, RecallResult
from src.modules.memory_manager.factory import MemoryFactory, MemoryConfig
from src.modules.memory_manager.coordinator import MemoryCoordinator

__all__ = [
    # 枚举
    "MemoryLevel",
    "MemorySource",
    # 数据结构
    "MemoryEntry",
    "RecallResult",
    # 管理器
    "BaseMemoryStorage",
    "InstantMemory",
    "ShortTermMemory",
    "LongTermMemory",
    "RelevanceScorer",
    # 协调器
    "MemoryCoordinator",
    # 工厂
    "MemoryFactory",
    "MemoryConfig",
]
