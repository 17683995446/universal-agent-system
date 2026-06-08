"""
无上限上下文Agent系统 - 长期记忆管理器

长期记忆：向量存储 + 语义检索按需召回。
生命周期 = 持久化，跨会话保留。

设计原则：
- 向量嵌入存储，支持语义相似度检索
- 四维相关性评分：语义 + 时间衰减 + 任务标签 + 行为权重
- Top-K召回 + 相似度阈值过滤
- Token配额限制召回量

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

from typing import Optional, Any
from collections import OrderedDict
from dataclasses import dataclass, field
import math

from src.modules.memory_manager.base import BaseMemoryStorage
from src.modules.memory_manager.memory_entry import (
    MemoryEntry, MemoryLevel, MemorySource,
)
from src.core.exceptions import MemoryStorageError, MemoryRecallError
from src.core.constants import (
    MEMORY_RECALL_TOP_K,
    MEMORY_SIMILARITY_THRESHOLD,
    MEMORY_TIME_HALFLIFE_HOURS,
    RELEVANCE_SEMANTIC_WEIGHT,
    RELEVANCE_TIME_WEIGHT,
    RELEVANCE_TASK_WEIGHT,
    RELEVANCE_BEHAVIOR_WEIGHT,
)


# =============================================================================
# 召回结果
# =============================================================================

@dataclass
class RecallResult:
    """
    记忆召回结果

    Attributes:
        query: 查询内容
        recalled: 召回的记忆列表（按综合评分降序）
        total_count: 候选数量（过滤前）
        semantic_threshold: 使用的语义相似度阈值
        top_k: 请求的Top-K数量
    """
    query: str
    recalled: list[MemoryEntry] = field(default_factory=list)
    total_count: int = 0
    semantic_threshold: float = MEMORY_SIMILARITY_THRESHOLD
    top_k: int = MEMORY_RECALL_TOP_K

    @property
    def total_tokens(self) -> int:
        """召回记忆的总Token数"""
        return sum(e.token_count for e in self.recalled)

    @property
    def count(self) -> int:
        """实际召回数量"""
        return len(self.recalled)


# =============================================================================
# 相关性评分器
# =============================================================================

class RelevanceScorer:
    """
    四维相关性评分器

    评分公式：
        total = W_semantic * semantic_score
              + W_time * time_decay_score
              + W_task * task_match_score
              + W_behavior * behavior_score
    """

    def __init__(
        self,
        semantic_weight: float = RELEVANCE_SEMANTIC_WEIGHT,
        time_weight: float = RELEVANCE_TIME_WEIGHT,
        task_weight: float = RELEVANCE_TASK_WEIGHT,
        behavior_weight: float = RELEVANCE_BEHAVIOR_WEIGHT,
        time_halflife_hours: float = MEMORY_TIME_HALFLIFE_HOURS,
    ):
        self.semantic_weight = semantic_weight
        self.time_weight = time_weight
        self.task_weight = task_weight
        self.behavior_weight = behavior_weight
        self.time_halflife_hours = time_halflife_hours

    def score(
        self,
        entry: MemoryEntry,
        semantic_score: float = 0.0,
        task_tags: Optional[list[str]] = None,
        behavior_boost: float = 0.0,
    ) -> float:
        """
        计算四维综合评分

        Args:
            entry: 记忆条目
            semantic_score: 语义相似度分数（0-1）
            task_tags: 当前任务标签列表
            behavior_boost: 行为权重加分（0-1）

        Returns:
            综合评分（0-1）
        """
        # 语义维度
        s_semantic = max(0.0, min(1.0, semantic_score))

        # 时间衰减维度
        s_time = entry.compute_time_decay(self.time_halflife_hours)

        # 任务标签维度
        s_task = 0.0
        if task_tags:
            entry_tags = entry.metadata.get("task_tags", [])
            if isinstance(entry_tags, list):
                if len(entry_tags) > 0 and len(task_tags) > 0:
                    overlap = len(set(entry_tags) & set(task_tags))
                    s_task = overlap / max(len(entry_tags), len(task_tags))

        # 行为权重维度
        s_behavior = max(0.0, min(1.0, behavior_boost))

        # 加权求和
        total = (
            self.semantic_weight * s_semantic
            + self.time_weight * s_time
            + self.task_weight * s_task
            + self.behavior_weight * s_behavior
        )

        return max(0.0, min(1.0, total))


# =============================================================================
# 长期记忆管理器
# =============================================================================

class LongTermMemory(BaseMemoryStorage):
    """
    长期记忆管理器

    特点：
    - 向量嵌入存储（本实现使用内存字典，生产环境替换为向量DB）
    - 语义相似度检索
    - 四维相关性评分
    - Top-K召回 + 阈值过滤
    """

    def __init__(
        self,
        top_k: int = MEMORY_RECALL_TOP_K,
        similarity_threshold: float = MEMORY_SIMILARITY_THRESHOLD,
        scorer: Optional[RelevanceScorer] = None,
    ):
        """
        Args:
            top_k: 召回数量上限
            similarity_threshold: 语义相似度阈值
            scorer: 相关性评分器（None则使用默认）
        """
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold
        self._scorer = scorer or RelevanceScorer()
        self._store: OrderedDict[str, MemoryEntry] = OrderedDict()

    # =========================================================================
    # 属性
    # =========================================================================

    @property
    def level(self) -> MemoryLevel:
        return MemoryLevel.LONG_TERM

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

        entry.level = MemoryLevel.LONG_TERM
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
            entry.token_count = 0
            # 内容变更后embedding失效
            entry.embedding = None

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
    # 语义召回
    # =========================================================================

    def recall(
        self,
        query: str,
        query_embedding: Optional[list[float]] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        task_tags: Optional[list[str]] = None,
        behavior_boost: float = 0.0,
    ) -> RecallResult:
        """
        语义召回记忆

        Args:
            query: 查询文本
            query_embedding: 查询向量（如果提供则使用向量检索，否则使用文本匹配）
            top_k: 召回数量（None则使用默认值）
            similarity_threshold: 相似度阈值（None则使用默认值）
            task_tags: 当前任务标签（用于任务维度评分）
            behavior_boost: 行为权重加分

        Returns:
            RecallResult 召回结果
        """
        k = top_k if top_k is not None else self._top_k
        threshold = (similarity_threshold if similarity_threshold is not None
                     else self._similarity_threshold)

        # 计算每个条目的语义相似度
        scored_entries: list[tuple[MemoryEntry, float, float]] = []

        for entry in self._store.values():
            # 跳过过期的
            if entry.is_expired():
                continue

            # 计算语义相似度
            if query_embedding is not None and entry.has_embedding():
                semantic_sim = self._cosine_similarity(
                    query_embedding, entry.embedding  # type: ignore
                )
            else:
                # 降级：简单的文本包含匹配
                semantic_sim = self._text_similarity(query, entry.content)

            # 阈值过滤
            if semantic_sim < threshold:
                continue

            # 计算四维综合评分
            total_score = self._scorer.score(
                entry=entry,
                semantic_score=semantic_sim,
                task_tags=task_tags,
                behavior_boost=behavior_boost,
            )

            scored_entries.append((entry, semantic_sim, total_score))

        # 按综合评分降序排序
        scored_entries.sort(key=lambda x: x[2], reverse=True)

        # Top-K截断
        top_entries = scored_entries[:k]

        # 更新被召回条目的访问信息
        recalled = []
        for entry, semantic_sim, total_score in top_entries:
            entry.touch()
            entry.update_relevance(total_score)
            recalled.append(entry)

        return RecallResult(
            query=query,
            recalled=recalled,
            total_count=len(scored_entries),
            semantic_threshold=threshold,
            top_k=k,
        )

    def recall_by_tags(
        self,
        tags: list[str],
        top_k: Optional[int] = None,
    ) -> RecallResult:
        """
        按标签召回记忆

        Args:
            tags: 标签列表
            top_k: 召回数量

        Returns:
            RecallResult
        """
        k = top_k or self._top_k
        matched: list[tuple[MemoryEntry, float]] = []

        for entry in self._store.values():
            if entry.is_expired():
                continue

            entry_tags = entry.metadata.get("task_tags", [])
            if isinstance(entry_tags, list):
                overlap = len(set(entry_tags) & set(tags))
                if overlap > 0:
                    score = overlap / len(tags)
                    matched.append((entry, score))

        matched.sort(key=lambda x: x[1], reverse=True)
        recalled = [entry for entry, _ in matched[:k]]

        for entry in recalled:
            entry.touch()

        return RecallResult(
            query=f"tags:{','.join(tags)}",
            recalled=recalled,
            total_count=len(matched),
            semantic_threshold=0.0,
            top_k=k,
        )

    # =========================================================================
    # 相似度计算（基础实现，生产环境替换为向量DB）
    # =========================================================================

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """
        计算余弦相似度

        Args:
            a: 向量A
            b: 向量B

        Returns:
            相似度（0-1）
        """
        if len(a) != len(b) or len(a) == 0:
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return max(0.0, min(1.0, dot_product / (norm_a * norm_b)))

    @staticmethod
    def _text_similarity(query: str, content: str) -> float:
        """
        简单的文本相似度（基于词重叠）

        注意：这是降级方案，生产环境应使用向量嵌入。

        Args:
            query: 查询文本
            content: 内容文本

        Returns:
            相似度（0-1）
        """
        if not query or not content:
            return 0.0

        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words:
            return 0.0

        overlap = query_words & content_words
        return len(overlap) / len(query_words)
