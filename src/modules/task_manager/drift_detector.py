"""
无上限上下文Agent系统 - 任务漂移检测器

检测任务目标是否发生显著偏离，支持语义相似度比较。
当检测到漂移时抛出TaskDriftDetected异常。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

from typing import Optional, Any
from dataclasses import dataclass
from datetime import datetime

from src.modules.task_manager.task_snapshot import TaskSnapshot
from src.core.exceptions import TaskDriftDetected
from src.core.constants import DRIFT_SIMILARITY_THRESHOLD


# =============================================================================
# 漂移检测结果
# =============================================================================

@dataclass
class DriftCheckResult:
    """
    漂移检测结果

    Attributes:
        has_drift: 是否检测到漂移
        similarity: 相似度分数（0-1）
        threshold: 使用的阈值
        original_goal: 原始目标
        current_intent: 当前意图
        task_id: 任务ID
    """
    has_drift: bool
    similarity: float
    threshold: float
    original_goal: str
    current_intent: str
    task_id: str

    def to_exception(self) -> TaskDriftDetected:
        """转换为异常"""
        return TaskDriftDetected(
            task_id=self.task_id,
            original_goal=self.original_goal,
            current_intent=self.current_intent,
            similarity=self.similarity
        )


# =============================================================================
# 漂移检测器
# =============================================================================

class DriftDetector:
    """
    任务漂移检测器

    检测当前用户意图与原始任务目标的偏离程度。
    使用语义相似度比较（当前为词重叠，生产环境可替换为向量相似度）。
    """

    def __init__(self, threshold: float = DRIFT_SIMILARITY_THRESHOLD):
        """
        Args:
            threshold: 漂移检测阈值（低于此值视为漂移）
        """
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    def check(
        self,
        snapshot: TaskSnapshot,
        current_intent: str,
        custom_threshold: Optional[float] = None,
    ) -> DriftCheckResult:
        """
        检查任务是否发生漂移

        Args:
            snapshot: 任务快照
            current_intent: 当前用户意图/输入
            custom_threshold: 自定义阈值（None则使用默认值）

        Returns:
            DriftCheckResult 检测结果
        """
        threshold = custom_threshold if custom_threshold is not None else self._threshold
        similarity = self._compute_similarity(snapshot.goal, current_intent)

        return DriftCheckResult(
            has_drift=similarity < threshold,
            similarity=similarity,
            threshold=threshold,
            original_goal=snapshot.goal,
            current_intent=current_intent,
            task_id=snapshot.task_id,
        )

    def check_and_raise(
        self,
        snapshot: TaskSnapshot,
        current_intent: str,
        custom_threshold: Optional[float] = None,
    ) -> None:
        """
        检查漂移，检测到则抛出异常

        Args:
            snapshot: 任务快照
            current_intent: 当前用户意图
            custom_threshold: 自定义阈值

        Raises:
            TaskDriftDetected: 检测到漂移时抛出
        """
        result = self.check(snapshot, current_intent, custom_threshold)
        if result.has_drift:
            raise result.to_exception()

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        计算两段文本的相似度

        当前实现：基于词集合的Jaccard相似度
        生产环境：可替换为向量余弦相似度

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            相似度（0-1）
        """
        if not text1 or not text2:
            return 0.0

        # 分词（简单空格分词，小写化）
        words1 = set(self._tokenize(text1))
        words2 = set(self._tokenize(text2))

        if not words1 or not words2:
            return 0.0

        # Jaccard相似度 = |交集| / |并集|
        intersection = words1 & words2
        union = words1 | words2

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _tokenize(self, text: str) -> list[str]:
        """
        简单分词

        - 转小写
        - 按空格分割
        - 过滤标点
        - 过滤停用词（可选）

        Args:
            text: 输入文本

        Returns:
            词列表
        """
        # 转小写并简单清理
        text = text.lower()

        # 基础标点替换为空格
        for char in ".,!?;:\"'()[]{}-":
            text = text.replace(char, " ")

        # 分词
        words = text.split()

        # 过滤短词（可选）
        words = [w for w in words if len(w) > 1]

        return words

    # =========================================================================
    # 批量检测
    # =========================================================================

    def check_batch(
        self,
        snapshots: list[TaskSnapshot],
        current_intents: list[str],
        custom_threshold: Optional[float] = None,
    ) -> list[DriftCheckResult]:
        """
        批量检测漂移

        Args:
            snapshots: 任务快照列表
            current_intents: 当前意图列表（与snapshots一一对应）
            custom_threshold: 自定义阈值

        Returns:
            检测结果列表
        """
        if len(snapshots) != len(current_intents):
            raise ValueError("snapshots和current_intents长度必须相同")

        return [
            self.check(snapshot, intent, custom_threshold)
            for snapshot, intent in zip(snapshots, current_intents)
        ]

    def get_drifted_tasks(
        self,
        snapshots: list[TaskSnapshot],
        current_intents: list[str],
        custom_threshold: Optional[float] = None,
    ) -> list[tuple[TaskSnapshot, DriftCheckResult]]:
        """
        获取所有发生漂移的任务

        Args:
            snapshots: 任务快照列表
            current_intents: 当前意图列表
            custom_threshold: 自定义阈值

        Returns:
            (快照, 检测结果) 元组列表
        """
        results = self.check_batch(snapshots, current_intents, custom_threshold)
        return [
            (snapshot, result)
            for snapshot, result in zip(snapshots, results)
            if result.has_drift
        ]


# =============================================================================
# 周期性漂移检测器
# =============================================================================

class PeriodicDriftDetector:
    """
    周期性漂移检测器

    按轮次间隔自动执行漂移检测。
    """

    def __init__(
        self,
        detector: Optional[DriftDetector] = None,
        check_interval: int = 5,  # 默认每5轮检测一次
    ):
        """
        Args:
            detector: 漂移检测器（None则使用默认）
            check_interval: 检测间隔（轮次）
        """
        self._detector = detector or DriftDetector()
        self._check_interval = check_interval
        self._round_counts: dict[str, int] = {}

    def should_check(self, task_id: str) -> bool:
        """
        检查是否应该执行漂移检测

        Args:
            task_id: 任务ID

        Returns:
            是否应该检测
        """
        count = self._round_counts.get(task_id, 0) + 1
        self._round_counts[task_id] = count
        return count % self._check_interval == 0

    def check_if_needed(
        self,
        snapshot: TaskSnapshot,
        current_intent: str,
    ) -> Optional[DriftCheckResult]:
        """
        按需执行漂移检测

        Args:
            snapshot: 任务快照
            current_intent: 当前意图

        Returns:
            检测结果（不需要检测时返回None）
        """
        if not self.should_check(snapshot.task_id):
            return None
        return self._detector.check(snapshot, current_intent)

    def reset_counter(self, task_id: str) -> None:
        """重置计数器"""
        if task_id in self._round_counts:
            del self._round_counts[task_id]

    def get_counter(self, task_id: str) -> int:
        """获取当前计数"""
        return self._round_counts.get(task_id, 0)
