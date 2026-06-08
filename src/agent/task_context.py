"""
无上限上下文Agent系统 - 任务上下文管理器

管理任务快照、漂移检测、任务状态追踪。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-26
"""

import logging
from typing import Optional, Any

from src.modules.task_manager.task_snapshot import TaskSnapshot, TaskState
from src.modules.task_manager.drift_detector import DriftDetector, DriftCheckResult, PeriodicDriftDetector
from src.modules.task_manager.snapshot_manager import SnapshotManager, InMemorySnapshotStorage
from src.core.utils.id_generator import generate_task_id, generate_session_id
from src.core.exceptions import TaskDriftDetected

logger = logging.getLogger(__name__)


class TaskContextManager:
    """
    任务上下文管理器

    统一管理任务快照和漂移检测：
    - 自动快照保存
    - 定期漂移检测
    - 任务状态追踪

    Attributes:
        snapshot_manager: 快照管理器
        drift_detector: 漂移检测器
        periodic_detector: 周期性漂移检测器
        current_task: 当前任务快照
    """

    def __init__(
        self,
        task_goal: str = "",
        auto_save: bool = True,
        drift_check_interval: int = 5,
        drift_threshold: float = 0.6,
    ):
        """
        Args:
            task_goal: 任务目标
            auto_save: 是否自动保存快照
            drift_check_interval: 漂移检测间隔（轮次）
            drift_threshold: 漂移检测阈值
        """
        self._snapshot_manager = SnapshotManager(storage=InMemorySnapshotStorage())
        self._drift_detector = DriftDetector(threshold=drift_threshold)
        self._periodic_detector = PeriodicDriftDetector(
            detector=self._drift_detector,
            check_interval=drift_check_interval,
        )
        self._auto_save = auto_save

        # 当前任务
        self._current_task: Optional[TaskSnapshot] = None
        if task_goal:
            self.create_task(task_goal)

        # 统计
        self._drift_checks: int = 0
        self._drifts_detected: int = 0

    # =========================================================================
    # 属性
    # =========================================================================

    @property
    def current_task(self) -> Optional[TaskSnapshot]:
        """当前任务快照"""
        return self._current_task

    @property
    def has_active_task(self) -> bool:
        """是否有活动任务"""
        return self._current_task is not None

    # =========================================================================
    # 任务管理
    # =========================================================================

    def create_task(self, goal: str, **kwargs) -> TaskSnapshot:
        """
        创建新任务

        Args:
            goal: 任务目标
            **kwargs: 其他TaskSnapshot参数

        Returns:
            创建的任务快照
        """
        task_id = kwargs.pop("task_id", generate_task_id())
        session_id = kwargs.pop("session_id", generate_session_id())

        self._current_task = TaskSnapshot(
            task_id=task_id,
            session_id=session_id,
            goal=goal,
            **kwargs
        )
        logger.info(f"Task created: {goal[:50]}...")

        if self._auto_save:
            self.save_snapshot()

        return self._current_task

    def set_task(self, snapshot: TaskSnapshot) -> None:
        """设置当前任务"""
        self._current_task = snapshot

    def update_task_goal(self, new_goal: str) -> None:
        """更新任务目标"""
        if self._current_task:
            self._current_task.goal = new_goal
            self._current_task._update_progress()

    def complete_task(self) -> None:
        """完成任务"""
        if self._current_task:
            self._current_task.state = TaskState.COMPLETED
            if self._auto_save:
                self.save_snapshot()
            logger.info("Task completed")

    def fail_task(self, error: str) -> None:
        """标记任务失败"""
        if self._current_task:
            self._current_task.state = TaskState.FAILED
            self._current_task.log_error("task_failure", error)
            if self._auto_save:
                self.save_snapshot()
            logger.error(f"Task failed: {error}")

    def clear_task(self) -> None:
        """清除当前任务"""
        self._current_task = None
        logger.info("Task cleared")

    # =========================================================================
    # 快照管理
    # =========================================================================

    def save_snapshot(self) -> bool:
        """
        保存当前任务快照

        Returns:
            是否保存成功
        """
        if not self._current_task:
            logger.warning("No active task to save")
            return False

        try:
            # 检查任务是否已在管理器中
            if self._snapshot_manager.exists(self._current_task.task_id):
                # 更新现有快照
                self._snapshot_manager.save(self._current_task.task_id)
            else:
                # 创建新快照
                self._snapshot_manager.create(
                    task_id=self._current_task.task_id,
                    session_id=self._current_task.session_id,
                    goal=self._current_task.goal,
                    state=self._current_task.state,
                    completed_steps=self._current_task.completed_steps,
                    pending_steps=self._current_task.pending_steps,
                    variables=self._current_task.variables,
                    constraints=self._current_task.constraints,
                    error_log=self._current_task.error_log,
                )
            logger.debug(f"Snapshot saved: {self._current_task.task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            return False

    def load_snapshot(self, task_id: str) -> Optional[TaskSnapshot]:
        """
        加载任务快照

        Args:
            task_id: 任务ID

        Returns:
            任务快照，不存在则返回None
        """
        try:
            snapshot = self._snapshot_manager.get_optional(task_id)
            if snapshot:
                self._current_task = snapshot
                logger.info(f"Snapshot loaded: {task_id}")
            return snapshot
        except Exception as e:
            logger.error(f"Failed to load snapshot: {e}")
            return None

    def restore_latest(self) -> Optional[TaskSnapshot]:
        """恢复最新的快照"""
        all_ids = self._snapshot_manager.list_all()
        if not all_ids:
            return None

        # 获取最新的快照（按更新时间）
        latest = None
        latest_time = None
        for task_id in all_ids:
            snapshot = self._snapshot_manager.get_optional(task_id)
            if snapshot:
                if latest_time is None or snapshot.updated_at > latest_time:
                    latest = snapshot
                    latest_time = snapshot.updated_at

        if latest:
            self._current_task = latest
            logger.info(f"Latest snapshot restored: {latest.task_id}")
        return latest

    # =========================================================================
    # 漂移检测
    # =========================================================================

    def check_drift(self, current_intent: str) -> Optional[DriftCheckResult]:
        """
        检查任务漂移

        Args:
            current_intent: 当前用户意图/输入

        Returns:
            检测结果，无任务时返回None
        """
        if not self._current_task:
            return None

        result = self._drift_detector.check(self._current_task, current_intent)

        if result.has_drift:
            self._drifts_detected += 1
            logger.warning(
                f"Drift detected: similarity={result.similarity:.2f}, "
                f"threshold={result.threshold}"
            )

        return result

    def check_drift_periodic(self, current_intent: str) -> Optional[DriftCheckResult]:
        """
        周期性检查漂移

        按配置的间隔轮次执行检测。

        Args:
            current_intent: 当前用户意图

        Returns:
            检测结果（不需要检测时返回None）
        """
        if not self._current_task:
            return None

        # 增加计数（无论是否实际检测）
        self._drift_checks += 1

        result = self._periodic_detector.check_if_needed(
            self._current_task, current_intent
        )

        if result and result.has_drift:
            self._drifts_detected += 1
            logger.warning(
                f"Periodic drift detected: similarity={result.similarity:.2f}"
            )

        return result

    def check_drift_and_raise(self, current_intent: str) -> None:
        """
        检查漂移，检测到则抛出异常

        Args:
            current_intent: 当前用户意图

        Raises:
            TaskDriftDetected: 检测到漂移时抛出
        """
        if not self._current_task:
            return

        self._drift_detector.check_and_raise(self._current_task, current_intent)

    def is_drifted(self, current_intent: str, threshold: Optional[float] = None) -> bool:
        """
        快速检查是否发生漂移

        Args:
            current_intent: 当前用户意图
            threshold: 自定义阈值

        Returns:
            是否发生漂移
        """
        if not self._current_task:
            return False

        result = self._drift_detector.check(
            self._current_task, current_intent, threshold
        )
        return result.has_drift

    # =========================================================================
    # 步骤管理
    # =========================================================================

    def add_step(self, description: str, **kwargs) -> Optional[Any]:
        """
        添加任务步骤

        Args:
            description: 步骤描述
            **kwargs: 其他参数

        Returns:
            创建的步骤，无任务时返回None
        """
        if not self._current_task:
            logger.warning("No active task to add step")
            return None

        step = self._current_task.add_step(description, **kwargs)

        if self._auto_save:
            self.save_snapshot()

        return step

    def start_step(self, step_id: str) -> bool:
        """
        开始执行步骤

        Args:
            step_id: 步骤ID

        Returns:
            是否成功
        """
        if not self._current_task:
            return False

        try:
            self._current_task.start_step(step_id)
            return True
        except ValueError:
            return False

    def complete_step(self, step_id: str, result: str) -> bool:
        """
        完成步骤

        Args:
            step_id: 步骤ID
            result: 执行结果

        Returns:
            是否成功
        """
        if not self._current_task:
            return False

        # 先尝试开始步骤（如果还没开始）
        try:
            self._current_task.start_step(step_id)
        except ValueError:
            # 可能已经在进行中或已完成
            pass

        completed = self._current_task.complete_step(step_id, result)

        if completed and self._auto_save:
            self.save_snapshot()

        return completed is not None

    # =========================================================================
    # 统计
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        return {
            "has_active_task": self.has_active_task,
            "task_id": self._current_task.task_id if self._current_task else None,
            "task_state": self._current_task.state.value if self._current_task else None,
            "task_progress": self._current_task.progress_ratio if self._current_task else 0,
            "drift_checks": self._drift_checks,
            "drifts_detected": self._drifts_detected,
            "snapshot_count": len(self._snapshot_manager.list_all()),
        }

    def reset_stats(self) -> None:
        """重置统计"""
        self._drift_checks = 0
        self._drifts_detected = 0
