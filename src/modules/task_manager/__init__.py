"""
无上限上下文Agent系统 - 任务管理器模块

提供任务快照和漂移检测功能。

主要组件：
- TaskStep: 任务步骤结构体
- TaskSnapshot: 任务快照结构体
- SnapshotManager: 快照管理器
- DriftDetector: 漂移检测器
"""

from src.modules.task_manager.task_step import (
    TaskStep,
    StepStatus,
    filter_by_status,
    filter_pending,
    filter_completed,
    filter_failed,
    get_completion_ratio,
    steps_to_json,
    steps_from_json,
)

from src.modules.task_manager.task_snapshot import (
    TaskSnapshot,
    TaskState,
)

from src.modules.task_manager.snapshot_manager import (
    SnapshotManager,
    SnapshotStorage,
    InMemorySnapshotStorage,
)

from src.modules.task_manager.drift_detector import (
    DriftDetector,
    PeriodicDriftDetector,
    DriftCheckResult,
)

__all__ = [
    # TaskStep
    "TaskStep",
    "StepStatus",
    "filter_by_status",
    "filter_pending",
    "filter_completed",
    "filter_failed",
    "get_completion_ratio",
    "steps_to_json",
    "steps_from_json",
    # TaskSnapshot
    "TaskSnapshot",
    "TaskState",
    # SnapshotManager
    "SnapshotManager",
    "SnapshotStorage",
    "InMemorySnapshotStorage",
    # DriftDetector
    "DriftDetector",
    "PeriodicDriftDetector",
    "DriftCheckResult",
]
