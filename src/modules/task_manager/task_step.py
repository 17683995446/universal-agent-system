"""
无上限上下文Agent系统 - 任务步骤模块

标准化TaskStep结构体，包含序列化、验证、状态流转。
TaskStep是任务快照的基本组成单元。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import copy

from src.core.utils.id_generator import generate_task_id


# =============================================================================
# 枚举定义
# =============================================================================

class StepStatus(str, Enum):
    """步骤状态枚举"""
    PENDING = "pending"           # 待执行
    IN_PROGRESS = "in_progress"   # 执行中
    COMPLETED = "completed"       # 已完成
    FAILED = "failed"             # 失败
    SKIPPED = "skipped"           # 跳过


# =============================================================================
# TaskStep 结构体
# =============================================================================

@dataclass
class TaskStep:
    """
    标准化任务步骤结构

    核心字段（必须）：
        - step_id: 步骤唯一标识
        - description: 步骤描述
        - status: 当前状态

    可选字段：
        - result: 执行结果
        - error: 错误信息
        - started_at: 开始时间
        - completed_at: 完成时间
        - metadata: 元数据（执行参数、中间结果等）
    """
    step_id: str
    description: str
    status: StepStatus = StepStatus.PENDING

    result: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理：确保枚举类型"""
        if isinstance(self.status, str):
            self.status = StepStatus(self.status)

    # =========================================================================
    # 序列化方法
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "step_id": self.step_id,
            "description": self.description,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskStep":
        """从字典创建TaskStep"""
        def _parse_dt(val: Any) -> Optional[datetime]:
            if val is None:
                return None
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                return datetime.fromisoformat(val)
            return None

        return cls(
            step_id=data.get("step_id", generate_task_id()),
            description=data.get("description", ""),
            status=StepStatus(data.get("status", "pending")),
            result=data.get("result"),
            error=data.get("error"),
            started_at=_parse_dt(data.get("started_at")),
            completed_at=_parse_dt(data.get("completed_at")),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "TaskStep":
        """从JSON字符串创建TaskStep"""
        return cls.from_dict(json.loads(json_str))

    # =========================================================================
    # 验证方法
    # =========================================================================

    def validate(self) -> tuple[bool, list[str]]:
        """
        验证步骤有效性

        Returns:
            (是否有效, 错误信息列表)
        """
        errors: list[str] = []

        if not self.step_id:
            errors.append("step_id不能为空")

        if not self.description:
            errors.append("description不能为空")

        if not isinstance(self.status, StepStatus):
            errors.append(f"status必须是StepStatus枚举，当前: {type(self.status)}")

        # 状态一致性检查
        if self.status == StepStatus.COMPLETED and not self.result:
            errors.append("已完成步骤应有result")

        if self.status == StepStatus.FAILED and not self.error:
            errors.append("失败步骤应有error")

        if self.completed_at and not self.started_at:
            errors.append("completed_at存在时started_at不能为None")

        if (self.started_at and self.completed_at
                and self.completed_at < self.started_at):
            errors.append("completed_at不能早于started_at")

        return len(errors) == 0, errors

    def is_valid(self) -> bool:
        """快速检查是否有效"""
        valid, _ = self.validate()
        return valid

    # =========================================================================
    # 状态流转
    # =========================================================================

    def start(self) -> "TaskStep":
        """开始执行（支持链式调用）"""
        if self.status not in [StepStatus.PENDING, StepStatus.FAILED]:
            raise ValueError(f"无法从{self.status}状态开始")
        self.status = StepStatus.IN_PROGRESS
        self.started_at = datetime.now()
        return self

    def complete(self, result: str) -> "TaskStep":
        """标记完成（支持链式调用）"""
        if self.status != StepStatus.IN_PROGRESS:
            raise ValueError(f"无法从{self.status}状态完成")
        self.status = StepStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now()
        return self

    def fail(self, error: str) -> "TaskStep":
        """标记失败（支持链式调用）"""
        if self.status != StepStatus.IN_PROGRESS:
            raise ValueError(f"无法从{self.status}状态失败")
        self.status = StepStatus.FAILED
        self.error = error
        self.completed_at = datetime.now()
        return self

    def skip(self, reason: Optional[str] = None) -> "TaskStep":
        """跳过步骤（支持链式调用）"""
        if self.status not in [StepStatus.PENDING, StepStatus.IN_PROGRESS]:
            raise ValueError(f"无法从{self.status}状态跳过")
        self.status = StepStatus.SKIPPED
        if reason:
            self.metadata["skip_reason"] = reason
        self.completed_at = datetime.now()
        return self

    def reset(self) -> "TaskStep":
        """重置为待执行状态（支持链式调用）"""
        self.status = StepStatus.PENDING
        self.result = None
        self.error = None
        self.started_at = None
        self.completed_at = None
        return self

    # =========================================================================
    # 状态检查
    # =========================================================================

    def is_pending(self) -> bool:
        return self.status == StepStatus.PENDING

    def is_in_progress(self) -> bool:
        return self.status == StepStatus.IN_PROGRESS

    def is_completed(self) -> bool:
        return self.status == StepStatus.COMPLETED

    def is_failed(self) -> bool:
        return self.status == StepStatus.FAILED

    def is_skipped(self) -> bool:
        return self.status == StepStatus.SKIPPED

    def is_terminal(self) -> bool:
        """是否处于终止状态（完成/失败/跳过）"""
        return self.status in [StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED]

    # =========================================================================
    # 工具方法
    # =========================================================================

    def copy(self) -> "TaskStep":
        """创建深拷贝"""
        return copy.deepcopy(self)

    def get_duration_seconds(self) -> Optional[float]:
        """获取执行时长（秒），未完成返回None"""
        if not self.started_at:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    def add_metadata(self, key: str, value: Any) -> "TaskStep":
        """添加元数据（支持链式调用）"""
        self.metadata[key] = value
        return self

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """获取元数据"""
        return self.metadata.get(key, default)

    # =========================================================================
    # 魔术方法
    # =========================================================================

    def __str__(self) -> str:
        return f"[{self.status.value}] {self.description[:50]}"

    def __repr__(self) -> str:
        return (f"TaskStep(id={self.step_id}, "
                f"status={self.status.value}, "
                f"desc={self.description[:30]!r})")

    def __hash__(self) -> int:
        return hash(self.step_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TaskStep):
            return False
        return self.step_id == other.step_id


# =============================================================================
# 步骤列表工具函数
# =============================================================================

def filter_by_status(steps: list[TaskStep], status: StepStatus) -> list[TaskStep]:
    """按状态过滤"""
    return [s for s in steps if s.status == status]


def filter_pending(steps: list[TaskStep]) -> list[TaskStep]:
    """过滤待执行步骤"""
    return filter_by_status(steps, StepStatus.PENDING)


def filter_completed(steps: list[TaskStep]) -> list[TaskStep]:
    """过滤已完成步骤"""
    return filter_by_status(steps, StepStatus.COMPLETED)


def filter_failed(steps: list[TaskStep]) -> list[TaskStep]:
    """过滤失败步骤"""
    return filter_by_status(steps, StepStatus.FAILED)


def get_completion_ratio(steps: list[TaskStep]) -> float:
    """
    获取完成比例

    Returns:
        完成比例（0-1），空列表返回0.0
    """
    if not steps:
        return 0.0
    terminal = [s for s in steps if s.is_terminal()]
    return len(terminal) / len(steps)


def get_total_duration_seconds(steps: list[TaskStep]) -> float:
    """获取所有步骤的总执行时长"""
    total = 0.0
    for step in steps:
        duration = step.get_duration_seconds()
        if duration is not None:
            total += duration
    return total


def validate_step_list(steps: list[TaskStep]) -> tuple[bool, list[str]]:
    """
    验证步骤列表

    Returns:
        (是否全部有效, 错误信息列表)
    """
    all_errors: list[str] = []
    for i, step in enumerate(steps):
        valid, errors = step.validate()
        if not valid:
            all_errors.extend([f"TaskStep[{i}]: {e}" for e in errors])
    return len(all_errors) == 0, all_errors


def steps_to_json(steps: list[TaskStep], indent: Optional[int] = None) -> str:
    """批量转换为JSON"""
    return json.dumps(
        [s.to_dict() for s in steps],
        indent=indent,
        ensure_ascii=False
    )


def steps_from_json(json_str: str) -> list[TaskStep]:
    """从JSON批量创建"""
    data = json.loads(json_str)
    return [TaskStep.from_dict(item) for item in data]
