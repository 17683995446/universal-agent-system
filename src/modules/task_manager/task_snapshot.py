"""
无上限上下文Agent系统 - 任务快照模块

标准化TaskSnapshot结构体，包含序列化、验证、工具方法。
TaskSnapshot是任务状态持久化的核心数据结构。

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

from src.core.utils.id_generator import generate_task_id, generate_session_id
from src.modules.task_manager.task_step import (
    TaskStep, StepStatus,
    filter_pending, filter_completed, filter_failed,
    get_completion_ratio, steps_to_json, steps_from_json,
)


# =============================================================================
# 任务状态枚举
# =============================================================================

class TaskState(str, Enum):
    """任务整体状态枚举"""
    PENDING = "pending"           # 待开始
    IN_PROGRESS = "in_progress"   # 进行中
    COMPLETED = "completed"       # 已完成
    FAILED = "failed"             # 失败
    SUSPENDED = "suspended"       # 暂停


# =============================================================================
# TaskSnapshot 结构体
# =============================================================================

@dataclass
class TaskSnapshot:
    """
    标准化任务状态快照

    核心字段（必须）：
        - task_id: 任务唯一标识
        - session_id: 所属会话标识
        - goal: 任务目标

    步骤字段：
        - completed_steps: 已完成步骤列表
        - pending_steps: 待执行步骤列表

    状态字段：
        - variables: 关键变量（任务上下文）
        - constraints: 约束条件
        - error_log: 错误记录
        - progress_ratio: 进度比例（0-1，自动计算）

    系统字段（自动生成）：
        - created_at: 创建时间
        - updated_at: 更新时间
        - version: 快照版本（用于乐观锁）
    """
    # 核心字段
    task_id: str
    session_id: str
    goal: str

    # 步骤字段
    completed_steps: list[TaskStep] = field(default_factory=list)
    pending_steps: list[TaskStep] = field(default_factory=list)

    # 状态字段
    variables: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    error_log: list[dict[str, Any]] = field(default_factory=list)
    state: TaskState = TaskState.PENDING

    # 系统字段
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1  # 乐观锁版本号

    def __post_init__(self):
        """初始化后处理"""
        if isinstance(self.state, str):
            self.state = TaskState(self.state)
        self._update_progress()

    # =========================================================================
    # 序列化方法
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "goal": self.goal,
            "completed_steps": [s.to_dict() for s in self.completed_steps],
            "pending_steps": [s.to_dict() for s in self.pending_steps],
            "variables": self.variables,
            "constraints": self.constraints,
            "error_log": self.error_log,
            "state": self.state.value,
            "progress_ratio": self.progress_ratio,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskSnapshot":
        """从字典创建TaskSnapshot"""
        def _parse_dt(val: Any) -> datetime:
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                return datetime.fromisoformat(val)
            return datetime.now()

        return cls(
            task_id=data.get("task_id", generate_task_id()),
            session_id=data.get("session_id", generate_session_id()),
            goal=data.get("goal", ""),
            completed_steps=[TaskStep.from_dict(s) for s in data.get("completed_steps", [])],
            pending_steps=[TaskStep.from_dict(s) for s in data.get("pending_steps", [])],
            variables=data.get("variables", {}),
            constraints=data.get("constraints", []),
            error_log=data.get("error_log", []),
            state=TaskState(data.get("state", "pending")),
            created_at=_parse_dt(data.get("created_at")),
            updated_at=_parse_dt(data.get("updated_at")),
            version=data.get("version", 1),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "TaskSnapshot":
        """从JSON字符串创建TaskSnapshot"""
        return cls.from_dict(json.loads(json_str))

    # =========================================================================
    # 验证方法
    # =========================================================================

    def validate(self) -> tuple[bool, list[str]]:
        """
        验证快照有效性

        Returns:
            (是否有效, 错误信息列表)
        """
        errors: list[str] = []

        if not self.task_id:
            errors.append("task_id不能为空")

        if not self.session_id:
            errors.append("session_id不能为空")

        if not self.goal:
            errors.append("goal不能为空")

        if not isinstance(self.state, TaskState):
            errors.append(f"state必须是TaskState枚举，当前: {type(self.state)}")

        if self.version < 1:
            errors.append(f"version必须>=1，当前: {self.version}")

        # 验证所有步骤
        all_steps = self.completed_steps + self.pending_steps
        for i, step in enumerate(all_steps):
            valid, step_errors = step.validate()
            if not valid:
                errors.extend([f"步骤[{i}]: {e}" for e in step_errors])

        # 检查步骤ID唯一性
        step_ids = [s.step_id for s in all_steps]
        if len(step_ids) != len(set(step_ids)):
            errors.append("存在重复的step_id")

        return len(errors) == 0, errors

    def is_valid(self) -> bool:
        """快速检查是否有效"""
        valid, _ = self.validate()
        return valid

    # =========================================================================
    # 进度管理
    # =========================================================================

    def _update_progress(self) -> None:
        """更新进度比例和更新时间"""
        all_steps = self.completed_steps + self.pending_steps
        if all_steps:
            completed = len([s for s in all_steps if s.is_terminal()])
            self.progress_ratio = completed / len(all_steps)
        else:
            self.progress_ratio = 0.0
        self.updated_at = datetime.now()

    def update_progress(self) -> "TaskSnapshot":
        """手动触发进度更新（支持链式调用）"""
        self._update_progress()
        return self

    def get_progress_percentage(self) -> float:
        """获取进度百分比（0-100）"""
        return self.progress_ratio * 100

    # =========================================================================
    # 步骤管理
    # =========================================================================

    def add_step(self, description: str, **kwargs) -> TaskStep:
        """
        添加新步骤到待执行列表

        Args:
            description: 步骤描述
            **kwargs: 其他TaskStep参数

        Returns:
            创建的TaskStep
        """
        step = TaskStep(
            step_id=generate_task_id(),
            description=description,
            **kwargs
        )
        self.pending_steps.append(step)
        self._update_progress()
        return step

    def start_step(self, step_id: str) -> Optional[TaskStep]:
        """
        开始执行指定步骤

        Args:
            step_id: 步骤ID

        Returns:
            开始的步骤，不存在则返回None
        """
        for step in self.pending_steps:
            if step.step_id == step_id:
                step.start()
                self.state = TaskState.IN_PROGRESS
                self._update_progress()
                return step
        return None

    def complete_step(self, step_id: str, result: str) -> Optional[TaskStep]:
        """
        完成指定步骤

        Args:
            step_id: 步骤ID
            result: 执行结果

        Returns:
            完成的步骤，不存在则返回None
        """
        for step in self.pending_steps:
            if step.step_id == step_id:
                step.complete(result)
                self.pending_steps.remove(step)
                self.completed_steps.append(step)
                self._update_progress()
                self._check_completion()
                return step
        return None

    def fail_step(self, step_id: str, error: str) -> Optional[TaskStep]:
        """
        标记步骤失败

        Args:
            step_id: 步骤ID
            error: 错误信息

        Returns:
            失败的步骤，不存在则返回None
        """
        for step in self.pending_steps:
            if step.step_id == step_id:
                step.fail(error)
                self.pending_steps.remove(step)
                self.completed_steps.append(step)
                self._log_error(f"步骤失败: {step_id}", error)
                self._update_progress()
                self._check_completion()
                return step
        return None

    def _check_completion(self) -> None:
        """检查任务是否完成"""
        if not self.pending_steps:
            # 检查是否有失败的步骤
            has_failed = any(s.is_failed() for s in self.completed_steps)
            if has_failed:
                self.state = TaskState.FAILED
            else:
                self.state = TaskState.COMPLETED

    # =========================================================================
    # 变量管理
    # =========================================================================

    def set_variable(self, key: str, value: Any) -> "TaskSnapshot":
        """设置变量（支持链式调用）"""
        self.variables[key] = value
        self._update_progress()
        return self

    def get_variable(self, key: str, default: Any = None) -> Any:
        """获取变量"""
        return self.variables.get(key, default)

    def remove_variable(self, key: str) -> "TaskSnapshot":
        """删除变量（支持链式调用）"""
        if key in self.variables:
            del self.variables[key]
        self._update_progress()
        return self

    # =========================================================================
    # 错误管理
    # =========================================================================

    def _log_error(self, context: str, error: str) -> None:
        """记录错误"""
        self.error_log.append({
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "error": error,
        })

    def log_error(self, context: str, error: str) -> "TaskSnapshot":
        """记录错误（支持链式调用）"""
        self._log_error(context, error)
        self._update_progress()
        return self

    def get_recent_errors(self, count: int = 5) -> list[dict[str, Any]]:
        """获取最近的错误记录"""
        return self.error_log[-count:] if self.error_log else []

    def has_errors(self) -> bool:
        """是否有错误记录"""
        return len(self.error_log) > 0

    def clear_errors(self) -> "TaskSnapshot":
        """清空错误记录（支持链式调用）"""
        self.error_log.clear()
        self._update_progress()
        return self

    # =========================================================================
    # 约束管理
    # =========================================================================

    def add_constraint(self, constraint: str) -> "TaskSnapshot":
        """添加约束条件（支持链式调用）"""
        if constraint not in self.constraints:
            self.constraints.append(constraint)
        self._update_progress()
        return self

    def remove_constraint(self, constraint: str) -> "TaskSnapshot":
        """移除约束条件（支持链式调用）"""
        if constraint in self.constraints:
            self.constraints.remove(constraint)
        self._update_progress()
        return self

    # =========================================================================
    # 状态检查
    # =========================================================================

    def is_pending(self) -> bool:
        return self.state == TaskState.PENDING

    def is_in_progress(self) -> bool:
        return self.state == TaskState.IN_PROGRESS

    def is_completed(self) -> bool:
        return self.state == TaskState.COMPLETED

    def is_failed(self) -> bool:
        return self.state == TaskState.FAILED

    def is_suspended(self) -> bool:
        return self.state == TaskState.SUSPENDED

    def is_terminal(self) -> bool:
        """是否处于终止状态"""
        return self.state in [TaskState.COMPLETED, TaskState.FAILED]

    # =========================================================================
    # 工具方法
    # =========================================================================

    def copy(self) -> "TaskSnapshot":
        """创建深拷贝"""
        return copy.deepcopy(self)

    def increment_version(self) -> "TaskSnapshot":
        """递增版本号（支持链式调用）"""
        self.version += 1
        self._update_progress()
        return self

    def get_all_steps(self) -> list[TaskStep]:
        """获取所有步骤（已完成+待执行）"""
        return self.completed_steps + self.pending_steps

    def get_step_by_id(self, step_id: str) -> Optional[TaskStep]:
        """通过ID查找步骤"""
        for step in self.get_all_steps():
            if step.step_id == step_id:
                return step
        return None

    def get_duration_seconds(self) -> float:
        """获取任务总执行时长"""
        return (self.updated_at - self.created_at).total_seconds()

    def suspend(self) -> "TaskSnapshot":
        """暂停任务（支持链式调用）"""
        self.state = TaskState.SUSPENDED
        self._update_progress()
        return self

    def resume(self) -> "TaskSnapshot":
        """恢复任务（支持链式调用）"""
        if self.state == TaskState.SUSPENDED:
            self.state = TaskState.IN_PROGRESS
        self._update_progress()
        return self

    # =========================================================================
    # 魔术方法
    # =========================================================================

    def __str__(self) -> str:
        return (f"[{self.state.value}] {self.goal[:50]} "
                f"({self.get_progress_percentage():.0f}%)")

    def __repr__(self) -> str:
        return (f"TaskSnapshot(id={self.task_id}, "
                f"state={self.state.value}, "
                f"progress={self.progress_ratio:.0%})")

    def __hash__(self) -> int:
        return hash(self.task_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TaskSnapshot):
            return False
        return self.task_id == other.task_id
