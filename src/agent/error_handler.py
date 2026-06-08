"""
无上限上下文Agent系统 - 错误处理器

分级错误处理策略、自动重试、错误记录与诊断。

核心设计：
- ErrorCategory: 错误分级（FATAL/RECOVERABLE/DEGRADED/WARNING）
- ErrorRecord: 结构化错误记录
- RetryPolicy: 可配置重试策略
- ErrorHandler: 统一错误处理入口

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-26
"""

import time
import logging
import traceback
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional, Any, Callable

from src.core.exceptions import (
    BaseAgentException,
    LLMClientError,
    TokenLimitExceeded,
    IntegrityCheckFailed,
    PurificationError,
    MemoryStorageError,
    MemoryRecallError,
    TaskDriftDetected,
    ConfigurationError,
    is_agent_exception,
    get_error_context,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 错误分级
# =============================================================================

class ErrorCategory(IntEnum):
    """错误分级枚举"""
    WARNING = 0        # 警告：不影响主流程
    DEGRADED = 1       # 降级：部分功能不可用，主流程继续
    RECOVERABLE = 2    # 可恢复：可通过重试解决
    FATAL = 3          # 致命：无法恢复，需要人工介入


# 错误类型到分级的映射
_ERROR_CATEGORY_MAP: dict[type[BaseAgentException], ErrorCategory] = {
    # 致命错误
    ConfigurationError: ErrorCategory.FATAL,
    # 可恢复错误
    LLMClientError: ErrorCategory.RECOVERABLE,
    TokenLimitExceeded: ErrorCategory.RECOVERABLE,
    IntegrityCheckFailed: ErrorCategory.RECOVERABLE,
    PurificationError: ErrorCategory.RECOVERABLE,
    # 降级错误
    MemoryStorageError: ErrorCategory.DEGRADED,
    MemoryRecallError: ErrorCategory.DEGRADED,
    TaskDriftDetected: ErrorCategory.DEGRADED,
}


def classify_error(error: Exception) -> ErrorCategory:
    """
    对异常进行分级

    Args:
        error: 异常对象

    Returns:
        ErrorCategory 错误级别
    """
    if isinstance(error, BaseAgentException):
        return _ERROR_CATEGORY_MAP.get(type(error), ErrorCategory.RECOVERABLE)
    return ErrorCategory.RECOVERABLE


# =============================================================================
# 错误记录
# =============================================================================

@dataclass
class ErrorRecord:
    """
    结构化错误记录

    Attributes:
        error_id: 错误唯一ID
        category: 错误级别
        error_type: 异常类型名
        message: 错误消息
        context: 错误上下文
        timestamp: 发生时间
        traceback_str: 堆栈信息
        turn_id: 关联的回合ID（可选）
        component: 发生错误的组件
        recovered: 是否已恢复
        recovery_method: 恢复方式
    """
    error_id: str = ""
    category: ErrorCategory = ErrorCategory.RECOVERABLE
    error_type: str = ""
    message: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    traceback_str: str = ""
    turn_id: str = ""
    component: str = ""
    recovered: bool = False
    recovery_method: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "error_id": self.error_id,
            "category": int(self.category),
            "category_name": self.category.name,
            "error_type": self.error_type,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp,
            "turn_id": self.turn_id,
            "component": self.component,
            "recovered": self.recovered,
            "recovery_method": self.recovery_method,
        }

    @property
    def age_seconds(self) -> float:
        """错误年龄"""
        return time.time() - self.timestamp


# =============================================================================
# 重试策略
# =============================================================================

@dataclass
class RetryPolicy:
    """
    重试策略配置

    Attributes:
        max_attempts: 最大重试次数（含首次调用）
        base_delay: 基础延迟（秒）
        max_delay: 最大延迟（秒）
        backoff_factor: 退避因子
        retryable_categories: 可重试的错误级别
        retryable_exceptions: 可重试的异常类型列表
    """
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    retryable_categories: set[ErrorCategory] = field(
        default_factory=lambda: {ErrorCategory.RECOVERABLE}
    )
    retryable_exceptions: tuple[type[Exception], ...] = ()

    def should_retry(
        self,
        error: Exception,
        attempt: int,
    ) -> bool:
        """
        判断是否应该重试

        Args:
            error: 异常对象
            attempt: 当前尝试次数（从1开始）

        Returns:
            是否应该重试
        """
        if attempt >= self.max_attempts:
            return False

        category = classify_error(error)
        if category not in self.retryable_categories:
            return False

        # 检查特定异常类型
        if self.retryable_exceptions and not isinstance(error, self.retryable_exceptions):
            return False

        return True

    def get_delay(self, attempt: int) -> float:
        """
        计算重试延迟

        Args:
            attempt: 当前尝试次数（从1开始）

        Returns:
            延迟秒数
        """
        delay = self.base_delay * (self.backoff_factor ** (attempt - 1))
        return min(delay, self.max_delay)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "max_attempts": self.max_attempts,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "backoff_factor": self.backoff_factor,
            "retryable_categories": [int(c) for c in self.retryable_categories],
        }


# =============================================================================
# 错误处理器
# =============================================================================

class ErrorHandler:
    """
    统一错误处理器

    提供分级错误处理、自动重试、错误记录和诊断功能。

    Attributes:
        retry_policy: 重试策略
        error_history: 错误历史记录
    """

    def __init__(
        self,
        retry_policy: Optional[RetryPolicy] = None,
        max_history: int = 100,
    ):
        """
        Args:
            retry_policy: 重试策略（None则使用默认）
            max_history: 最大错误记录数
        """
        self._retry_policy = retry_policy or RetryPolicy()
        self._error_history: list[ErrorRecord] = []
        self._max_history = max_history
        self._total_errors: int = 0
        self._total_recovered: int = 0

    # =========================================================================
    # 属性
    # =========================================================================

    @property
    def retry_policy(self) -> RetryPolicy:
        """重试策略"""
        return self._retry_policy

    @property
    def error_history(self) -> list[ErrorRecord]:
        """错误历史（只读副本）"""
        return self._error_history.copy()

    # =========================================================================
    # 核心方法
    # =========================================================================

    def record_error(
        self,
        error: Exception,
        component: str = "",
        turn_id: str = "",
        context: Optional[dict[str, Any]] = None,
    ) -> ErrorRecord:
        """
        记录错误

        Args:
            error: 异常对象
            component: 组件名称
            turn_id: 关联回合ID
            context: 额外上下文

        Returns:
            ErrorRecord
        """
        category = classify_error(error)

        # 提取已有上下文
        error_context = {}
        if is_agent_exception(error):
            error_context = get_error_context(error) or {}

        if context:
            error_context.update(context)

        record = ErrorRecord(
            error_id=f"err_{int(time.time() * 1000)}_{id(error) % 10000:04d}",
            category=category,
            error_type=type(error).__name__,
            message=str(error),
            context=error_context,
            traceback_str=traceback.format_exc(),
            turn_id=turn_id,
            component=component,
        )

        self._add_to_history(record)
        return record

    def execute_with_retry(
        self,
        fn: Callable[..., Any],
        component: str = "",
        turn_id: str = "",
        context: Optional[dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        带重试的函数执行

        Args:
            fn: 要执行的函数
            component: 组件名称
            turn_id: 关联回合ID
            context: 额外上下文
            *args: 函数位置参数
            **kwargs: 函数关键字参数

        Returns:
            函数返回值

        Raises:
            最后一次异常（重试耗尽后）
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, self._retry_policy.max_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_error = e
                record = self.record_error(e, component, turn_id, context)

                if not self._retry_policy.should_retry(e, attempt):
                    logger.error(
                        f"[{component}] Error not retryable "
                        f"(category={record.category.name}, attempt={attempt}): {e}"
                    )
                    raise

                delay = self._retry_policy.get_delay(attempt)
                logger.warning(
                    f"[{component}] Retryable error (attempt={attempt}/{self._retry_policy.max_attempts}), "
                    f"retrying in {delay:.1f}s: {e}"
                )
                time.sleep(delay)

        # 所有重试耗尽
        if last_error:
            record = self.record_error(last_error, component, turn_id, {
                **(context or {}),
                "retry_exhausted": True,
                "total_attempts": self._retry_policy.max_attempts,
            })
            raise last_error

    def handle_error(
        self,
        error: Exception,
        component: str = "",
        turn_id: str = "",
        context: Optional[dict[str, Any]] = None,
    ) -> ErrorRecord:
        """
        处理错误（记录 + 分类处理）

        根据错误级别采取不同策略：
        - WARNING: 仅记录日志
        - DEGRADED: 记录日志 + 标记降级
        - RECOVERABLE: 记录日志 + 建议重试
        - FATAL: 记录日志 + 标记致命

        Args:
            error: 异常对象
            component: 组件名称
            turn_id: 关联回合ID
            context: 额外上下文

        Returns:
            ErrorRecord
        """
        record = self.record_error(error, component, turn_id, context)

        if record.category == ErrorCategory.WARNING:
            logger.warning(f"[{component}] Warning: {error}")
        elif record.category == ErrorCategory.DEGRADED:
            logger.warning(
                f"[{component}] Degraded mode: {error}. "
                f"Continuing with reduced functionality."
            )
        elif record.category == ErrorCategory.RECOVERABLE:
            logger.error(
                f"[{component}] Recoverable error: {error}. "
                f"Retry may resolve this issue."
            )
        elif record.category == ErrorCategory.FATAL:
            logger.critical(
                f"[{component}] Fatal error: {error}. "
                f"Manual intervention may be required."
            )

        return record

    def mark_recovered(
        self,
        error_id: str,
        method: str = "automatic",
    ) -> bool:
        """
        标记错误已恢复

        Args:
            error_id: 错误ID
            method: 恢复方式

        Returns:
            是否成功标记
        """
        for record in self._error_history:
            if record.error_id == error_id:
                record.recovered = True
                record.recovery_method = method
                self._total_recovered += 1
                logger.info(f"Error {error_id} recovered via {method}")
                return True
        return False

    # =========================================================================
    # 历史管理
    # =========================================================================

    def _add_to_history(self, record: ErrorRecord) -> None:
        """添加到错误历史"""
        self._error_history.append(record)
        self._total_errors += 1

        # 限制历史长度
        if len(self._error_history) > self._max_history:
            self._error_history = self._error_history[-self._max_history:]

    # =========================================================================
    # 统计与诊断
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """获取错误统计"""
        category_counts = {}
        component_counts = {}
        for record in self._error_history:
            cat = record.category.name
            category_counts[cat] = category_counts.get(cat, 0) + 1
            if record.component:
                component_counts[record.component] = component_counts.get(record.component, 0) + 1

        return {
            "total_errors": self._total_errors,
            "total_recovered": self._total_recovered,
            "recovery_rate": (
                self._total_recovered / self._total_errors
                if self._total_errors > 0
                else 0.0
            ),
            "by_category": category_counts,
            "by_component": component_counts,
            "history_size": len(self._error_history),
            "retry_policy": self._retry_policy.to_dict(),
        }

    def get_recent_errors(self, limit: int = 10) -> list[dict[str, Any]]:
        """获取最近的错误记录"""
        recent = self._error_history[-limit:]
        return [r.to_dict() for r in recent]

    def get_unrecovered_errors(self) -> list[ErrorRecord]:
        """获取未恢复的错误"""
        return [r for r in self._error_history if not r.recovered]

    def get_diagnosis(self) -> dict[str, Any]:
        """
        获取错误诊断报告

        Returns:
            诊断信息
        """
        unrecovered = self.get_unrecovered_errors()
        recent = self._error_history[-5:] if self._error_history else []

        return {
            "status": "healthy" if not unrecovered else "degraded",
            "total_errors": self._total_errors,
            "unrecovered_count": len(unrecovered),
            "recovery_rate": (
                self._total_recovered / self._total_errors
                if self._total_errors > 0
                else 1.0
            ),
            "recent_errors": [
                {
                    "error_id": r.error_id,
                    "category": r.category.name,
                    "message": r.message,
                    "component": r.component,
                    "age_seconds": round(r.age_seconds, 1),
                }
                for r in recent
            ],
            "recommendations": self._generate_recommendations(unrecovered),
        }

    def _generate_recommendations(self, unrecovered: list[ErrorRecord]) -> list[str]:
        """生成修复建议"""
        recommendations = []

        if not unrecovered:
            return ["系统运行正常，无未恢复错误。"]

        # 按组件分组
        by_component: dict[str, list[ErrorRecord]] = {}
        for r in unrecovered:
            comp = r.component or "unknown"
            by_component.setdefault(comp, []).append(r)

        for comp, errors in by_component.items():
            fatal = [e for e in errors if e.category == ErrorCategory.FATAL]
            if fatal:
                recommendations.append(
                    f"[{comp}] 存在{len(fatal)}个致命错误，建议检查配置和依赖。"
                )
            recoverable = [e for e in errors if e.category == ErrorCategory.RECOVERABLE]
            if recoverable:
                recommendations.append(
                    f"[{comp}] 存在{len(recoverable)}个可恢复错误，建议重试或检查网络连接。"
                )

        return recommendations

    def clear_history(self) -> None:
        """清空错误历史"""
        self._error_history.clear()
        logger.info("Error history cleared")

    def reset_stats(self) -> None:
        """重置统计"""
        self._total_errors = 0
        self._total_recovered = 0
        self.clear_history()
