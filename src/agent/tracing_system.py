# =============================================================================
# Agent日志与追踪系统 - 阶段30: Agent Logging & Tracing
# =============================================================================
#
# 功能：实现Agent日志与追踪系统，支持结构化日志和分布式追踪
# 原则：图难于其易，为大于其细
#
# =============================================================================

from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional


# =============================================================================
# 日志级别
# =============================================================================

class LogLevel(str, Enum):
    """日志级别"""
    TRACE = "trace"       # 最详细追踪
    DEBUG = "debug"       # 调试信息
    INFO = "info"         # 一般信息
    WARN = "warn"         # 警告
    ERROR = "error"       # 错误
    FATAL = "fatal"       # 致命错误

    @property
    def priority(self) -> int:
        """优先级（数字越大越重要）"""
        _priorities = {
            LogLevel.TRACE: 0,
            LogLevel.DEBUG: 1,
            LogLevel.INFO: 2,
            LogLevel.WARN: 3,
            LogLevel.ERROR: 4,
            LogLevel.FATAL: 5,
        }
        return _priorities[self]


# =============================================================================
# Span状态
# =============================================================================

class SpanStatus(str, Enum):
    """Span状态"""
    STARTED = "started"       # 已开始
    FINISHED = "finished"     # 已完成
    ERROR = "error"           # 出错


# =============================================================================
# 日志记录
# =============================================================================

@dataclass
class LogRecord:
    """
    日志记录
    
    表示一条结构化日志
    """
    record_id: str = ""
    timestamp: float = field(default_factory=time.time)
    level: LogLevel = LogLevel.INFO
    message: str = ""
    logger_name: str = "root"
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    context: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    source: str = ""

    def __post_init__(self):
        if not self.record_id:
            self.record_id = f"log_{uuid.uuid4().hex[:8]}"

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "level": self.level.value,
            "message": self.message,
            "logger_name": self.logger_name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "context": self.context,
            "error": self.error,
            "source": self.source,
        }


# =============================================================================
# Span（追踪跨度）
# =============================================================================

@dataclass
class Span:
    """
    追踪Span
    
    表示一个操作的时间跨度
    """
    span_id: str = ""
    trace_id: str = ""
    parent_id: Optional[str] = None
    operation_name: str = ""
    status: SpanStatus = SpanStatus.STARTED
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    tags: dict[str, Any] = field(default_factory=dict)
    logs: list[LogRecord] = field(default_factory=list)
    service_name: str = "agent"

    def __post_init__(self):
        if not self.span_id:
            self.span_id = f"span_{uuid.uuid4().hex[:12]}"

    def finish(self, status: SpanStatus = SpanStatus.FINISHED) -> None:
        """结束Span"""
        self.status = status
        self.end_time = time.time()

    def add_tag(self, key: str, value: Any) -> None:
        """添加标签"""
        self.tags[key] = value

    def add_log(self, record: LogRecord) -> None:
        """添加日志"""
        self.logs.append(record)

    @property
    def duration_ms(self) -> Optional[float]:
        """获取持续时间（毫秒）"""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "operation_name": self.operation_name,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "tags": self.tags,
            "logs_count": len(self.logs),
            "service_name": self.service_name,
        }


# =============================================================================
# Trace（追踪链）
# =============================================================================

@dataclass
class Trace:
    """
    追踪链
    
    表示一个完整的请求追踪
    """
    trace_id: str = ""
    name: str = ""
    service_name: str = "agent"
    spans: list[Span] = field(default_factory=list)
    tags: dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "active"

    def __post_init__(self):
        if not self.trace_id:
            self.trace_id = f"trace_{uuid.uuid4().hex[:12]}"

    def add_span(self, span: Span) -> None:
        """添加Span"""
        span.trace_id = self.trace_id
        self.spans.append(span)

    def finish(self) -> None:
        """结束追踪"""
        self.end_time = time.time()
        self.status = "completed"

    def finish_error(self) -> None:
        """以错误结束追踪"""
        self.end_time = time.time()
        self.status = "error"

    @property
    def root_span(self) -> Optional[Span]:
        """获取根Span"""
        for span in self.spans:
            if span.parent_id is None:
                return span
        return None

    @property
    def duration_ms(self) -> Optional[float]:
        """获取总持续时间"""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "service_name": self.service_name,
            "spans_count": len(self.spans),
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "tags": self.tags,
        }


# =============================================================================
# 结构化日志器
# =============================================================================

class StructuredLogger:
    """
    结构化日志器
    
    提供分级、带上下文的日志记录
    """

    def __init__(
        self,
        name: str = "root",
        min_level: LogLevel = LogLevel.DEBUG,
        max_records: int = 5000,
    ):
        self._name = name
        self._min_level = min_level
        self._records: list[LogRecord] = []
        self._max_records = max_records
        self._lock = threading.RLock()
        self._handlers: list[Callable[[LogRecord], None]] = []
        self._current_trace_id: Optional[str] = None
        self._current_span_id: Optional[str] = None

    @property
    def name(self) -> str:
        return self._name

    def set_trace_context(self, trace_id: Optional[str], span_id: Optional[str] = None) -> None:
        """设置追踪上下文"""
        self._current_trace_id = trace_id
        self._current_span_id = span_id

    def _log(
        self,
        level: LogLevel,
        message: str,
        context: Optional[dict] = None,
        error: Optional[str] = None,
        source: str = "",
    ) -> LogRecord:
        """内部日志方法"""
        if level.priority < self._min_level.priority:
            # 返回空记录（不记录）
            return LogRecord(level=level, message=message)

        record = LogRecord(
            level=level,
            message=message,
            logger_name=self._name,
            trace_id=self._current_trace_id,
            span_id=self._current_span_id,
            context=context or {},
            error=error,
            source=source,
        )

        with self._lock:
            self._records.append(record)
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records:]

        for handler in self._handlers:
            try:
                handler(record)
            except Exception:
                pass

        return record

    def trace(self, message: str, **kwargs) -> LogRecord:
        """TRACE级别"""
        return self._log(LogLevel.TRACE, message, **kwargs)

    def debug(self, message: str, **kwargs) -> LogRecord:
        """DEBUG级别"""
        return self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> LogRecord:
        """INFO级别"""
        return self._log(LogLevel.INFO, message, **kwargs)

    def warn(self, message: str, **kwargs) -> LogRecord:
        """WARN级别"""
        return self._log(LogLevel.WARN, message, **kwargs)

    def error(self, message: str, **kwargs) -> LogRecord:
        """ERROR级别"""
        return self._log(LogLevel.ERROR, message, **kwargs)

    def fatal(self, message: str, **kwargs) -> LogRecord:
        """FATAL级别"""
        return self._log(LogLevel.FATAL, message, **kwargs)

    def add_handler(self, handler: Callable[[LogRecord], None]) -> None:
        """添加日志处理器"""
        self._handlers.append(handler)

    def get_records(
        self,
        min_level: Optional[LogLevel] = None,
        limit: int = 100,
        trace_id: Optional[str] = None,
    ) -> list[LogRecord]:
        """获取日志记录"""
        with self._lock:
            records = self._records.copy()

        if min_level:
            records = [r for r in records if r.level.priority >= min_level.priority]
        if trace_id:
            records = [r for r in records if r.trace_id == trace_id]

        return records[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """获取日志统计"""
        with self._lock:
            level_counts: dict[str, int] = defaultdict(int)
            for r in self._records:
                level_counts[r.level.value] += 1

            return {
                "logger_name": self._name,
                "total_records": len(self._records),
                "level_distribution": dict(level_counts),
                "min_level": self._min_level.value,
            }

    def clear(self) -> int:
        """清空日志"""
        with self._lock:
            count = len(self._records)
            self._records.clear()
            return count


# =============================================================================
# 分布式追踪器
# =============================================================================

class Tracer:
    """
    分布式追踪器
    
    管理追踪链和Span的创建
    """

    def __init__(self, service_name: str = "agent", max_traces: int = 1000):
        self._service_name = service_name
        self._traces: dict[str, Trace] = {}
        self._active_spans: dict[str, Span] = {}
        self._max_traces = max_traces
        self._lock = threading.RLock()
        self._handlers: list[Callable[[Trace], None]] = []

    def start_trace(self, name: str = "", tags: Optional[dict] = None) -> Trace:
        """
        开始新追踪
        
        Args:
            name: 追踪名称
            tags: 标签
        
        Returns:
            追踪对象
        """
        trace = Trace(
            name=name or "unnamed",
            service_name=self._service_name,
            tags=tags or {},
        )

        with self._lock:
            self._traces[trace.trace_id] = trace
            if len(self._traces) > self._max_traces:
                # 清理最旧的已完成追踪
                completed_ids = [
                    tid for tid, t in self._traces.items()
                    if t.end_time is not None
                ]
                for tid in completed_ids[:len(self._traces) - self._max_traces]:
                    del self._traces[tid]

        return trace

    def start_span(
        self,
        operation_name: str,
        trace_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        tags: Optional[dict] = None,
    ) -> Span:
        """
        开始新Span
        
        Args:
            operation_name: 操作名称
            trace_id: 追踪ID（None则创建新追踪）
            parent_id: 父Span ID
            tags: 标签
        
        Returns:
            Span对象
        """
        if trace_id and trace_id in self._traces:
            trace = self._traces[trace_id]
        else:
            trace = self.start_trace(name=operation_name, tags=tags)

        span = Span(
            trace_id=trace.trace_id,
            parent_id=parent_id,
            operation_name=operation_name,
            tags=tags or {},
            service_name=self._service_name,
        )

        trace.add_span(span)

        with self._lock:
            self._active_spans[span.span_id] = span

        return span

    def finish_span(self, span_id: str, status: SpanStatus = SpanStatus.FINISHED) -> Optional[Span]:
        """
        结束Span
        
        Args:
            span_id: Span ID
            status: 结束状态
        
        Returns:
            Span对象或None
        """
        with self._lock:
            span = self._active_spans.pop(span_id, None)

        if span:
            span.finish(status)

            # 检查是否所有span都完成了
            if span.trace_id in self._traces:
                trace = self._traces[span.trace_id]
                all_finished = all(s.end_time is not None for s in trace.spans)
                if all_finished and not trace.end_time:
                    if status == SpanStatus.ERROR:
                        trace.finish_error()
                    else:
                        trace.finish()
                    # 通知处理器
                    for handler in self._handlers:
                        try:
                            handler(trace)
                        except Exception:
                            pass

        return span

    def get_active_span(self, span_id: str) -> Optional[Span]:
        """获取活跃Span"""
        with self._lock:
            return self._active_spans.get(span_id)

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """获取追踪"""
        with self._lock:
            return self._traces.get(trace_id)

    def get_all_traces(self) -> list[Trace]:
        """获取所有追踪"""
        with self._lock:
            return list(self._traces.values())

    def get_active_trace_ids(self) -> list[str]:
        """获取活跃追踪ID"""
        with self._lock:
            return [
                tid for tid, t in self._traces.items()
                if t.end_time is None
            ]

    def add_trace_handler(self, handler: Callable[[Trace], None]) -> None:
        """添加追踪完成处理器"""
        self._handlers.append(handler)

    def get_tracer_stats(self) -> dict[str, Any]:
        """获取追踪器统计"""
        with self._lock:
            total = len(self._traces)
            active = sum(1 for t in self._traces.values() if t.end_time is None)
            completed = total - active
            error_count = sum(1 for t in self._traces.values() if t.status == "error")

            return {
                "service_name": self._service_name,
                "total_traces": total,
                "active_traces": active,
                "completed_traces": completed,
                "error_traces": error_count,
                "active_spans": len(self._active_spans),
            }


# =============================================================================
# 追踪管理器
# =============================================================================

class TraceManager:
    """
    追踪管理器
    
    统一管理日志器和追踪器
    """

    def __init__(self, service_name: str = "agent"):
        self._logger = StructuredLogger(name=service_name)
        self._tracer = Tracer(service_name=service_name)
        self._service_name = service_name

    @property
    def logger(self) -> StructuredLogger:
        """获取日志器"""
        return self._logger

    @property
    def tracer(self) -> Tracer:
        """获取追踪器"""
        return self._tracer

    def start_trace(self, name: str = "", tags: Optional[dict] = None) -> Trace:
        """开始追踪"""
        trace = self._tracer.start_trace(name=name, tags=tags)
        self._logger.set_trace_context(trace.trace_id)
        self._logger.info(f"Trace started: {trace.trace_id}", context={"trace_name": name})
        return trace

    def start_span(
        self,
        operation_name: str,
        trace_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        tags: Optional[dict] = None,
    ) -> Span:
        """开始Span"""
        span = self._tracer.start_span(
            operation_name=operation_name,
            trace_id=trace_id,
            parent_id=parent_id,
            tags=tags,
        )
        self._logger.set_trace_context(span.trace_id, span.span_id)
        self._logger.debug(
            f"Span started: {span.operation_name}",
            context={"span_id": span.span_id},
        )
        return span

    def finish_span(
        self, span_id: str, status: SpanStatus = SpanStatus.FINISHED,
    ) -> Optional[Span]:
        """结束Span"""
        span = self._tracer.finish_span(span_id, status)
        if span:
            level = LogLevel.WARN if status == SpanStatus.ERROR else LogLevel.DEBUG
            self._logger._log(
                level,
                f"Span finished: {span.operation_name} ({span.duration_ms:.2f}ms)",
                context={"span_id": span.span_id, "status": status.value},
            )
        return span

    def finish_trace(self, trace_id: str) -> Optional[Trace]:
        """结束追踪"""
        trace = self._tracer.get_trace(trace_id)
        if trace and not trace.end_time:
            trace.finish()
            self._logger.info(
                f"Trace finished: {trace_id} ({trace.duration_ms:.2f}ms)",
                context={"spans_count": len(trace.spans)},
            )
        return trace

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """获取追踪"""
        return self._tracer.get_trace(trace_id)

    def get_stats(self) -> dict[str, Any]:
        """获取统计"""
        return {
            "service_name": self._service_name,
            "logger_stats": self._logger.get_stats(),
            "tracer_stats": self._tracer.get_tracer_stats(),
        }
