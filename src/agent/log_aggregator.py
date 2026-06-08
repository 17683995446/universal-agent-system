"""
阶段41: Agent日志聚合系统

遵循"图难于其易，为大于其细"的原则，实现多源日志收集、聚合、查询、分析。
"""

import collections
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.agent.tracing_system import LogLevel


# ============================================================
# 41.1 日志模型
# ============================================================

class LogSourceType(str, Enum):
    """日志来源类型"""
    AGENT = "agent"
    SYSTEM = "system"
    PLUGIN = "plugin"
    WORKFLOW = "workflow"
    EXTERNAL = "external"


class LogAggregationStrategy(str, Enum):
    """日志聚合策略"""
    ALL = "all"
    ERROR_ONLY = "error_only"
    WARN_AND_ABOVE = "warn_and_above"
    SAMPLED = "sampled"
    FILTERED = "filtered"


@dataclass
class LogEntry:
    """日志条目"""
    id: str = ""
    timestamp: float = field(default_factory=time.time)
    level: str = "INFO"
    source: str = "agent"
    source_type: LogSourceType = LogSourceType.AGENT
    message: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    agent_id: str = ""
    trace_id: str = ""
    span_id: str = ""
    duration_ms: float = 0.0
    error_type: str = ""
    stack_trace: str = ""

    def __post_init__(self):
        if not self.id:
            import uuid
            self.id = str(uuid.uuid4())[:12]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "level": self.level,
            "source": self.source,
            "source_type": self.source_type.value,
            "message": self.message,
            "context": self.context,
            "tags": self.tags,
            "agent_id": self.agent_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "duration_ms": self.duration_ms,
            "error_type": self.error_type,
            "stack_trace": self.stack_trace,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogEntry":
        data = data.copy()
        if "source_type" in data and isinstance(data["source_type"], str):
            data["source_type"] = LogSourceType(data["source_type"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LogQuery:
    """日志查询条件"""
    source: Optional[str] = None
    source_type: Optional[LogSourceType] = None
    level: Optional[str] = None
    min_level: Optional[str] = None
    tags: Optional[List[str]] = None
    agent_id: Optional[str] = None
    trace_id: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    message_contains: Optional[str] = None
    limit: int = 100
    offset: int = 0


@dataclass
class LogStats:
    """日志统计"""
    total_count: int = 0
    by_level: Dict[str, int] = field(default_factory=dict)
    by_source: Dict[str, int] = field(default_factory=dict)
    by_source_type: Dict[str, int] = field(default_factory=dict)
    error_count: int = 0
    warn_count: int = 0
    time_range: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_count": self.total_count,
            "by_level": self.by_level,
            "by_source": self.by_source,
            "by_source_type": self.by_source_type,
            "error_count": self.error_count,
            "warn_count": self.warn_count,
            "time_range": self.time_range,
        }


# ============================================================
# 41.2 日志收集器
# ============================================================

class LogCollector:
    """日志收集器"""

    def __init__(self, buffer_size: int = 10000):
        self._buffer: List[LogEntry] = []
        self._buffer_size = buffer_size
        self._lock = threading.RLock()
        self._total_collected = 0
        self._total_dropped = 0

    def collect(self, entry: LogEntry) -> bool:
        """收集日志条目"""
        with self._lock:
            if len(self._buffer) >= self._buffer_size:
                self._buffer.pop(0)
                self._total_dropped += 1
            self._buffer.append(entry)
            self._total_collected += 1
            return True

    def collect_many(self, entries: List[LogEntry]) -> int:
        """批量收集"""
        count = 0
        for entry in entries:
            if self.collect(entry):
                count += 1
        return count

    def get_buffer(self) -> List[LogEntry]:
        """获取缓冲区"""
        with self._lock:
            return list(self._buffer)

    def clear(self) -> int:
        """清空缓冲区"""
        with self._lock:
            count = len(self._buffer)
            self._buffer.clear()
            return count

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "buffer_size": len(self._buffer),
                "max_buffer_size": self._buffer_size,
                "total_collected": self._total_collected,
                "total_dropped": self._total_dropped,
            }


# ============================================================
# 41.3 日志聚合器
# ============================================================

class LogAggregator:
    """日志聚合器"""

    LEVEL_ORDER = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3, "CRITICAL": 4}

    def __init__(self, strategy: LogAggregationStrategy = LogAggregationStrategy.ALL):
        self._strategy = strategy
        self._collector = LogCollector()
        self._subscribers: Dict[str, Callable[[LogEntry], None]] = {}
        self._lock = threading.RLock()
        self._filters: List[Callable[[LogEntry], bool]] = []

    def add_log(self, entry: LogEntry) -> bool:
        """添加日志"""
        if not self._should_accept(entry):
            return False
        self._collector.collect(entry)
        self._notify_subscribers(entry)
        return True

    def add_log_simple(
        self, level: str, message: str, source: str = "agent",
        context: Optional[Dict] = None, tags: Optional[List[str]] = None,
    ) -> bool:
        """简单添加日志"""
        entry = LogEntry(
            level=level, message=message, source=source,
            context=context or {}, tags=tags or [],
        )
        return self.add_log(entry)

    def query(self, query: LogQuery) -> List[LogEntry]:
        """查询日志"""
        entries = self._collector.get_buffer()
        results = []

        for entry in entries:
            if self._matches_query(entry, query):
                results.append(entry)

        # 按时间排序（最新在前）
        results.sort(key=lambda e: e.timestamp, reverse=True)

        # 分页
        return results[query.offset:query.offset + query.limit]

    def query_all(self) -> List[LogEntry]:
        """查询所有日志"""
        return self._collector.get_buffer()

    def get_stats(self) -> LogStats:
        """获取统计"""
        entries = self._collector.get_buffer()
        if not entries:
            return LogStats()

        by_level: Dict[str, int] = collections.defaultdict(int)
        by_source: Dict[str, int] = collections.defaultdict(int)
        by_source_type: Dict[str, int] = collections.defaultdict(int)
        error_count = 0
        warn_count = 0
        times = []

        for e in entries:
            by_level[e.level] += 1
            by_source[e.source] += 1
            by_source_type[e.source_type.value] += 1
            if e.level in ("ERROR", "CRITICAL"):
                error_count += 1
            if e.level == "WARN":
                warn_count += 1
            times.append(e.timestamp)

        return LogStats(
            total_count=len(entries),
            by_level=dict(by_level),
            by_source=dict(by_source),
            by_source_type=dict(by_source_type),
            error_count=error_count,
            warn_count=warn_count,
            time_range=(min(times), max(times)) if times else (0, 0),
        )

    def subscribe(self, name: str, callback: Callable[[LogEntry], None]) -> None:
        """订阅日志"""
        with self._lock:
            self._subscribers[name] = callback

    def unsubscribe(self, name: str) -> None:
        """取消订阅"""
        with self._lock:
            self._subscribers.pop(name, None)

    def add_filter(self, filter_fn: Callable[[LogEntry], bool]) -> None:
        """添加过滤器"""
        self._filters.append(filter_fn)

    def clear(self) -> int:
        return self._collector.clear()

    def _should_accept(self, entry: LogEntry) -> bool:
        if self._strategy == LogAggregationStrategy.ALL:
            pass
        elif self._strategy == LogAggregationStrategy.ERROR_ONLY:
            if entry.level not in ("ERROR", "CRITICAL"):
                return False
        elif self._strategy == LogAggregationStrategy.WARN_AND_ABOVE:
            if self.LEVEL_ORDER.get(entry.level, 0) < self.LEVEL_ORDER.get("WARN", 0):
                return False
        elif self._strategy == LogAggregationStrategy.SAMPLED:
            import random
            if random.random() > 0.1:
                return False
        for f in self._filters:
            if not f(entry):
                return False
        return True

    def _matches_query(self, entry: LogEntry, query: LogQuery) -> bool:
        if query.source and entry.source != query.source:
            return False
        if query.source_type and entry.source_type != query.source_type:
            return False
        if query.level and entry.level != query.level:
            return False
        if query.min_level:
            if self.LEVEL_ORDER.get(entry.level, 0) < self.LEVEL_ORDER.get(query.min_level, 0):
                return False
        if query.tags:
            if not any(t in entry.tags for t in query.tags):
                return False
        if query.agent_id and entry.agent_id != query.agent_id:
            return False
        if query.trace_id and entry.trace_id != query.trace_id:
            return False
        if query.start_time and entry.timestamp < query.start_time:
            return False
        if query.end_time and entry.timestamp > query.end_time:
            return False
        if query.message_contains and query.message_contains not in entry.message:
            return False
        return True

    def _notify_subscribers(self, entry: LogEntry) -> None:
        with self._lock:
            for callback in self._subscribers.values():
                try:
                    callback(entry)
                except Exception:
                    pass


# ============================================================
# 41.4 日志分析器
# ============================================================

class LogAnalyzer:
    """日志分析器"""

    def __init__(self, aggregator: LogAggregator):
        self._aggregator = aggregator

    def get_error_summary(self) -> List[Dict[str, Any]]:
        """获取错误摘要"""
        query = LogQuery(min_level="ERROR", limit=100)
        errors = self._aggregator.query(query)
        return [
            {
                "id": e.id,
                "timestamp": e.timestamp,
                "level": e.level,
                "source": e.source,
                "message": e.message,
                "error_type": e.error_type,
            }
            for e in errors
        ]

    def get_source_distribution(self) -> Dict[str, int]:
        """获取来源分布"""
        stats = self._aggregator.get_stats()
        return stats.by_source

    def get_level_distribution(self) -> Dict[str, int]:
        """获取级别分布"""
        stats = self._aggregator.get_stats()
        return stats.by_level

    def search(self, keyword: str, limit: int = 50) -> List[LogEntry]:
        """搜索日志"""
        query = LogQuery(message_contains=keyword, limit=limit)
        return self._aggregator.query(query)

    def get_recent(self, count: int = 20) -> List[LogEntry]:
        """获取最近日志"""
        query = LogQuery(limit=count)
        return self._aggregator.query(query)

    def count_by_tag(self, tag: str) -> int:
        """按标签计数"""
        query = LogQuery(tags=[tag], limit=10000)
        return len(self._aggregator.query(query))


# ============================================================
# 便捷函数
# ============================================================

def create_log_aggregator(
    strategy: LogAggregationStrategy = LogAggregationStrategy.ALL,
    buffer_size: int = 10000,
) -> LogAggregator:
    """创建日志聚合器"""
    agg = LogAggregator(strategy)
    agg._collector = LogCollector(buffer_size)
    return agg


def create_log_analyzer(aggregator: LogAggregator) -> LogAnalyzer:
    """创建日志分析器"""
    return LogAnalyzer(aggregator)
