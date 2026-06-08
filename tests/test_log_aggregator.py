"""
阶段41: Agent日志聚合系统 - 单元测试
"""
import time
import pytest
from src.agent.log_aggregator import (
    LogSourceType, LogAggregationStrategy,
    LogEntry, LogQuery, LogStats,
    LogCollector, LogAggregator, LogAnalyzer,
    create_log_aggregator, create_log_analyzer,
)


class TestLogSourceType:
    def test_values(self):
        assert LogSourceType.AGENT == "agent"
        assert LogSourceType.SYSTEM == "system"


class TestLogEntry:
    def test_creation(self):
        e = LogEntry(level="INFO", message="hello", source="test")
        assert e.level == "INFO"
        assert e.message == "hello"
        assert len(e.id) == 12

    def test_to_dict(self):
        e = LogEntry(level="ERROR", message="fail", source="agent")
        d = e.to_dict()
        assert d["level"] == "ERROR"
        assert d["source_type"] == "agent"

    def test_from_dict(self):
        d = {"level": "INFO", "message": "test", "source": "agent", "source_type": "system"}
        e = LogEntry.from_dict(d)
        assert e.source_type == LogSourceType.SYSTEM


class TestLogCollector:
    def test_collect(self):
        c = LogCollector()
        e = LogEntry(level="INFO", message="test")
        assert c.collect(e) is True
        assert c.size == 1

    def test_buffer_overflow(self):
        c = LogCollector(buffer_size=2)
        c.collect(LogEntry(level="INFO", message="a"))
        c.collect(LogEntry(level="INFO", message="b"))
        c.collect(LogEntry(level="INFO", message="c"))
        assert c.size == 2
        assert c.stats["total_dropped"] == 1

    def test_clear(self):
        c = LogCollector()
        c.collect(LogEntry(level="INFO", message="test"))
        assert c.clear() == 1
        assert c.size == 0


class TestLogAggregator:
    def setup_method(self):
        self.agg = create_log_aggregator()

    def test_add_log(self):
        assert self.agg.add_log_simple("INFO", "hello") is True
        assert len(self.agg.query_all()) == 1

    def test_error_only_strategy(self):
        agg = create_log_aggregator(strategy=LogAggregationStrategy.ERROR_ONLY)
        agg.add_log_simple("INFO", "info")
        agg.add_log_simple("ERROR", "err")
        assert len(agg.query_all()) == 1

    def test_query_by_source(self):
        self.agg.add_log_simple("INFO", "a", source="s1")
        self.agg.add_log_simple("INFO", "b", source="s2")
        q = LogQuery(source="s1")
        results = self.agg.query(q)
        assert len(results) == 1

    def test_query_by_level(self):
        self.agg.add_log_simple("INFO", "a")
        self.agg.add_log_simple("ERROR", "b")
        q = LogQuery(level="ERROR")
        results = self.agg.query(q)
        assert len(results) == 1

    def test_query_by_message(self):
        self.agg.add_log_simple("INFO", "hello world")
        self.agg.add_log_simple("INFO", "goodbye")
        q = LogQuery(message_contains="hello")
        results = self.agg.query(q)
        assert len(results) == 1

    def test_query_by_tag(self):
        self.agg.add_log(LogEntry(level="INFO", message="test", tags=["important"]))
        self.agg.add_log(LogEntry(level="INFO", message="test2", tags=["normal"]))
        q = LogQuery(tags=["important"])
        results = self.agg.query(q)
        assert len(results) == 1

    def test_query_pagination(self):
        for i in range(10):
            self.agg.add_log_simple("INFO", f"msg_{i}")
        q = LogQuery(limit=3, offset=2)
        results = self.agg.query(q)
        assert len(results) == 3

    def test_get_stats(self):
        self.agg.add_log_simple("INFO", "a", source="s1")
        self.agg.add_log_simple("ERROR", "b", source="s2")
        self.agg.add_log_simple("WARN", "c", source="s1")
        stats = self.agg.get_stats()
        assert stats.total_count == 3
        assert stats.error_count == 1
        assert stats.warn_count == 1
        assert stats.by_source["s1"] == 2

    def test_subscribe(self):
        received = []
        self.agg.subscribe("test", lambda e: received.append(e))
        self.agg.add_log_simple("INFO", "hello")
        assert len(received) == 1

    def test_unsubscribe(self):
        received = []
        self.agg.subscribe("test", lambda e: received.append(e))
        self.agg.unsubscribe("test")
        self.agg.add_log_simple("INFO", "hello")
        assert len(received) == 0

    def test_filter(self):
        self.agg.add_filter(lambda e: "skip" not in e.message)
        self.agg.add_log_simple("INFO", "keep this")
        self.agg.add_log_simple("INFO", "skip this")
        assert len(self.agg.query_all()) == 1

    def test_clear(self):
        self.agg.add_log_simple("INFO", "a")
        assert self.agg.clear() == 1
        assert len(self.agg.query_all()) == 0


class TestLogAnalyzer:
    def setup_method(self):
        self.agg = create_log_aggregator()
        self.analyzer = create_log_analyzer(self.agg)

    def test_error_summary(self):
        self.agg.add_log_simple("INFO", "ok")
        self.agg.add_log_simple("ERROR", "fail", source="agent")
        summary = self.analyzer.get_error_summary()
        assert len(summary) == 1
        assert summary[0]["message"] == "fail"

    def test_source_distribution(self):
        self.agg.add_log_simple("INFO", "a", source="s1")
        self.agg.add_log_simple("INFO", "b", source="s1")
        self.agg.add_log_simple("INFO", "c", source="s2")
        dist = self.analyzer.get_source_distribution()
        assert dist["s1"] == 2

    def test_search(self):
        self.agg.add_log_simple("INFO", "hello world")
        results = self.analyzer.search("hello")
        assert len(results) == 1

    def test_recent(self):
        for i in range(10):
            self.agg.add_log_simple("INFO", f"msg_{i}")
        recent = self.analyzer.get_recent(3)
        assert len(recent) == 3

    def test_count_by_tag(self):
        self.agg.add_log(LogEntry(level="INFO", message="a", tags=["tag1"]))
        self.agg.add_log(LogEntry(level="INFO", message="b", tags=["tag1"]))
        self.agg.add_log(LogEntry(level="INFO", message="c", tags=["tag2"]))
        assert self.analyzer.count_by_tag("tag1") == 2


class TestConvenienceFunctions:
    def test_create_aggregator(self):
        agg = create_log_aggregator()
        assert isinstance(agg, LogAggregator)

    def test_create_analyzer(self):
        agg = create_log_aggregator()
        analyzer = create_log_analyzer(agg)
        assert isinstance(analyzer, LogAnalyzer)
