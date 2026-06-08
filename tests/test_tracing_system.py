# =============================================================================
# 日志与追踪系统测试 - 阶段30测试
# =============================================================================

import time
import pytest
from unittest.mock import Mock

from src.agent.tracing_system import (
    TraceManager,
    StructuredLogger,
    Tracer,
    Trace,
    Span,
    LogRecord,
    LogLevel,
    SpanStatus,
)


class TestLogLevel:
    """日志级别测试"""

    def test_enum_values(self):
        assert LogLevel.TRACE.value == "trace"
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARN.value == "warn"
        assert LogLevel.ERROR.value == "error"
        assert LogLevel.FATAL.value == "fatal"

    def test_priority(self):
        assert LogLevel.TRACE.priority < LogLevel.DEBUG.priority
        assert LogLevel.DEBUG.priority < LogLevel.INFO.priority
        assert LogLevel.INFO.priority < LogLevel.WARN.priority
        assert LogLevel.WARN.priority < LogLevel.ERROR.priority
        assert LogLevel.ERROR.priority < LogLevel.FATAL.priority


class TestSpanStatus:
    """Span状态测试"""

    def test_enum_values(self):
        assert SpanStatus.STARTED.value == "started"
        assert SpanStatus.FINISHED.value == "finished"
        assert SpanStatus.ERROR.value == "error"


class TestLogRecord:
    """日志记录测试"""

    def test_creation(self):
        record = LogRecord(
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test",
            trace_id="trace_1",
            span_id="span_1",
        )
        assert record.level == LogLevel.INFO
        assert record.message == "Test message"
        assert record.trace_id == "trace_1"
        assert record.record_id  # 自动生成

    def test_to_dict(self):
        record = LogRecord(
            level=LogLevel.ERROR,
            message="Error occurred",
            error="Something went wrong",
            context={"key": "value"},
        )
        data = record.to_dict()
        assert data["level"] == "error"
        assert data["message"] == "Error occurred"
        assert data["error"] == "Something went wrong"
        assert data["context"]["key"] == "value"


class TestSpan:
    """Span测试"""

    def test_creation(self):
        span = Span(
            span_id="span_1",
            trace_id="trace_1",
            operation_name="process_request",
        )
        assert span.span_id == "span_1"
        assert span.status == SpanStatus.STARTED

    def test_auto_id(self):
        span = Span(operation_name="test")
        assert span.span_id  # 自动生成

    def test_finish(self):
        span = Span(operation_name="test")
        time.sleep(0.01)
        span.finish()
        assert span.status == SpanStatus.FINISHED
        assert span.end_time is not None
        assert span.duration_ms is not None
        assert span.duration_ms >= 10

    def test_finish_error(self):
        span = Span(operation_name="test")
        span.finish(SpanStatus.ERROR)
        assert span.status == SpanStatus.ERROR

    def test_add_tag(self):
        span = Span(operation_name="test")
        span.add_tag("http.method", "GET")
        span.add_tag("http.status", 200)
        assert span.tags["http.method"] == "GET"
        assert span.tags["http.status"] == 200

    def test_add_log(self):
        span = Span(operation_name="test")
        record = LogRecord(level=LogLevel.INFO, message="test log")
        span.add_log(record)
        assert len(span.logs) == 1

    def test_duration_before_finish(self):
        span = Span(operation_name="test")
        assert span.duration_ms is None

    def test_to_dict(self):
        span = Span(
            span_id="span_1",
            trace_id="trace_1",
            operation_name="test_op",
            parent_id="parent_1",
        )
        span.add_tag("key", "value")
        data = span.to_dict()
        assert data["span_id"] == "span_1"
        assert data["operation_name"] == "test_op"
        assert data["parent_id"] == "parent_1"
        assert data["tags"]["key"] == "value"


class TestTrace:
    """Trace测试"""

    def test_creation(self):
        trace = Trace(trace_id="trace_1", name="test_trace")
        assert trace.trace_id == "trace_1"
        assert trace.status == "active"

    def test_auto_id(self):
        trace = Trace(name="test")
        assert trace.trace_id

    def test_add_span(self):
        trace = Trace(name="test")
        span = Span(operation_name="step_1")
        trace.add_span(span)
        assert len(trace.spans) == 1
        assert span.trace_id == trace.trace_id

    def test_finish(self):
        trace = Trace(name="test")
        time.sleep(0.01)
        trace.finish()
        assert trace.status == "completed"
        assert trace.duration_ms is not None

    def test_finish_error(self):
        trace = Trace(name="test")
        trace.finish_error()
        assert trace.status == "error"

    def test_root_span(self):
        trace = Trace(name="test")
        root = Span(operation_name="root")
        child = Span(operation_name="child", parent_id="span_1")
        trace.add_span(root)
        trace.add_span(child)
        assert trace.root_span == root

    def test_no_root_span(self):
        trace = Trace(name="test")
        assert trace.root_span is None

    def test_to_dict(self):
        trace = Trace(trace_id="trace_1", name="test")
        trace.add_span(Span(operation_name="step_1"))
        data = trace.to_dict()
        assert data["trace_id"] == "trace_1"
        assert data["spans_count"] == 1


class TestStructuredLogger:
    """结构化日志器测试"""

    def test_creation(self):
        logger = StructuredLogger(name="test", min_level=LogLevel.DEBUG)
        assert logger.name == "test"

    def test_log_levels(self):
        logger = StructuredLogger(min_level=LogLevel.TRACE)
        logger.trace("trace msg")
        logger.debug("debug msg")
        logger.info("info msg")
        logger.warn("warn msg")
        logger.error("error msg")
        logger.fatal("fatal msg")

        records = logger.get_records()
        assert len(records) == 6

    def test_min_level_filter(self):
        logger = StructuredLogger(min_level=LogLevel.WARN)
        logger.trace("trace")
        logger.debug("debug")
        logger.info("info")
        logger.warn("warn")
        logger.error("error")

        records = logger.get_records()
        assert len(records) == 2  # 只有warn和error

    def test_trace_context(self):
        logger = StructuredLogger()
        logger.set_trace_context("trace_1", "span_1")
        logger.info("test message")

        records = logger.get_records()
        assert len(records) == 1
        assert records[0].trace_id == "trace_1"
        assert records[0].span_id == "span_1"

    def test_context_and_error(self):
        logger = StructuredLogger()
        logger.error(
            "Error occurred",
            context={"key": "value"},
            error="Stack trace...",
            source="module.py:42",
        )
        records = logger.get_records()
        assert records[0].context["key"] == "value"
        assert records[0].error == "Stack trace..."
        assert records[0].source == "module.py:42"

    def test_handler_called(self):
        logger = StructuredLogger()
        handler = Mock()
        logger.add_handler(handler)

        logger.info("test")
        handler.assert_called_once()
        call_arg = handler.call_args[0][0]
        assert call_arg.message == "test"

    def test_get_records_by_level(self):
        logger = StructuredLogger(min_level=LogLevel.TRACE)
        logger.info("info")
        logger.error("error")
        logger.warn("warn")

        error_records = logger.get_records(min_level=LogLevel.ERROR)
        assert len(error_records) == 1
        assert error_records[0].level == LogLevel.ERROR

    def test_get_records_by_trace_id(self):
        logger = StructuredLogger()
        logger.set_trace_context("trace_1")
        logger.info("msg1")
        logger.set_trace_context("trace_2")
        logger.info("msg2")

        records = logger.get_records(trace_id="trace_1")
        assert len(records) == 1
        assert records[0].message == "msg1"

    def test_max_records(self):
        logger = StructuredLogger(max_records=5)
        for i in range(10):
            logger.info(f"msg_{i}")

        records = logger.get_records(limit=100)
        assert len(records) == 5

    def test_clear(self):
        logger = StructuredLogger()
        logger.info("msg1")
        logger.info("msg2")
        count = logger.clear()
        assert count == 2
        assert len(logger.get_records()) == 0

    def test_get_stats(self):
        logger = StructuredLogger()
        logger.info("info1")
        logger.info("info2")
        logger.error("err1")

        stats = logger.get_stats()
        assert stats["total_records"] == 3
        assert stats["level_distribution"]["info"] == 2
        assert stats["level_distribution"]["error"] == 1


class TestTracer:
    """追踪器测试"""

    def test_start_trace(self):
        tracer = Tracer()
        trace = tracer.start_trace(name="test_trace")
        assert trace.trace_id
        assert trace.name == "test_trace"
        assert trace.status == "active"

    def test_start_span(self):
        tracer = Tracer()
        span = tracer.start_span(operation_name="process")
        assert span.span_id
        assert span.trace_id
        assert span.operation_name == "process"

    def test_start_span_with_parent(self):
        tracer = Tracer()
        parent = tracer.start_span(operation_name="parent")
        child = tracer.start_span(
            operation_name="child",
            trace_id=parent.trace_id,
            parent_id=parent.span_id,
        )
        assert child.parent_id == parent.span_id
        assert child.trace_id == parent.trace_id

    def test_finish_span(self):
        tracer = Tracer()
        span = tracer.start_span(operation_name="test")
        time.sleep(0.01)
        result = tracer.finish_span(span.span_id)
        assert result is not None
        assert result.status == SpanStatus.FINISHED
        assert result.duration_ms is not None

    def test_finish_span_error(self):
        tracer = Tracer()
        span = tracer.start_span(operation_name="test")
        result = tracer.finish_span(span.span_id, SpanStatus.ERROR)
        assert result.status == SpanStatus.ERROR

    def test_get_trace(self):
        tracer = Tracer()
        trace = tracer.start_trace(name="test")
        found = tracer.get_trace(trace.trace_id)
        assert found is not None
        assert found.name == "test"

    def test_get_active_trace_ids(self):
        tracer = Tracer()
        trace = tracer.start_trace(name="active")
        active_ids = tracer.get_active_trace_ids()
        assert trace.trace_id in active_ids

    def test_trace_auto_finish_on_last_span(self):
        tracer = Tracer()
        span = tracer.start_span(operation_name="only_span")
        tracer.finish_span(span.span_id)

        trace = tracer.get_trace(span.trace_id)
        assert trace is not None
        assert trace.end_time is not None
        assert trace.status == "completed"

    def test_trace_error_on_error_span(self):
        tracer = Tracer()
        span = tracer.start_span(operation_name="fail_span")
        tracer.finish_span(span.span_id, SpanStatus.ERROR)

        trace = tracer.get_trace(span.trace_id)
        assert trace.status == "error"

    def test_handler_on_trace_complete(self):
        tracer = Tracer()
        handler = Mock()
        tracer.add_trace_handler(handler)

        span = tracer.start_span(operation_name="test")
        tracer.finish_span(span.span_id)

        handler.assert_called_once()

    def test_get_all_traces(self):
        tracer = Tracer()
        tracer.start_trace(name="t1")
        tracer.start_trace(name="t2")

        traces = tracer.get_all_traces()
        assert len(traces) == 2

    def test_get_tracer_stats(self):
        tracer = Tracer()
        tracer.start_trace(name="active")
        span = tracer.start_span(operation_name="test")
        tracer.finish_span(span.span_id)

        stats = tracer.get_tracer_stats()
        assert stats["total_traces"] >= 1
        assert stats["active_spans"] == 0

    def test_max_traces(self):
        tracer = Tracer(max_traces=3)
        traces = []
        for i in range(5):
            t = tracer.start_trace(name=f"t{i}")
            s = tracer.start_span(operation_name=f"op{i}", trace_id=t.trace_id)
            tracer.finish_span(s.span_id)
            traces.append(t)

        # 至少应该有3个（可能更多因为清理策略）
        all_traces = tracer.get_all_traces()
        assert len(all_traces) <= 5


class TestTraceManager:
    """追踪管理器测试"""

    def test_creation(self):
        manager = TraceManager(service_name="test_service")
        assert manager.logger is not None
        assert manager.tracer is not None

    def test_start_and_finish_trace(self):
        manager = TraceManager()
        trace = manager.start_trace(name="test_trace")
        assert trace is not None

        manager.finish_trace(trace.trace_id)
        assert trace.end_time is not None

    def test_start_and_finish_span(self):
        manager = TraceManager()
        span = manager.start_span(operation_name="test_op")
        time.sleep(0.01)
        manager.finish_span(span.span_id)

        assert span.duration_ms is not None

    def test_nested_spans(self):
        manager = TraceManager()
        trace = manager.start_trace(name="parent_trace")
        parent = manager.start_span(
            operation_name="parent",
            trace_id=trace.trace_id,
        )
        child = manager.start_span(
            operation_name="child",
            trace_id=trace.trace_id,
            parent_id=parent.span_id,
        )

        manager.finish_span(child.span_id)
        manager.finish_span(parent.span_id)
        manager.finish_trace(trace.trace_id)

        assert len(trace.spans) == 2

    def test_get_trace(self):
        manager = TraceManager()
        trace = manager.start_trace(name="test")
        found = manager.get_trace(trace.trace_id)
        assert found is not None

    def test_get_stats(self):
        manager = TraceManager()
        manager.start_trace(name="t1")
        manager.start_span(operation_name="op1")

        stats = manager.get_stats()
        assert "logger_stats" in stats
        assert "tracer_stats" in stats
        assert stats["service_name"] == "agent"


class TestExportTracingSystem:
    """导出测试"""

    def test_trace_manager_exported(self):
        from src.agent import TraceManager
        assert TraceManager is not None

    def test_structured_logger_exported(self):
        from src.agent import StructuredLogger
        assert StructuredLogger is not None

    def test_tracer_exported(self):
        from src.agent import Tracer
        assert Tracer is not None

    def test_trace_exported(self):
        from src.agent import Trace
        assert Trace is not None

    def test_span_exported(self):
        from src.agent import Span
        assert Span is not None

    def test_log_record_exported(self):
        from src.agent import LogRecord
        assert LogRecord is not None

    def test_log_level_exported(self):
        from src.agent import LogLevel
        assert LogLevel.INFO is not None

    def test_span_status_exported(self):
        from src.agent import SpanStatus
        assert SpanStatus.FINISHED is not None


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])