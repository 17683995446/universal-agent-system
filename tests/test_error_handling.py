"""
阶段19测试：错误处理与恢复机制

测试ErrorCategory、ErrorRecord、RetryPolicy、ErrorHandler及Agent集成。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-26
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.agent.error_handler import (
    ErrorCategory,
    ErrorRecord,
    RetryPolicy,
    ErrorHandler,
    classify_error,
)
from src.core.exceptions import (
    LLMClientError,
    TokenLimitExceeded,
    IntegrityCheckFailed,
    MemoryStorageError,
    MemoryRecallError,
    ConfigurationError,
    PurificationError,
    TaskDriftDetected,
)


class TestErrorCategory:
    """ErrorCategory枚举测试"""

    def test_enum_values(self):
        assert ErrorCategory.WARNING == 0
        assert ErrorCategory.DEGRADED == 1
        assert ErrorCategory.RECOVERABLE == 2
        assert ErrorCategory.FATAL == 3

    def test_enum_count(self):
        assert len(ErrorCategory) == 4


class TestClassifyError:
    """classify_error函数测试"""

    def test_classify_llm_error(self):
        error = LLMClientError(model="gpt-4", error="timeout")
        assert classify_error(error) == ErrorCategory.RECOVERABLE

    def test_classify_token_limit(self):
        error = TokenLimitExceeded(current=5000, limit=4096)
        assert classify_error(error) == ErrorCategory.RECOVERABLE

    def test_classify_memory_storage(self):
        error = MemoryStorageError(operation="store", error="disk full")
        assert classify_error(error) == ErrorCategory.DEGRADED

    def test_classify_memory_recall(self):
        error = MemoryRecallError(query="test", error="not found")
        assert classify_error(error) == ErrorCategory.DEGRADED

    def test_classify_config_error(self):
        error = ConfigurationError(key="api_key", value="", error="missing")
        assert classify_error(error) == ErrorCategory.FATAL

    def test_classify_integrity(self):
        error = IntegrityCheckFailed(
            original_tokens=1000, preserved_ratio=0.5, required_ratio=0.7
        )
        assert classify_error(error) == ErrorCategory.RECOVERABLE

    def test_classify_purification(self):
        error = PurificationError(text="test", level=2, error="failed")
        assert classify_error(error) == ErrorCategory.RECOVERABLE

    def test_classify_drift(self):
        error = TaskDriftDetected(
            task_id="t1", original_goal="A", current_intent="B", similarity=0.3
        )
        assert classify_error(error) == ErrorCategory.DEGRADED

    def test_classify_generic_error(self):
        error = RuntimeError("something went wrong")
        assert classify_error(error) == ErrorCategory.RECOVERABLE


class TestErrorRecord:
    """ErrorRecord测试"""

    def test_creation(self):
        record = ErrorRecord(
            error_id="err_001",
            category=ErrorCategory.RECOVERABLE,
            error_type="LLMClientError",
            message="timeout",
        )
        assert record.error_id == "err_001"
        assert record.category == ErrorCategory.RECOVERABLE
        assert record.recovered == False

    def test_serialization(self):
        record = ErrorRecord(
            error_id="err_002",
            category=ErrorCategory.FATAL,
            error_type="ConfigurationError",
            message="missing key",
            component="Agent",
        )
        data = record.to_dict()
        assert data["error_id"] == "err_002"
        assert data["category"] == 3
        assert data["category_name"] == "FATAL"
        assert data["component"] == "Agent"

    def test_age_seconds(self):
        record = ErrorRecord(timestamp=0.0)
        assert record.age_seconds > 0


class TestRetryPolicy:
    """RetryPolicy测试"""

    def test_default_policy(self):
        policy = RetryPolicy()
        assert policy.max_attempts == 3
        assert policy.base_delay == 1.0
        assert policy.backoff_factor == 2.0

    def test_should_retry_recoverable(self):
        policy = RetryPolicy(max_attempts=3)
        error = LLMClientError(model="gpt-4", error="timeout")
        assert policy.should_retry(error, 1) == True
        assert policy.should_retry(error, 2) == True
        assert policy.should_retry(error, 3) == False  # 已达上限

    def test_should_retry_fatal(self):
        policy = RetryPolicy()
        error = ConfigurationError(key="k", value="", error="bad")
        assert policy.should_retry(error, 1) == False

    def test_should_retry_degraded(self):
        policy = RetryPolicy()
        error = MemoryStorageError(operation="store", error="fail")
        assert policy.should_retry(error, 1) == False

    def test_get_delay(self):
        policy = RetryPolicy(base_delay=1.0, backoff_factor=2.0, max_delay=30.0)
        assert policy.get_delay(1) == 1.0
        assert policy.get_delay(2) == 2.0
        assert policy.get_delay(3) == 4.0

    def test_get_delay_max_cap(self):
        policy = RetryPolicy(base_delay=1.0, backoff_factor=10.0, max_delay=5.0)
        assert policy.get_delay(3) == 5.0  # 被max_delay限制

    def test_serialization(self):
        policy = RetryPolicy(max_attempts=5)
        data = policy.to_dict()
        assert data["max_attempts"] == 5
        assert "retryable_categories" in data


class TestErrorHandler:
    """ErrorHandler核心功能测试"""

    def test_creation(self):
        handler = ErrorHandler()
        assert handler.retry_policy.max_attempts == 3
        assert len(handler.error_history) == 0

    def test_record_error(self):
        handler = ErrorHandler()
        error = LLMClientError(model="gpt-4", error="timeout")
        record = handler.record_error(error, component="test")
        assert record.error_type == "LLMClientError"
        assert record.category == ErrorCategory.RECOVERABLE
        assert record.component == "test"
        assert len(handler.error_history) == 1

    def test_handle_error_warning(self):
        handler = ErrorHandler()
        # MemoryRecallError is DEGRADED
        error = MemoryRecallError(query="test", error="not found")
        record = handler.handle_error(error, component="memory")
        assert record.category == ErrorCategory.DEGRADED

    def test_handle_error_fatal(self):
        handler = ErrorHandler()
        error = ConfigurationError(key="k", value="", error="bad")
        record = handler.handle_error(error, component="config")
        assert record.category == ErrorCategory.FATAL

    def test_mark_recovered(self):
        handler = ErrorHandler()
        error = LLMClientError(model="gpt-4", error="timeout")
        record = handler.record_error(error)
        assert record.recovered == False

        result = handler.mark_recovered(record.error_id, "retry")
        assert result == True
        assert handler.error_history[0].recovered == True
        assert handler.error_history[0].recovery_method == "retry"

    def test_mark_recovered_not_found(self):
        handler = ErrorHandler()
        result = handler.mark_recovered("nonexistent", "retry")
        assert result == False

    def test_get_stats(self):
        handler = ErrorHandler()
        error = LLMClientError(model="gpt-4", error="timeout")
        handler.record_error(error)
        handler.record_error(error)

        stats = handler.get_stats()
        assert stats["total_errors"] == 2
        assert stats["history_size"] == 2
        assert "by_category" in stats

    def test_get_recent_errors(self):
        handler = ErrorHandler()
        for i in range(5):
            handler.record_error(RuntimeError(f"error_{i}"))

        recent = handler.get_recent_errors(limit=3)
        assert len(recent) == 3

    def test_get_unrecovered_errors(self):
        handler = ErrorHandler()
        e1 = LLMClientError(model="gpt-4", error="timeout")
        r1 = handler.record_error(e1)
        handler.record_error(RuntimeError("error2"))

        handler.mark_recovered(r1.error_id)
        unrecovered = handler.get_unrecovered_errors()
        assert len(unrecovered) == 1

    def test_get_diagnosis_healthy(self):
        handler = ErrorHandler()
        diagnosis = handler.get_diagnosis()
        assert diagnosis["status"] == "healthy"
        assert diagnosis["unrecovered_count"] == 0

    def test_get_diagnosis_degraded(self):
        handler = ErrorHandler()
        handler.record_error(LLMClientError(model="gpt-4", error="timeout"))
        diagnosis = handler.get_diagnosis()
        assert diagnosis["status"] == "degraded"
        assert diagnosis["unrecovered_count"] == 1
        assert len(diagnosis["recommendations"]) > 0

    def test_clear_history(self):
        handler = ErrorHandler()
        handler.record_error(RuntimeError("test"))
        handler.clear_history()
        assert len(handler.error_history) == 0

    def test_max_history_limit(self):
        handler = ErrorHandler(max_history=5)
        for i in range(10):
            handler.record_error(RuntimeError(f"error_{i}"))
        assert len(handler.error_history) == 5

    def test_execute_with_retry_success(self):
        handler = ErrorHandler()
        mock_fn = Mock(return_value="success")

        result = handler.execute_with_retry(mock_fn, component="test")
        assert result == "success"
        assert mock_fn.call_count == 1

    def test_execute_with_retry_eventual_success(self):
        handler = ErrorHandler(retry_policy=RetryPolicy(
            max_attempts=3, base_delay=0.01
        ))
        mock_fn = Mock(side_effect=[RuntimeError("fail"), RuntimeError("fail"), "ok"])

        result = handler.execute_with_retry(mock_fn, component="test")
        assert result == "ok"
        assert mock_fn.call_count == 3

    def test_execute_with_retry_all_fail(self):
        handler = ErrorHandler(retry_policy=RetryPolicy(
            max_attempts=2, base_delay=0.01
        ))
        mock_fn = Mock(side_effect=RuntimeError("always fail"))

        with pytest.raises(RuntimeError):
            handler.execute_with_retry(mock_fn, component="test")

        assert mock_fn.call_count == 2
        assert len(handler.error_history) == 2

    def test_execute_with_retry_non_retryable(self):
        handler = ErrorHandler()
        mock_fn = Mock(side_effect=ConfigurationError(key="k", value="", error="bad"))

        with pytest.raises(ConfigurationError):
            handler.execute_with_retry(mock_fn, component="test")

        assert mock_fn.call_count == 1  # 不重试


class TestAgentErrorIntegration:
    """Agent错误处理集成测试"""

    @patch('src.infrastructure.llm_client.LLMClient')
    @patch('src.modules.purification.pipeline.PurificationPipeline')
    def test_agent_has_error_handler(self, mock_pipeline, mock_llm):
        mock_pipeline.return_value = Mock()

        from src.agent.agent import Agent
        from src.agent.agent_config import AgentConfig

        config = AgentConfig()
        with patch('src.agent.agent.LLMClient', mock_llm):
            agent = Agent(config=config)

        assert agent.error_handler is not None
        assert isinstance(agent.error_handler, ErrorHandler)

    @patch('src.infrastructure.llm_client.LLMClient')
    @patch('src.modules.purification.pipeline.PurificationPipeline')
    def test_summary_includes_errors(self, mock_pipeline, mock_llm):
        mock_pipeline.return_value = Mock()

        from src.agent.agent import Agent
        from src.agent.agent_config import AgentConfig

        config = AgentConfig()
        with patch('src.agent.agent.LLMClient', mock_llm):
            agent = Agent(config=config)

        summary = agent.get_summary()
        assert "errors" in summary
        assert "total_errors" in summary["errors"]


class TestExportErrorHandler:
    """导出测试"""

    def test_error_handler_exported(self):
        from src.agent import ErrorHandler
        assert ErrorHandler is not None

    def test_error_category_exported(self):
        from src.agent import ErrorCategory
        assert ErrorCategory.RECOVERABLE == 2

    def test_error_record_exported(self):
        from src.agent import ErrorRecord
        record = ErrorRecord(error_id="test")
        assert isinstance(record, ErrorRecord)

    def test_retry_policy_exported(self):
        from src.agent import RetryPolicy
        policy = RetryPolicy()
        assert policy.max_attempts == 3


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
