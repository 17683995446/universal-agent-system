"""
阶段22测试：Agent性能监控

测试MetricType、PerformanceCollector、PerformanceReport、PerformanceAlert及Agent集成。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-27
"""

import pytest
from unittest.mock import Mock, patch

from src.agent.performance_monitor import (
    MetricType,
    MetricCategory,
    AlertLevel,
    PerformanceMetric,
    PerformanceCollector,
    PerformanceReport,
    PerformanceReportGenerator,
    PerformanceAlertConfig,
    PerformanceAlertManager,
    PerformanceMonitor,
)


class TestMetricType:
    """MetricType枚举测试"""

    def test_timing_metrics(self):
        """测试时间类指标"""
        assert MetricType.TURN_DURATION.value == "turn_duration"
        assert MetricType.LLM_LATENCY.value == "llm_latency"
        assert MetricType.TOOL_EXECUTION_TIME.value == "tool_execution"

    def test_count_metrics(self):
        """测试数量类指标"""
        assert MetricType.TOTAL_TURNS.value == "total_turns"
        assert MetricType.TOTAL_TOKENS.value == "total_tokens"
        assert MetricType.ERROR_COUNT.value == "error_count"

    def test_rate_metrics(self):
        """测试比率类指标"""
        assert MetricType.ERROR_RATE.value == "error_rate"
        assert MetricType.COMPRESSION_RATIO.value == "compression_ratio"


class TestMetricCategory:
    """MetricCategory枚举测试"""

    def test_categories(self):
        assert MetricCategory.TIMING.value == "timing"
        assert MetricCategory.COUNT.value == "count"
        assert MetricCategory.RATE.value == "rate"
        assert MetricCategory.RESOURCE.value == "resource"


class TestAlertLevel:
    """AlertLevel枚举测试"""

    def test_levels(self):
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.CRITICAL.value == "critical"


class TestPerformanceMetric:
    """PerformanceMetric测试"""

    def test_creation(self):
        metric = PerformanceMetric(
            metric_type=MetricType.TURN_DURATION,
            value=100.5,
            category=MetricCategory.TIMING,
            unit="ms",
        )
        assert metric.metric_type == MetricType.TURN_DURATION
        assert metric.value == 100.5
        assert metric.unit == "ms"

    def test_serialization(self):
        metric = PerformanceMetric(
            metric_type=MetricType.ERROR_RATE,
            value=0.05,
            category=MetricCategory.RATE,
        )
        data = metric.to_dict()
        assert data["metric_type"] == "error_rate"
        assert data["value"] == 0.05


class TestPerformanceCollector:
    """PerformanceCollector测试"""

    def test_creation(self):
        collector = PerformanceCollector()
        assert len(collector.metrics) == 0

    def test_record(self):
        collector = PerformanceCollector()
        metric = collector.record(MetricType.TURN_DURATION, 150.0)

        assert len(collector.metrics) == 1
        assert collector.metrics[0].value == 150.0

    def test_record_timing(self):
        collector = PerformanceCollector()
        collector.record_timing(MetricType.LLM_LATENCY, 200.0)

        assert collector.get_count(MetricType.LLM_LATENCY) == 1

    def test_record_count(self):
        collector = PerformanceCollector()
        collector.record_count(MetricType.TOTAL_TURNS, 5)

        assert collector.get_latest(MetricType.TOTAL_TURNS) == 5

    def test_record_rate(self):
        collector = PerformanceCollector()
        collector.record_rate(MetricType.ERROR_RATE, 0.02)

        assert collector.get_latest(MetricType.ERROR_RATE) == 0.02

    def test_aggregation(self):
        collector = PerformanceCollector()

        for i in [100, 200, 150, 180]:
            collector.record(MetricType.TURN_DURATION, i)

        assert collector.get_average(MetricType.TURN_DURATION) == 157.5
        assert collector.get_min(MetricType.TURN_DURATION) == 100
        assert collector.get_max(MetricType.TURN_DURATION) == 200

    def test_get_summary(self):
        collector = PerformanceCollector()

        for i in [100, 200, 150]:
            collector.record(MetricType.LLM_LATENCY, i)

        summary = collector.get_summary(MetricType.LLM_LATENCY)
        assert summary["count"] == 3
        assert summary["average"] == 150.0
        assert summary["min"] == 100
        assert summary["max"] == 200

    def test_get_metrics_filter(self):
        collector = PerformanceCollector()

        collector.record(MetricType.TURN_DURATION, 100)
        collector.record(MetricType.LLM_LATENCY, 200)
        collector.record(MetricType.TURN_DURATION, 150)

        filtered = collector.get_metrics(metric_type=MetricType.TURN_DURATION)
        assert len(filtered) == 2

    def test_max_metrics_limit(self):
        collector = PerformanceCollector(max_metrics=5)

        for i in range(10):
            collector.record(MetricType.TURN_DURATION, i)

        assert len(collector.metrics) == 5

    def test_clear(self):
        collector = PerformanceCollector()
        collector.record(MetricType.TURN_DURATION, 100)
        collector.clear()

        assert len(collector.metrics) == 0

    def test_get_stats(self):
        collector = PerformanceCollector()
        collector.record(MetricType.TURN_DURATION, 100)
        collector.record(MetricType.LLM_LATENCY, 200)

        stats = collector.get_stats()
        assert stats["total_collected"] == 2
        assert "summaries" in stats


class TestPerformanceReport:
    """PerformanceReport测试"""

    def test_creation(self):
        report = PerformanceReport(
            report_id="perf_001",
            recommendations=["Performance is good"],
        )
        assert report.report_id == "perf_001"
        assert len(report.recommendations) == 1

    def test_serialization(self):
        report = PerformanceReport(
            report_id="perf_002",
            metrics_summary={"turn_duration": {"average": 150}},
        )
        data = report.to_dict()
        assert data["report_id"] == "perf_002"
        assert "metrics_summary" in data


class TestPerformanceReportGenerator:
    """PerformanceReportGenerator测试"""

    def test_generate_empty(self):
        collector = PerformanceCollector()
        generator = PerformanceReportGenerator(collector)

        report = generator.generate()
        assert report.report_id != ""
        assert len(report.recommendations) > 0

    def test_generate_with_metrics(self):
        collector = PerformanceCollector()
        collector.record(MetricType.TURN_DURATION, 100)
        collector.record(MetricType.LLM_LATENCY, 200)

        generator = PerformanceReportGenerator(collector)
        report = generator.generate()

        assert "turn_duration" in report.metrics_summary
        assert "llm_latency" in report.metrics_summary


class TestPerformanceAlertManager:
    """PerformanceAlertManager测试"""

    def test_creation(self):
        collector = PerformanceCollector()
        manager = PerformanceAlertManager(collector)

        assert len(manager.DEFAULT_ALERTS) > 0

    def test_check_no_alert(self):
        collector = PerformanceCollector()
        manager = PerformanceAlertManager(collector)

        # 记录正常值
        collector.record(MetricType.LLM_LATENCY, 500)  # 低于阈值

        alerts = manager.check()
        assert len(alerts) == 0

    def test_check_alert_triggered(self):
        collector = PerformanceCollector()
        configs = [
            PerformanceAlertConfig(
                metric_type=MetricType.LLM_LATENCY,
                threshold=1000.0,
                level=AlertLevel.WARNING,
                comparison="above",
                message="LLM延迟过高",
            )
        ]
        manager = PerformanceAlertManager(collector, configs=configs)

        # 记录超阈值值
        collector.record(MetricType.LLM_LATENCY, 1500)

        alerts = manager.check()
        assert len(alerts) == 1
        assert alerts[0]["level"] == "warning"

    def test_get_stats(self):
        collector = PerformanceCollector()
        manager = PerformanceAlertManager(collector)

        stats = manager.get_stats()
        assert "total_alerts" in stats
        assert "configs_count" in stats

    def test_add_config(self):
        collector = PerformanceCollector()
        manager = PerformanceAlertManager(collector)

        new_config = PerformanceAlertConfig(
            metric_type=MetricType.ERROR_RATE,
            threshold=0.5,
            level=AlertLevel.CRITICAL,
        )
        manager.add_config(new_config)

        assert len(manager._configs) > len(manager.DEFAULT_ALERTS)


class TestPerformanceMonitor:
    """PerformanceMonitor统一入口测试"""

    def test_creation(self):
        monitor = PerformanceMonitor()
        assert monitor.collector is not None
        assert monitor.alert_manager is not None

    def test_record(self):
        monitor = PerformanceMonitor()
        monitor.record(MetricType.TURN_DURATION, 100)

        assert monitor.collector.get_count(MetricType.TURN_DURATION) == 1

    def test_generate_report(self):
        monitor = PerformanceMonitor()
        monitor.record(MetricType.TURN_DURATION, 100)

        report = monitor.generate_report()
        assert report is not None
        assert report.report_id != ""

    def test_get_stats(self):
        monitor = PerformanceMonitor()
        monitor.record(MetricType.TURN_DURATION, 100)

        stats = monitor.get_stats()
        assert "collector" in stats
        assert "alerts" in stats

    def test_get_summary(self):
        monitor = PerformanceMonitor()

        for i in [100, 200, 150]:
            monitor.record(MetricType.LLM_LATENCY, i)

        summary = monitor.get_summary(MetricType.LLM_LATENCY)
        assert summary["average"] == 150.0

    def test_clear(self):
        monitor = PerformanceMonitor()
        monitor.record(MetricType.TURN_DURATION, 100)
        monitor.clear()

        assert len(monitor.collector.metrics) == 0


class TestAgentPerformanceIntegration:
    """Agent性能集成测试"""

    @patch('src.infrastructure.llm_client.LLMClient')
    @patch('src.modules.purification.pipeline.PurificationPipeline')
    def test_agent_has_performance_monitor(self, mock_pipeline, mock_llm):
        mock_pipeline.return_value = Mock()

        from src.agent.agent import Agent
        from src.agent.agent_config import AgentConfig

        config = AgentConfig()
        with patch('src.agent.agent.LLMClient', mock_llm):
            agent = Agent(config=config)

        assert agent.performance_monitor is not None

    @patch('src.infrastructure.llm_client.LLMClient')
    @patch('src.modules.purification.pipeline.PurificationPipeline')
    def test_agent_get_performance_report(self, mock_pipeline, mock_llm):
        mock_pipeline.return_value = Mock()

        from src.agent.agent import Agent
        from src.agent.agent_config import AgentConfig

        config = AgentConfig()
        with patch('src.agent.agent.LLMClient', mock_llm):
            agent = Agent(config=config)

        report = agent.get_performance_report()
        assert report is not None

    @patch('src.infrastructure.llm_client.LLMClient')
    @patch('src.modules.purification.pipeline.PurificationPipeline')
    def test_agent_get_performance_stats(self, mock_pipeline, mock_llm):
        mock_pipeline.return_value = Mock()

        from src.agent.agent import Agent
        from src.agent.agent_config import AgentConfig

        config = AgentConfig()
        with patch('src.agent.agent.LLMClient', mock_llm):
            agent = Agent(config=config)

        stats = agent.get_performance_stats()
        assert "collector" in stats


class TestExportPerformanceMonitor:
    """导出测试"""

    def test_performance_monitor_exported(self):
        from src.agent import PerformanceMonitor
        assert PerformanceMonitor is not None

    def test_metric_type_exported(self):
        from src.agent import MetricType
        assert MetricType.TURN_DURATION.value == "turn_duration"

    def test_alert_level_exported(self):
        from src.agent import AlertLevel
        assert AlertLevel.WARNING.value == "warning"


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])