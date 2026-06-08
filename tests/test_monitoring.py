"""阶段75: Agent机器学习监控告警系统 - 单元测试"""
import pytest
from src.agent.monitoring import (
    AlertSeverity, AlertStatus, MetricPoint, AlertRule, Alert,
    MetricCollector, AnomalyDetector, AlertManager, MonitoringSystem, create_monitoring_system,
)


class TestMetricCollector:
    def test_record_and_get_series(self):
        c = MetricCollector()
        c.record("cpu", 50.0)
        c.record("cpu", 60.0)
        series = c.get_series("cpu")
        assert len(series) == 2
        assert series[0].value == 50.0

    def test_get_latest(self):
        c = MetricCollector()
        c.record("cpu", 50.0)
        c.record("cpu", 60.0)
        latest = c.get_latest("cpu")
        assert latest is not None
        assert latest.value == 60.0

    def test_get_latest_empty(self):
        c = MetricCollector()
        assert c.get_latest("cpu") is None

    def test_get_average(self):
        c = MetricCollector()
        c.record("cpu", 50.0)
        c.record("cpu", 70.0)
        assert c.get_average("cpu", window_sec=60.0) == 60.0

    def test_list_metrics(self):
        c = MetricCollector()
        c.record("cpu", 50.0)
        c.record("mem", 80.0)
        assert set(c.list_metrics()) == {"cpu", "mem"}

    def test_get_stats(self):
        c = MetricCollector()
        c.record("cpu", 50.0)
        c.record("cpu", 70.0)
        stats = c.get_stats("cpu")
        assert stats["count"] == 2
        assert stats["min"] == 50.0
        assert stats["max"] == 70.0
        assert stats["avg"] == 60.0

    def test_max_history(self):
        c = MetricCollector(max_history=3)
        for i in range(5):
            c.record("cpu", float(i))
        assert len(c.get_series("cpu")) == 3


class TestAnomalyDetector:
    def test_detect_normal(self):
        d = AnomalyDetector(window_size=5, threshold_sigma=3.0)
        values = [10.0] * 20
        results = d.detect(values)
        assert not any(results)

    def test_detect_anomaly(self):
        d = AnomalyDetector(window_size=5, threshold_sigma=1.0)
        values = [10.0, 10.1, 9.9, 10.0, 10.1, 100.0]
        results = d.detect(values)
        assert results[-1] is True

    def test_is_anomaly_insufficient_history(self):
        d = AnomalyDetector(window_size=10)
        assert not d.is_anomaly(100.0, [1.0, 2.0])

    def test_is_anomaly_true(self):
        d = AnomalyDetector(window_size=5, threshold_sigma=1.0)
        history = [10.0, 10.1, 9.9, 10.0, 10.1]
        assert d.is_anomaly(100.0, history)

    def test_is_anomaly_false(self):
        d = AnomalyDetector(window_size=5, threshold_sigma=3.0)
        history = [10.0, 10.1, 9.9, 10.0, 10.1]
        assert not d.is_anomaly(10.2, history)


class TestAlertManager:
    def test_add_rule(self):
        mgr = AlertManager()
        rule = mgr.add_rule("high_cpu", "cpu", "gt", 80.0, AlertSeverity.CRITICAL)
        assert rule.name == "high_cpu"
        assert rule.condition == "gt"

    def test_remove_rule(self):
        mgr = AlertManager()
        rule = mgr.add_rule("high_cpu", "cpu", "gt", 80.0)
        assert mgr.remove_rule(rule.rule_id)
        assert not mgr.remove_rule("nonexistent")

    def test_evaluate_trigger(self):
        mgr = AlertManager()
        rule = mgr.add_rule("high_cpu", "cpu", "gt", 80.0, AlertSeverity.WARNING)
        alerts = mgr.evaluate("cpu", 90.0)
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING

    def test_evaluate_no_trigger(self):
        mgr = AlertManager()
        rule = mgr.add_rule("high_cpu", "cpu", "gt", 80.0)
        alerts = mgr.evaluate("cpu", 70.0)
        assert len(alerts) == 0

    def test_evaluate_resolve(self):
        mgr = AlertManager()
        rule = mgr.add_rule("high_cpu", "cpu", "gt", 80.0)
        mgr.evaluate("cpu", 90.0)
        mgr.evaluate("cpu", 70.0)
        active = mgr.get_active_alerts()
        assert len(active) == 0

    def test_silence_rule(self):
        mgr = AlertManager()
        rule = mgr.add_rule("high_cpu", "cpu", "gt", 80.0)
        mgr.silence_rule(rule.rule_id)
        alerts = mgr.evaluate("cpu", 90.0)
        assert len(alerts) == 0

    def test_unsilence_rule(self):
        mgr = AlertManager()
        rule = mgr.add_rule("high_cpu", "cpu", "gt", 80.0)
        mgr.silence_rule(rule.rule_id)
        mgr.unsilence_rule(rule.rule_id)
        alerts = mgr.evaluate("cpu", 90.0)
        assert len(alerts) == 1

    def test_notifier(self):
        mgr = AlertManager()
        notified = []
        mgr.add_notifier(lambda a: notified.append(a.alert_id))
        mgr.add_rule("high_cpu", "cpu", "gt", 80.0)
        mgr.evaluate("cpu", 90.0)
        assert len(notified) == 1

    def test_get_stats(self):
        mgr = AlertManager()
        mgr.add_rule("high_cpu", "cpu", "gt", 80.0)
        mgr.evaluate("cpu", 90.0)
        stats = mgr.get_stats()
        assert stats["total_alerts"] == 1
        assert stats["active_alerts"] == 1
        assert stats["rules_count"] == 1


class TestMonitoringSystem:
    def test_record_and_auto_evaluate(self):
        sys = MonitoringSystem()
        sys.alert_manager.add_rule("high_cpu", "cpu", "gt", 80.0)
        sys.record("cpu", 90.0)
        assert len(sys.alert_manager.get_active_alerts()) == 1

    def test_check_anomaly(self):
        sys = MonitoringSystem()
        for i in range(10):
            sys.record("cpu", 50.0 + i * 0.1)
        assert not sys.check_anomaly("cpu")
        sys.record("cpu", 200.0)
        assert sys.check_anomaly("cpu")

    def test_get_health_healthy(self):
        sys = MonitoringSystem()
        health = sys.get_health()
        assert health["status"] == "healthy"
        assert health["active_alerts"] == 0

    def test_get_health_warning(self):
        sys = MonitoringSystem()
        sys.alert_manager.add_rule("high_cpu", "cpu", "gt", 80.0)
        sys.record("cpu", 90.0)
        health = sys.get_health()
        assert health["status"] == "warning"

    def test_get_health_critical(self):
        sys = MonitoringSystem()
        sys.alert_manager.add_rule("high_cpu", "cpu", "gt", 80.0, AlertSeverity.CRITICAL)
        sys.record("cpu", 90.0)
        health = sys.get_health()
        assert health["status"] == "critical"

    def test_get_dashboard(self):
        sys = MonitoringSystem()
        sys.record("cpu", 50.0)
        dash = sys.get_dashboard()
        assert "metrics" in dash
        assert "health" in dash
        assert "alert_stats" in dash

    def test_create_monitoring_system(self):
        sys = create_monitoring_system()
        assert isinstance(sys, MonitoringSystem)
