"""阶段42: Agent指标仪表盘系统 - 单元测试"""
import pytest
from src.agent.metrics_dashboard import (
    MetricKind, MetricPoint, MetricSnapshot, DashboardPanel, Dashboard,
    MetricCollector, DashboardManager, create_metric_collector, create_dashboard_manager,
)

class TestMetricKind:
    def test_values(self):
        assert MetricKind.COUNTER == "counter"
        assert MetricKind.GAUGE == "gauge"

class TestMetricPoint:
    def test_creation(self):
        p = MetricPoint(name="cpu", value=0.5)
        assert p.name == "cpu"
        assert p.value == 0.5

class TestMetricSnapshot:
    def test_to_dict(self):
        s = MetricSnapshot(name="cpu", kind=MetricKind.GAUGE, value=0.5, count=10)
        d = s.to_dict()
        assert d["name"] == "cpu"
        assert d["kind"] == "gauge"

class TestMetricCollector:
    def test_record_gauge(self):
        c = MetricCollector()
        c.record("cpu", 0.5)
        snap = c.get_snapshot("cpu")
        assert snap.value == 0.5
        assert snap.kind == MetricKind.GAUGE

    def test_record_counter(self):
        c = MetricCollector()
        c.increment("requests")
        c.increment("requests")
        snap = c.get_snapshot("requests")
        assert snap.value == 2.0
        assert snap.kind == MetricKind.COUNTER

    def test_min_max_avg(self):
        c = MetricCollector()
        for v in [10, 20, 30]:
            c.record("temp", v)
        snap = c.get_snapshot("temp")
        assert snap.min_value == 10
        assert snap.max_value == 30
        assert snap.avg_value == 20.0

    def test_get_all_snapshots(self):
        c = MetricCollector()
        c.record("a", 1)
        c.record("b", 2)
        snaps = c.get_all_snapshots()
        assert len(snaps) == 2

    def test_clear_single(self):
        c = MetricCollector()
        c.record("a", 1)
        c.record("b", 2)
        c.clear("a")
        assert c.get_snapshot("a") is None
        assert c.get_snapshot("b") is not None

    def test_clear_all(self):
        c = MetricCollector()
        c.record("a", 1)
        c.clear()
        assert len(c.get_all_snapshots()) == 0

class TestDashboardManager:
    def test_create_dashboard(self):
        c = MetricCollector()
        m = DashboardManager(c)
        db = m.create_dashboard("main", "主仪表盘")
        assert db.name == "main"
        assert "main" in m.list_dashboards()

    def test_add_panel(self):
        c = MetricCollector()
        m = DashboardManager(c)
        m.create_dashboard("main")
        assert m.add_panel("main", "CPU", ["cpu_usage"])
        data = m.get_dashboard_data("main")
        assert len(data["panels"]) == 1

    def test_get_nonexistent(self):
        c = MetricCollector()
        m = DashboardManager(c)
        assert m.get_dashboard_data("nope") is None

    def test_delete_dashboard(self):
        c = MetricCollector()
        m = DashboardManager(c)
        m.create_dashboard("temp")
        assert m.delete_dashboard("temp")
        assert "temp" not in m.list_dashboards()

    def test_full_dashboard_data(self):
        c = MetricCollector()
        c.record("cpu", 0.5)
        c.record("mem", 1024)
        m = DashboardManager(c)
        m.create_dashboard("sys")
        m.add_panel("sys", "资源", ["cpu", "mem"])
        data = m.get_dashboard_data("sys")
        assert data["panels"][0]["metrics"][0]["value"] == 0.5
        assert data["panels"][0]["metrics"][1]["value"] == 1024

class TestConvenienceFunctions:
    def test_create_collector(self):
        c = create_metric_collector()
        assert isinstance(c, MetricCollector)

    def test_create_manager(self):
        c = create_metric_collector()
        m = create_dashboard_manager(c)
        assert isinstance(m, DashboardManager)
