"""阶段43: Agent通知与告警系统 - 单元测试"""
import pytest
from src.agent.alert_system import (
    AlertSeverity, AlertStatus, NotificationChannel,
    AlertRule, Alert, Notification, AlertManager, create_alert_manager,
)

class TestAlertManager:
    def test_add_and_evaluate_rule(self):
        m = AlertManager()
        m.add_rule(AlertRule(name="high_cpu", condition=lambda ctx: ctx.get("cpu", 0) > 90,
                             message_template="CPU过高: {cpu}%"))
        alerts = m.evaluate({"cpu": 95})
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING

    def test_no_trigger(self):
        m = AlertManager()
        m.add_rule(AlertRule(name="high_cpu", condition=lambda ctx: ctx.get("cpu", 0) > 90))
        alerts = m.evaluate({"cpu": 50})
        assert len(alerts) == 0

    def test_cooldown(self):
        m = AlertManager()
        m.add_rule(AlertRule(name="r", condition=lambda ctx: True, cooldown=100))
        m.evaluate({})
        m.evaluate({})
        assert len(m.get_all_alerts()) == 1

    def test_acknowledge(self):
        m = AlertManager()
        m.add_rule(AlertRule(name="r", condition=lambda ctx: True))
        alerts = m.evaluate({})
        assert m.acknowledge(alerts[0].id)
        assert m.get_all_alerts()[0].status == AlertStatus.ACKNOWLEDGED

    def test_resolve(self):
        m = AlertManager()
        m.add_rule(AlertRule(name="r", condition=lambda ctx: True))
        alerts = m.evaluate({})
        assert m.resolve(alerts[0].id)
        assert m.get_all_alerts()[0].status == AlertStatus.RESOLVED

    def test_disabled_rule(self):
        m = AlertManager()
        m.add_rule(AlertRule(name="r", condition=lambda ctx: True, enabled=False))
        assert len(m.evaluate({})) == 0

    def test_callback(self):
        received = []
        m = AlertManager()
        m.register_callback("test", lambda a: received.append(a))
        m.add_rule(AlertRule(name="r", condition=lambda ctx: True))
        m.evaluate({})
        assert len(received) == 1

    def test_clear_resolved(self):
        m = AlertManager()
        m.add_rule(AlertRule(name="r", condition=lambda ctx: True))
        alerts = m.evaluate({})
        m.resolve(alerts[0].id)
        count = m.clear_resolved()
        assert count == 1

    def test_remove_rule(self):
        m = AlertManager()
        m.add_rule(AlertRule(name="r", condition=lambda ctx: True))
        assert m.remove_rule("r")
        assert len(m.evaluate({})) == 0

    def test_alert_to_dict(self):
        a = Alert(rule_name="test", message="hello")
        d = a.to_dict()
        assert d["rule_name"] == "test"

    def test_create_alert_manager(self):
        m = create_alert_manager()
        assert isinstance(m, AlertManager)
