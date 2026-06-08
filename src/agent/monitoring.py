"""
阶段75: Agent机器学习监控告警系统

遵循"图难于其易，为大于其细"的原则，实现指标采集、阈值告警、异常检测、通知推送。
"""
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    SILENCED = "silenced"


@dataclass
class MetricPoint:
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "value": self.value,
            "labels": self.labels,
        }


@dataclass
class AlertRule:
    rule_id: str
    name: str
    metric_name: str
    condition: str  # 'gt', 'lt', 'eq', 'ne'
    threshold: float
    severity: AlertSeverity = AlertSeverity.WARNING
    duration_sec: float = 0.0  # 持续多久触发
    labels: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "metric_name": self.metric_name,
            "condition": self.condition,
            "threshold": self.threshold,
            "severity": self.severity.value,
        }


@dataclass
class Alert:
    alert_id: str
    rule_id: str
    metric_name: str
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "rule_id": self.rule_id,
            "metric_name": self.metric_name,
            "severity": self.severity.value,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "status": self.status.value,
        }


class MetricCollector:
    """指标采集器"""

    def __init__(self, max_history: int = 10000):
        self._metrics: Dict[str, List[MetricPoint]] = {}
        self._max_history = max_history

    def record(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        if name not in self._metrics:
            self._metrics[name] = []
        self._metrics[name].append(MetricPoint(
            timestamp=time.time(),
            value=value,
            labels=labels or {},
        ))
        # 限制历史长度
        if len(self._metrics[name]) > self._max_history:
            self._metrics[name] = self._metrics[name][-self._max_history:]

    def get_series(self, name: str) -> List[MetricPoint]:
        return list(self._metrics.get(name, []))

    def get_latest(self, name: str) -> Optional[MetricPoint]:
        series = self._metrics.get(name)
        return series[-1] if series else None

    def get_average(self, name: str, window_sec: float = 60.0) -> float:
        series = self._metrics.get(name, [])
        now = time.time()
        values = [p.value for p in series if now - p.timestamp <= window_sec]
        return sum(values) / len(values) if values else 0.0

    def list_metrics(self) -> List[str]:
        return list(self._metrics.keys())

    def get_stats(self, name: str) -> Dict[str, Any]:
        series = self._metrics.get(name, [])
        if not series:
            return {"count": 0}
        values = [p.value for p in series]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1],
        }


class AnomalyDetector:
    """异常检测器"""

    def __init__(self, window_size: int = 10, threshold_sigma: float = 3.0):
        self._window_size = window_size
        self._threshold_sigma = threshold_sigma

    def detect(self, values: List[float]) -> List[bool]:
        """对每个点返回是否异常"""
        if len(values) < self._window_size:
            return [False] * len(values)
        results = []
        for i in range(len(values)):
            window = values[max(0, i - self._window_size):i]
            if not window:
                results.append(False)
                continue
            mean = sum(window) / len(window)
            variance = sum((x - mean) ** 2 for x in window) / len(window)
            std = variance ** 0.5
            if std == 0:
                results.append(False)
            else:
                results.append(abs(values[i] - mean) > self._threshold_sigma * std)
        return results

    def is_anomaly(self, value: float, history: List[float]) -> bool:
        if len(history) < self._window_size:
            return False
        window = history[-self._window_size:]
        mean = sum(window) / len(window)
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        std = variance ** 0.5
        if std == 0:
            return False
        return abs(value - mean) > self._threshold_sigma * std


class AlertManager:
    """告警管理器"""

    def __init__(self):
        self._rules: Dict[str, AlertRule] = {}
        self._alerts: Dict[str, Alert] = {}
        self._notifiers: List[Callable[[Alert], None]] = []
        self._silenced_rules: set = set()

    def add_rule(self, name: str, metric_name: str, condition: str,
                 threshold: float, severity: AlertSeverity = AlertSeverity.WARNING,
                 duration_sec: float = 0.0) -> AlertRule:
        rule_id = str(uuid.uuid4())[:8]
        rule = AlertRule(
            rule_id=rule_id,
            name=name,
            metric_name=metric_name,
            condition=condition,
            threshold=threshold,
            severity=severity,
            duration_sec=duration_sec,
        )
        self._rules[rule_id] = rule
        return rule

    def remove_rule(self, rule_id: str) -> bool:
        return self._rules.pop(rule_id, None) is not None

    def list_rules(self) -> List[AlertRule]:
        return list(self._rules.values())

    def add_notifier(self, notifier: Callable[[Alert], None]) -> None:
        self._notifiers.append(notifier)

    def evaluate(self, metric_name: str, value: float,
                 labels: Optional[Dict[str, str]] = None) -> List[Alert]:
        triggered = []
        for rule in self._rules.values():
            if not rule.enabled or rule.metric_name != metric_name:
                continue
            if rule.rule_id in self._silenced_rules:
                continue
            matched = self._check_condition(value, rule.condition, rule.threshold)
            if matched:
                alert = self._trigger_alert(rule, value)
                triggered.append(alert)
            else:
                self._resolve_alert(rule.rule_id)
        return triggered

    def _check_condition(self, value: float, condition: str, threshold: float) -> bool:
        if condition == "gt":
            return value > threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "eq":
            return abs(value - threshold) < 0.0001
        elif condition == "ne":
            return abs(value - threshold) >= 0.0001
        elif condition == "gte":
            return value >= threshold
        elif condition == "lte":
            return value <= threshold
        return False

    def _trigger_alert(self, rule: AlertRule, value: float) -> Alert:
        alert_id = f"{rule.rule_id}_{int(time.time())}"
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            metric_name=rule.metric_name,
            severity=rule.severity,
            message=f"{rule.name}: {rule.metric_name}={value} {rule.condition} {rule.threshold}",
            value=value,
            threshold=rule.threshold,
        )
        self._alerts[alert_id] = alert
        for notifier in self._notifiers:
            try:
                notifier(alert)
            except Exception:
                pass
        return alert

    def _resolve_alert(self, rule_id: str) -> None:
        for alert in list(self._alerts.values()):
            if alert.rule_id == rule_id and alert.status == AlertStatus.ACTIVE:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = time.time()

    def get_active_alerts(self) -> List[Alert]:
        return [a for a in self._alerts.values() if a.status == AlertStatus.ACTIVE]

    def get_alerts(self) -> List[Alert]:
        return list(self._alerts.values())

    def silence_rule(self, rule_id: str) -> bool:
        if rule_id in self._rules:
            self._silenced_rules.add(rule_id)
            return True
        return False

    def unsilence_rule(self, rule_id: str) -> bool:
        self._silenced_rules.discard(rule_id)
        return True

    def get_stats(self) -> Dict[str, Any]:
        total = len(self._alerts)
        active = sum(1 for a in self._alerts.values() if a.status == AlertStatus.ACTIVE)
        return {
            "total_alerts": total,
            "active_alerts": active,
            "resolved_alerts": total - active,
            "rules_count": len(self._rules),
        }


class MonitoringSystem:
    """监控系统"""

    def __init__(self):
        self._collector = MetricCollector()
        self._detector = AnomalyDetector()
        self._alert_manager = AlertManager()

    @property
    def collector(self) -> MetricCollector:
        return self._collector

    @property
    def detector(self) -> AnomalyDetector:
        return self._detector

    @property
    def alert_manager(self) -> AlertManager:
        return self._alert_manager

    def record(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        self._collector.record(name, value, labels)
        # 自动触发告警评估
        self._alert_manager.evaluate(name, value, labels)

    def check_anomaly(self, name: str) -> bool:
        series = self._collector.get_series(name)
        values = [p.value for p in series]
        if not values:
            return False
        return self._detector.is_anomaly(values[-1], values[:-1])

    def get_health(self) -> Dict[str, Any]:
        active = self._alert_manager.get_active_alerts()
        critical = sum(1 for a in active if a.severity == AlertSeverity.CRITICAL)
        warning = sum(1 for a in active if a.severity == AlertSeverity.WARNING)
        return {
            "status": "critical" if critical > 0 else "warning" if warning > 0 else "healthy",
            "active_alerts": len(active),
            "critical": critical,
            "warning": warning,
        }

    def get_dashboard(self) -> Dict[str, Any]:
        return {
            "metrics": self._collector.list_metrics(),
            "health": self.get_health(),
            "alert_stats": self._alert_manager.get_stats(),
        }


def create_monitoring_system() -> MonitoringSystem:
    return MonitoringSystem()
