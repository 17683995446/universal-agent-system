"""
阶段43: Agent通知与告警系统
"""
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertStatus(str, Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SILENCED = "silenced"

class NotificationChannel(str, Enum):
    LOG = "log"
    CALLBACK = "callback"
    WEBHOOK = "webhook"

@dataclass
class AlertRule:
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity = AlertSeverity.WARNING
    message_template: str = ""
    cooldown: float = 60.0
    enabled: bool = True
    tags: List[str] = field(default_factory=list)

@dataclass
class Alert:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    rule_name: str = ""
    severity: AlertSeverity = AlertSeverity.WARNING
    status: AlertStatus = AlertStatus.ACTIVE
    message: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "rule_name": self.rule_name,
            "severity": self.severity.value, "status": self.status.value,
            "message": self.message, "context": self.context,
            "created_at": self.created_at,
        }

@dataclass
class Notification:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    channel: NotificationChannel = NotificationChannel.LOG
    title: str = ""
    message: str = ""
    severity: AlertSeverity = AlertSeverity.INFO
    created_at: float = field(default_factory=time.time)
    delivered: bool = False

class AlertManager:
    def __init__(self):
        self._rules: Dict[str, AlertRule] = {}
        self._alerts: List[Alert] = []
        self._notifications: List[Notification] = []
        self._callbacks: Dict[str, Callable] = {}
        self._last_triggered: Dict[str, float] = {}
        self._lock = threading.RLock()

    def add_rule(self, rule: AlertRule) -> None:
        with self._lock:
            self._rules[rule.name] = rule

    def remove_rule(self, name: str) -> bool:
        with self._lock:
            return self._rules.pop(name, None) is not None

    def evaluate(self, context: Dict[str, Any]) -> List[Alert]:
        with self._lock:
            new_alerts = []
            now = time.time()
            for rule in self._rules.values():
                if not rule.enabled:
                    continue
                if now - self._last_triggered.get(rule.name, 0) < rule.cooldown:
                    continue
                try:
                    if rule.condition(context):
                        msg = rule.message_template.format(**context) if rule.message_template else f"规则 {rule.name} 触发"
                        alert = Alert(rule_name=rule.name, severity=rule.severity, message=msg, context=context)
                        self._alerts.append(alert)
                        self._last_triggered[rule.name] = now
                        new_alerts.append(alert)
                        self._send_notification(alert)
                except Exception:
                    pass
            return new_alerts

    def acknowledge(self, alert_id: str) -> bool:
        with self._lock:
            for a in self._alerts:
                if a.id == alert_id and a.status == AlertStatus.ACTIVE:
                    a.status = AlertStatus.ACKNOWLEDGED
                    a.acknowledged_at = time.time()
                    return True
            return False

    def resolve(self, alert_id: str) -> bool:
        with self._lock:
            for a in self._alerts:
                if a.id == alert_id and a.status in (AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED):
                    a.status = AlertStatus.RESOLVED
                    a.resolved_at = time.time()
                    return True
            return False

    def get_active_alerts(self) -> List[Alert]:
        with self._lock:
            return [a for a in self._alerts if a.status == AlertStatus.ACTIVE]

    def get_all_alerts(self) -> List[Alert]:
        with self._lock:
            return list(self._alerts)

    def register_callback(self, name: str, callback: Callable[[Alert], None]) -> None:
        with self._lock:
            self._callbacks[name] = callback

    def _send_notification(self, alert: Alert) -> None:
        notif = Notification(
            channel=NotificationChannel.CALLBACK,
            title=f"[{alert.severity.value}] {alert.rule_name}",
            message=alert.message,
            severity=alert.severity,
        )
        with self._lock:
            self._notifications.append(notif)
            for cb in self._callbacks.values():
                try:
                    cb(alert)
                except Exception:
                    pass
            notif.delivered = True

    def get_notifications(self, limit: int = 50) -> List[Notification]:
        with self._lock:
            return list(self._notifications[-limit:])

    def clear_resolved(self) -> int:
        with self._lock:
            before = len(self._alerts)
            self._alerts = [a for a in self._alerts if a.status != AlertStatus.RESOLVED]
            return before - len(self._alerts)

def create_alert_manager() -> AlertManager:
    return AlertManager()
