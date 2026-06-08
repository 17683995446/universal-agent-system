"""
阶段42: Agent指标仪表盘系统

遵循"图难于其易，为大于其细"的原则，实现指标收集、聚合、仪表盘展示。
"""
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

class MetricKind(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"

@dataclass
class MetricPoint:
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    kind: MetricKind = MetricKind.GAUGE

@dataclass
class MetricSnapshot:
    name: str
    kind: MetricKind
    value: float
    min_value: float = 0.0
    max_value: float = 0.0
    avg_value: float = 0.0
    count: int = 0
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind.value,
            "value": self.value, "min": self.min_value,
            "max": self.max_value, "avg": self.avg_value,
            "count": self.count, "tags": self.tags,
        }

@dataclass
class DashboardPanel:
    name: str
    metric_names: List[str] = field(default_factory=list)
    description: str = ""

@dataclass
class Dashboard:
    name: str
    panels: List[DashboardPanel] = field(default_factory=list)
    description: str = ""
    created_at: float = field(default_factory=time.time)

class MetricCollector:
    """指标收集器"""
    def __init__(self, history_size: int = 1000):
        self._metrics: Dict[str, deque] = {}
        self._counters: Dict[str, float] = {}
        self._history_size = history_size
        self._lock = threading.RLock()

    def record(self, name: str, value: float, kind: MetricKind = MetricKind.GAUGE,
               tags: Optional[Dict[str, str]] = None) -> None:
        with self._lock:
            point = MetricPoint(name=name, value=value, tags=tags or {}, kind=kind)
            if name not in self._metrics:
                self._metrics[name] = deque(maxlen=self._history_size)
            self._metrics[name].append(point)
            if kind == MetricKind.COUNTER:
                self._counters[name] = self._counters.get(name, 0) + value

    def increment(self, name: str, amount: float = 1.0, tags: Optional[Dict] = None) -> None:
        self.record(name, amount, MetricKind.COUNTER, tags)

    def get_snapshot(self, name: str) -> Optional[MetricSnapshot]:
        with self._lock:
            if name not in self._metrics or not self._metrics[name]:
                return None
            points = list(self._metrics[name])
            values = [p.value for p in points]
            kind = points[-1].kind
            tags = points[-1].tags
            if kind == MetricKind.COUNTER:
                return MetricSnapshot(
                    name=name, kind=kind,
                    value=self._counters.get(name, 0),
                    count=len(points), tags=tags,
                )
            return MetricSnapshot(
                name=name, kind=kind,
                value=values[-1], min_value=min(values),
                max_value=max(values), avg_value=sum(values)/len(values),
                count=len(points), tags=tags,
            )

    def get_all_snapshots(self) -> List[MetricSnapshot]:
        with self._lock:
            results = []
            for name in self._metrics:
                snap = self.get_snapshot(name)
                if snap:
                    results.append(snap)
            return results

    def get_metric_names(self) -> List[str]:
        with self._lock:
            return list(self._metrics.keys())

    def clear(self, name: Optional[str] = None) -> None:
        with self._lock:
            if name:
                self._metrics.pop(name, None)
                self._counters.pop(name, None)
            else:
                self._metrics.clear()
                self._counters.clear()

class DashboardManager:
    """仪表盘管理器"""
    def __init__(self, collector: MetricCollector):
        self._collector = collector
        self._dashboards: Dict[str, Dashboard] = {}
        self._lock = threading.RLock()

    def create_dashboard(self, name: str, description: str = "") -> Dashboard:
        with self._lock:
            db = Dashboard(name=name, description=description)
            self._dashboards[name] = db
            return db

    def add_panel(self, dashboard_name: str, panel_name: str,
                  metric_names: List[str], description: str = "") -> bool:
        with self._lock:
            db = self._dashboards.get(dashboard_name)
            if not db:
                return False
            db.panels.append(DashboardPanel(
                name=panel_name, metric_names=metric_names, description=description,
            ))
            return True

    def get_dashboard_data(self, name: str) -> Optional[Dict[str, Any]]:
        db = self._dashboards.get(name)
        if not db:
            return None
        panels_data = []
        for panel in db.panels:
            metrics = []
            for mn in panel.metric_names:
                snap = self._collector.get_snapshot(mn)
                metrics.append(snap.to_dict() if snap else {"name": mn, "value": None})
            panels_data.append({
                "name": panel.name, "description": panel.description, "metrics": metrics,
            })
        return {"name": db.name, "description": db.description, "panels": panels_data}

    def list_dashboards(self) -> List[str]:
        return list(self._dashboards.keys())

    def delete_dashboard(self, name: str) -> bool:
        with self._lock:
            return self._dashboards.pop(name, None) is not None

def create_metric_collector(history_size: int = 1000) -> MetricCollector:
    return MetricCollector(history_size)

def create_dashboard_manager(collector: MetricCollector) -> DashboardManager:
    return DashboardManager(collector)
