# 阶段22：Agent性能监控

## 概述

实现性能指标收集、报告生成和告警功能，为Agent系统提供全面的性能监控能力。

**原则遵循**："图难于其易，为大于其细。天下难事，必作于易；天下大事，必作于细。"

---

## 核心设计

### 指标类型（MetricType）

| 分类 | 指标 | 说明 |
|------|------|------|
| TIMING | TURN_DURATION | 回合耗时（ms） |
| TIMING | LLM_LATENCY | LLM调用延迟（ms） |
| TIMING | TOOL_EXECUTION_TIME | 工具执行时间（ms） |
| TIMING | MEMORY_RECALL_TIME | 记忆召回时间（ms） |
| TIMING | CONTEXT_BUILD_TIME | 上下文构建时间（ms） |
| COUNT | TOTAL_TURNS | 总回合数 |
| COUNT | TOTAL_TOKENS | 总Token数 |
| COUNT | TOOL_CALLS | 工具调用次数 |
| COUNT | ERROR_COUNT | 错误次数 |
| RATE | ERROR_RATE | 错误率（0-1） |
| RATE | TOKEN_EFFICIENCY | Token效率 |
| RATE | COMPRESSION_RATIO | 压缩比率 |
| RESOURCE | MEMORY_USAGE | 内存使用（MB） |
| RESOURCE | CONTEXT_SIZE | 上下文大小（Token） |
| RESOURCE | HISTORY_LENGTH | 历史长度 |

### 告警级别（AlertLevel）

```
INFO      → 信息提示
WARNING   → 警告（需关注）
CRITICAL  → 严重（需立即处理）
```

---

## 实现内容

### 22.1 MetricType + MetricCategory枚举

**文件**：`/workspace/src/agent/performance_monitor.py`

```python
class MetricType(str, Enum):
    TURN_DURATION = "turn_duration"
    LLM_LATENCY = "llm_latency"
    # ... 共15种指标

class MetricCategory(str, Enum):
    TIMING = "timing"
    COUNT = "count"
    RATE = "rate"
    RESOURCE = "resource"
```

### 22.2 PerformanceMetric指标数据

```python
@dataclass
class PerformanceMetric:
    metric_type: MetricType
    value: float
    timestamp: float
    category: MetricCategory
    unit: str
    metadata: dict

    def to_dict() -> dict
```

### 22.3 PerformanceCollector收集器

```python
class PerformanceCollector:
    def record(metric_type, value, category, unit, metadata) -> PerformanceMetric
    def record_timing(metric_type, duration_ms) -> PerformanceMetric
    def record_count(metric_type, count) -> PerformanceMetric
    def record_rate(metric_type, rate) -> PerformanceMetric

    def get_average(metric_type) -> float
    def get_min(metric_type) -> float
    def get_max(metric_type) -> float
    def get_latest(metric_type) -> float
    def get_count(metric_type) -> int
    def get_summary(metric_type) -> dict

    def get_metrics(metric_type, category, limit) -> list
    def get_stats() -> dict
    def clear()
```

### 22.4 PerformanceReport报告

```python
@dataclass
class PerformanceReport:
    report_id: str
    generated_at: float
    period_start: float
    period_end: float
    metrics_summary: dict
    recommendations: list[str]

class PerformanceReportGenerator:
    def generate(period_start, period_end) -> PerformanceReport
    def _generate_recommendations(metrics_summary) -> list[str]
```

### 22.5 PerformanceAlert告警

```python
class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class PerformanceAlertConfig:
    metric_type: MetricType
    threshold: float
    level: AlertLevel
    comparison: str  # "above"/"below"
    message: str

class PerformanceAlertManager:
    DEFAULT_ALERTS: list[PerformanceAlertConfig]  # 默认告警配置

    def check() -> list[dict]  # 检查告警
    def get_triggered_alerts(limit) -> list[dict]
    def get_stats() -> dict
    def add_config(config)
    def clear_triggered()
```

### 22.6 PerformanceMonitor统一入口

```python
class PerformanceMonitor:
    def __init__(max_metrics, alert_configs)

    @property
    def collector -> PerformanceCollector
    @property
    def alert_manager -> PerformanceAlertManager

    def record(metric_type, value) -> PerformanceMetric
    def record_timing(metric_type, duration_ms) -> PerformanceMetric
    def generate_report() -> PerformanceReport
    def check_alerts() -> list[dict]
    def get_alerts(limit) -> list[dict]
    def get_stats() -> dict
    def get_summary(metric_type) -> dict
    def clear()
```

### 22.7 Agent集成

**修改文件**：`/workspace/src/agent/agent.py`

```python
class Agent:
    def _init_performance_monitor(self):
        self._performance_monitor = PerformanceMonitor()

    def chat(self, user_input):
        turn_start_time = time.time()
        ...
        # 记录性能指标
        if self._performance_monitor:
            turn_duration = (time.time() - turn_start_time) * 1000
            self._performance_monitor.record_timing(
                MetricType.TURN_DURATION, turn_duration
            )

    @property
    def performance_monitor -> Optional[PerformanceMonitor]

    def get_performance_report() -> Optional[PerformanceReport]
    def get_performance_stats() -> dict
```

---

## 测试结果

**文件**：`/workspace/tests/test_performance_monitor.py`

| 测试类 | 测试数 | 结果 |
|--------|--------|------|
| TestMetricType | 3 | ✅ |
| TestMetricCategory | 1 | ✅ |
| TestAlertLevel | 1 | ✅ |
| TestPerformanceMetric | 2 | ✅ |
| TestPerformanceCollector | 11 | ✅ |
| TestPerformanceReport | 2 | ✅ |
| TestPerformanceReportGenerator | 2 | ✅ |
| TestPerformanceAlertManager | 5 | ✅ |
| TestPerformanceMonitor | 6 | ✅ |
| TestAgentPerformanceIntegration | 3 | ✅ |
| TestExportPerformanceMonitor | 3 | ✅ |

**总计**：39 passed ✅

**全系统测试**：839 passed（新增39个），2 failed（阶段15遗留）。

---

## 使用示例

```python
from src.agent import Agent, AgentConfig, MetricType

# 创建Agent
config = AgentConfig()
agent = Agent(config=config)

# 运行对话（自动记录性能指标）
agent.chat("Hello")

# 获取性能报告
report = agent.get_performance_report()
print(f"Report ID: {report.report_id}")
print(f"Summary: {report.metrics_summary}")
print(f"Recommendations: {report.recommendations}")

# 获取性能统计
stats = agent.get_performance_stats()
print(f"Total collected: {stats['collector']['total_collected']}")

# 获取特定指标摘要
summary = agent.performance_monitor.get_summary(MetricType.TURN_DURATION)
print(f"Average turn duration: {summary['average']}ms")

# 检查告警
alerts = agent.performance_monitor.check_alerts()
for alert in alerts:
    print(f"[{alert['level']}] {alert['message']}: {alert['value']}")
```

---

## 架构关系

```
Agent
  ├── _performance_monitor: PerformanceMonitor
  │       ├── _collector: PerformanceCollector
  │       │       ├── metrics: list[PerformanceMetric]
  │       │       ├── _aggregations: dict[MetricType, list[float]]
  │       │       └── record/get_average/get_summary...
  │       │
  │       ├── _report_generator: PerformanceReportGenerator
  │       │       └── generate() → PerformanceReport
  │       │
  │       └── _alert_manager: PerformanceAlertManager
  │               ├── _configs: list[PerformanceAlertConfig]
  │               ├── _triggered_alerts: list[dict]
  │               └── check() → 触发告警列表
  │
  ├── chat() → 记录 TURN_DURATION, TOTAL_TOKENS
  ├── get_performance_report() → PerformanceReport
  └── get_performance_stats() → dict
```

---

## 下阶段规划

**阶段23**：Agent多实例管理

- 实现Agent池管理
- 支持多Agent并发
- 实现Agent间通信

---

## 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `/workspace/src/agent/performance_monitor.py` | 新建 | 性能监控核心 |
| `/workspace/src/agent/agent.py` | 修改 | 集成性能监控 |
| `/workspace/src/agent/__init__.py` | 修改 | 导出性能监控类 |
| `/workspace/tests/test_performance_monitor.py` | 新建 | 39个测试 |

---

**阶段22完成** ✅