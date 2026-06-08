"""
无上限上下文Agent系统 - 性能监控

实现性能指标收集、报告生成和告警功能。

核心设计：
- MetricType: 指标类型枚举
- PerformanceMetric: 单个指标数据
- PerformanceCollector: 指标收集器
- PerformanceReport: 报告生成器
- PerformanceAlert: 告警管理器

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-27
"""

import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# 指标类型
# =============================================================================

class MetricType(str, Enum):
    """性能指标类型"""
    # 时间指标
    TURN_DURATION = "turn_duration"           # 回合耗时（毫秒）
    LLM_LATENCY = "llm_latency"               # LLM调用延迟（毫秒）
    TOOL_EXECUTION_TIME = "tool_execution"    # 工具执行时间（毫秒）
    MEMORY_RECALL_TIME = "memory_recall"      # 记忆召回时间（毫秒）
    CONTEXT_BUILD_TIME = "context_build"      # 上下文构建时间（毫秒）

    # 数量指标
    TOTAL_TURNS = "total_turns"               # 总回合数
    TOTAL_TOKENS = "total_tokens"             # 总Token数
    TOOL_CALLS = "tool_calls"                 # 工具调用次数
    ERROR_COUNT = "error_count"               # 错误次数

    # 比率指标
    ERROR_RATE = "error_rate"                 # 错误率（0-1）
    TOKEN_EFFICIENCY = "token_efficiency"     # Token效率
    COMPRESSION_RATIO = "compression_ratio"   # 压缩比率

    # 资源指标
    MEMORY_USAGE = "memory_usage"             # 内存使用（MB）
    CONTEXT_SIZE = "context_size"             # 上下文大小（Token）
    HISTORY_LENGTH = "history_length"         # 历史长度


class MetricCategory(str, Enum):
    """指标分类"""
    TIMING = "timing"       # 时间类
    COUNT = "count"         # 数量类
    RATE = "rate"           # 比率类
    RESOURCE = "resource"   # 资源类


# =============================================================================
# 指标数据
# =============================================================================

@dataclass
class PerformanceMetric:
    """
    单个性能指标

    Attributes:
        metric_type: 指标类型
        value: 指标值
        timestamp: 记录时间
        category: 指标分类
        unit: 单位
        metadata: 元数据
    """
    metric_type: MetricType
    value: float
    timestamp: float = field(default_factory=time.time)
    category: MetricCategory = MetricCategory.TIMING
    unit: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "metric_type": self.metric_type.value,
            "value": self.value,
            "timestamp": self.timestamp,
            "category": self.category.value,
            "unit": self.unit,
            "metadata": self.metadata,
        }


# =============================================================================
# 指标收集器
# =============================================================================

class PerformanceCollector:
    """
    性能指标收集器

    收集、存储和聚合性能指标。

    Attributes:
        metrics: 收集的指标列表
    """

    def __init__(self, max_metrics: int = 1000):
        """
        Args:
            max_metrics: 最大指标存储数量
        """
        self._metrics: list[PerformanceMetric] = []
        self._max_metrics = max_metrics
        self._aggregations: dict[MetricType, list[float]] = defaultdict(list)
        self._total_collected: int = 0

    # =========================================================================
    # 收集
    # =========================================================================

    def record(
        self,
        metric_type: MetricType,
        value: float,
        category: Optional[MetricCategory] = None,
        unit: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> PerformanceMetric:
        """
        记录单个指标

        Args:
            metric_type: 指标类型
            value: 指标值
            category: 分类（None则自动推断）
            unit: 单位
            metadata: 元数据

        Returns:
            PerformanceMetric
        """
        # 自动推断分类
        if category is None:
            category = self._infer_category(metric_type)

        # 自动推断单位
        if unit == "":
            unit = self._infer_unit(metric_type)

        metric = PerformanceMetric(
            metric_type=metric_type,
            value=value,
            category=category,
            unit=unit,
            metadata=metadata or {},
        )

        self._metrics.append(metric)
        self._aggregations[metric_type].append(value)
        self._total_collected += 1

        # 限制存储数量
        if len(self._metrics) > self._max_metrics:
            self._metrics = self._metrics[-self._max_metrics:]

        return metric

    def record_timing(
        self,
        metric_type: MetricType,
        duration_ms: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> PerformanceMetric:
        """便捷记录时间指标"""
        return self.record(
            metric_type=metric_type,
            value=duration_ms,
            category=MetricCategory.TIMING,
            unit="ms",
            metadata=metadata,
        )

    def record_count(
        self,
        metric_type: MetricType,
        count: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> PerformanceMetric:
        """便捷记录数量指标"""
        return self.record(
            metric_type=metric_type,
            value=count,
            category=MetricCategory.COUNT,
            unit="",
            metadata=metadata,
        )

    def record_rate(
        self,
        metric_type: MetricType,
        rate: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> PerformanceMetric:
        """便捷记录比率指标"""
        return self.record(
            metric_type=metric_type,
            value=rate,
            category=MetricCategory.RATE,
            unit="%",
            metadata=metadata,
        )

    # =========================================================================
    # 查询
    # =========================================================================

    @property
    def metrics(self) -> list[PerformanceMetric]:
        """所有指标（只读副本）"""
        return self._metrics.copy()

    def get_metrics(
        self,
        metric_type: Optional[MetricType] = None,
        category: Optional[MetricCategory] = None,
        limit: int = 100,
    ) -> list[PerformanceMetric]:
        """
        获取指标

        Args:
            metric_type: 筛选类型（None则全部）
            category: 筛选分类（None则全部）
            limit: 最大返回数

        Returns:
            指标列表
        """
        filtered = self._metrics

        if metric_type is not None:
            filtered = [m for m in filtered if m.metric_type == metric_type]

        if category is not None:
            filtered = [m for m in filtered if m.category == category]

        return filtered[-limit:]

    # =========================================================================
    # 聚合
    # =========================================================================

    def get_average(self, metric_type: MetricType) -> Optional[float]:
        """获取指标平均值"""
        values = self._aggregations.get(metric_type, [])
        if not values:
            return None
        return sum(values) / len(values)

    def get_min(self, metric_type: MetricType) -> Optional[float]:
        """获取指标最小值"""
        values = self._aggregations.get(metric_type, [])
        if not values:
            return None
        return min(values)

    def get_max(self, metric_type: MetricType) -> Optional[float]:
        """获取指标最大值"""
        values = self._aggregations.get(metric_type, [])
        if not values:
            return None
        return max(values)

    def get_latest(self, metric_type: MetricType) -> Optional[float]:
        """获取最新指标值"""
        values = self._aggregations.get(metric_type, [])
        if not values:
            return None
        return values[-1]

    def get_count(self, metric_type: MetricType) -> int:
        """获取指标记录次数"""
        return len(self._aggregations.get(metric_type, []))

    def get_summary(self, metric_type: MetricType) -> dict[str, Any]:
        """获取指标摘要"""
        values = self._aggregations.get(metric_type, [])
        if not values:
            return {"count": 0}

        return {
            "count": len(values),
            "average": round(sum(values) / len(values), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "latest": round(values[-1], 2),
        }

    # =========================================================================
    # 统计
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """获取收集器统计"""
        summaries = {}
        for metric_type in self._aggregations:
            summaries[metric_type.value] = self.get_summary(metric_type)

        return {
            "total_collected": self._total_collected,
            "metrics_count": len(self._metrics),
            "metric_types": list(self._aggregations.keys()),
            "summaries": summaries,
        }

    def clear(self) -> None:
        """清空所有指标"""
        self._metrics.clear()
        self._aggregations.clear()
        logger.debug("Performance metrics cleared")

    # =========================================================================
    # 辅助方法
    # =========================================================================

    def _infer_category(self, metric_type: MetricType) -> MetricCategory:
        """推断指标分类"""
        timing_types = {
            MetricType.TURN_DURATION,
            MetricType.LLM_LATENCY,
            MetricType.TOOL_EXECUTION_TIME,
            MetricType.MEMORY_RECALL_TIME,
            MetricType.CONTEXT_BUILD_TIME,
        }
        count_types = {
            MetricType.TOTAL_TURNS,
            MetricType.TOTAL_TOKENS,
            MetricType.TOOL_CALLS,
            MetricType.ERROR_COUNT,
            MetricType.HISTORY_LENGTH,
        }
        rate_types = {
            MetricType.ERROR_RATE,
            MetricType.TOKEN_EFFICIENCY,
            MetricType.COMPRESSION_RATIO,
        }

        if metric_type in timing_types:
            return MetricCategory.TIMING
        elif metric_type in count_types:
            return MetricCategory.COUNT
        elif metric_type in rate_types:
            return MetricCategory.RATE
        else:
            return MetricCategory.RESOURCE

    def _infer_unit(self, metric_type: MetricType) -> str:
        """推断指标单位"""
        units = {
            MetricType.TURN_DURATION: "ms",
            MetricType.LLM_LATENCY: "ms",
            MetricType.TOOL_EXECUTION_TIME: "ms",
            MetricType.MEMORY_RECALL_TIME: "ms",
            MetricType.CONTEXT_BUILD_TIME: "ms",
            MetricType.ERROR_RATE: "%",
            MetricType.TOKEN_EFFICIENCY: "%",
            MetricType.COMPRESSION_RATIO: "%",
            MetricType.MEMORY_USAGE: "MB",
            MetricType.CONTEXT_SIZE: "tokens",
        }
        return units.get(metric_type, "")


# =============================================================================
# 性能报告
# =============================================================================

@dataclass
class PerformanceReport:
    """
    性能报告

    Attributes:
        report_id: 报告ID
        generated_at: 生成时间
        period_start: 统计开始时间
        period_end: 统计结束时间
        metrics_summary: 指标摘要
        alerts: 告警列表
        recommendations: 建议
    """
    report_id: str = ""
    generated_at: float = field(default_factory=time.time)
    period_start: float = 0.0
    period_end: float = 0.0
    metrics_summary: dict[str, Any] = field(default_factory=dict)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "metrics_summary": self.metrics_summary,
            "alerts": self.alerts,
            "recommendations": self.recommendations,
        }


class PerformanceReportGenerator:
    """性能报告生成器"""

    def __init__(self, collector: PerformanceCollector):
        """
        Args:
            collector: 指标收集器
        """
        self._collector = collector

    def generate(
        self,
        period_start: Optional[float] = None,
        period_end: Optional[float] = None,
    ) -> PerformanceReport:
        """
        生成性能报告

        Args:
            period_start: 统计开始时间（None则使用最早指标时间）
            period_end: 统计结束时间（None则使用当前时间）

        Returns:
            PerformanceReport
        """
        metrics = self._collector.metrics

        if not metrics:
            return PerformanceReport(
                report_id=f"perf_{int(time.time() * 1000)}",
                recommendations=["暂无性能数据，建议运行更多回合后生成报告。"],
            )

        # 确定时间范围
        if period_start is None:
            period_start = min(m.timestamp for m in metrics)
        if period_end is None:
            period_end = time.time()

        # 生成摘要
        metrics_summary = self._collector.get_stats()["summaries"]

        # 生成建议
        recommendations = self._generate_recommendations(metrics_summary)

        report = PerformanceReport(
            report_id=f"perf_{int(time.time() * 1000)}",
            period_start=period_start,
            period_end=period_end,
            metrics_summary=metrics_summary,
            recommendations=recommendations,
        )

        return report

    def _generate_recommendations(
        self,
        metrics_summary: dict[str, Any],
    ) -> list[str]:
        """生成性能建议"""
        recommendations = []

        # 检查LLM延迟
        llm_summary = metrics_summary.get("llm_latency", {})
        if llm_summary.get("average", 0) > 2000:
            recommendations.append(
                f"LLM平均延迟较高（{llm_summary['average']}ms），"
                f"建议检查网络连接或考虑使用更快的模型。"
            )

        # 检查错误率
        error_rate = metrics_summary.get("error_rate", {})
        if error_rate.get("latest", 0) > 0.1:
            recommendations.append(
                f"错误率较高（{error_rate['latest']}），"
                f"建议检查错误日志并优化错误处理策略。"
            )

        # 检查回合耗时
        turn_summary = metrics_summary.get("turn_duration", {})
        if turn_summary.get("average", 0) > 5000:
            recommendations.append(
                f"回合平均耗时较长（{turn_summary['average']}ms），"
                f"建议优化上下文构建或减少工具调用次数。"
            )

        if not recommendations:
            recommendations.append("性能表现良好，建议继续保持当前配置。")

        return recommendations


# =============================================================================
# 性能告警
# =============================================================================

class AlertLevel(str, Enum):
    """告警级别"""
    INFO = "info"       # 信息
    WARNING = "warning" # 警告
    CRITICAL = "critical"  # 严重


@dataclass
class PerformanceAlertConfig:
    """告警配置"""
    metric_type: MetricType
    threshold: float
    level: AlertLevel = AlertLevel.WARNING
    comparison: str = "above"  # above/below
    message: str = ""


class PerformanceAlertManager:
    """性能告警管理器"""

    # 默认告警配置
    DEFAULT_ALERTS: list[PerformanceAlertConfig] = [
        PerformanceAlertConfig(
            metric_type=MetricType.LLM_LATENCY,
            threshold=3000.0,
            level=AlertLevel.WARNING,
            comparison="above",
            message="LLM延迟超过阈值",
        ),
        PerformanceAlertConfig(
            metric_type=MetricType.ERROR_RATE,
            threshold=0.15,
            level=AlertLevel.WARNING,
            comparison="above",
            message="错误率超过阈值",
        ),
        PerformanceAlertConfig(
            metric_type=MetricType.ERROR_RATE,
            threshold=0.3,
            level=AlertLevel.CRITICAL,
            comparison="above",
            message="错误率严重超标",
        ),
        PerformanceAlertConfig(
            metric_type=MetricType.TURN_DURATION,
            threshold=10000.0,
            level=AlertLevel.WARNING,
            comparison="above",
            message="回合耗时过长",
        ),
    ]

    def __init__(
        self,
        collector: PerformanceCollector,
        configs: Optional[list[PerformanceAlertConfig]] = None,
    ):
        """
        Args:
            collector: 指标收集器
            configs: 告警配置（None则使用默认）
        """
        self._collector = collector
        self._configs = configs or self.DEFAULT_ALERTS.copy()
        self._triggered_alerts: list[dict[str, Any]] = []
        self._alert_count: int = 0

    def check(self) -> list[dict[str, Any]]:
        """
        检查所有告警

        Returns:
            触发的告警列表
        """
        triggered = []

        for config in self._configs:
            latest = self._collector.get_latest(config.metric_type)
            if latest is None:
                continue

            is_triggered = False
            if config.comparison == "above" and latest > config.threshold:
                is_triggered = True
            elif config.comparison == "below" and latest < config.threshold:
                is_triggered = True

            if is_triggered:
                alert = {
                    "metric_type": config.metric_type.value,
                    "value": round(latest, 2),
                    "threshold": config.threshold,
                    "level": config.level.value,
                    "message": config.message,
                    "timestamp": time.time(),
                }
                triggered.append(alert)
                self._triggered_alerts.append(alert)
                self._alert_count += 1

                # 限制历史长度
                if len(self._triggered_alerts) > 100:
                    self._triggered_alerts = self._triggered_alerts[-100:]

                logger.warning(
                    f"[{config.level.value.upper()}] {config.message}: "
                    f"{latest} > {config.threshold}"
                )

        return triggered

    def get_triggered_alerts(self, limit: int = 20) -> list[dict[str, Any]]:
        """获取触发的告警"""
        return self._triggered_alerts[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """获取告警统计"""
        level_counts = defaultdict(int)
        for alert in self._triggered_alerts:
            level_counts[alert["level"]] += 1

        return {
            "total_alerts": self._alert_count,
            "active_alerts": len(self._triggered_alerts),
            "by_level": dict(level_counts),
            "configs_count": len(self._configs),
        }

    def add_config(self, config: PerformanceAlertConfig) -> None:
        """添加告警配置"""
        self._configs.append(config)

    def clear_triggered(self) -> None:
        """清空触发历史"""
        self._triggered_alerts.clear()


# =============================================================================
# 性能监控器（统一入口）
# =============================================================================

class PerformanceMonitor:
    """
    性能监控器

    统一管理指标收集、报告生成和告警。

    Attributes:
        collector: 指标收集器
        report_generator: 报告生成器
        alert_manager: 告警管理器
    """

    def __init__(
        self,
        max_metrics: int = 1000,
        alert_configs: Optional[list[PerformanceAlertConfig]] = None,
    ):
        """
        Args:
            max_metrics: 最大指标存储数量
            alert_configs: 告警配置
        """
        self._collector = PerformanceCollector(max_metrics=max_metrics)
        self._report_generator = PerformanceReportGenerator(self._collector)
        self._alert_manager = PerformanceAlertManager(
            self._collector,
            configs=alert_configs,
        )

    # =========================================================================
    # 收集
    # =========================================================================

    def record(
        self,
        metric_type: MetricType,
        value: float,
        **kwargs: Any,
    ) -> PerformanceMetric:
        """记录指标"""
        metric = self._collector.record(metric_type, value, **kwargs)

        # 自动检查告警
        self._alert_manager.check()

        return metric

    def record_timing(
        self,
        metric_type: MetricType,
        duration_ms: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> PerformanceMetric:
        """记录时间指标"""
        return self.record(
            metric_type=metric_type,
            value=duration_ms,
            category=MetricCategory.TIMING,
            unit="ms",
            metadata=metadata,
        )

    # =========================================================================
    # 报告
    # =========================================================================

    def generate_report(
        self,
        period_start: Optional[float] = None,
        period_end: Optional[float] = None,
    ) -> PerformanceReport:
        """生成性能报告"""
        report = self._report_generator.generate(period_start, period_end)
        # 添加告警信息
        report.alerts = self._alert_manager.get_triggered_alerts()
        return report

    # =========================================================================
    # 告警
    # =========================================================================

    def check_alerts(self) -> list[dict[str, Any]]:
        """检查告警"""
        return self._alert_manager.check()

    def get_alerts(self, limit: int = 20) -> list[dict[str, Any]]:
        """获取告警列表"""
        return self._alert_manager.get_triggered_alerts(limit)

    # =========================================================================
    # 统计
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """获取监控统计"""
        return {
            "collector": self._collector.get_stats(),
            "alerts": self._alert_manager.get_stats(),
        }

    def get_summary(self, metric_type: MetricType) -> dict[str, Any]:
        """获取指标摘要"""
        return self._collector.get_summary(metric_type)

    # =========================================================================
    # 属性
    # =========================================================================

    @property
    def collector(self) -> PerformanceCollector:
        """指标收集器"""
        return self._collector

    @property
    def alert_manager(self) -> PerformanceAlertManager:
        """告警管理器"""
        return self._alert_manager

    # =========================================================================
    # 清理
    # =========================================================================

    def clear(self) -> None:
        """清空所有数据"""
        self._collector.clear()
        self._alert_manager.clear_triggered()