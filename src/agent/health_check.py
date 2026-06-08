"""
阶段31: Agent健康检查系统

遵循"图难于其易，为大于其细"的原则，实现全面的Agent健康检查机制。
"""

import asyncio
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Union


class HealthStatus(str, Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"       # 健康
    DEGRADED = "degraded"     # 降级运行
    UNHEALTHY = "unhealthy"   # 不健康
    UNKNOWN = "unknown"       # 未知


class CheckType(str, Enum):
    """检查类型枚举"""
    LIVENESS = "liveness"     # 存活检查
    READINESS = "readiness"   # 就绪检查
    STARTUP = "startup"       # 启动检查
    CUSTOM = "custom"         # 自定义检查


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    name: str
    status: HealthStatus
    check_type: CheckType
    message: str = ""
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "status": self.status.value,
            "check_type": self.check_type.value,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "error": str(self.error) if self.error else None
        }


@dataclass
class ComponentHealth:
    """组件健康状态"""
    component_id: str
    component_type: str
    status: HealthStatus
    checks: List[HealthCheckResult] = field(default_factory=list)
    last_check_time: datetime = field(default_factory=datetime.now)
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "status": self.status.value,
            "checks": [c.to_dict() for c in self.checks],
            "last_check_time": self.last_check_time.isoformat(),
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "metadata": self.metadata
        }


@dataclass
class SystemHealth:
    """系统整体健康状态"""
    system_id: str
    status: HealthStatus
    components: Dict[str, ComponentHealth] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    uptime_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "system_id": self.system_id,
            "status": self.status.value,
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "uptime_seconds": self.uptime_seconds
        }


# 健康检查函数类型
HealthCheckFunc = Callable[[], Union[HealthCheckResult, Coroutine[Any, Any, HealthCheckResult]]]


class HealthChecker:
    """健康检查器 - 执行单个组件的健康检查"""

    def __init__(self, name: str, check_type: CheckType = CheckType.CUSTOM):
        self._name = name
        self._check_type = check_type
        self._checks: Dict[str, HealthCheckFunc] = {}
        self._lock = threading.RLock()
        self._timeout_seconds = 30.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def check_type(self) -> CheckType:
        return self._check_type

    def register_check(self, check_name: str, check_func: HealthCheckFunc) -> None:
        """注册健康检查函数"""
        with self._lock:
            self._checks[check_name] = check_func

    def unregister_check(self, check_name: str) -> bool:
        """注销健康检查函数"""
        with self._lock:
            if check_name in self._checks:
                del self._checks[check_name]
                return True
            return False

    def set_timeout(self, timeout_seconds: float) -> None:
        """设置检查超时时间"""
        self._timeout_seconds = timeout_seconds

    async def check(self) -> List[HealthCheckResult]:
        """执行所有健康检查"""
        results = []

        with self._lock:
            checks = dict(self._checks)

        for check_name, check_func in checks.items():
            start_time = time.time()
            try:
                # 检查是否为异步函数
                if asyncio.iscoroutinefunction(check_func):
                    result = await asyncio.wait_for(
                        check_func(),
                        timeout=self._timeout_seconds
                    )
                else:
                    # 在线程池中运行同步函数
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, check_func),
                        timeout=self._timeout_seconds
                    )

                duration_ms = (time.time() - start_time) * 1000

                if not isinstance(result, HealthCheckResult):
                    result = HealthCheckResult(
                        name=check_name,
                        status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                        check_type=self._check_type,
                        message="Check passed" if result else "Check failed",
                        duration_ms=duration_ms
                    )
                else:
                    result.duration_ms = duration_ms

            except asyncio.TimeoutError:
                duration_ms = (time.time() - start_time) * 1000
                result = HealthCheckResult(
                    name=check_name,
                    status=HealthStatus.UNHEALTHY,
                    check_type=self._check_type,
                    message=f"Check timed out after {self._timeout_seconds}s",
                    duration_ms=duration_ms,
                    error=TimeoutError(f"Check exceeded {self._timeout_seconds}s timeout")
                )
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                result = HealthCheckResult(
                    name=check_name,
                    status=HealthStatus.UNHEALTHY,
                    check_type=self._check_type,
                    message=f"Check failed with error: {str(e)}",
                    duration_ms=duration_ms,
                    error=e
                )

            results.append(result)

        return results

    def check_sync(self) -> List[HealthCheckResult]:
        """同步执行所有健康检查"""
        return asyncio.run(self.check())


class HealthEndpoint:
    """健康端点 - 暴露健康检查HTTP端点"""

    def __init__(self, path: str = "/health"):
        self._path = path
        self._checkers: Dict[str, HealthChecker] = {}
        self._lock = threading.RLock()
        self._enabled_checks: Set[CheckType] = {
            CheckType.LIVENESS,
            CheckType.READINESS,
            CheckType.STARTUP
        }

    @property
    def path(self) -> str:
        return self._path

    def register_checker(self, name: str, checker: HealthChecker) -> None:
        """注册健康检查器"""
        with self._lock:
            self._checkers[name] = checker

    def unregister_checker(self, name: str) -> bool:
        """注销健康检查器"""
        with self._lock:
            if name in self._checkers:
                del self._checkers[name]
                return True
            return False

    def enable_check_type(self, check_type: CheckType) -> None:
        """启用检查类型"""
        self._enabled_checks.add(check_type)

    def disable_check_type(self, check_type: CheckType) -> None:
        """禁用检查类型"""
        self._enabled_checks.discard(check_type)

    async def get_liveness(self) -> Dict[str, Any]:
        """获取存活状态"""
        return await self._get_check_status(CheckType.LIVENESS)

    async def get_readiness(self) -> Dict[str, Any]:
        """获取就绪状态"""
        return await self._get_check_status(CheckType.READINESS)

    async def get_startup(self) -> Dict[str, Any]:
        """获取启动状态"""
        return await self._get_check_status(CheckType.STARTUP)

    async def get_health(self) -> Dict[str, Any]:
        """获取完整健康状态"""
        with self._lock:
            checkers = dict(self._checkers)

        all_results = []
        component_statuses = {}

        for name, checker in checkers.items():
            if checker.check_type in self._enabled_checks:
                results = await checker.check()
                all_results.extend(results)

                # 确定组件状态
                statuses = [r.status for r in results]
                if HealthStatus.UNHEALTHY in statuses:
                    component_statuses[name] = HealthStatus.UNHEALTHY
                elif HealthStatus.DEGRADED in statuses:
                    component_statuses[name] = HealthStatus.DEGRADED
                elif HealthStatus.UNKNOWN in statuses:
                    component_statuses[name] = HealthStatus.UNKNOWN
                else:
                    component_statuses[name] = HealthStatus.HEALTHY

        # 确定整体状态
        if HealthStatus.UNHEALTHY in component_statuses.values():
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in component_statuses.values():
            overall_status = HealthStatus.DEGRADED
        elif HealthStatus.UNKNOWN in component_statuses.values():
            overall_status = HealthStatus.UNKNOWN
        else:
            overall_status = HealthStatus.HEALTHY

        return {
            "status": overall_status.value,
            "components": {k: v.value for k, v in component_statuses.items()},
            "checks": [r.to_dict() for r in all_results],
            "timestamp": datetime.now().isoformat()
        }

    async def _get_check_status(self, check_type: CheckType) -> Dict[str, Any]:
        """获取特定类型的检查状态"""
        with self._lock:
            checkers = {
                name: checker
                for name, checker in self._checkers.items()
                if checker.check_type == check_type
            }

        all_results = []
        for name, checker in checkers.items():
            results = await checker.check()
            all_results.extend(results)

        # 确定状态
        statuses = [r.status for r in all_results]
        if HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        elif HealthStatus.UNKNOWN in statuses:
            overall_status = HealthStatus.UNKNOWN
        else:
            overall_status = HealthStatus.HEALTHY

        return {
            "status": overall_status.value,
            "checks": [r.to_dict() for r in all_results],
            "timestamp": datetime.now().isoformat()
        }


class HealthAlert:
    """健康警报"""

    def __init__(
        self,
        alert_id: str,
        component_id: str,
        severity: str,
        message: str,
        created_at: datetime = None
    ):
        self.alert_id = alert_id
        self.component_id = component_id
        self.severity = severity
        self.message = message
        self.created_at = created_at or datetime.now()
        self.resolved_at: Optional[datetime] = None
        self.resolved = False

    def resolve(self) -> None:
        """解决警报"""
        self.resolved = True
        self.resolved_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "alert_id": self.alert_id,
            "component_id": self.component_id,
            "severity": self.severity,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved": self.resolved
        }


class HealthMonitor:
    """健康监控器 - 持续监控组件健康状态"""

    def __init__(self, check_interval_seconds: float = 30.0):
        self._check_interval = check_interval_seconds
        self._checkers: Dict[str, HealthChecker] = {}
        self._component_health: Dict[str, ComponentHealth] = {}
        self._alerts: List[HealthAlert] = []
        self._alert_handlers: List[Callable[[HealthAlert], None]] = []
        self._lock = threading.RLock()
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._failure_threshold = 3
        self._success_threshold = 2

    def register_checker(
        self,
        component_id: str,
        component_type: str,
        checker: HealthChecker
    ) -> None:
        """注册组件健康检查器"""
        with self._lock:
            self._checkers[component_id] = checker
            if component_id not in self._component_health:
                self._component_health[component_id] = ComponentHealth(
                    component_id=component_id,
                    component_type=component_type,
                    status=HealthStatus.UNKNOWN
                )

    def unregister_checker(self, component_id: str) -> bool:
        """注销组件健康检查器"""
        with self._lock:
            if component_id in self._checkers:
                del self._checkers[component_id]
                if component_id in self._component_health:
                    del self._component_health[component_id]
                return True
            return False

    def add_alert_handler(self, handler: Callable[[HealthAlert], None]) -> None:
        """添加警报处理器"""
        self._alert_handlers.append(handler)

    def remove_alert_handler(self, handler: Callable[[HealthAlert], None]) -> bool:
        """移除警报处理器"""
        if handler in self._alert_handlers:
            self._alert_handlers.remove(handler)
            return True
        return False

    def set_thresholds(self, failure: int = 3, success: int = 2) -> None:
        """设置健康状态切换阈值"""
        self._failure_threshold = failure
        self._success_threshold = success

    async def start(self) -> None:
        """启动健康监控"""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        """停止健康监控"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

    async def _monitor_loop(self) -> None:
        """监控循环"""
        while self._running:
            try:
                await self._check_all_components()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                # 记录错误但继续监控
                await asyncio.sleep(self._check_interval)

    async def _check_all_components(self) -> None:
        """检查所有组件"""
        with self._lock:
            checkers = dict(self._checkers)

        for component_id, checker in checkers.items():
            try:
                results = await checker.check()
                await self._update_component_health(component_id, results)
            except Exception as e:
                # 检查失败，更新为不健康状态
                await self._update_component_health(component_id, [], error=e)

    async def _update_component_health(
        self,
        component_id: str,
        results: List[HealthCheckResult],
        error: Optional[Exception] = None
    ) -> None:
        """更新组件健康状态"""
        with self._lock:
            if component_id not in self._component_health:
                return

            health = self._component_health[component_id]
            health.checks = results
            health.last_check_time = datetime.now()

            if error:
                health.status = HealthStatus.UNHEALTHY
                health.consecutive_failures += 1
                health.consecutive_successes = 0
            else:
                # 根据检查结果确定状态
                statuses = [r.status for r in results]
                if HealthStatus.UNHEALTHY in statuses:
                    new_status = HealthStatus.UNHEALTHY
                    health.consecutive_failures += 1
                    health.consecutive_successes = 0
                elif HealthStatus.DEGRADED in statuses:
                    new_status = HealthStatus.DEGRADED
                    health.consecutive_failures += 1
                    health.consecutive_successes = 0
                elif HealthStatus.UNKNOWN in statuses:
                    new_status = HealthStatus.UNKNOWN
                else:
                    new_status = HealthStatus.HEALTHY
                    health.consecutive_successes += 1
                    health.consecutive_failures = 0

                # 应用阈值
                if health.consecutive_failures >= self._failure_threshold:
                    health.status = HealthStatus.UNHEALTHY
                elif health.consecutive_successes >= self._success_threshold:
                    health.status = HealthStatus.HEALTHY
                else:
                    health.status = new_status

            # 检查是否需要触发警报
            await self._check_alerts(component_id, health)

    async def _check_alerts(self, component_id: str, health: ComponentHealth) -> None:
        """检查并触发警报"""
        # 查找未解决的警报
        existing_alert = None
        for alert in self._alerts:
            if alert.component_id == component_id and not alert.resolved:
                existing_alert = alert
                break

        if health.status == HealthStatus.UNHEALTHY and not existing_alert:
            # 创建新警报
            alert = HealthAlert(
                alert_id=f"alert-{component_id}-{int(time.time())}",
                component_id=component_id,
                severity="critical",
                message=f"Component {component_id} is unhealthy"
            )
            self._alerts.append(alert)
            await self._notify_alert_handlers(alert)

        elif health.status == HealthStatus.HEALTHY and existing_alert:
            # 解决警报
            existing_alert.resolve()

    async def _notify_alert_handlers(self, alert: HealthAlert) -> None:
        """通知警报处理器"""
        for handler in self._alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception:
                # 忽略处理器错误
                pass

    def get_component_health(self, component_id: str) -> Optional[ComponentHealth]:
        """获取组件健康状态"""
        with self._lock:
            return self._component_health.get(component_id)

    def get_all_health(self) -> Dict[str, ComponentHealth]:
        """获取所有组件健康状态"""
        with self._lock:
            return dict(self._component_health)

    def get_alerts(
        self,
        resolved: Optional[bool] = None,
        component_id: Optional[str] = None
    ) -> List[HealthAlert]:
        """获取警报列表"""
        with self._lock:
            alerts = self._alerts

        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]

        if component_id:
            alerts = [a for a in alerts if a.component_id == component_id]

        return alerts

    def clear_resolved_alerts(self) -> int:
        """清除已解决的警报"""
        with self._lock:
            original_count = len(self._alerts)
            self._alerts = [a for a in self._alerts if not a.resolved]
            return original_count - len(self._alerts)


class HealthCheckBuilder:
    """健康检查构建器 - 简化健康检查创建"""

    @staticmethod
    def create_simple_check(
        name: str,
        check_func: Callable[[], bool],
        check_type: CheckType = CheckType.CUSTOM
    ) -> HealthChecker:
        """创建简单健康检查"""
        checker = HealthChecker(name, check_type)
        checker.register_check(name, lambda: HealthCheckResult(
            name=name,
            status=HealthStatus.HEALTHY if check_func() else HealthStatus.UNHEALTHY,
            check_type=check_type,
            message="Check passed" if check_func() else "Check failed"
        ))
        return checker

    @staticmethod
    def create_http_check(
        name: str,
        url: str,
        expected_status: int = 200,
        timeout_seconds: float = 5.0
    ) -> HealthChecker:
        """创建HTTP健康检查"""
        async def http_check() -> HealthCheckResult:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=timeout_seconds) as response:
                        if response.status == expected_status:
                            return HealthCheckResult(
                                name=name,
                                status=HealthStatus.HEALTHY,
                                check_type=CheckType.LIVENESS,
                                message=f"HTTP {response.status} OK"
                            )
                        else:
                            return HealthCheckResult(
                                name=name,
                                status=HealthStatus.UNHEALTHY,
                                check_type=CheckType.LIVENESS,
                                message=f"Unexpected status: {response.status}"
                            )
            except Exception as e:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    check_type=CheckType.LIVENESS,
                    message=f"HTTP check failed: {str(e)}",
                    error=e
                )

        checker = HealthChecker(name, CheckType.LIVENESS)
        checker.register_check(name, http_check)
        return checker

    @staticmethod
    def create_resource_check(
        name: str,
        max_memory_percent: float = 90.0,
        max_cpu_percent: float = 95.0
    ) -> HealthChecker:
        """创建资源使用健康检查"""
        def resource_check() -> HealthCheckResult:
            try:
                import psutil
                memory = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=0.1)

                issues = []
                if memory.percent > max_memory_percent:
                    issues.append(f"Memory usage {memory.percent}% exceeds {max_memory_percent}%")
                if cpu > max_cpu_percent:
                    issues.append(f"CPU usage {cpu}% exceeds {max_cpu_percent}%")

                if issues:
                    return HealthCheckResult(
                        name=name,
                        status=HealthStatus.DEGRADED,
                        check_type=CheckType.READINESS,
                        message="; ".join(issues),
                        metadata={"memory_percent": memory.percent, "cpu_percent": cpu}
                    )
                else:
                    return HealthCheckResult(
                        name=name,
                        status=HealthStatus.HEALTHY,
                        check_type=CheckType.READINESS,
                        message="Resource usage within limits",
                        metadata={"memory_percent": memory.percent, "cpu_percent": cpu}
                    )
            except Exception as e:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    check_type=CheckType.READINESS,
                    message=f"Resource check failed: {str(e)}",
                    error=e
                )

        checker = HealthChecker(name, CheckType.READINESS)
        checker.register_check(name, resource_check)
        return checker


# 便捷函数
def create_health_checker(name: str, check_type: CheckType = CheckType.CUSTOM) -> HealthChecker:
    """创建健康检查器"""
    return HealthChecker(name, check_type)


def create_health_endpoint(path: str = "/health") -> HealthEndpoint:
    """创建健康端点"""
    return HealthEndpoint(path)


def create_health_monitor(check_interval_seconds: float = 30.0) -> HealthMonitor:
    """创建健康监控器"""
    return HealthMonitor(check_interval_seconds)
