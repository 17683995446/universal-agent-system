"""
阶段31测试: Agent健康检查系统

遵循"图难于其易，为大于其细"的原则，全面测试健康检查系统。
"""

import asyncio
import pytest
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.agent.health_check import (
    HealthStatus,
    CheckType,
    HealthCheckResult,
    ComponentHealth,
    SystemHealth,
    HealthAlert,
    HealthChecker,
    HealthEndpoint,
    HealthMonitor,
    HealthCheckBuilder,
    create_health_checker,
    create_health_endpoint,
    create_health_monitor,
)


class TestHealthStatus:
    """测试健康状态枚举"""

    def test_health_status_values(self):
        """测试健康状态值"""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"

    def test_health_status_comparison(self):
        """测试健康状态比较"""
        assert HealthStatus.HEALTHY == HealthStatus("healthy")
        assert HealthStatus.UNHEALTHY == HealthStatus("unhealthy")


class TestCheckType:
    """测试检查类型枚举"""

    def test_check_type_values(self):
        """测试检查类型值"""
        assert CheckType.LIVENESS.value == "liveness"
        assert CheckType.READINESS.value == "readiness"
        assert CheckType.STARTUP.value == "startup"
        assert CheckType.CUSTOM.value == "custom"


class TestHealthCheckResult:
    """测试健康检查结果"""

    def test_result_creation(self):
        """测试结果创建"""
        result = HealthCheckResult(
            name="test_check",
            status=HealthStatus.HEALTHY,
            check_type=CheckType.LIVENESS,
            message="All good"
        )
        assert result.name == "test_check"
        assert result.status == HealthStatus.HEALTHY
        assert result.check_type == CheckType.LIVENESS
        assert result.message == "All good"
        assert result.duration_ms == 0.0
        assert result.error is None

    def test_result_to_dict(self):
        """测试结果转字典"""
        result = HealthCheckResult(
            name="test_check",
            status=HealthStatus.HEALTHY,
            check_type=CheckType.LIVENESS,
            message="OK",
            duration_ms=100.0,
            metadata={"key": "value"}
        )
        data = result.to_dict()
        assert data["name"] == "test_check"
        assert data["status"] == "healthy"
        assert data["check_type"] == "liveness"
        assert data["message"] == "OK"
        assert data["duration_ms"] == 100.0
        assert data["metadata"] == {"key": "value"}
        assert data["error"] is None

    def test_result_with_error(self):
        """测试带错误的结果"""
        error = ValueError("Test error")
        result = HealthCheckResult(
            name="failing_check",
            status=HealthStatus.UNHEALTHY,
            check_type=CheckType.READINESS,
            message="Failed",
            error=error
        )
        data = result.to_dict()
        assert data["error"] == "Test error"


class TestComponentHealth:
    """测试组件健康状态"""

    def test_component_health_creation(self):
        """测试组件健康创建"""
        health = ComponentHealth(
            component_id="comp1",
            component_type="service",
            status=HealthStatus.HEALTHY
        )
        assert health.component_id == "comp1"
        assert health.component_type == "service"
        assert health.status == HealthStatus.HEALTHY
        assert health.checks == []
        assert health.consecutive_failures == 0
        assert health.consecutive_successes == 0

    def test_component_health_with_checks(self):
        """测试带检查结果的组件健康"""
        check = HealthCheckResult(
            name="check1",
            status=HealthStatus.HEALTHY,
            check_type=CheckType.LIVENESS
        )
        health = ComponentHealth(
            component_id="comp1",
            component_type="service",
            status=HealthStatus.HEALTHY,
            checks=[check],
            consecutive_successes=5
        )
        assert len(health.checks) == 1
        assert health.consecutive_successes == 5

    def test_component_health_to_dict(self):
        """测试组件健康转字典"""
        health = ComponentHealth(
            component_id="comp1",
            component_type="service",
            status=HealthStatus.HEALTHY
        )
        data = health.to_dict()
        assert data["component_id"] == "comp1"
        assert data["component_type"] == "service"
        assert data["status"] == "healthy"
        assert data["consecutive_failures"] == 0
        assert data["consecutive_successes"] == 0


class TestSystemHealth:
    """测试系统健康状态"""

    def test_system_health_creation(self):
        """测试系统健康创建"""
        health = SystemHealth(
            system_id="sys1",
            status=HealthStatus.HEALTHY
        )
        assert health.system_id == "sys1"
        assert health.status == HealthStatus.HEALTHY
        assert health.version == "1.0.0"
        assert health.uptime_seconds == 0.0

    def test_system_health_with_components(self):
        """测试带组件的系统健康"""
        comp = ComponentHealth(
            component_id="comp1",
            component_type="service",
            status=HealthStatus.HEALTHY
        )
        health = SystemHealth(
            system_id="sys1",
            status=HealthStatus.HEALTHY,
            components={"comp1": comp}
        )
        assert "comp1" in health.components

    def test_system_health_to_dict(self):
        """测试系统健康转字典"""
        health = SystemHealth(
            system_id="sys1",
            status=HealthStatus.HEALTHY,
            version="2.0.0",
            uptime_seconds=3600.0
        )
        data = health.to_dict()
        assert data["system_id"] == "sys1"
        assert data["status"] == "healthy"
        assert data["version"] == "2.0.0"
        assert data["uptime_seconds"] == 3600.0


class TestHealthAlert:
    """测试健康警报"""

    def test_alert_creation(self):
        """测试警报创建"""
        alert = HealthAlert(
            alert_id="alert1",
            component_id="comp1",
            severity="critical",
            message="Component is down"
        )
        assert alert.alert_id == "alert1"
        assert alert.component_id == "comp1"
        assert alert.severity == "critical"
        assert alert.message == "Component is down"
        assert alert.resolved is False
        assert alert.resolved_at is None

    def test_alert_resolve(self):
        """测试警报解决"""
        alert = HealthAlert(
            alert_id="alert1",
            component_id="comp1",
            severity="critical",
            message="Component is down"
        )
        alert.resolve()
        assert alert.resolved is True
        assert alert.resolved_at is not None

    def test_alert_to_dict(self):
        """测试警报转字典"""
        alert = HealthAlert(
            alert_id="alert1",
            component_id="comp1",
            severity="critical",
            message="Component is down"
        )
        data = alert.to_dict()
        assert data["alert_id"] == "alert1"
        assert data["component_id"] == "comp1"
        assert data["severity"] == "critical"
        assert data["message"] == "Component is down"
        assert data["resolved"] is False
        assert data["resolved_at"] is None


class TestHealthChecker:
    """测试健康检查器"""

    @pytest.mark.asyncio
    async def test_checker_creation(self):
        """测试检查器创建"""
        checker = HealthChecker("test_checker", CheckType.LIVENESS)
        assert checker.name == "test_checker"
        assert checker.check_type == CheckType.LIVENESS

    @pytest.mark.asyncio
    async def test_register_check(self):
        """测试注册检查"""
        checker = HealthChecker("test_checker")

        def simple_check():
            return True

        checker.register_check("simple", simple_check)
        results = await checker.check()
        assert len(results) == 1
        assert results[0].name == "simple"
        assert results[0].status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_register_async_check(self):
        """测试注册异步检查"""
        checker = HealthChecker("test_checker")

        async def async_check():
            return HealthCheckResult(
                name="async_check",
                status=HealthStatus.HEALTHY,
                check_type=CheckType.CUSTOM
            )

        checker.register_check("async", async_check)
        results = await checker.check()
        assert len(results) == 1
        assert results[0].name == "async_check"

    @pytest.mark.asyncio
    async def test_unregister_check(self):
        """测试注销检查"""
        checker = HealthChecker("test_checker")

        def simple_check():
            return True

        checker.register_check("simple", simple_check)
        assert checker.unregister_check("simple") is True
        assert checker.unregister_check("simple") is False

        results = await checker.check()
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_check_with_error(self):
        """测试检查异常处理"""
        checker = HealthChecker("test_checker")

        def failing_check():
            raise ValueError("Test error")

        checker.register_check("failing", failing_check)
        results = await checker.check()
        assert len(results) == 1
        assert results[0].status == HealthStatus.UNHEALTHY
        assert "Test error" in results[0].message
        assert results[0].error is not None

    @pytest.mark.asyncio
    async def test_check_timeout(self):
        """测试检查超时"""
        checker = HealthChecker("test_checker")
        checker.set_timeout(0.1)

        async def slow_check():
            await asyncio.sleep(1.0)
            return True

        checker.register_check("slow", slow_check)
        results = await checker.check()
        assert len(results) == 1
        assert results[0].status == HealthStatus.UNHEALTHY
        assert "timed out" in results[0].message

    def test_check_sync(self):
        """测试同步检查"""
        checker = HealthChecker("test_checker")

        def simple_check():
            return True

        checker.register_check("simple", simple_check)
        results = checker.check_sync()
        assert len(results) == 1
        assert results[0].status == HealthStatus.HEALTHY


class TestHealthEndpoint:
    """测试健康端点"""

    @pytest.mark.asyncio
    async def test_endpoint_creation(self):
        """测试端点创建"""
        endpoint = HealthEndpoint("/health")
        assert endpoint.path == "/health"

    @pytest.mark.asyncio
    async def test_register_checker(self):
        """测试注册检查器"""
        endpoint = HealthEndpoint()
        checker = HealthChecker("test", CheckType.LIVENESS)

        def simple_check():
            return True

        checker.register_check("check1", simple_check)
        endpoint.register_checker("test", checker)

        health = await endpoint.get_health()
        assert health["status"] == "healthy"
        assert "test" in health["components"]

    @pytest.mark.asyncio
    async def test_unregister_checker(self):
        """测试注销检查器"""
        endpoint = HealthEndpoint()
        checker = HealthChecker("test")
        endpoint.register_checker("test", checker)

        assert endpoint.unregister_checker("test") is True
        assert endpoint.unregister_checker("test") is False

    @pytest.mark.asyncio
    async def test_get_liveness(self):
        """测试获取存活状态"""
        endpoint = HealthEndpoint()
        checker = HealthChecker("test", CheckType.LIVENESS)

        def simple_check():
            return True

        checker.register_check("check1", simple_check)
        endpoint.register_checker("test", checker)

        liveness = await endpoint.get_liveness()
        assert liveness["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_get_readiness(self):
        """测试获取就绪状态"""
        endpoint = HealthEndpoint()
        checker = HealthChecker("test", CheckType.READINESS)

        def simple_check():
            return True

        checker.register_check("check1", simple_check)
        endpoint.register_checker("test", checker)

        readiness = await endpoint.get_readiness()
        assert readiness["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_disable_check_type(self):
        """测试禁用检查类型"""
        endpoint = HealthEndpoint()
        checker = HealthChecker("test", CheckType.CUSTOM)

        def simple_check():
            return True

        checker.register_check("check1", simple_check)
        endpoint.register_checker("test", checker)
        endpoint.disable_check_type(CheckType.CUSTOM)

        health = await endpoint.get_health()
        assert health["status"] == "healthy"
        assert health["components"] == {}

    @pytest.mark.asyncio
    async def test_unhealthy_component(self):
        """测试不健康组件"""
        endpoint = HealthEndpoint()
        checker = HealthChecker("test", CheckType.LIVENESS)

        def failing_check():
            return False

        checker.register_check("check1", failing_check)
        endpoint.register_checker("test", checker)

        health = await endpoint.get_health()
        assert health["status"] == "unhealthy"
        assert health["components"]["test"] == "unhealthy"


class TestHealthMonitor:
    """测试健康监控器"""

    def test_monitor_creation(self):
        """测试监控器创建"""
        monitor = HealthMonitor(check_interval_seconds=10.0)
        assert monitor._check_interval == 10.0

    def test_register_checker(self):
        """测试注册检查器"""
        monitor = HealthMonitor()
        checker = HealthChecker("test")
        monitor.register_checker("comp1", "service", checker)

        health = monitor.get_component_health("comp1")
        assert health is not None
        assert health.component_id == "comp1"
        assert health.component_type == "service"

    def test_unregister_checker(self):
        """测试注销检查器"""
        monitor = HealthMonitor()
        checker = HealthChecker("test")
        monitor.register_checker("comp1", "service", checker)

        assert monitor.unregister_checker("comp1") is True
        assert monitor.unregister_checker("comp1") is False
        assert monitor.get_component_health("comp1") is None

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """测试启动和停止"""
        monitor = HealthMonitor(check_interval_seconds=0.1)
        await monitor.start()
        assert monitor._running is True

        await monitor.stop()
        assert monitor._running is False
        assert monitor._monitor_task is None

    @pytest.mark.asyncio
    async def test_alert_handler(self):
        """测试警报处理器"""
        monitor = HealthMonitor()
        handler = MagicMock()
        monitor.add_alert_handler(handler)

        checker = HealthChecker("test")

        def failing_check():
            return False

        checker.register_check("check1", failing_check)
        monitor.register_checker("comp1", "service", checker)
        monitor.set_thresholds(failure=1, success=1)

        await monitor._check_all_components()

        # 检查警报是否被触发
        alerts = monitor.get_alerts(resolved=False)
        assert len(alerts) == 1
        assert alerts[0].component_id == "comp1"

    def test_get_alerts_filtered(self):
        """测试过滤获取警报"""
        monitor = HealthMonitor()

        # 手动添加警报
        alert1 = HealthAlert("a1", "comp1", "critical", "Error")
        alert2 = HealthAlert("a2", "comp2", "warning", "Warning")
        alert2.resolve()

        monitor._alerts = [alert1, alert2]

        unresolved = monitor.get_alerts(resolved=False)
        assert len(unresolved) == 1
        assert unresolved[0].alert_id == "a1"

        resolved = monitor.get_alerts(resolved=True)
        assert len(resolved) == 1
        assert resolved[0].alert_id == "a2"

        comp1_alerts = monitor.get_alerts(component_id="comp1")
        assert len(comp1_alerts) == 1
        assert comp1_alerts[0].component_id == "comp1"

    def test_clear_resolved_alerts(self):
        """测试清除已解决警报"""
        monitor = HealthMonitor()

        alert1 = HealthAlert("a1", "comp1", "critical", "Error")
        alert2 = HealthAlert("a2", "comp2", "warning", "Warning")
        alert2.resolve()

        monitor._alerts = [alert1, alert2]

        cleared = monitor.clear_resolved_alerts()
        assert cleared == 1
        assert len(monitor._alerts) == 1


class TestHealthCheckBuilder:
    """测试健康检查构建器"""

    def test_create_simple_check(self):
        """测试创建简单检查"""
        checker = HealthCheckBuilder.create_simple_check(
            "simple",
            lambda: True,
            CheckType.LIVENESS
        )
        assert checker.name == "simple"
        assert checker.check_type == CheckType.LIVENESS

        results = checker.check_sync()
        assert len(results) == 1
        assert results[0].status == HealthStatus.HEALTHY

    def test_create_simple_check_failing(self):
        """测试创建失败的简单检查"""
        checker = HealthCheckBuilder.create_simple_check(
            "failing",
            lambda: False
        )
        results = checker.check_sync()
        assert results[0].status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_create_http_check(self):
        """测试创建HTTP检查"""
        # 注意：这个测试需要aiohttp，如果没有安装会跳过
        pytest.importorskip("aiohttp")

        checker = HealthCheckBuilder.create_http_check(
            "http_check",
            "http://localhost:9999/health",
            timeout_seconds=0.1
        )
        assert checker.name == "http_check"
        assert checker.check_type == CheckType.LIVENESS

        # 由于端口未监听，应该失败
        results = await checker.check()
        assert results[0].status == HealthStatus.UNHEALTHY

    def test_create_resource_check(self):
        """测试创建资源检查"""
        pytest.importorskip("psutil")

        checker = HealthCheckBuilder.create_resource_check(
            "resource",
            max_memory_percent=100.0,
            max_cpu_percent=100.0
        )
        assert checker.name == "resource"
        assert checker.check_type == CheckType.READINESS

        results = checker.check_sync()
        assert len(results) == 1
        assert results[0].status == HealthStatus.HEALTHY
        assert "memory_percent" in results[0].metadata


class TestConvenienceFunctions:
    """测试便捷函数"""

    def test_create_health_checker(self):
        """测试创建健康检查器"""
        checker = create_health_checker("test", CheckType.LIVENESS)
        assert isinstance(checker, HealthChecker)
        assert checker.name == "test"
        assert checker.check_type == CheckType.LIVENESS

    def test_create_health_endpoint(self):
        """测试创建健康端点"""
        endpoint = create_health_endpoint("/custom/health")
        assert isinstance(endpoint, HealthEndpoint)
        assert endpoint.path == "/custom/health"

    def test_create_health_monitor(self):
        """测试创建健康监控器"""
        monitor = create_health_monitor(60.0)
        assert isinstance(monitor, HealthMonitor)
        assert monitor._check_interval == 60.0


class TestIntegration:
    """集成测试"""

    @pytest.mark.asyncio
    async def test_full_health_check_flow(self):
        """测试完整健康检查流程"""
        # 创建监控器
        monitor = HealthMonitor(check_interval_seconds=0.1)

        # 创建检查器
        checker = HealthChecker("api_service", CheckType.LIVENESS)

        def api_check():
            return True

        checker.register_check("api", api_check)
        monitor.register_checker("api_service", "service", checker)

        # 启动监控
        await monitor.start()
        await asyncio.sleep(0.15)

        # 检查健康状态
        health = monitor.get_component_health("api_service")
        assert health is not None

        # 停止监控
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_endpoint_with_monitor(self):
        """测试端点与监控器集成"""
        endpoint = HealthEndpoint()
        monitor = HealthMonitor()

        # 创建检查器并注册到两者
        checker = HealthChecker("service1", CheckType.LIVENESS)

        def health_check():
            return True

        checker.register_check("health", health_check)
        endpoint.register_checker("service1", checker)
        monitor.register_checker("service1", "microservice", checker)

        # 通过端点获取健康状态
        health = await endpoint.get_health()
        assert health["status"] == "healthy"
        assert "service1" in health["components"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
