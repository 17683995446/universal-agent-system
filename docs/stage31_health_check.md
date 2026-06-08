# 阶段31: Agent健康检查系统

> 遵循"图难于其易，为大于其细。天下难事，必作于易；天下大事，必作于细。"的原则

## 概述

阶段31实现了完整的Agent健康检查系统，提供组件健康监控、健康端点暴露、自动警报等功能，确保系统可观测性和高可用性。

## 核心组件

### 1. 健康状态模型

#### HealthStatus 枚举
```python
class HealthStatus(str, Enum):
    HEALTHY = "healthy"       # 健康
    DEGRADED = "degraded"     # 降级运行
    UNHEALTHY = "unhealthy"   # 不健康
    UNKNOWN = "unknown"       # 未知
```

#### CheckType 枚举
```python
class CheckType(str, Enum):
    LIVENESS = "liveness"     # 存活检查
    READINESS = "readiness"   # 就绪检查
    STARTUP = "startup"       # 启动检查
    CUSTOM = "custom"         # 自定义检查
```

#### 核心数据类
- **HealthCheckResult**: 单个健康检查结果
- **ComponentHealth**: 组件健康状态
- **SystemHealth**: 系统整体健康状态
- **HealthAlert**: 健康警报

### 2. HealthChecker - 健康检查器

执行单个组件的健康检查，支持同步和异步检查函数。

```python
checker = HealthChecker("api_service", CheckType.LIVENESS)

# 注册检查函数
checker.register_check("api", lambda: True)

# 执行检查
results = await checker.check()
```

**特性**:
- 支持同步和异步检查函数
- 可配置超时时间
- 自动异常处理
- 执行时间测量

### 3. HealthEndpoint - 健康端点

暴露健康检查HTTP端点，支持多种检查类型。

```python
endpoint = HealthEndpoint("/health")
endpoint.register_checker("api", api_checker)

# 获取完整健康状态
health = await endpoint.get_health()

# 获取特定检查类型状态
liveness = await endpoint.get_liveness()
readiness = await endpoint.get_readiness()
```

### 4. HealthMonitor - 健康监控器

持续监控组件健康状态，自动触发警报。

```python
monitor = HealthMonitor(check_interval_seconds=30.0)
monitor.register_checker("comp1", "service", checker)

# 添加警报处理器
monitor.add_alert_handler(lambda alert: print(f"Alert: {alert.message}"))

# 启动监控
await monitor.start()
```

**特性**:
- 定期自动检查
- 连续失败/成功阈值
- 自动警报触发和解决
- 可配置的警报处理器

### 5. HealthCheckBuilder - 健康检查构建器

简化健康检查的创建。

```python
# 简单检查
checker = HealthCheckBuilder.create_simple_check("simple", lambda: True)

# HTTP检查
checker = HealthCheckBuilder.create_http_check(
    "http_check",
    "http://localhost:8080/health"
)

# 资源检查
checker = HealthCheckBuilder.create_resource_check(
    "resource",
    max_memory_percent=90.0
)
```

## 使用示例

### 基本使用

```python
from src.agent.health_check import (
    HealthChecker, HealthEndpoint, HealthMonitor,
    HealthStatus, CheckType
)

# 创建检查器
checker = HealthChecker("my_service", CheckType.LIVENESS)
checker.register_check("health", lambda: True)

# 创建端点
endpoint = HealthEndpoint()
endpoint.register_checker("my_service", checker)

# 获取健康状态
health = await endpoint.get_health()
print(health["status"])  # "healthy"
```

### 完整监控示例

```python
# 创建监控器
monitor = HealthMonitor(check_interval_seconds=10.0)

# 创建检查器
checker = HealthChecker("api", CheckType.LIVENESS)
checker.register_check("api_check", check_api)

# 注册到监控器
monitor.register_checker("api", "service", checker)

# 添加警报处理器
async def on_alert(alert):
    print(f"ALERT: {alert.severity} - {alert.message}")

monitor.add_alert_handler(on_alert)

# 启动监控
await monitor.start()

# ... 运行一段时间后

# 停止监控
await monitor.stop()
```

## 测试覆盖

共43个测试用例，覆盖：
- 健康状态枚举 (2 tests)
- 检查类型枚举 (1 test)
- 健康检查结果 (3 tests)
- 组件健康状态 (3 tests)
- 系统健康状态 (3 tests)
- 健康警报 (3 tests)
- 健康检查器 (7 tests)
- 健康端点 (8 tests)
- 健康监控器 (8 tests)
- 健康检查构建器 (4 tests)
- 便捷函数 (3 tests)
- 集成测试 (2 tests)

## 文件清单

| 文件 | 说明 | 行数 |
|------|------|------|
| `/workspace/src/agent/health_check.py` | 健康检查系统实现 | 713 |
| `/workspace/tests/test_health_check.py` | 单元测试 | 684 |
| `/workspace/docs/stage31_health_check.md` | 本文档 | - |

## 与前一阶段的集成

健康检查系统与阶段30的追踪系统集成：
- 健康检查结果可记录到追踪系统
- 警报事件可生成追踪span
- 监控指标可与性能监控联动

## 下一步（阶段32）

下一阶段将实现 **Agent配置中心**，提供集中式配置管理、配置版本控制、灰度发布等功能。

## 总结

阶段31成功实现了完整的Agent健康检查系统，包括：

1. ✅ 定义了健康状态模型（HealthStatus, CheckType, HealthCheckResult等）
2. ✅ 实现了健康检查器（HealthChecker）
3. ✅ 实现了健康端点（HealthEndpoint）
4. ✅ 实现了健康监控器（HealthMonitor）
5. ✅ 集成了便捷函数和构建器
6. ✅ 编写了43个单元测试
7. ✅ 所有测试通过

系统现在具备了完整的健康检查能力，可以监控组件状态、自动触发警报、暴露健康端点供外部监控使用。
