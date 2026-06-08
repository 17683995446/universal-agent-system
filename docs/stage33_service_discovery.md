# 阶段33: Agent服务发现系统

> 遵循"图难于其易，为大于其细。天下难事，必作于易；天下大事，必作于细。"的原则

## 概述

阶段33实现了完整的Agent服务发现系统，提供服务注册、注销、心跳续约、多维度服务发现和五种负载均衡策略。

## 核心组件

### 1. 服务注册模型

- **ServiceStatus**: UP / DOWN / STARTING / DRAINING / OUT_OF_SERVICE
- **LoadBalancePolicy**: ROUND_ROBIN / RANDOM / WEIGHTED / CONSISTENT_HASH / LEAST_CONNECTIONS
- **ServiceEndpoint**: 服务端点（host, port, protocol, path, tls）
- **ServiceInstance**: 服务实例（含权重、区域、标签、连接数）
- **ServiceRegistration**: 注册信息
- **ServiceChangeEvent**: 变更事件

### 2. ServiceRegistry - 服务注册中心

```python
registry = ServiceRegistry(heartbeat_ttl_seconds=30.0)

# 注册
registry.register(create_registration("api", "10.0.0.1", 8080, zone="a"))

# 心跳续约
registry.heartbeat("api", instance_id)

# 注销
registry.deregister("api", instance_id)

# 状态管理
registry.set_status("api", instance_id, ServiceStatus.DRAINING)

# 过期清理（自动后台线程）
registry.start_eviction(interval_seconds=10.0)
registry.stop_eviction()
```

### 3. ServiceDiscovery - 服务发现

```python
discovery = ServiceDiscovery(registry)

# 多维度过滤
instances = discovery.discover("api", healthy_only=True, zone="a", region="us-east", tags={"v": "2"})

# 负载均衡选择单个实例
inst = discovery.discover_one("api", policy=LoadBalancePolicy.WEIGHTED, key="user-123")
```

### 4. 负载均衡策略

| 策略 | 说明 |
|------|------|
| ROUND_ROBIN | 轮询，均匀分配 |
| RANDOM | 随机选择 |
| WEIGHTED | 加权随机 |
| CONSISTENT_HASH | 一致性哈希（同key命中同实例） |
| LEAST_CONNECTIONS | 最少连接数优先 |

## 测试覆盖

共41个测试用例，覆盖模型、注册中心、服务发现、负载均衡、便捷函数和集成测试。

## 文件清单

| 文件 | 说明 | 行数 |
|------|------|------|
| `src/agent/service_discovery.py` | 服务发现实现 | ~370 |
| `tests/test_service_discovery.py` | 单元测试 | ~390 |
| `docs/stage33_service_discovery.md` | 本文档 | - |

## 与前一阶段的集成

- 服务发现与阶段32配置中心联动：配置变更可触发服务重新注册
- 服务实例可通过阶段31健康检查系统监控健康状态

## 下一步（阶段34）

下一阶段将实现 **Agent限流与熔断系统**，提供请求限流、熔断降级、服务隔离等功能。

## 总结

1. ✅ 定义了服务注册模型（ServiceStatus, LoadBalancePolicy, ServiceEndpoint等）
2. ✅ 实现了服务注册中心（注册、注销、心跳、过期清理）
3. ✅ 实现了服务发现（按区域/标签/健康状态过滤）
4. ✅ 实现了五种负载均衡策略
5. ✅ 编写了41个单元测试，全部通过
6. ✅ 集成到系统导出
