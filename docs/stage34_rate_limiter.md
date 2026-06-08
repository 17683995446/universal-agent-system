# 阶段34: Agent限流与熔断系统

> 遵循"图难于其易，为大于其细。天下难事，必作于易；天下大事，必作于细。"的原则

## 概述

阶段34实现了完整的Agent限流与熔断系统，提供四种限流算法、三态熔断器和舱壁隔离，保障系统在高负载下的稳定性。

## 核心组件

### 1. 模型定义

- **RateLimitAlgorithm**: TOKEN_BUCKET / SLIDING_WINDOW / FIXED_WINDOW / LEAKY_BUCKET
- **CircuitState**: CLOSED（正常）/ OPEN（熔断）/ HALF_OPEN（试探）
- **BulkheadState**: HEALTHY / REJECTED / DEGRADED
- **RateLimitResult**: 限流结果（是否允许、剩余配额、重试时间）
- **CircuitBreakerConfig/Stats**: 熔断器配置与统计
- **BulkheadConfig/Stats**: 隔离舱配置与统计

### 2. 四种限流算法

| 算法 | 特点 | 适用场景 |
|------|------|----------|
| 令牌桶 | 允许突发流量，匀速补充 | API限流、通用场景 |
| 滑动窗口 | 精确计数，时间窗口滑动 | 精确限流 |
| 固定窗口 | 简单高效，窗口重置 | 简单配额控制 |
| 漏桶 | 平滑输出，恒定速率 | 流量整形 |

```python
# 令牌桶：容量100，每秒补充10个
rl = create_rate_limiter(RateLimitAlgorithm.TOKEN_BUCKET, capacity=100, refill_rate=10.0)
result = rl.acquire(permits=1)
if result.allowed:
    process_request()
else:
    wait(result.retry_after_seconds)
```

### 3. 熔断器（三态状态机）

```
CLOSED ──失败率超阈值──> OPEN ──超时──> HALF_OPEN ──成功──> CLOSED
                              ↑                    │
                              └────失败────────────┘
```

```python
cb = create_circuit_breaker("api", CircuitBreakerConfig(
    failure_threshold=5,       # 5次失败触发熔断
    success_threshold=3,       # 3次成功恢复
    timeout_seconds=30.0,      # 30秒后试探
    failure_rate_threshold=0.5, # 50%失败率触发
))

if cb.allow_request():
    try:
        result = call_service()
        cb.record_success()
    except Exception:
        cb.record_failure()
```

### 4. 舱壁隔离（Bulkhead）

```python
bh = create_bulkhead("api", BulkheadConfig(
    max_concurrent_calls=10,   # 最大并发
    max_pending_calls=20,      # 最大等待
    timeout_seconds=5.0,       # 等待超时
))

if bh.try_acquire():
    try:
        process_request()
    finally:
        bh.release()
```

## 测试覆盖

共50个测试用例，覆盖模型、四种限流器、熔断器状态机、隔离舱并发、集成测试。

## 文件清单

| 文件 | 说明 |
|------|------|
| `src/agent/rate_limiter.py` | 限流与熔断实现 |
| `tests/test_rate_limiter.py` | 单元测试 |
| `docs/stage34_rate_limiter.md` | 本文档 |

## 与前一阶段的集成

- 限流器可应用于阶段33服务发现的请求分发
- 熔断器可联动阶段31健康检查系统
- 隔离舱可保护阶段29 API网关

## 下一步（阶段35）

下一阶段将实现 **Agent消息队列系统**，提供异步消息传递、发布订阅、消息持久化等功能。

## 总结

1. ✅ 定义了限流与熔断模型（4种算法枚举、3种状态枚举、配置/统计数据类）
2. ✅ 实现了四种限流器（令牌桶、滑动窗口、固定窗口、漏桶）
3. ✅ 实现了三态熔断器（CLOSED→OPEN→HALF_OPEN状态机）
4. ✅ 实现了舱壁隔离（并发限制、非阻塞/阻塞获取）
5. ✅ 编写了50个单元测试，全部通过
6. ✅ 集成到系统导出
