# 阶段36: Agent缓存系统

> 遵循"图难于其易，为大于其细。天下难事，必作于易；天下大事，必作于细。"的原则

## 概述

阶段36实现了完整的Agent缓存系统，提供多级缓存、多种淘汰策略、TTL过期、缓存策略管理和后台刷新功能。

## 核心组件

### 1. 缓存模型

- **EvictionPolicy**: LRU / LFU / FIFO / TTL / NONE
- **CacheLevel**: L1 / L2 / L3
- **CacheEntry**: 缓存项（key, value, TTL, access_count等）
- **CacheStats**: 缓存统计（hits, misses, hit_rate, utilization等）
- **CacheConfig**: 缓存配置（max_size, ttl, eviction_policy等）

### 2. MemoryCache - 内存缓存

```python
cache = create_cache(
    name="user_cache",
    max_size=1000,
    ttl_seconds=60,
    eviction_policy=EvictionPolicy.LRU,
)

# 基本操作
cache.put("user:1", {"name": "Alice"})
value = cache.get("user:1")
cache.delete("user:1")
cache.exists("user:1")

# TTL
cache.put("session:abc", "data", ttl_seconds=30)

# 批量失效
cache.invalidate("user:")  # 失效所有user:开头的键

# Loader（穿透防护）
cache.set_loader(lambda key: load_from_db(key))

# 淘汰回调
cache.set_on_evict(lambda key, value: log_eviction(key))
```

### 3. CachePolicy - 缓存策略

```python
policy = create_cache_policy()
policy.add_rule("user_", ttl_seconds=60, refresh_interval=30)
policy.add_rule("session_", ttl_seconds=120)

ttl = policy.get_ttl("user:123", default=30)
```

### 4. CacheRefreshManager - 后台刷新

```python
manager = CacheRefreshManager(cache)
manager.schedule_refresh("hot_key", interval_seconds=10)
manager.start()  # 启动后台刷新线程
manager.stop()
```

### 5. MultiLevelCache - 多级缓存

```python
mlc = create_multi_level_cache(l1_size=100, l2_size=1000)

mlc.put("key", "value")  # 同时写入L1和L2
value = mlc.get("key")   # 先查L1，未命中则查L2并回填

mlc.invalidate("prefix")  # 所有层失效
mlc.clear()               # 清空所有层
```

## 淘汰策略

| 策略 | 说明 |
|------|------|
| LRU | 最近最少使用，淘汰最久未访问 |
| LFU | 最不经常使用，淘汰访问次数最少 |
| FIFO | 先进先出，淘汰最早插入 |
| TTL | 时间过期优先淘汰 |
| NONE | 不淘汰，容量满时拒绝写入 |

## 测试覆盖

共47个测试用例，覆盖模型、缓存操作、淘汰策略、多级缓存、集成测试。

## 文件清单

| 文件 | 说明 |
|------|------|
| `src/agent/cache_system.py` | 缓存系统实现 |
| `tests/test_cache_system.py` | 单元测试 |
| `docs/stage36_cache_system.md` | 本文档 |

## 与前一阶段的集成

- 缓存可与阶段35消息队列结合，缓存热点消息
- 可与阶段34限流器结合，缓存限流计数
- 可与阶段33服务发现结合，缓存服务实例列表

## 下一步（阶段37）

下一阶段将实现 **Agent任务调度系统**，提供定时任务、延迟任务、任务队列等功能。

## 总结

1. ✅ 定义了缓存模型（淘汰策略、缓存层级、缓存项、统计、配置）
2. ✅ 实现了内存缓存（LRU/LFU/FIFO/TTL淘汰、TTL过期、Loader穿透防护）
3. ✅ 实现了缓存策略（规则管理、TTL获取）
4. ✅ 实现了后台刷新管理器（定时刷新、启动/停止）
5. ✅ 实现了多级缓存（L1/L2分层、回填、统一操作）
6. ✅ 编写了47个单元测试，全部通过
7. ✅ 集成到系统导出