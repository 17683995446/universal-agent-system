"""
阶段36测试: Agent缓存系统
"""

import time
import threading
import pytest
from datetime import datetime, timedelta

from src.agent.cache_system import (
    EvictionPolicy,
    CacheLevel,
    CacheEntry,
    CacheStats,
    CacheConfig,
    MemoryCache,
    CachePolicy,
    CacheRefreshManager,
    MultiLevelCache,
    create_cache,
    create_multi_level_cache,
    create_cache_policy,
)


# ============================================================
# 模型测试
# ============================================================

class TestEvictionPolicy:
    def test_values(self):
        assert EvictionPolicy.LRU.value == "lru"
        assert EvictionPolicy.LFU.value == "lfu"
        assert EvictionPolicy.FIFO.value == "fifo"
        assert EvictionPolicy.TTL.value == "ttl"
        assert EvictionPolicy.NONE.value == "none"


class TestCacheLevel:
    def test_values(self):
        assert CacheLevel.L1.value == "l1"
        assert CacheLevel.L2.value == "l2"
        assert CacheLevel.L3.value == "l3"


class TestCacheEntry:
    def test_creation(self):
        entry = CacheEntry(key="k1", value="v1", ttl_seconds=60)
        assert entry.key == "k1"
        assert entry.value == "v1"
        assert entry.ttl_seconds == 60
        assert entry.access_count == 0

    def test_is_expired(self):
        entry = CacheEntry(key="k1", value="v1", expires_at=datetime.now() - timedelta(seconds=1))
        assert entry.is_expired() is True

        entry2 = CacheEntry(key="k2", value="v2", expires_at=datetime.now() + timedelta(seconds=60))
        assert entry2.is_expired() is False

    def test_touch(self):
        entry = CacheEntry(key="k1", value="v1")
        entry.touch()
        assert entry.access_count == 1
        entry.touch()
        assert entry.access_count == 2

    def test_to_dict(self):
        entry = CacheEntry(key="k1", value="v1", access_count=5)
        d = entry.to_dict()
        assert d["key"] == "k1"
        assert d["access_count"] == 5


class TestCacheStats:
    def test_hit_rate(self):
        stats = CacheStats(name="test", level=CacheLevel.L1, hits=80, misses=20)
        assert abs(stats.hit_rate - 0.8) < 0.001

    def test_hit_rate_zero(self):
        stats = CacheStats(name="test", level=CacheLevel.L1)
        assert stats.hit_rate == 0.0

    def test_utilization(self):
        stats = CacheStats(name="test", level=CacheLevel.L1, size=50, max_size=100)
        assert abs(stats.utilization - 0.5) < 0.001

    def test_to_dict(self):
        stats = CacheStats(name="test", level=CacheLevel.L1, hits=10, misses=5)
        d = stats.to_dict()
        assert d["name"] == "test"
        assert d["hits"] == 10


class TestCacheConfig:
    def test_creation(self):
        config = CacheConfig(name="my_cache", max_size=500, default_ttl_seconds=120)
        assert config.name == "my_cache"
        assert config.max_size == 500
        assert config.default_ttl_seconds == 120


# ============================================================
# MemoryCache 测试
# ============================================================

class TestMemoryCache:
    def setup_method(self):
        self.cache = create_cache("test", max_size=10)

    def test_put_and_get(self):
        self.cache.put("k1", "v1")
        assert self.cache.get("k1") == "v1"
        assert self.cache.stats.hits == 1

    def test_get_nonexistent(self):
        assert self.cache.get("nonexistent") is None
        assert self.cache.stats.misses == 1

    def test_delete(self):
        self.cache.put("k1", "v1")
        assert self.cache.delete("k1") is True
        assert self.cache.get("k1") is None
        assert self.cache.delete("k1") is False

    def test_exists(self):
        self.cache.put("k1", "v1")
        assert self.cache.exists("k1") is True
        assert self.cache.exists("nonexistent") is False

    def test_clear(self):
        self.cache.put("k1", "v1")
        self.cache.put("k2", "v2")
        count = self.cache.clear()
        assert count == 2
        assert self.cache.stats.size == 0

    def test_keys_values(self):
        self.cache.put("k1", "v1")
        self.cache.put("k2", "v2")
        assert set(self.cache.keys()) == {"k1", "k2"}
        assert set(self.cache.values()) == {"v1", "v2"}

    def test_ttl_expiration(self):
        cache = create_cache("ttl_test", max_size=10, ttl_seconds=0.5)
        cache.put("k1", "v1")
        assert cache.get("k1") == "v1"
        time.sleep(0.6)
        assert cache.get("k1") is None

    def test_ttl_on_put(self):
        self.cache.put("k1", "v1", ttl_seconds=0.5)
        assert self.cache.get("k1") == "v1"
        time.sleep(0.6)
        assert self.cache.get("k1") is None

    def test_lru_eviction(self):
        cache = create_cache("lru_test", max_size=3, eviction_policy=EvictionPolicy.LRU)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.get("a")  # 访问a
        cache.put("d", 4)  # 应淘汰b（最久未访问）
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3
        assert cache.get("d") == 4

    def test_lfu_eviction(self):
        cache = create_cache("lfu_test", max_size=3, eviction_policy=EvictionPolicy.LFU)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.get("a")  # a访问1次
        cache.get("a")  # a访问2次
        cache.get("b")  # b访问1次
        cache.put("d", 4)  # 应淘汰c（访问0次）
        assert cache.get("c") is None
        assert cache.get("a") == 1

    def test_fifo_eviction(self):
        cache = create_cache("fifo_test", max_size=3, eviction_policy=EvictionPolicy.FIFO)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.put("d", 4)  # 应淘汰a（最早插入）
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_loader(self):
        cache = create_cache("loader_test", max_size=10)
        cache.set_loader(lambda k: f"loaded-{k}")
        value = cache.get("new_key")
        assert value == "loaded-new_key"
        # 加载后数据被缓存，第二次访问是命中
        assert cache.get("new_key") == "loaded-new_key"
        assert cache.stats.hits == 1

    def test_on_evict_callback(self):
        evicted = []
        cache = create_cache("callback_test", max_size=2)
        cache.set_on_evict(lambda k, v: evicted.append((k, v)))
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # 淘汰a
        assert ("a", 1) in evicted

    def test_refresh(self):
        cache = create_cache("refresh_test", max_size=10)
        cache.put("k1", "v1")
        cache.refresh("k1", lambda k: "v2")
        assert cache.get("k1") == "v2"
        assert cache.stats.refreshes == 1

    def test_invalidate(self):
        self.cache.put("prefix_a", 1)
        self.cache.put("prefix_b", 2)
        self.cache.put("other", 3)
        count = self.cache.invalidate("prefix_")
        assert count == 2
        assert self.cache.get("prefix_a") is None
        assert self.cache.get("other") == 3

    def test_stats(self):
        self.cache.put("k1", "v1")
        self.cache.get("k1")
        self.cache.get("nonexistent")
        stats = self.cache.stats
        assert stats.puts == 1
        assert stats.hits == 1
        assert stats.misses == 1


# ============================================================
# CachePolicy 测试
# ============================================================

class TestCachePolicy:
    def setup_method(self):
        self.policy = create_cache_policy()

    def test_add_rule(self):
        self.policy.add_rule("user_", ttl_seconds=60, refresh_interval=30)
        rule = self.policy.get_rule("user_123")
        assert rule is not None
        assert rule["ttl_seconds"] == 60

    def test_remove_rule(self):
        self.policy.add_rule("user_", ttl_seconds=60)
        assert self.policy.remove_rule("user_") is True
        assert self.policy.get_rule("user_123") is None

    def test_get_ttl(self):
        self.policy.add_rule("session_", ttl_seconds=120)
        ttl = self.policy.get_ttl("session_abc", default=60)
        assert ttl == 120

    def test_get_ttl_default(self):
        ttl = self.policy.get_ttl("unknown", default=60)
        assert ttl == 60


# ============================================================
# CacheRefreshManager 测试
# ============================================================

class TestCacheRefreshManager:
    def setup_method(self):
        self.cache = create_cache("refresh_test", max_size=10)
        self.manager = CacheRefreshManager(self.cache)

    def test_schedule_and_cancel(self):
        self.manager.schedule_refresh("k1", 5.0)
        assert "k1" in self.manager._refresh_tasks
        assert self.manager.cancel_refresh("k1") is True
        assert "k1" not in self.manager._refresh_tasks

    def test_start_stop(self):
        self.manager.start()
        assert self.manager._running is True
        self.manager.stop()
        assert self.manager._running is False


# ============================================================
# MultiLevelCache 测试
# ============================================================

class TestMultiLevelCache:
    def setup_method(self):
        self.mlc = create_multi_level_cache(l1_size=5, l2_size=10)

    def test_put_and_get(self):
        self.mlc.put("k1", "v1")
        assert self.mlc.get("k1") == "v1"

    def test_l1_l2_hierarchy(self):
        # 写入会同时写入L1和L2
        self.mlc.put("k1", "v1")
        l1 = self.mlc.get_cache(0)
        l2 = self.mlc.get_cache(1)
        assert l1.get("k1") == "v1"
        assert l2.get("k1") == "v1"

    def test_backfill(self):
        # 只写入L2，从L1获取时会回填
        l2 = self.mlc.get_cache(1)
        l2.put("k2", "v2")
        # L1没有
        assert self.mlc.get_cache(0).get("k2") is None
        # 从多级缓存获取，应该从L2获取并回填L1
        value = self.mlc.get("k2")
        assert value == "v2"
        # 现在L1应该有了
        assert self.mlc.get_cache(0).get("k2") == "v2"

    def test_delete(self):
        self.mlc.put("k1", "v1")
        assert self.mlc.delete("k1") is True
        assert self.mlc.get("k1") is None

    def test_invalidate(self):
        self.mlc.put("a1", 1)
        self.mlc.put("a2", 2)
        self.mlc.put("b1", 3)
        count = self.mlc.invalidate("a")
        assert count >= 2

    def test_clear(self):
        self.mlc.put("k1", "v1")
        self.mlc.put("k2", "v2")
        count = self.mlc.clear()
        assert count >= 2

    def test_stats(self):
        self.mlc.put("k1", "v1")
        self.mlc.get("k1")
        stats = self.mlc.stats
        assert stats.puts >= 1
        assert stats.hits >= 1

    def test_get_cache(self):
        assert self.mlc.get_cache(0) is not None
        assert self.mlc.get_cache(1) is not None
        assert self.mlc.get_cache(99) is None


# ============================================================
# 便捷函数测试
# ============================================================

class TestConvenienceFunctions:
    def test_create_cache(self):
        cache = create_cache("test", max_size=100, ttl_seconds=60)
        assert isinstance(cache, MemoryCache)
        assert cache.name == "test"

    def test_create_multi_level_cache(self):
        mlc = create_multi_level_cache()
        assert isinstance(mlc, MultiLevelCache)

    def test_create_cache_policy(self):
        policy = create_cache_policy()
        assert isinstance(policy, CachePolicy)


# ============================================================
# 集成测试
# ============================================================

class TestIntegration:
    def test_cache_with_policy(self):
        cache = create_cache("policy_test", max_size=10)
        policy = create_cache_policy()
        policy.add_rule("user_", ttl_seconds=0.5)

        cache.put("user_1", "data")
        ttl = policy.get_ttl("user_1", default=60)
        assert ttl == 0.5

        # 使用policy的TTL
        cache.put("user_2", "data2", ttl_seconds=ttl)
        time.sleep(0.6)
        assert cache.get("user_2") is None

    def test_multi_level_with_eviction(self):
        mlc = create_multi_level_cache(l1_size=2, l2_size=5)
        # L1容量小，会触发淘汰
        for i in range(5):
            mlc.put(f"k{i}", f"v{i}")

        # L1应该只保留最新的几个
        l1 = mlc.get_cache(0)
        assert l1.stats.size <= 2

        # L2应该有全部
        l2 = mlc.get_cache(1)
        assert l2.stats.size == 5

    def test_concurrent_access(self):
        cache = create_cache("concurrent_test", max_size=100)
        errors = []

        def writer():
            for i in range(50):
                try:
                    cache.put(f"k{i}", f"v{i}")
                except Exception as e:
                    errors.append(e)

        def reader():
            for i in range(50):
                try:
                    cache.get(f"k{i}")
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])