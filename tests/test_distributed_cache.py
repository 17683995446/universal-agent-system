"""阶段54: Agent分布式缓存 - 单元测试"""
import time
import pytest
from src.agent.distributed_cache import (
    CacheEntry, DistributedCache, ConsistentHashCache, CacheManager, create_cache_manager,
)

class TestCacheEntry:
    def test_expiry(self):
        e = CacheEntry(key="k", value="v", ttl=0.01)
        time.sleep(0.02)
        assert e.is_expired

    def test_no_expiry(self):
        e = CacheEntry(key="k", value="v", ttl=0)
        assert not e.is_expired

class TestDistributedCache:
    def test_get_set(self):
        c = DistributedCache("c1")
        c.set("a", 1)
        assert c.get("a") == 1

    def test_miss(self):
        c = DistributedCache("c1")
        assert c.get("nope") is None

    def test_ttl(self):
        c = DistributedCache("c1", default_ttl=0.01)
        c.set("a", 1)
        time.sleep(0.02)
        assert c.get("a") is None

    def test_delete(self):
        c = DistributedCache("c1")
        c.set("a", 1)
        assert c.delete("a")
        assert c.get("a") is None

    def test_lru_eviction(self):
        c = DistributedCache("c1", max_size=2)
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)
        assert c.get("a") is None
        assert c.get("b") == 2
        assert c.get("c") == 3

    def test_stats(self):
        c = DistributedCache("c1")
        c.set("a", 1)
        c.get("a")
        c.get("b")
        stats = c.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

class TestConsistentHashCache:
    def test_add_remove_node(self):
        ch = ConsistentHashCache()
        c1 = DistributedCache("n1")
        ch.add_node("n1", c1)
        assert ch.get_node("key") is not None
        ch.remove_node("n1")
        assert ch.get_node("key") is None

    def test_set_get(self):
        ch = ConsistentHashCache()
        ch.add_node("n1", DistributedCache("n1"))
        ch.add_node("n2", DistributedCache("n2"))
        ch.set("a", 1)
        assert ch.get("a") == 1

class TestCacheManager:
    def test_get_cache(self):
        m = CacheManager()
        c = m.get_cache("c1")
        assert isinstance(c, DistributedCache)

    def test_remove_cache(self):
        m = CacheManager()
        m.get_cache("c1")
        assert m.remove_cache("c1")

    def test_create_cache_manager(self):
        m = create_cache_manager()
        assert isinstance(m, CacheManager)
