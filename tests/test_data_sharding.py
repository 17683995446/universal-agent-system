"""阶段61: Agent数据分片系统 - 单元测试"""
import pytest
from src.agent.data_sharding import (
    ShardStrategy, Shard, ShardRouter, ShardingManager, create_sharding_manager,
)

class TestShard:
    def test_get_set(self):
        s = Shard(shard_id="s1")
        s.set("a", 1)
        assert s.get("a") == 1

    def test_delete(self):
        s = Shard(shard_id="s1")
        s.set("a", 1)
        assert s.delete("a")
        assert s.get("a") is None

class TestShardRouter:
    def test_hash_route(self):
        r = ShardRouter(strategy=ShardStrategy.HASH)
        r.add_shard(Shard(shard_id="s0"))
        r.add_shard(Shard(shard_id="s1"))
        shard = r.route("key1")
        assert shard is not None
        shard.set("key1", "value1")
        assert r.route("key1").get("key1") == "value1"

    def test_range_route(self):
        r = ShardRouter(strategy=ShardStrategy.RANGE)
        r.add_shard(Shard(shard_id="s0", key_range=("a", "m")))
        r.add_shard(Shard(shard_id="s1", key_range=("n", "z")))
        assert r.route("apple").shard_id == "s0"
        assert r.route("orange").shard_id == "s1"

    def test_list_route(self):
        r = ShardRouter(strategy=ShardStrategy.LIST)
        r.add_shard(Shard(shard_id="s0", key_list=["a", "b"]))
        r.add_shard(Shard(shard_id="s1", key_list=["c", "d"]))
        assert r.route("a").shard_id == "s0"
        assert r.route("c").shard_id == "s1"

    def test_rebalance(self):
        r = ShardRouter(strategy=ShardStrategy.HASH)
        r.add_shard(Shard(shard_id="s0"))
        r.add_shard(Shard(shard_id="s1"))
        r.route("k1").set("k1", "v1")
        r.route("k2").set("k2", "v2")
        migration = r.rebalance(4)
        assert r.get_stats()["shard_count"] == 4
        assert r.get_stats()["total_keys"] == 2

    def test_stats(self):
        r = ShardRouter(strategy=ShardStrategy.HASH)
        r.add_shard(Shard(shard_id="s0"))
        r.route("k1").set("k1", "v1")
        stats = r.get_stats()
        assert stats["total_keys"] == 1

class TestShardingManager:
    def test_create_router(self):
        m = ShardingManager()
        r = m.create_router("users", shard_count=2)
        assert r.get_stats()["shard_count"] == 2

    def test_get_router(self):
        m = ShardingManager()
        m.create_router("users")
        assert m.get_router("users") is not None

    def test_create_sharding_manager(self):
        m = create_sharding_manager()
        assert isinstance(m, ShardingManager)
