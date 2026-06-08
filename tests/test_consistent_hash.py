"""阶段60: Agent分布式一致性哈希 - 单元测试"""
import pytest
from src.agent.consistent_hash import (
    ConsistentHashRing, HashRouter, create_consistent_hash_ring, create_hash_router,
)

class TestConsistentHashRing:
    def test_add_node(self):
        ring = ConsistentHashRing(virtual_nodes=10)
        ring.add_node("n1", {"host": "localhost"})
        assert ring.node_count == 1
        assert ring.ring_size == 10

    def test_get_node(self):
        ring = ConsistentHashRing(virtual_nodes=10)
        ring.add_node("n1")
        ring.add_node("n2")
        node = ring.get_node("key1")
        assert node is not None
        assert node[0] in ("n1", "n2")

    def test_remove_node(self):
        ring = ConsistentHashRing(virtual_nodes=10)
        ring.add_node("n1")
        assert ring.remove_node("n1")
        assert ring.node_count == 0

    def test_get_nodes(self):
        ring = ConsistentHashRing(virtual_nodes=10)
        ring.add_node("n1")
        ring.add_node("n2")
        ring.add_node("n3")
        nodes = ring.get_nodes("key1", count=2)
        assert len(nodes) == 2
        assert nodes[0][0] != nodes[1][0]

    def test_distribution(self):
        ring = ConsistentHashRing(virtual_nodes=10)
        ring.add_node("n1")
        ring.add_node("n2")
        dist = ring.get_node_distribution()
        assert dist["n1"] == 10
        assert dist["n2"] == 10

    def test_empty_ring(self):
        ring = ConsistentHashRing()
        assert ring.get_node("key") is None

class TestHashRouter:
    def test_route(self):
        router = HashRouter(virtual_nodes=10)
        router.add_node("n1", {"host": "h1"})
        router.add_node("n2", {"host": "h2"})
        node = router.route("key")
        assert node in ("n1", "n2")

    def test_replica_nodes(self):
        router = HashRouter(virtual_nodes=10)
        router.add_node("n1")
        router.add_node("n2")
        router.add_node("n3")
        nodes = router.get_replica_nodes("key", replicas=2)
        assert len(nodes) == 2

    def test_stats(self):
        router = HashRouter(virtual_nodes=10)
        router.add_node("n1")
        stats = router.get_stats()
        assert stats["nodes"] == 1
        assert stats["virtual_nodes"] == 10

    def test_create_ring(self):
        r = create_consistent_hash_ring(50)
        assert isinstance(r, ConsistentHashRing)

    def test_create_router(self):
        r = create_hash_router(50)
        assert isinstance(r, HashRouter)
