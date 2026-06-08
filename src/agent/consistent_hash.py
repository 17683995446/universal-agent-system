"""
阶段60: Agent分布式一致性哈希

遵循"图难于其易，为大于其细"的原则，实现一致性哈希环、虚拟节点、节点增删、负载均衡。
"""
import hashlib
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

class ConsistentHashRing:
    """一致性哈希环"""
    def __init__(self, virtual_nodes: int = 150):
        self._virtual_nodes = virtual_nodes
        self._ring: Dict[int, str] = {}  # hash -> node_id
        self._nodes: Dict[str, Any] = {}  # node_id -> node_data
        self._sorted_hashes: List[int] = []
        self._lock = threading.RLock()

    def add_node(self, node_id: str, node_data: Any = None) -> None:
        with self._lock:
            if node_id in self._nodes:
                return
            self._nodes[node_id] = node_data
            for i in range(self._virtual_nodes):
                key = f"{node_id}:{i}"
                h = self._hash(key)
                self._ring[h] = node_id
            self._sorted_hashes = sorted(self._ring.keys())

    def remove_node(self, node_id: str) -> bool:
        with self._lock:
            if node_id not in self._nodes:
                return False
            del self._nodes[node_id]
            to_remove = [h for h, n in self._ring.items() if n == node_id]
            for h in to_remove:
                del self._ring[h]
            self._sorted_hashes = sorted(self._ring.keys())
            return True

    def get_node(self, key: str) -> Optional[Tuple[str, Any]]:
        with self._lock:
            if not self._ring:
                return None
            h = self._hash(key)
            for sh in self._sorted_hashes:
                if sh >= h:
                    node_id = self._ring[sh]
                    return (node_id, self._nodes.get(node_id))
            # wrap around
            node_id = self._ring[self._sorted_hashes[0]]
            return (node_id, self._nodes.get(node_id))

    def get_nodes(self, key: str, count: int = 1) -> List[Tuple[str, Any]]:
        """获取key对应的多个节点（用于副本）"""
        with self._lock:
            if not self._ring:
                return []
            h = self._hash(key)
            result = []
            seen = set()
            # 从hash位置开始遍历
            idx = 0
            for i, sh in enumerate(self._sorted_hashes):
                if sh >= h:
                    idx = i
                    break
            for i in range(len(self._sorted_hashes)):
                sh = self._sorted_hashes[(idx + i) % len(self._sorted_hashes)]
                node_id = self._ring[sh]
                if node_id not in seen:
                    seen.add(node_id)
                    result.append((node_id, self._nodes.get(node_id)))
                    if len(result) >= count:
                        break
            return result

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def ring_size(self) -> int:
        return len(self._ring)

    def get_node_distribution(self) -> Dict[str, int]:
        """获取每个节点的虚拟节点数量"""
        with self._lock:
            dist: Dict[str, int] = {}
            for node_id in self._nodes:
                dist[node_id] = sum(1 for h, n in self._ring.items() if n == node_id)
            return dist

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

class HashRouter:
    """哈希路由器"""
    def __init__(self, virtual_nodes: int = 150):
        self._ring = ConsistentHashRing(virtual_nodes)
        self._lock = threading.RLock()

    def add_node(self, node_id: str, node_data: Any = None) -> None:
        self._ring.add_node(node_id, node_data)

    def remove_node(self, node_id: str) -> bool:
        return self._ring.remove_node(node_id)

    def route(self, key: str) -> Optional[str]:
        result = self._ring.get_node(key)
        return result[0] if result else None

    def route_with_data(self, key: str) -> Optional[Tuple[str, Any]]:
        return self._ring.get_node(key)

    def get_replica_nodes(self, key: str, replicas: int = 3) -> List[str]:
        nodes = self._ring.get_nodes(key, replicas)
        return [n[0] for n in nodes]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "nodes": self._ring.node_count,
            "virtual_nodes": self._ring.ring_size,
            "distribution": self._ring.get_node_distribution(),
        }

def create_consistent_hash_ring(virtual_nodes: int = 150) -> ConsistentHashRing:
    return ConsistentHashRing(virtual_nodes)

def create_hash_router(virtual_nodes: int = 150) -> HashRouter:
    return HashRouter(virtual_nodes)
