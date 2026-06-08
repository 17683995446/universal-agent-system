"""
阶段54: Agent分布式缓存

遵循"图难于其易，为大于其细"的原则，实现缓存条目、TTL、LRU淘汰、一致性哈希分片。
"""
import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class CacheEntry:
    key: str
    value: Any = None
    ttl: float = 0.0  # 0 means no expiry
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0

    @property
    def is_expired(self) -> bool:
        if self.ttl <= 0:
            return False
        return time.time() - self.created_at > self.ttl

class DistributedCache:
    """分布式缓存（本地模拟）"""
    def __init__(self, cache_id: str, max_size: int = 1000, default_ttl: float = 300.0):
        self.cache_id = cache_id
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._data: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired:
                del self._data[key]
                self._misses += 1
                return None
            entry.accessed_at = time.time()
            entry.access_count += 1
            self._data.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        with self._lock:
            self._evict_if_needed()
            self._data[key] = CacheEntry(
                key=key, value=value,
                ttl=ttl if ttl is not None else self._default_ttl,
            )

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def clear(self) -> int:
        with self._lock:
            count = len(self._data)
            self._data.clear()
            return count

    def keys(self) -> List[str]:
        with self._lock:
            return [k for k in list(self._data.keys()) if not self._data[k].is_expired]

    def size(self) -> int:
        with self._lock:
            return len(self._data)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._data),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0,
            }

    def _evict_if_needed(self) -> None:
        while len(self._data) >= self._max_size:
            self._data.popitem(last=False)

class ConsistentHashCache:
    """一致性哈希缓存"""
    def __init__(self, virtual_nodes: int = 150):
        self._virtual_nodes = virtual_nodes
        self._ring: Dict[int, str] = {}
        self._nodes: set = set()
        self._caches: Dict[str, DistributedCache] = {}
        self._lock = threading.RLock()

    def add_node(self, node_id: str, cache: DistributedCache) -> None:
        with self._lock:
            self._nodes.add(node_id)
            self._caches[node_id] = cache
            for i in range(self._virtual_nodes):
                key = f"{node_id}:{i}"
                h = int(hashlib.md5(key.encode()).hexdigest(), 16)
                self._ring[h] = node_id

    def remove_node(self, node_id: str) -> None:
        with self._lock:
            self._nodes.discard(node_id)
            self._caches.pop(node_id, None)
            to_remove = [h for h, n in self._ring.items() if n == node_id]
            for h in to_remove:
                del self._ring[h]

    def get_node(self, key: str) -> Optional[str]:
        with self._lock:
            if not self._ring:
                return None
            h = int(hashlib.md5(key.encode()).hexdigest(), 16)
            sorted_hashes = sorted(self._ring.keys())
            for sh in sorted_hashes:
                if sh >= h:
                    return self._ring[sh]
            return self._ring[sorted_hashes[0]]

    def get(self, key: str) -> Any:
        node = self.get_node(key)
        if node and node in self._caches:
            return self._caches[node].get(key)
        return None

    def set(self, key: str, value: Any) -> None:
        node = self.get_node(key)
        if node and node in self._caches:
            self._caches[node].set(key, value)

class CacheManager:
    """缓存管理器"""
    def __init__(self):
        self._caches: Dict[str, DistributedCache] = {}
        self._lock = threading.RLock()

    def get_cache(self, cache_id: str, max_size: int = 1000) -> DistributedCache:
        with self._lock:
            if cache_id not in self._caches:
                self._caches[cache_id] = DistributedCache(cache_id, max_size)
            return self._caches[cache_id]

    def remove_cache(self, cache_id: str) -> bool:
        with self._lock:
            return self._caches.pop(cache_id, None) is not None

    def list_caches(self) -> List[str]:
        return list(self._caches.keys())

def create_cache_manager() -> CacheManager:
    return CacheManager()
