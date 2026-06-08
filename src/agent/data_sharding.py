"""
阶段61: Agent数据分片系统

遵循"图难于其易，为大于其细"的原则，实现分片策略、分片路由、分片迁移、再平衡。
"""
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

class ShardStrategy(str):
    HASH = "hash"
    RANGE = "range"
    LIST = "list"

@dataclass
class Shard:
    shard_id: str
    strategy: str = ShardStrategy.HASH
    key_range: Optional[Tuple[str, str]] = None
    key_list: Optional[List[str]] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def delete(self, key: str) -> bool:
        if key in self.data:
            del self.data[key]
            return True
        return False

    def keys(self) -> List[str]:
        return list(self.data.keys())

@dataclass
class ShardRouter:
    """分片路由器"""
    strategy: str = ShardStrategy.HASH
    shard_count: int = 4
    _shards: Dict[str, Shard] = field(default_factory=dict)
    _hash_ring: Optional[Any] = None

    def add_shard(self, shard: Shard) -> None:
        self._shards[shard.shard_id] = shard

    def remove_shard(self, shard_id: str) -> bool:
        return self._shards.pop(shard_id, None) is not None

    def route(self, key: str) -> Optional[Shard]:
        if self.strategy == ShardStrategy.HASH:
            return self._hash_route(key)
        elif self.strategy == ShardStrategy.RANGE:
            return self._range_route(key)
        elif self.strategy == ShardStrategy.LIST:
            return self._list_route(key)
        return None

    def _hash_route(self, key: str) -> Optional[Shard]:
        if not self._shards:
            return None
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        shard_ids = sorted(self._shards.keys())
        idx = h % len(shard_ids)
        return self._shards[shard_ids[idx]]

    def _range_route(self, key: str) -> Optional[Shard]:
        for shard in self._shards.values():
            if shard.key_range and shard.key_range[0] <= key <= shard.key_range[1]:
                return shard
        return None

    def _list_route(self, key: str) -> Optional[Shard]:
        for shard in self._shards.values():
            if shard.key_list and key in shard.key_list:
                return shard
        return None

    def rebalance(self, new_shard_count: int) -> Dict[str, List[str]]:
        """重新平衡分片，返回迁移计划"""
        old_shards = dict(self._shards)
        # 收集所有key
        all_keys = []
        for shard in old_shards.values():
            all_keys.extend(shard.keys())

        # 创建新分片
        self._shards.clear()
        for i in range(new_shard_count):
            shard_id = f"shard_{i}"
            self._shards[shard_id] = Shard(shard_id=shard_id, strategy=self.strategy)

        # 重新分配
        migration: Dict[str, List[str]] = {}
        for key in all_keys:
            new_shard = self.route(key)
            if new_shard:
                # 找到旧数据
                for old_id, old_shard in old_shards.items():
                    if key in old_shard.data:
                        if old_id != new_shard.shard_id:
                            if old_id not in migration:
                                migration[old_id] = []
                            migration[old_id].append(key)
                        new_shard.set(key, old_shard.data[key])
                        break
        return migration

    def get_stats(self) -> Dict[str, Any]:
        return {
            "shard_count": len(self._shards),
            "total_keys": sum(len(s.keys()) for s in self._shards.values()),
            "strategy": self.strategy,
        }

class ShardingManager:
    """分片管理器"""
    def __init__(self):
        self._routers: Dict[str, ShardRouter] = {}
        self._lock = threading.RLock()

    def create_router(self, name: str, strategy: str = ShardStrategy.HASH,
                      shard_count: int = 4) -> ShardRouter:
        with self._lock:
            router = ShardRouter(strategy=strategy, shard_count=shard_count)
            for i in range(shard_count):
                shard = Shard(shard_id=f"{name}_shard_{i}", strategy=strategy)
                router.add_shard(shard)
            self._routers[name] = router
            return router

    def get_router(self, name: str) -> Optional[ShardRouter]:
        return self._routers.get(name)

    def remove_router(self, name: str) -> bool:
        with self._lock:
            return self._routers.pop(name, None) is not None

    def list_routers(self) -> List[str]:
        return list(self._routers.keys())

def create_sharding_manager() -> ShardingManager:
    return ShardingManager()
