"""
阶段36: Agent缓存系统

遵循"图难于其易，为大于其细"的原则，实现多级缓存、缓存策略、缓存淘汰。
"""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar

K = TypeVar('K')
V = TypeVar('V')


# ============================================================
# 36.1 缓存模型
# ============================================================

class EvictionPolicy(str, Enum):
    """淘汰策略"""
    LRU = "lru"           # 最近最少使用
    LFU = "lfu"           # 最不经常使用
    FIFO = "fifo"         # 先进先出
    TTL = "ttl"           # 时间过期
    NONE = "none"         # 不淘汰


class CacheLevel(str, Enum):
    """缓存层级"""
    L1 = "l1"             # 本地内存缓存（最快）
    L2 = "l2"             # 二级缓存（如Redis）
    L3 = "l3"             # 三级缓存（如分布式存储）


@dataclass
class CacheEntry(Generic[V]):
    """缓存项"""
    key: str
    value: V
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    ttl_seconds: Optional[float] = None
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at is None:
            return False
        return datetime.now() >= self.expires_at

    def touch(self) -> None:
        """更新访问信息"""
        self.access_count += 1
        self.last_access = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "ttl_seconds": self.ttl_seconds,
            "access_count": self.access_count,
            "last_access": self.last_access.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class CacheStats:
    """缓存统计"""
    name: str
    level: CacheLevel
    size: int = 0
    max_size: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    puts: int = 0
    deletes: int = 0
    refreshes: int = 0
    errors: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    @property
    def utilization(self) -> float:
        if self.max_size == 0:
            return 0.0
        return self.size / self.max_size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level.value,
            "size": self.size,
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "evictions": self.evictions,
            "puts": self.puts,
            "deletes": self.deletes,
            "refreshes": self.refreshes,
            "errors": self.errors,
            "utilization": round(self.utilization, 4),
        }


@dataclass
class CacheConfig:
    """缓存配置"""
    name: str = "default"
    max_size: int = 1000
    default_ttl_seconds: Optional[float] = None
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    refresh_on_access: bool = False
    refresh_ttl_on_access: bool = True
    preload: bool = False
    preload_keys: List[str] = field(default_factory=list)


# ============================================================
# 36.2 内存缓存
# ============================================================

class MemoryCache(Generic[K, V]):
    """内存缓存 - 支持多种淘汰策略"""

    def __init__(self, config: Optional[CacheConfig] = None):
        self._config = config or CacheConfig()
        self._entries: Dict[str, CacheEntry[V]] = {}
        self._access_order: OrderedDict = OrderedDict()  # LRU/FIFO顺序
        self._frequency: Dict[str, int] = {}  # LFU频率
        self._lock = threading.RLock()
        self._stats = CacheStats(
            name=self._config.name,
            level=CacheLevel.L1,
            max_size=self._config.max_size,
        )
        self._loader: Optional[Callable[[str], V]] = None
        self._on_evict: Optional[Callable[[str, V], None]] = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def stats(self) -> CacheStats:
        with self._lock:
            self._stats.size = len(self._entries)
            return CacheStats(
                name=self._stats.name,
                level=self._stats.level,
                size=self._stats.size,
                max_size=self._stats.max_size,
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                puts=self._stats.puts,
                deletes=self._stats.deletes,
                refreshes=self._stats.refreshes,
                errors=self._stats.errors,
            )

    def set_loader(self, loader: Callable[[str], V]) -> None:
        """设置数据加载器（用于缓存穿透）"""
        self._loader = loader

    def set_on_evict(self, callback: Callable[[str, V], None]) -> None:
        """设置淘汰回调"""
        self._on_evict = callback

    def get(self, key: str) -> Optional[V]:
        """获取缓存值"""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._stats.misses += 1
                # 尝试从loader加载
                if self._loader:
                    try:
                        value = self._loader(key)
                        self.put(key, value)
                        return value
                    except Exception:
                        self._stats.errors += 1
                return None

            if entry.is_expired():
                self._remove_entry(key)
                self._stats.misses += 1
                self._stats.evictions += 1
                return None

            entry.touch()
            self._update_access_order(key)
            self._update_frequency(key)

            if self._config.refresh_ttl_on_access and entry.ttl_seconds:
                entry.expires_at = datetime.now() + timedelta(seconds=entry.ttl_seconds)

            self._stats.hits += 1
            return entry.value

    def put(self, key: str, value: V, ttl_seconds: Optional[float] = None) -> bool:
        """写入缓存"""
        with self._lock:
            ttl = ttl_seconds or self._config.default_ttl_seconds
            expires_at = None
            if ttl:
                expires_at = datetime.now() + timedelta(seconds=ttl)

            entry = CacheEntry(
                key=key,
                value=value,
                ttl_seconds=ttl,
                expires_at=expires_at,
            )

            # 如果已存在，先移除
            if key in self._entries:
                self._remove_entry(key)

            # 检查容量
            while len(self._entries) >= self._config.max_size:
                self._evict_one()

            self._entries[key] = entry
            self._access_order[key] = None
            self._frequency[key] = 1
            self._stats.puts += 1
            return True

    def delete(self, key: str) -> bool:
        """删除缓存"""
        with self._lock:
            if key in self._entries:
                self._remove_entry(key)
                self._stats.deletes += 1
                return True
            return False

    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                self._remove_entry(key)
                return False
            return True

    def clear(self) -> int:
        """清空缓存"""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._access_order.clear()
            self._frequency.clear()
            return count

    def keys(self) -> List[str]:
        """获取所有键"""
        with self._lock:
            self._expire_check()
            return list(self._entries.keys())

    def values(self) -> List[V]:
        """获取所有值"""
        with self._lock:
            self._expire_check()
            return [e.value for e in self._entries.values()]

    def entries(self) -> List[CacheEntry[V]]:
        """获取所有缓存项"""
        with self._lock:
            self._expire_check()
            return list(self._entries.values())

    def refresh(self, key: str, loader: Optional[Callable[[str], V]] = None) -> bool:
        """刷新缓存"""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False

            try:
                new_value = (loader or self._loader)(key) if (loader or self._loader) else entry.value
                entry.value = new_value
                entry.updated_at = datetime.now()
                if entry.ttl_seconds:
                    entry.expires_at = datetime.now() + timedelta(seconds=entry.ttl_seconds)
                self._stats.refreshes += 1
                return True
            except Exception:
                self._stats.errors += 1
                return False

    def invalidate(self, pattern: str = "") -> int:
        """失效匹配的缓存"""
        with self._lock:
            keys_to_delete = [
                k for k in self._entries.keys()
                if pattern == "" or k.startswith(pattern)
            ]
            for key in keys_to_delete:
                self._remove_entry(key)
                self._stats.deletes += 1
            return len(keys_to_delete)

    # --- 内部方法 ---

    def _remove_entry(self, key: str) -> None:
        """移除缓存项"""
        entry = self._entries.pop(key, None)
        self._access_order.pop(key, None)
        self._frequency.pop(key, None)
        if entry and self._on_evict:
            try:
                self._on_evict(key, entry.value)
            except Exception:
                pass

    def _update_access_order(self, key: str) -> None:
        """更新访问顺序（LRU）"""
        if self._config.eviction_policy == EvictionPolicy.LRU:
            self._access_order.move_to_end(key)

    def _update_frequency(self, key: str) -> None:
        """更新频率（LFU）"""
        if self._config.eviction_policy == EvictionPolicy.LFU:
            self._frequency[key] = self._frequency.get(key, 0) + 1

    def _evict_one(self) -> Optional[str]:
        """淘汰一个缓存项"""
        if not self._entries:
            return None

        policy = self._config.eviction_policy
        key_to_evict = None

        if policy == EvictionPolicy.LRU:
            # 淘汰最久未访问
            key_to_evict = next(iter(self._access_order))
        elif policy == EvictionPolicy.LFU:
            # 淘汰访问频率最低
            min_freq = min(self._frequency.values())
            for k, f in self._frequency.items():
                if f == min_freq:
                    key_to_evict = k
                    break
        elif policy == EvictionPolicy.FIFO:
            # 淘汰最早插入
            key_to_evict = next(iter(self._access_order))
        elif policy == EvictionPolicy.TTL:
            # 淘汰最早过期
            earliest = None
            for k, e in self._entries.items():
                if e.expires_at:
                    if earliest is None or e.expires_at < earliest:
                        earliest = e.expires_at
                        key_to_evict = k
            if key_to_evict is None:
                key_to_evict = next(iter(self._access_order))
        else:
            key_to_evict = next(iter(self._access_order))

        if key_to_evict:
            self._remove_entry(key_to_evict)
            self._stats.evictions += 1
        return key_to_evict

    def _expire_check(self) -> int:
        """检查并清理过期项"""
        expired_keys = [
            k for k, e in self._entries.items()
            if e.is_expired()
        ]
        for key in expired_keys:
            self._remove_entry(key)
            self._stats.evictions += 1
        return len(expired_keys)


# ============================================================
# 36.3 缓存策略
# ============================================================

class CachePolicy:
    """缓存策略管理"""

    def __init__(self):
        self._rules: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def add_rule(
        self,
        key_pattern: str,
        ttl_seconds: Optional[float] = None,
        refresh_interval: Optional[float] = None,
        max_idle_seconds: Optional[float] = None,
    ) -> None:
        """添加缓存规则"""
        with self._lock:
            self._rules[key_pattern] = {
                "ttl_seconds": ttl_seconds,
                "refresh_interval": refresh_interval,
                "max_idle_seconds": max_idle_seconds,
            }

    def remove_rule(self, key_pattern: str) -> bool:
        """移除缓存规则"""
        with self._lock:
            if key_pattern in self._rules:
                del self._rules[key_pattern]
                return True
            return False

    def get_rule(self, key: str) -> Optional[Dict[str, Any]]:
        """获取匹配的规则"""
        with self._lock:
            for pattern, rule in self._rules.items():
                if key.startswith(pattern) or key == pattern:
                    return rule
            return None

    def get_ttl(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """获取键的TTL"""
        rule = self.get_rule(key)
        if rule and rule.get("ttl_seconds"):
            return rule["ttl_seconds"]
        return default


class CacheRefreshManager:
    """缓存刷新管理器"""

    def __init__(self, cache: MemoryCache):
        self._cache = cache
        self._refresh_tasks: Dict[str, float] = {}  # key -> refresh_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def schedule_refresh(self, key: str, interval_seconds: float) -> None:
        """安排定期刷新"""
        with self._lock:
            self._refresh_tasks[key] = interval_seconds

    def cancel_refresh(self, key: str) -> bool:
        """取消刷新"""
        with self._lock:
            if key in self._refresh_tasks:
                del self._refresh_tasks[key]
                return True
            return False

    def start(self) -> None:
        """启动刷新线程"""
        if self._running:
            return
        self._running = True

        def _refresh_loop():
            while self._running:
                with self._lock:
                    tasks = dict(self._refresh_tasks)
                for key, interval in tasks.items():
                    self._cache.refresh(key)
                time.sleep(1.0)

        self._thread = threading.Thread(target=_refresh_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止刷新"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None


# ============================================================
# 36.4 多级缓存
# ============================================================

class MultiLevelCache(Generic[K, V]):
    """多级缓存 - L1/L2分层"""

    def __init__(self, caches: List[MemoryCache[K, V]]):
        self._caches = caches
        self._lock = threading.RLock()
        self._stats = CacheStats(name="multi_level", level=CacheLevel.L1)

    @property
    def stats(self) -> CacheStats:
        total_hits = sum(c.stats.hits for c in self._caches)
        total_misses = sum(c.stats.misses for c in self._caches)
        return CacheStats(
            name="multi_level",
            level=CacheLevel.L1,
            size=sum(c.stats.size for c in self._caches),
            max_size=sum(c.stats.max_size for c in self._caches),
            hits=total_hits,
            misses=total_misses,
            evictions=sum(c.stats.evictions for c in self._caches),
            puts=sum(c.stats.puts for c in self._caches),
            deletes=sum(c.stats.deletes for c in self._caches),
        )

    def get(self, key: str) -> Optional[V]:
        """从多级缓存获取"""
        with self._lock:
            for i, cache in enumerate(self._caches):
                value = cache.get(key)
                if value is not None:
                    # 回填到更高级缓存
                    for j in range(i):
                        self._caches[j].put(key, value)
                    return value
            return None

    def put(self, key: str, value: V, ttl_seconds: Optional[float] = None) -> bool:
        """写入所有缓存层"""
        with self._lock:
            for cache in self._caches:
                cache.put(key, value, ttl_seconds)
            return True

    def delete(self, key: str) -> bool:
        """从所有层删除"""
        with self._lock:
            deleted = False
            for cache in self._caches:
                if cache.delete(key):
                    deleted = True
            return deleted

    def invalidate(self, pattern: str = "") -> int:
        """失效所有层"""
        total = 0
        for cache in self._caches:
            total += cache.invalidate(pattern)
        return total

    def clear(self) -> int:
        """清空所有层"""
        total = 0
        for cache in self._caches:
            total += cache.clear()
        return total

    def get_cache(self, level: int) -> Optional[MemoryCache[K, V]]:
        """获取特定层缓存"""
        if 0 <= level < len(self._caches):
            return self._caches[level]
        return None


# ============================================================
# 便捷函数
# ============================================================

def create_cache(
    name: str = "default",
    max_size: int = 1000,
    ttl_seconds: Optional[float] = None,
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
) -> MemoryCache:
    """创建内存缓存"""
    config = CacheConfig(
        name=name,
        max_size=max_size,
        default_ttl_seconds=ttl_seconds,
        eviction_policy=eviction_policy,
    )
    return MemoryCache(config)


def create_multi_level_cache(
    l1_size: int = 100,
    l2_size: int = 1000,
    ttl_seconds: Optional[float] = None,
) -> MultiLevelCache:
    """创建多级缓存"""
    l1 = create_cache("l1", l1_size, ttl_seconds, EvictionPolicy.LRU)
    l2 = create_cache("l2", l2_size, ttl_seconds, EvictionPolicy.LFU)
    return MultiLevelCache([l1, l2])


def create_cache_policy() -> CachePolicy:
    """创建缓存策略"""
    return CachePolicy()


# 导入timedelta用于TTL计算
from datetime import timedelta