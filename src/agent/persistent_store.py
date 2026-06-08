"""
阶段39: Agent数据持久化存储系统

遵循"图难于其易，为大于其细"的原则，在阶段38序列化系统基础上，
实现二进制持久化存储、快照管理、数据恢复能力。
"""

import hashlib
import json
import os
import shutil
import struct
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.agent.serialization import (
    SerializationFormat,
    CompressionType,
    SerializationManager,
    SerializationResult,
    VersionInfo,
    VersionedSerializer,
)


# ============================================================
# 39.1 持久化模型
# ============================================================

class PersistentStoreType(str, Enum):
    """持久化存储类型"""
    MEMORY = "memory"
    FILE_BINARY = "file_binary"
    FILE_JSON = "file_json"


class StoreStatus(str, Enum):
    """存储状态"""
    ACTIVE = "active"
    READ_ONLY = "read_only"
    CLOSED = "closed"
    ERROR = "error"


@dataclass
class StoreEntry:
    """持久化存储条目"""
    key: str
    value: Any
    namespace: str = "default"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: int = 1
    checksum: str = ""
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    ttl: Optional[float] = None

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def compute_checksum(self, data: bytes) -> str:
        """计算数据校验和"""
        self.checksum = hashlib.sha256(data).hexdigest()[:16]
        return self.checksum

    def touch(self) -> None:
        """更新时间戳和版本"""
        self.updated_at = time.time()
        self.version += 1

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "key": self.key,
            "namespace": self.namespace,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata,
            "ttl": self.ttl,
        }


@dataclass
class StoreConfig:
    """存储配置"""
    store_type: PersistentStoreType = PersistentStoreType.MEMORY
    base_path: str = "./data/persistent"
    serialization_format: SerializationFormat = SerializationFormat.JSON
    compression: CompressionType = CompressionType.NONE
    max_entries: int = 100000
    auto_flush_interval: float = 30.0
    enable_checksum: bool = True
    enable_versioning: bool = True
    max_versions_per_key: int = 10


@dataclass
class SnapshotInfo:
    """快照信息"""
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: float = field(default_factory=time.time)
    entry_count: int = 0
    total_size: int = 0
    store_type: str = ""
    description: str = ""
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "entry_count": self.entry_count,
            "total_size": self.total_size,
            "store_type": self.store_type,
            "description": self.description,
            "version": self.version,
        }


@dataclass
class RecoveryResult:
    """恢复结果"""
    success: bool
    recovered_entries: int = 0
    failed_entries: int = 0
    errors: List[str] = field(default_factory=list)
    snapshot_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "recovered_entries": self.recovered_entries,
            "failed_entries": self.failed_entries,
            "errors": self.errors,
            "snapshot_id": self.snapshot_id,
        }


@dataclass
class StoreStats:
    """存储统计"""
    total_entries: int = 0
    total_size_bytes: int = 0
    namespace_count: int = 0
    oldest_entry_time: float = 0
    newest_entry_time: float = 0
    snapshot_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_entries": self.total_entries,
            "total_size_bytes": self.total_size_bytes,
            "namespace_count": self.namespace_count,
            "oldest_entry_time": self.oldest_entry_time,
            "newest_entry_time": self.newest_entry_time,
            "snapshot_count": self.snapshot_count,
        }


# ============================================================
# 39.2 持久化存储抽象接口
# ============================================================

class PersistentStore(ABC):
    """持久化存储抽象接口"""

    @abstractmethod
    def get(self, key: str, namespace: str = "default") -> Optional[StoreEntry]:
        """获取条目"""
        pass

    @abstractmethod
    def put(self, key: str, value: Any, namespace: str = "default",
            ttl: Optional[float] = None, metadata: Optional[Dict] = None) -> bool:
        """存储条目"""
        pass

    @abstractmethod
    def delete(self, key: str, namespace: str = "default") -> bool:
        """删除条目"""
        pass

    @abstractmethod
    def exists(self, key: str, namespace: str = "default") -> bool:
        """检查条目是否存在"""
        pass

    @abstractmethod
    def keys(self, namespace: str = "default") -> List[str]:
        """获取所有键"""
        pass

    @abstractmethod
    def count(self, namespace: str = "default") -> int:
        """获取条目数量"""
        pass

    @abstractmethod
    def namespaces(self) -> List[str]:
        """获取所有命名空间"""
        pass

    @abstractmethod
    def clear(self, namespace: Optional[str] = None) -> int:
        """清除条目"""
        pass

    @abstractmethod
    def flush(self) -> bool:
        """刷新到持久层"""
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭存储"""
        pass

    @abstractmethod
    def get_stats(self) -> StoreStats:
        """获取统计信息"""
        pass


# ============================================================
# 39.3 内存持久化存储
# ============================================================

class MemoryPersistentStore(PersistentStore):
    """内存持久化存储"""

    def __init__(self, config: Optional[StoreConfig] = None):
        self._config = config or StoreConfig()
        self._store: Dict[str, StoreEntry] = {}
        self._namespaces: Dict[str, set] = {}
        self._lock = threading.RLock()
        self._status = StoreStatus.ACTIVE
        self._total_size = 0

    def _make_key(self, key: str, namespace: str) -> str:
        return f"{namespace}://{key}"

    def get(self, key: str, namespace: str = "default") -> Optional[StoreEntry]:
        with self._lock:
            if self._status == StoreStatus.CLOSED:
                return None
            store_key = self._make_key(key, namespace)
            entry = self._store.get(store_key)
            if entry is None:
                return None
            if entry.is_expired():
                self._remove_entry(store_key, namespace)
                return None
            return entry

    def put(self, key: str, value: Any, namespace: str = "default",
            ttl: Optional[float] = None, metadata: Optional[Dict] = None) -> bool:
        with self._lock:
            if self._status == StoreStatus.CLOSED:
                return False
            if self._status == StoreStatus.READ_ONLY:
                return False

            store_key = self._make_key(key, namespace)

            # 检查容量
            if store_key not in self._store and len(self._store) >= self._config.max_entries:
                return False

            # 更新大小
            old_entry = self._store.get(store_key)
            if old_entry:
                self._total_size -= old_entry.size_bytes

            # 序列化计算大小
            value_bytes = json.dumps(value, default=str).encode("utf-8")
            size = len(value_bytes)

            entry = StoreEntry(
                key=key,
                value=value,
                namespace=namespace,
                ttl=ttl,
                metadata=metadata or {},
                size_bytes=size,
            )
            if self._config.enable_checksum:
                entry.compute_checksum(value_bytes)
            if self._config.enable_versioning and old_entry:
                entry.version = old_entry.version + 1

            self._store[store_key] = entry
            self._total_size += size

            # 记录命名空间
            if namespace not in self._namespaces:
                self._namespaces[namespace] = set()
            self._namespaces[namespace].add(key)

            return True

    def delete(self, key: str, namespace: str = "default") -> bool:
        with self._lock:
            if self._status == StoreStatus.CLOSED:
                return False
            return self._remove_entry(self._make_key(key, namespace), namespace)

    def _remove_entry(self, store_key: str, namespace: str) -> bool:
        """内部删除条目"""
        entry = self._store.pop(store_key, None)
        if entry:
            self._total_size -= entry.size_bytes
            if namespace in self._namespaces:
                self._namespaces[namespace].discard(entry.key)
                if not self._namespaces[namespace]:
                    del self._namespaces[namespace]
            return True
        return False

    def exists(self, key: str, namespace: str = "default") -> bool:
        with self._lock:
            return self.get(key, namespace) is not None

    def keys(self, namespace: str = "default") -> List[str]:
        with self._lock:
            prefix = f"{namespace}://"
            result = []
            expired = []
            for store_key, entry in self._store.items():
                if store_key.startswith(prefix):
                    if entry.is_expired():
                        expired.append(store_key)
                    else:
                        result.append(entry.key)
            for k in expired:
                self._remove_entry(k, namespace)
            return result

    def count(self, namespace: str = "default") -> int:
        return len(self.keys(namespace))

    def namespaces(self) -> List[str]:
        with self._lock:
            return list(self._namespaces.keys())

    def clear(self, namespace: Optional[str] = None) -> int:
        with self._lock:
            if namespace is None:
                count = len(self._store)
                self._store.clear()
                self._namespaces.clear()
                self._total_size = 0
                return count
            keys = self.keys(namespace)
            for key in keys:
                self.delete(key, namespace)
            return len(keys)

    def flush(self) -> bool:
        """内存存储无需flush"""
        return True

    def close(self) -> None:
        with self._lock:
            self._status = StoreStatus.CLOSED
            self._store.clear()
            self._namespaces.clear()
            self._total_size = 0

    def get_stats(self) -> StoreStats:
        with self._lock:
            entries = list(self._store.values())
            if not entries:
                return StoreStats(
                    namespace_count=len(self._namespaces),
                )
            times = [e.created_at for e in entries]
            return StoreStats(
                total_entries=len(entries),
                total_size_bytes=self._total_size,
                namespace_count=len(self._namespaces),
                oldest_entry_time=min(times),
                newest_entry_time=max(times),
            )

    @property
    def status(self) -> StoreStatus:
        return self._status


# ============================================================
# 39.4 文件持久化存储
# ============================================================

class FilePersistentStore(PersistentStore):
    """文件持久化存储（基于序列化系统）"""

    def __init__(self, config: Optional[StoreConfig] = None):
        self._config = config or StoreConfig(store_type=PersistentStoreType.FILE_BINARY)
        self._base_path = Path(self._config.base_path)
        self._serializer = SerializationManager()
        self._serializer.set_default_format(self._config.serialization_format)
        self._serializer.set_default_compression(self._config.compression)
        self._lock = threading.RLock()
        self._status = StoreStatus.ACTIVE
        self._cache: Dict[str, StoreEntry] = {}
        self._dirty: set = set()
        self._namespaces_cache: Optional[List[str]] = None
        self._ensure_base_path()

    def _ensure_base_path(self) -> None:
        self._base_path.mkdir(parents=True, exist_ok=True)

    def _safe_name(self, name: str) -> str:
        """生成安全文件名"""
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)

    def _namespace_path(self, namespace: str) -> Path:
        return self._base_path / self._safe_name(namespace)

    def _entry_path(self, key: str, namespace: str) -> Path:
        return self._namespace_path(namespace) / f"{self._safe_name(key)}.bin"

    def _meta_path(self, key: str, namespace: str) -> Path:
        return self._namespace_path(namespace) / f"{self._safe_name(key)}.meta.json"

    def _index_path(self, namespace: str) -> Path:
        return self._namespace_path(namespace) / "_index.json"

    def _read_meta(self, key: str, namespace: str) -> Optional[Dict[str, Any]]:
        """读取条目元数据"""
        meta_path = self._meta_path(key, namespace)
        if not meta_path.exists():
            return None
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            return None

    def _write_meta(self, entry: StoreEntry) -> None:
        """写入条目元数据"""
        ns_path = self._namespace_path(entry.namespace)
        ns_path.mkdir(parents=True, exist_ok=True)
        meta_path = self._meta_path(entry.key, entry.namespace)
        meta_path.write_text(
            json.dumps(entry.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get(self, key: str, namespace: str = "default") -> Optional[StoreEntry]:
        with self._lock:
            if self._status == StoreStatus.CLOSED:
                return None

            # 先查缓存
            store_key = f"{namespace}://{key}"
            if store_key in self._cache:
                entry = self._cache[store_key]
                if not entry.is_expired():
                    return entry
                else:
                    self.delete(key, namespace)
                    return None

            # 从文件读取
            entry_path = self._entry_path(key, namespace)
            if not entry_path.exists():
                return None

            try:
                data = entry_path.read_bytes()
                meta = self._read_meta(key, namespace)
                if meta is None:
                    return None

                # 反序列化
                result = SerializationResult(
                    data=data,
                    format=self._config.serialization_format,
                    compression=self._config.compression,
                    original_size=meta.get("size_bytes", len(data)),
                    compressed_size=len(data),
                    version=meta.get("version", "1.0"),
                )
                value = self._serializer.deserialize(result)

                entry = StoreEntry(
                    key=key,
                    value=value,
                    namespace=namespace,
                    created_at=meta.get("created_at", time.time()),
                    updated_at=meta.get("updated_at", time.time()),
                    version=meta.get("version", 1),
                    checksum=meta.get("checksum", ""),
                    size_bytes=meta.get("size_bytes", 0),
                    metadata=meta.get("metadata", {}),
                    ttl=meta.get("ttl"),
                )

                if entry.is_expired():
                    self.delete(key, namespace)
                    return None

                # 放入缓存
                self._cache[store_key] = entry
                return entry

            except Exception:
                return None

    def put(self, key: str, value: Any, namespace: str = "default",
            ttl: Optional[float] = None, metadata: Optional[Dict] = None) -> bool:
        with self._lock:
            if self._status in (StoreStatus.CLOSED, StoreStatus.READ_ONLY):
                return False

            store_key = f"{namespace}://{key}"
            old_entry = self._cache.get(store_key)

            # 序列化
            result = self._serializer.serialize(
                value,
                format=self._config.serialization_format,
                compression=self._config.compression,
            )

            # 写入数据文件
            ns_path = self._namespace_path(namespace)
            ns_path.mkdir(parents=True, exist_ok=True)
            entry_path = self._entry_path(key, namespace)

            try:
                entry_path.write_bytes(result.data)

                # 创建条目
                entry = StoreEntry(
                    key=key,
                    value=value,
                    namespace=namespace,
                    ttl=ttl,
                    metadata=metadata or {},
                    size_bytes=result.original_size,
                    version=(old_entry.version + 1) if old_entry and self._config.enable_versioning else 1,
                )
                if self._config.enable_checksum:
                    entry.compute_checksum(result.data)

                # 写入元数据
                self._write_meta(entry)

                # 更新缓存
                self._cache[store_key] = entry
                self._dirty.discard(store_key)
                self._namespaces_cache = None

                return True

            except (IOError, OSError):
                return False

    def delete(self, key: str, namespace: str = "default") -> bool:
        with self._lock:
            if self._status == StoreStatus.CLOSED:
                return False

            store_key = f"{namespace}://{key}"
            entry_path = self._entry_path(key, namespace)
            meta_path = self._meta_path(key, namespace)

            deleted = False
            if entry_path.exists():
                entry_path.unlink(missing_ok=True)
                deleted = True
            if meta_path.exists():
                meta_path.unlink(missing_ok=True)
                deleted = True

            self._cache.pop(store_key, None)
            self._dirty.discard(store_key)
            self._namespaces_cache = None

            return deleted

    def exists(self, key: str, namespace: str = "default") -> bool:
        with self._lock:
            return self.get(key, namespace) is not None

    def keys(self, namespace: str = "default") -> List[str]:
        with self._lock:
            ns_path = self._namespace_path(namespace)
            if not ns_path.exists():
                return []
            result = []
            for file_path in ns_path.glob("*.meta.json"):
                try:
                    meta = json.loads(file_path.read_text(encoding="utf-8"))
                    key = meta.get("key", "")
                    ttl = meta.get("ttl")
                    created_at = meta.get("created_at", time.time())
                    if ttl is not None and time.time() - created_at > ttl:
                        # 过期条目，清理
                        safe_key = self._safe_name(key)
                        data_path = ns_path / f"{safe_key}.bin"
                        data_path.unlink(missing_ok=True)
                        file_path.unlink(missing_ok=True)
                    else:
                        result.append(key)
                except (json.JSONDecodeError, IOError):
                    continue
            return result

    def count(self, namespace: str = "default") -> int:
        return len(self.keys(namespace))

    def namespaces(self) -> List[str]:
        with self._lock:
            if self._namespaces_cache is not None:
                return self._namespaces_cache
            result = []
            if self._base_path.exists():
                for p in self._base_path.iterdir():
                    if p.is_dir():
                        result.append(p.name)
            self._namespaces_cache = result
            return result

    def clear(self, namespace: Optional[str] = None) -> int:
        with self._lock:
            if namespace is None:
                if self._base_path.exists():
                    count = sum(1 for _ in self._base_path.rglob("*.bin"))
                    shutil.rmtree(self._base_path, ignore_errors=True)
                    self._ensure_base_path()
                else:
                    count = 0
                self._cache.clear()
                self._dirty.clear()
                self._namespaces_cache = None
                return count

            ns_path = self._namespace_path(namespace)
            if ns_path.exists():
                count = sum(1 for _ in ns_path.glob("*.bin"))
                shutil.rmtree(ns_path, ignore_errors=True)
            else:
                count = 0

            # 清理缓存
            prefix = f"{namespace}://"
            to_remove = [k for k in self._cache if k.startswith(prefix)]
            for k in to_remove:
                del self._cache[k]
            self._dirty -= set(to_remove)
            self._namespaces_cache = None

            return count

    def flush(self) -> bool:
        """文件存储每次put已直接写入，flush为空操作"""
        return True

    def close(self) -> None:
        with self._lock:
            self._status = StoreStatus.CLOSED
            self._cache.clear()
            self._dirty.clear()

    def get_stats(self) -> StoreStats:
        with self._lock:
            all_namespaces = self.namespaces()
            total_entries = 0
            total_size = 0
            oldest = float("inf")
            newest = 0

            for ns in all_namespaces:
                ns_path = self._namespace_path(ns)
                if not ns_path.exists():
                    continue
                for meta_path in ns_path.glob("*.meta.json"):
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        total_entries += 1
                        total_size += meta.get("size_bytes", 0)
                        ct = meta.get("created_at", 0)
                        if ct < oldest:
                            oldest = ct
                        ut = meta.get("updated_at", 0)
                        if ut > newest:
                            newest = ut
                    except (json.JSONDecodeError, IOError):
                        continue

            return StoreStats(
                total_entries=total_entries,
                total_size_bytes=total_size,
                namespace_count=len(all_namespaces),
                oldest_entry_time=oldest if oldest != float("inf") else 0,
                newest_entry_time=newest,
            )

    @property
    def status(self) -> StoreStatus:
        return self._status


# ============================================================
# 39.5 快照管理器
# ============================================================

class SnapshotManager:
    """快照管理器"""

    def __init__(self, store: PersistentStore, snapshot_dir: Optional[str] = None):
        self._store = store
        self._snapshot_dir = Path(snapshot_dir or "./data/snapshots")
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._snapshots: Dict[str, SnapshotInfo] = {}

    def create_snapshot(self, description: str = "",
                         namespaces: Optional[List[str]] = None) -> SnapshotInfo:
        """创建快照"""
        with self._lock:
            snapshot_id = str(uuid.uuid4())[:8]
            snapshot_path = self._snapshot_dir / snapshot_id
            snapshot_path.mkdir(parents=True, exist_ok=True)

            # 确定要快照的命名空间
            target_ns = namespaces or self._store.namespaces()

            entry_count = 0
            total_size = 0
            manifest: List[Dict[str, Any]] = []

            for ns in target_ns:
                keys = self._store.keys(ns)
                for key in keys:
                    entry = self._store.get(key, ns)
                    if entry is None:
                        continue

                    # 序列化条目
                    serializer = SerializationManager()
                    result = serializer.serialize(
                        entry.value,
                        format=SerializationFormat.JSON,
                        compression=CompressionType.GZIP,
                    )

                    # 写入快照文件
                    ns_dir = snapshot_path / self._safe_name(ns)
                    ns_dir.mkdir(parents=True, exist_ok=True)
                    entry_file = ns_dir / f"{self._safe_name(key)}.snap.gz"
                    entry_file.write_bytes(result.data)

                    # 写入元数据
                    meta_file = ns_dir / f"{self._safe_name(key)}.meta.json"
                    meta_file.write_text(
                        json.dumps(entry.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                    entry_count += 1
                    total_size += entry.size_bytes
                    manifest.append({
                        "namespace": ns,
                        "key": key,
                        "size": entry.size_bytes,
                        "version": entry.version,
                    })

            # 写入清单
            manifest_file = snapshot_path / "manifest.json"
            info = SnapshotInfo(
                snapshot_id=snapshot_id,
                entry_count=entry_count,
                total_size=total_size,
                store_type=type(self._store).__name__,
                description=description,
            )
            manifest_file.write_text(
                json.dumps({
                    "info": info.to_dict(),
                    "entries": manifest,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            self._snapshots[snapshot_id] = info
            return info

    def restore_snapshot(self, snapshot_id: str,
                          target_store: Optional[PersistentStore] = None) -> RecoveryResult:
        """恢复快照"""
        with self._lock:
            store = target_store or self._store
            snapshot_path = self._snapshot_dir / snapshot_id
            manifest_file = snapshot_path / "manifest.json"

            if not manifest_file.exists():
                return RecoveryResult(
                    success=False,
                    errors=[f"快照 {snapshot_id} 不存在"],
                )

            try:
                manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError) as e:
                return RecoveryResult(
                    success=False,
                    errors=[f"读取清单失败: {e}"],
                )

            serializer = SerializationManager()
            recovered = 0
            failed = 0
            errors = []

            for entry_info in manifest_data.get("entries", []):
                ns = entry_info["namespace"]
                key = entry_info["key"]

                ns_dir = snapshot_path / self._safe_name(ns)
                entry_file = ns_dir / f"{self._safe_name(key)}.snap.gz"
                meta_file = ns_dir / f"{self._safe_name(key)}.meta.json"

                if not entry_file.exists():
                    failed += 1
                    errors.append(f"条目文件缺失: {ns}/{key}")
                    continue

                try:
                    data = entry_file.read_bytes()
                    result = SerializationResult(
                        data=data,
                        format=SerializationFormat.JSON,
                        compression=CompressionType.GZIP,
                        original_size=entry_info.get("size", len(data)),
                        compressed_size=len(data),
                    )
                    value = serializer.deserialize(result)

                    # 读取元数据
                    ttl = None
                    metadata = {}
                    if meta_file.exists():
                        meta = json.loads(meta_file.read_text(encoding="utf-8"))
                        ttl = meta.get("ttl")
                        metadata = meta.get("metadata", {})

                    store.put(key, value, namespace=ns, ttl=ttl, metadata=metadata)
                    recovered += 1

                except Exception as e:
                    failed += 1
                    errors.append(f"恢复 {ns}/{key} 失败: {e}")

            return RecoveryResult(
                success=failed == 0,
                recovered_entries=recovered,
                failed_entries=failed,
                errors=errors,
                snapshot_id=snapshot_id,
            )

    def list_snapshots(self) -> List[SnapshotInfo]:
        """列出所有快照"""
        with self._lock:
            result = []
            if not self._snapshot_dir.exists():
                return result

            for snapshot_dir in self._snapshot_dir.iterdir():
                if not snapshot_dir.is_dir():
                    continue
                manifest_file = snapshot_dir / "manifest.json"
                if not manifest_file.exists():
                    continue
                try:
                    data = json.loads(manifest_file.read_text(encoding="utf-8"))
                    info_data = data.get("info", {})
                    info = SnapshotInfo(
                        snapshot_id=info_data.get("snapshot_id", snapshot_dir.name),
                        created_at=info_data.get("created_at", 0),
                        entry_count=info_data.get("entry_count", 0),
                        total_size=info_data.get("total_size", 0),
                        store_type=info_data.get("store_type", ""),
                        description=info_data.get("description", ""),
                        version=info_data.get("version", "1.0"),
                    )
                    result.append(info)
                    self._snapshots[info.snapshot_id] = info
                except (json.JSONDecodeError, IOError):
                    continue

            return sorted(result, key=lambda x: x.created_at, reverse=True)

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """删除快照"""
        with self._lock:
            snapshot_path = self._snapshot_dir / snapshot_id
            if not snapshot_path.exists():
                return False
            shutil.rmtree(snapshot_path, ignore_errors=True)
            self._snapshots.pop(snapshot_id, None)
            return True

    def get_snapshot_info(self, snapshot_id: str) -> Optional[SnapshotInfo]:
        """获取快照信息"""
        with self._lock:
            if snapshot_id in self._snapshots:
                return self._snapshots[snapshot_id]
            snapshots = self.list_snapshots()
            for s in snapshots:
                if s.snapshot_id == snapshot_id:
                    return s
            return None

    def _safe_name(self, name: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


# ============================================================
# 持久化存储管理器
# ============================================================

class PersistentStoreManager:
    """持久化存储管理器"""

    def __init__(self, default_store: Optional[PersistentStore] = None):
        self._stores: Dict[str, PersistentStore] = {}
        self._default_name: Optional[str] = None
        self._lock = threading.RLock()
        self._snapshot_managers: Dict[str, SnapshotManager] = {}

        if default_store:
            self.register_store("default", default_store)

    def register_store(self, name: str, store: PersistentStore) -> None:
        """注册存储"""
        with self._lock:
            self._stores[name] = store
            if self._default_name is None:
                self._default_name = name

    def unregister_store(self, name: str) -> bool:
        """注销存储"""
        with self._lock:
            if name not in self._stores:
                return False
            self._stores[name].close()
            del self._stores[name]
            self._snapshot_managers.pop(name, None)
            if self._default_name == name:
                self._default_name = next(iter(self._stores), None)
            return True

    def get_store(self, name: Optional[str] = None) -> Optional[PersistentStore]:
        """获取存储"""
        with self._lock:
            store_name = name or self._default_name
            return self._stores.get(store_name) if store_name else None

    def get_snapshot_manager(self, store_name: Optional[str] = None) -> Optional[SnapshotManager]:
        """获取快照管理器"""
        with self._lock:
            name = store_name or self._default_name
            if name is None:
                return None
            if name not in self._snapshot_managers:
                store = self._stores.get(name)
                if store:
                    self._snapshot_managers[name] = SnapshotManager(store)
            return self._snapshot_managers.get(name)

    # 便捷操作
    def get(self, key: str, namespace: str = "default") -> Optional[StoreEntry]:
        store = self.get_store()
        return store.get(key, namespace) if store else None

    def put(self, key: str, value: Any, namespace: str = "default",
            ttl: Optional[float] = None, metadata: Optional[Dict] = None) -> bool:
        store = self.get_store()
        return store.put(key, value, namespace, ttl, metadata) if store else False

    def delete(self, key: str, namespace: str = "default") -> bool:
        store = self.get_store()
        return store.delete(key, namespace) if store else False

    def exists(self, key: str, namespace: str = "default") -> bool:
        store = self.get_store()
        return store.exists(key, namespace) if store else False

    def keys(self, namespace: str = "default") -> List[str]:
        store = self.get_store()
        return store.keys(namespace) if store else []

    def count(self, namespace: str = "default") -> int:
        store = self.get_store()
        return store.count(namespace) if store else 0

    def close_all(self) -> None:
        """关闭所有存储"""
        with self._lock:
            for store in self._stores.values():
                store.close()
            self._stores.clear()
            self._snapshot_managers.clear()
            self._default_name = None


# ============================================================
# 便捷函数
# ============================================================

def create_memory_store(config: Optional[StoreConfig] = None) -> MemoryPersistentStore:
    """创建内存持久化存储"""
    cfg = config or StoreConfig(store_type=PersistentStoreType.MEMORY)
    return MemoryPersistentStore(cfg)


def create_file_store(
    base_path: str = "./data/persistent",
    format: SerializationFormat = SerializationFormat.JSON,
    compression: CompressionType = CompressionType.NONE,
) -> FilePersistentStore:
    """创建文件持久化存储"""
    cfg = StoreConfig(
        store_type=PersistentStoreType.FILE_BINARY,
        base_path=base_path,
        serialization_format=format,
        compression=compression,
    )
    return FilePersistentStore(cfg)


def create_persistent_manager(
    default_store: Optional[PersistentStore] = None,
) -> PersistentStoreManager:
    """创建持久化存储管理器"""
    return PersistentStoreManager(default_store)
