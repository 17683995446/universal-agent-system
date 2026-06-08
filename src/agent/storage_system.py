# =============================================================================
# Agent存储与持久化 - 阶段28: Agent Storage & Persistence
# =============================================================================
#
# 功能：实现Agent存储与持久化系统，支持多种存储后端
# 原则：图难于其易，为大于其细
#
# =============================================================================

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional, Callable


# =============================================================================
# 存储后端类型
# =============================================================================

class StorageBackendType(str, Enum):
    """存储后端类型"""
    MEMORY = "memory"       # 内存存储
    FILE = "file"           # 文件存储


# =============================================================================
# 存储条目
# =============================================================================

@dataclass
class StorageEntry:
    """
    存储条目
    
    表示一个存储的数据项
    """
    key: str
    value: Any
    namespace: str = "default"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    ttl: Optional[float] = None  # 生存时间（秒），None表示永不过期
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def touch(self) -> None:
        """更新时间戳"""
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "key": self.key,
            "namespace": self.namespace,
            "value": self.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "ttl": self.ttl,
            "metadata": self.metadata,
        }


# =============================================================================
# 存储后端抽象接口
# =============================================================================

class StorageBackend(ABC):
    """
    存储后端抽象接口
    
    所有存储后端必须实现此接口
    """

    @abstractmethod
    def get(self, key: str, namespace: str = "default") -> Optional[StorageEntry]:
        """获取存储条目"""
        pass

    @abstractmethod
    def put(self, entry: StorageEntry) -> bool:
        """存储条目"""
        pass

    @abstractmethod
    def delete(self, key: str, namespace: str = "default") -> bool:
        """删除存储条目"""
        pass

    @abstractmethod
    def exists(self, key: str, namespace: str = "default") -> bool:
        """检查条目是否存在"""
        pass

    @abstractmethod
    def keys(self, namespace: str = "default") -> list[str]:
        """获取所有键"""
        pass

    @abstractmethod
    def count(self, namespace: str = "default") -> int:
        """获取条目数量"""
        pass

    @abstractmethod
    def clear(self, namespace: Optional[str] = None) -> int:
        """清除条目，返回清除数量"""
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭存储后端"""
        pass


# =============================================================================
# 内存存储
# =============================================================================

class MemoryStorage(StorageBackend):
    """
    内存存储后端
    
    数据存储在内存中，速度快但重启后丢失
    """

    def __init__(self):
        self._store: dict[str, StorageEntry] = {}
        self._lock = threading.RLock()

    def _make_key(self, key: str, namespace: str) -> str:
        """生成存储键"""
        return f"{namespace}:{key}"

    def get(self, key: str, namespace: str = "default") -> Optional[StorageEntry]:
        """获取存储条目"""
        with self._lock:
            store_key = self._make_key(key, namespace)
            entry = self._store.get(store_key)
            if entry is None:
                return None
            if entry.is_expired():
                del self._store[store_key]
                return None
            return entry

    def put(self, entry: StorageEntry) -> bool:
        """存储条目"""
        with self._lock:
            store_key = self._make_key(entry.key, entry.namespace)
            entry.touch()
            self._store[store_key] = entry
            return True

    def delete(self, key: str, namespace: str = "default") -> bool:
        """删除存储条目"""
        with self._lock:
            store_key = self._make_key(key, namespace)
            if store_key in self._store:
                del self._store[store_key]
                return True
            return False

    def exists(self, key: str, namespace: str = "default") -> bool:
        """检查条目是否存在"""
        with self._lock:
            store_key = self._make_key(key, namespace)
            entry = self._store.get(store_key)
            if entry is None:
                return False
            if entry.is_expired():
                del self._store[store_key]
                return False
            return True

    def keys(self, namespace: str = "default") -> list[str]:
        """获取所有键"""
        with self._lock:
            prefix = f"{namespace}:"
            result = []
            expired_keys = []
            for store_key, entry in self._store.items():
                if store_key.startswith(prefix):
                    if entry.is_expired():
                        expired_keys.append(store_key)
                    else:
                        result.append(entry.key)
            # 清理过期键
            for k in expired_keys:
                del self._store[k]
            return result

    def count(self, namespace: str = "default") -> int:
        """获取条目数量"""
        return len(self.keys(namespace))

    def clear(self, namespace: Optional[str] = None) -> int:
        """清除条目"""
        with self._lock:
            if namespace is None:
                count = len(self._store)
                self._store.clear()
                return count
            prefix = f"{namespace}:"
            to_delete = [k for k in self._store if k.startswith(prefix)]
            for k in to_delete:
                del self._store[k]
            return len(to_delete)

    def close(self) -> None:
        """关闭存储"""
        with self._lock:
            self._store.clear()


# =============================================================================
# 文件存储
# =============================================================================

class FileStorage(StorageBackend):
    """
    文件存储后端
    
    数据持久化到文件系统
    """

    def __init__(self, base_path: str | Path = "./data/storage"):
        self._base_path = Path(base_path)
        self._lock = threading.RLock()
        self._ensure_base_path()

    def _ensure_base_path(self) -> None:
        """确保基础路径存在"""
        self._base_path.mkdir(parents=True, exist_ok=True)

    def _namespace_path(self, namespace: str) -> Path:
        """获取命名空间路径"""
        # 清理命名空间中的特殊字符
        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in namespace
        )
        return self._base_path / safe_name

    def _entry_path(self, key: str, namespace: str) -> Path:
        """获取条目文件路径"""
        safe_key = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in key
        )
        return self._namespace_path(namespace) / f"{safe_key}.json"

    def get(self, key: str, namespace: str = "default") -> Optional[StorageEntry]:
        """获取存储条目"""
        with self._lock:
            path = self._entry_path(key, namespace)
            if not path.exists():
                return None

            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                entry = StorageEntry(
                    key=data["key"],
                    value=data["value"],
                    namespace=data["namespace"],
                    created_at=data["created_at"],
                    updated_at=data["updated_at"],
                    ttl=data.get("ttl"),
                    metadata=data.get("metadata", {}),
                )
                if entry.is_expired():
                    path.unlink(missing_ok=True)
                    return None
                return entry
            except (json.JSONDecodeError, KeyError):
                return None

    def put(self, entry: StorageEntry) -> bool:
        """存储条目"""
        with self._lock:
            ns_path = self._namespace_path(entry.namespace)
            ns_path.mkdir(parents=True, exist_ok=True)

            path = self._entry_path(entry.key, entry.namespace)
            entry.touch()

            data = entry.to_dict()
            # 序列化value（确保可JSON化）
            data["value"] = self._serialize_value(entry.value)

            try:
                path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return True
            except (IOError, TypeError):
                return False

    def delete(self, key: str, namespace: str = "default") -> bool:
        """删除存储条目"""
        with self._lock:
            path = self._entry_path(key, namespace)
            if path.exists():
                path.unlink()
                return True
            return False

    def exists(self, key: str, namespace: str = "default") -> bool:
        """检查条目是否存在"""
        with self._lock:
            path = self._entry_path(key, namespace)
            if not path.exists():
                return False
            entry = self.get(key, namespace)
            return entry is not None

    def keys(self, namespace: str = "default") -> list[str]:
        """获取所有键"""
        with self._lock:
            ns_path = self._namespace_path(namespace)
            if not ns_path.exists():
                return []
            result = []
            for file_path in ns_path.glob("*.json"):
                try:
                    data = json.loads(file_path.read_text(encoding="utf-8"))
                    entry = StorageEntry(
                        key=data["key"],
                        value=data["value"],
                        namespace=data["namespace"],
                        created_at=data["created_at"],
                        updated_at=data["updated_at"],
                        ttl=data.get("ttl"),
                    )
                    if entry.is_expired():
                        file_path.unlink(missing_ok=True)
                    else:
                        result.append(data["key"])
                except (json.JSONDecodeError, KeyError):
                    continue
            return result

    def count(self, namespace: str = "default") -> int:
        """获取条目数量"""
        return len(self.keys(namespace))

    def clear(self, namespace: Optional[str] = None) -> int:
        """清除条目"""
        with self._lock:
            if namespace is None:
                if self._base_path.exists():
                    count = sum(1 for _ in self._base_path.rglob("*.json"))
                    shutil.rmtree(self._base_path, ignore_errors=True)
                    self._ensure_base_path()
                    return count
                return 0
            ns_path = self._namespace_path(namespace)
            if ns_path.exists():
                count = sum(1 for _ in ns_path.glob("*.json"))
                shutil.rmtree(ns_path, ignore_errors=True)
                return count
            return 0

    def close(self) -> None:
        """关闭存储（文件存储无需特殊处理）"""
        pass

    def _serialize_value(self, value: Any) -> Any:
        """序列化值，确保可JSON化"""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._serialize_value(v) for k, v in value.items()}
        return str(value)


# =============================================================================
# 存储管理器
# =============================================================================

class StorageManager:
    """
    存储管理器
    
    统一管理多个存储后端，提供便捷的存储操作接口
    """

    def __init__(self, default_backend: Optional[StorageBackend] = None):
        self._backends: dict[str, StorageBackend] = {}
        self._default_name: Optional[str] = None
        self._lock = threading.RLock()

        if default_backend:
            self.register_backend("default", default_backend)

    def register_backend(self, name: str, backend: StorageBackend) -> None:
        """
        注册存储后端
        
        Args:
            name: 后端名称
            backend: 存储后端实例
        """
        with self._lock:
            self._backends[name] = backend
            if self._default_name is None:
                self._default_name = name

    def unregister_backend(self, name: str) -> bool:
        """
        注销存储后端
        
        Args:
            name: 后端名称
        
        Returns:
            是否成功
        """
        with self._lock:
            if name not in self._backends:
                return False
            self._backends[name].close()
            del self._backends[name]
            if self._default_name == name:
                self._default_name = next(iter(self._backends), None)
            return True

    def set_default(self, name: str) -> bool:
        """设置默认后端"""
        with self._lock:
            if name in self._backends:
                self._default_name = name
                return True
            return False

    def get_backend(self, name: Optional[str] = None) -> Optional[StorageBackend]:
        """获取存储后端"""
        with self._lock:
            backend_name = name or self._default_name
            if backend_name is None:
                return None
            return self._backends.get(backend_name)

    # =====================================================================
    # 便捷操作方法（使用默认后端）
    # =====================================================================

    def get(self, key: str, namespace: str = "default") -> Optional[StorageEntry]:
        """获取存储条目"""
        backend = self.get_backend()
        if backend:
            return backend.get(key, namespace)
        return None

    def get_value(self, key: str, namespace: str = "default", default: Any = None) -> Any:
        """获取存储值"""
        entry = self.get(key, namespace)
        return entry.value if entry else default

    def put(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        ttl: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> bool:
        """存储值"""
        backend = self.get_backend()
        if not backend:
            return False

        entry = StorageEntry(
            key=key,
            value=value,
            namespace=namespace,
            ttl=ttl,
            metadata=metadata or {},
        )
        return backend.put(entry)

    def delete(self, key: str, namespace: str = "default") -> bool:
        """删除存储条目"""
        backend = self.get_backend()
        if backend:
            return backend.delete(key, namespace)
        return False

    def exists(self, key: str, namespace: str = "default") -> bool:
        """检查条目是否存在"""
        backend = self.get_backend()
        if backend:
            return backend.exists(key, namespace)
        return False

    def keys(self, namespace: str = "default") -> list[str]:
        """获取所有键"""
        backend = self.get_backend()
        if backend:
            return backend.keys(namespace)
        return []

    def count(self, namespace: str = "default") -> int:
        """获取条目数量"""
        backend = self.get_backend()
        if backend:
            return backend.count(namespace)
        return 0

    def clear(self, namespace: Optional[str] = None) -> int:
        """清除条目"""
        backend = self.get_backend()
        if backend:
            return backend.clear(namespace)
        return 0

    # =====================================================================
    # 跨后端操作
    # =====================================================================

    def get_all_backends(self) -> dict[str, StorageBackend]:
        """获取所有后端"""
        with self._lock:
            return self._backends.copy()

    def get_backend_names(self) -> list[str]:
        """获取所有后端名称"""
        with self._lock:
            return list(self._backends.keys())

    def copy_between(
        self,
        key: str,
        from_backend: str,
        to_backend: str,
        namespace: str = "default",
    ) -> bool:
        """
        在后端间复制条目
        
        Args:
            key: 键
            from_backend: 源后端
            to_backend: 目标后端
            namespace: 命名空间
        
        Returns:
            是否成功
        """
        source = self.get_backend(from_backend)
        target = self.get_backend(to_backend)
        if not source or not target:
            return False

        entry = source.get(key, namespace)
        if entry is None:
            return False

        return target.put(entry)

    def sync_to(
        self,
        target_backend: str,
        namespace: str = "default",
    ) -> int:
        """
        将默认后端数据同步到目标后端
        
        Returns:
            同步的条目数
        """
        source = self.get_backend()
        target = self.get_backend(target_backend)
        if not source or not target:
            return 0

        count = 0
        for key in source.keys(namespace):
            entry = source.get(key, namespace)
            if entry:
                target.put(entry)
                count += 1
        return count

    def get_stats(self) -> dict[str, Any]:
        """
        获取存储统计
        
        Returns:
            统计信息
        """
        with self._lock:
            backend_stats = {}
            for name, backend in self._backends.items():
                backend_stats[name] = {
                    "type": type(backend).__name__,
                    "namespaces": self._get_namespaces(backend),
                    "total_keys": sum(backend.count(ns) for ns in self._get_namespaces(backend)),
                }

            return {
                "default_backend": self._default_name,
                "backends_count": len(self._backends),
                "backends": backend_stats,
            }

    def _get_namespaces(self, backend: StorageBackend) -> list[str]:
        """获取后端中的所有命名空间"""
        # 通过检查default命名空间来推断
        namespaces = ["default"]
        return namespaces

    def close_all(self) -> None:
        """关闭所有后端"""
        with self._lock:
            for backend in self._backends.values():
                backend.close()
            self._backends.clear()
            self._default_name = None


# =============================================================================
# 存储异常
# =============================================================================

class StorageError(Exception):
    """存储错误"""
    pass


class StorageNotFoundError(StorageError):
    """存储条目未找到"""
    pass


class StorageSerializationError(StorageError):
    """序列化错误"""
    pass