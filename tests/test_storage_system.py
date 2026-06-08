# =============================================================================
# 存储系统测试 - 阶段28测试
# =============================================================================

import time
import pytest
import tempfile
import shutil
from pathlib import Path

from src.agent.storage_system import (
    StorageManager,
    StorageBackend,
    MemoryStorage,
    FileStorage,
    StorageEntry,
    StorageBackendType,
    StorageError,
    StorageNotFoundError,
    StorageSerializationError,
)


class TestStorageBackendType:
    """存储后端类型枚举测试"""

    def test_enum_values(self):
        assert StorageBackendType.MEMORY.value == "memory"
        assert StorageBackendType.FILE.value == "file"


class TestStorageEntry:
    """存储条目测试"""

    def test_creation(self):
        entry = StorageEntry(
            key="key_1",
            value={"data": "test"},
            namespace="default",
            metadata={"source": "test"},
        )
        assert entry.key == "key_1"
        assert entry.value == {"data": "test"}
        assert entry.namespace == "default"

    def test_not_expired(self):
        entry = StorageEntry(key="key_1", value="data", ttl=60)
        assert entry.is_expired() is False

    def test_expired(self):
        entry = StorageEntry(key="key_1", value="data", ttl=0.01)
        time.sleep(0.02)
        assert entry.is_expired() is True

    def test_no_ttl_never_expires(self):
        entry = StorageEntry(key="key_1", value="data")
        time.sleep(0.01)
        assert entry.is_expired() is False

    def test_touch(self):
        entry = StorageEntry(key="key_1", value="data")
        old_time = entry.updated_at
        time.sleep(0.01)
        entry.touch()
        assert entry.updated_at > old_time

    def test_to_dict(self):
        entry = StorageEntry(
            key="key_1",
            value="data",
            namespace="ns1",
            ttl=60,
            metadata={"key": "val"},
        )
        result = entry.to_dict()
        assert result["key"] == "key_1"
        assert result["namespace"] == "ns1"
        assert result["ttl"] == 60
        assert result["metadata"]["key"] == "val"


class TestMemoryStorage:
    """内存存储测试"""

    def setup_method(self):
        self.storage = MemoryStorage()

    def test_put_and_get(self):
        entry = StorageEntry(key="key_1", value="value_1")
        self.storage.put(entry)

        result = self.storage.get("key_1")
        assert result is not None
        assert result.value == "value_1"

    def test_get_nonexistent(self):
        result = self.storage.get("nonexistent")
        assert result is None

    def test_delete(self):
        entry = StorageEntry(key="key_1", value="value_1")
        self.storage.put(entry)

        result = self.storage.delete("key_1")
        assert result is True
        assert self.storage.get("key_1") is None

    def test_delete_nonexistent(self):
        result = self.storage.delete("nonexistent")
        assert result is False

    def test_exists(self):
        entry = StorageEntry(key="key_1", value="value_1")
        self.storage.put(entry)

        assert self.storage.exists("key_1") is True
        assert self.storage.exists("nonexistent") is False

    def test_keys(self):
        self.storage.put(StorageEntry(key="key_1", value="v1"))
        self.storage.put(StorageEntry(key="key_2", value="v2"))
        self.storage.put(StorageEntry(key="key_3", value="v3", namespace="other"))

        keys = self.storage.keys()
        assert len(keys) == 2
        assert "key_1" in keys
        assert "key_2" in keys

    def test_count(self):
        self.storage.put(StorageEntry(key="key_1", value="v1"))
        self.storage.put(StorageEntry(key="key_2", value="v2"))

        assert self.storage.count() == 2

    def test_clear_namespace(self):
        self.storage.put(StorageEntry(key="key_1", value="v1", namespace="ns1"))
        self.storage.put(StorageEntry(key="key_2", value="v2", namespace="ns1"))
        self.storage.put(StorageEntry(key="key_3", value="v3", namespace="ns2"))

        count = self.storage.clear(namespace="ns1")
        assert count == 2
        assert self.storage.count(namespace="ns1") == 0
        assert self.storage.count(namespace="ns2") == 1

    def test_clear_all(self):
        self.storage.put(StorageEntry(key="key_1", value="v1"))
        self.storage.put(StorageEntry(key="key_2", value="v2"))

        count = self.storage.clear()
        assert count == 2
        assert self.storage.count() == 0

    def test_expired_entry(self):
        entry = StorageEntry(key="key_1", value="v1", ttl=0.01)
        self.storage.put(entry)
        time.sleep(0.02)

        result = self.storage.get("key_1")
        assert result is None

    def test_expired_entry_not_returned_in_keys(self):
        entry = StorageEntry(key="key_1", value="v1", ttl=0.01)
        self.storage.put(entry)
        time.sleep(0.02)

        keys = self.storage.keys()
        assert "key_1" not in keys

    def test_close(self):
        self.storage.put(StorageEntry(key="key_1", value="v1"))
        self.storage.close()
        assert self.storage.count() == 0


class TestFileStorage:
    """文件存储测试"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.storage = FileStorage(base_path=self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_put_and_get(self):
        entry = StorageEntry(key="key_1", value="value_1")
        self.storage.put(entry)

        result = self.storage.get("key_1")
        assert result is not None
        assert result.value == "value_1"
        assert result.key == "key_1"

    def test_get_nonexistent(self):
        result = self.storage.get("nonexistent")
        assert result is None

    def test_delete(self):
        entry = StorageEntry(key="key_1", value="value_1")
        self.storage.put(entry)

        result = self.storage.delete("key_1")
        assert result is True
        assert self.storage.get("key_1") is None

    def test_delete_nonexistent(self):
        result = self.storage.delete("nonexistent")
        assert result is False

    def test_exists(self):
        entry = StorageEntry(key="key_1", value="value_1")
        self.storage.put(entry)

        assert self.storage.exists("key_1") is True
        assert self.storage.exists("nonexistent") is False

    def test_keys(self):
        self.storage.put(StorageEntry(key="key_1", value="v1"))
        self.storage.put(StorageEntry(key="key_2", value="v2"))
        self.storage.put(StorageEntry(key="key_3", value="v3", namespace="other"))

        keys = self.storage.keys()
        assert len(keys) == 2
        assert "key_1" in keys

    def test_count(self):
        self.storage.put(StorageEntry(key="key_1", value="v1"))
        self.storage.put(StorageEntry(key="key_2", value="v2"))

        assert self.storage.count() == 2

    def test_clear_namespace(self):
        self.storage.put(StorageEntry(key="key_1", value="v1", namespace="ns1"))
        self.storage.put(StorageEntry(key="key_2", value="v2", namespace="ns1"))

        count = self.storage.clear(namespace="ns1")
        assert count == 2

    def test_clear_all(self):
        self.storage.put(StorageEntry(key="key_1", value="v1"))
        self.storage.put(StorageEntry(key="key_2", value="v2"))

        count = self.storage.clear()
        assert count == 2

    def test_expired_entry(self):
        entry = StorageEntry(key="key_1", value="v1", ttl=0.01)
        self.storage.put(entry)
        time.sleep(0.02)

        result = self.storage.get("key_1")
        assert result is None

    def test_complex_value(self):
        entry = StorageEntry(
            key="complex",
            value={"list": [1, 2, 3], "nested": {"key": "value"}},
        )
        self.storage.put(entry)

        result = self.storage.get("complex")
        assert result is not None
        assert result.value["list"] == [1, 2, 3]

    def test_file_created(self):
        entry = StorageEntry(key="key_1", value="value_1")
        self.storage.put(entry)

        ns_path = Path(self.temp_dir) / "default"
        assert ns_path.exists()
        assert (ns_path / "key_1.json").exists()

    def test_close(self):
        self.storage.close()
        # 文件存储close无特殊操作
        assert True


class TestStorageManager:
    """存储管理器测试"""

    def test_creation_with_default(self):
        backend = MemoryStorage()
        manager = StorageManager(default_backend=backend)
        assert manager.get_backend() is not None

    def test_creation_without_default(self):
        manager = StorageManager()
        assert manager.get_backend() is None

    def test_register_backend(self):
        manager = StorageManager()
        backend = MemoryStorage()
        manager.register_backend("memory", backend)

        assert manager.get_backend("memory") is not None
        assert manager.get_backend() is not None  # 自动设为默认

    def test_unregister_backend(self):
        manager = StorageManager()
        backend = MemoryStorage()
        manager.register_backend("memory", backend)

        result = manager.unregister_backend("memory")
        assert result is True
        assert manager.get_backend("memory") is None

    def test_set_default(self):
        manager = StorageManager()
        b1 = MemoryStorage()
        b2 = MemoryStorage()
        manager.register_backend("b1", b1)
        manager.register_backend("b2", b2)

        manager.set_default("b2")
        assert manager.get_backend() is b2

    def test_put_and_get_value(self):
        backend = MemoryStorage()
        manager = StorageManager(default_backend=backend)

        manager.put("key_1", "value_1")
        result = manager.get_value("key_1")

        assert result == "value_1"

    def test_get_value_default(self):
        backend = MemoryStorage()
        manager = StorageManager(default_backend=backend)

        result = manager.get_value("nonexistent", default="default_val")
        assert result == "default_val"

    def test_put_with_ttl_and_metadata(self):
        backend = MemoryStorage()
        manager = StorageManager(default_backend=backend)

        manager.put("key_1", "value_1", ttl=60, metadata={"source": "test"})

        entry = manager.get("key_1")
        assert entry is not None
        assert entry.ttl == 60
        assert entry.metadata["source"] == "test"

    def test_delete(self):
        backend = MemoryStorage()
        manager = StorageManager(default_backend=backend)

        manager.put("key_1", "value_1")
        result = manager.delete("key_1")

        assert result is True
        assert manager.get_value("key_1") is None

    def test_exists(self):
        backend = MemoryStorage()
        manager = StorageManager(default_backend=backend)

        manager.put("key_1", "value_1")
        assert manager.exists("key_1") is True
        assert manager.exists("nonexistent") is False

    def test_keys(self):
        backend = MemoryStorage()
        manager = StorageManager(default_backend=backend)

        manager.put("key_1", "v1")
        manager.put("key_2", "v2")

        keys = manager.keys()
        assert len(keys) == 2

    def test_count(self):
        backend = MemoryStorage()
        manager = StorageManager(default_backend=backend)

        manager.put("key_1", "v1")
        manager.put("key_2", "v2")

        assert manager.count() == 2

    def test_clear(self):
        backend = MemoryStorage()
        manager = StorageManager(default_backend=backend)

        manager.put("key_1", "v1")
        manager.put("key_2", "v2")
        count = manager.clear()

        assert count == 2
        assert manager.count() == 0

    def test_copy_between(self):
        manager = StorageManager()
        b1 = MemoryStorage()
        b2 = MemoryStorage()
        manager.register_backend("b1", b1)
        manager.register_backend("b2", b2)

        manager.put("key_1", "value_1", namespace="ns1")
        # 直接操作b1（因为默认是b1）
        b1.put(StorageEntry(key="key_1", value="value_1", namespace="ns1"))

        result = manager.copy_between("key_1", "b1", "b2", namespace="ns1")
        assert result is True

        entry = b2.get("key_1", "ns1")
        assert entry is not None
        assert entry.value == "value_1"

    def test_sync_to(self):
        manager = StorageManager()
        b1 = MemoryStorage()
        b2 = MemoryStorage()
        manager.register_backend("b1", b1)
        manager.register_backend("b2", b2)
        manager.set_default("b1")

        b1.put(StorageEntry(key="key_1", value="v1"))
        b1.put(StorageEntry(key="key_2", value="v2"))

        count = manager.sync_to("b2")
        assert count == 2
        assert b2.count() == 2

    def test_get_stats(self):
        manager = StorageManager()
        b1 = MemoryStorage()
        manager.register_backend("memory", b1)

        b1.put(StorageEntry(key="key_1", value="v1"))

        stats = manager.get_stats()
        assert stats["default_backend"] == "memory"
        assert stats["backends_count"] == 1
        assert "memory" in stats["backends"]

    def test_get_backend_names(self):
        manager = StorageManager()
        manager.register_backend("b1", MemoryStorage())
        manager.register_backend("b2", MemoryStorage())

        names = manager.get_backend_names()
        assert sorted(names) == ["b1", "b2"]

    def test_close_all(self):
        manager = StorageManager()
        manager.register_backend("b1", MemoryStorage())
        manager.register_backend("b2", MemoryStorage())

        manager.close_all()
        assert manager.get_backend_names() == []

    def test_no_backend_operations(self):
        manager = StorageManager()

        assert manager.get("key_1") is None
        assert manager.get_value("key_1") is None
        assert manager.put("key_1", "v1") is False
        assert manager.delete("key_1") is False
        assert manager.exists("key_1") is False
        assert manager.keys() == []
        assert manager.count() == 0
        assert manager.clear() == 0


class TestStorageExceptions:
    """存储异常测试"""

    def test_storage_error(self):
        error = StorageError("存储错误")
        assert str(error) == "存储错误"

    def test_storage_not_found_error(self):
        error = StorageNotFoundError("未找到")
        assert str(error) == "未找到"
        assert isinstance(error, StorageError)

    def test_storage_serialization_error(self):
        error = StorageSerializationError("序列化失败")
        assert str(error) == "序列化失败"
        assert isinstance(error, StorageError)


class TestExportStorageSystem:
    """导出测试"""

    def test_storage_manager_exported(self):
        from src.agent import StorageManager
        assert StorageManager is not None

    def test_storage_backend_exported(self):
        from src.agent import StorageBackend
        assert StorageBackend is not None

    def test_memory_storage_exported(self):
        from src.agent import MemoryStorage
        assert MemoryStorage is not None

    def test_file_storage_exported(self):
        from src.agent import FileStorage
        assert FileStorage is not None

    def test_storage_entry_exported(self):
        from src.agent import StorageEntry
        assert StorageEntry is not None

    def test_storage_backend_type_exported(self):
        from src.agent import StorageBackendType
        assert StorageBackendType.MEMORY is not None

    def test_storage_error_exported(self):
        from src.agent import StorageError
        assert StorageError is not None

    def test_storage_not_found_error_exported(self):
        from src.agent import StorageNotFoundError
        assert StorageNotFoundError is not None

    def test_storage_serialization_error_exported(self):
        from src.agent import StorageSerializationError
        assert StorageSerializationError is not None


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])