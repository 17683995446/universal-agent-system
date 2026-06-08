"""
阶段39: Agent数据持久化存储系统 - 单元测试

遵循"图难于其易，为大于其细"的原则，全面测试持久化存储功能。
"""

import os
import shutil
import tempfile
import time

import pytest

from src.agent.persistent_store import (
    PersistentStoreType,
    StoreStatus,
    StoreEntry,
    StoreConfig,
    SnapshotInfo,
    RecoveryResult,
    StoreStats,
    MemoryPersistentStore,
    FilePersistentStore,
    SnapshotManager,
    PersistentStoreManager,
    create_memory_store,
    create_file_store,
    create_persistent_manager,
)
from src.agent.serialization import SerializationFormat, CompressionType


# ============================================================
# 39.1 持久化模型测试
# ============================================================

class TestPersistentStoreType:
    """测试持久化存储类型枚举"""

    def test_values(self):
        assert PersistentStoreType.MEMORY == "memory"
        assert PersistentStoreType.FILE_BINARY == "file_binary"
        assert PersistentStoreType.FILE_JSON == "file_json"


class TestStoreStatus:
    """测试存储状态枚举"""

    def test_values(self):
        assert StoreStatus.ACTIVE == "active"
        assert StoreStatus.READ_ONLY == "read_only"
        assert StoreStatus.CLOSED == "closed"
        assert StoreStatus.ERROR == "error"


class TestStoreEntry:
    """测试存储条目"""

    def test_creation(self):
        entry = StoreEntry(key="test", value="hello")
        assert entry.key == "test"
        assert entry.value == "hello"
        assert entry.namespace == "default"
        assert entry.version == 1
        assert isinstance(entry.created_at, float)
        assert isinstance(entry.updated_at, float)

    def test_is_expired_no_ttl(self):
        entry = StoreEntry(key="test", value="hello", ttl=None)
        assert not entry.is_expired()

    def test_is_expired_with_ttl(self):
        entry = StoreEntry(key="test", value="hello", ttl=0.001)
        time.sleep(0.01)
        assert entry.is_expired()

    def test_is_expired_not_yet(self):
        entry = StoreEntry(key="test", value="hello", ttl=100)
        assert not entry.is_expired()

    def test_compute_checksum(self):
        entry = StoreEntry(key="test", value="hello")
        checksum = entry.compute_checksum(b"hello")
        assert len(checksum) == 16
        assert entry.checksum == checksum

    def test_touch(self):
        entry = StoreEntry(key="test", value="hello")
        old_version = entry.version
        old_time = entry.updated_at
        time.sleep(0.01)
        entry.touch()
        assert entry.version == old_version + 1
        assert entry.updated_at > old_time

    def test_to_dict(self):
        entry = StoreEntry(
            key="test", value="hello", namespace="ns1",
            metadata={"tag": "important"},
        )
        d = entry.to_dict()
        assert d["key"] == "test"
        assert d["namespace"] == "ns1"
        assert d["version"] == 1
        assert d["metadata"] == {"tag": "important"}


class TestStoreConfig:
    """测试存储配置"""

    def test_defaults(self):
        config = StoreConfig()
        assert config.store_type == PersistentStoreType.MEMORY
        assert config.base_path == "./data/persistent"
        assert config.serialization_format == SerializationFormat.JSON
        assert config.compression == CompressionType.NONE
        assert config.max_entries == 100000
        assert config.enable_checksum is True
        assert config.enable_versioning is True


class TestSnapshotInfo:
    """测试快照信息"""

    def test_creation(self):
        info = SnapshotInfo(description="test snapshot")
        assert len(info.snapshot_id) == 8
        assert info.description == "test snapshot"
        assert info.entry_count == 0

    def test_to_dict(self):
        info = SnapshotInfo(entry_count=10, total_size=1024)
        d = info.to_dict()
        assert d["entry_count"] == 10
        assert d["total_size"] == 1024


class TestRecoveryResult:
    """测试恢复结果"""

    def test_success(self):
        result = RecoveryResult(success=True, recovered_entries=10)
        assert result.success is True
        assert result.recovered_entries == 10
        assert result.failed_entries == 0

    def test_failure(self):
        result = RecoveryResult(
            success=False,
            errors=["entry missing"],
            snapshot_id="abc123",
        )
        assert result.success is False
        assert "entry missing" in result.errors
        assert result.snapshot_id == "abc123"


class TestStoreStats:
    """测试存储统计"""

    def test_defaults(self):
        stats = StoreStats()
        assert stats.total_entries == 0
        assert stats.total_size_bytes == 0

    def test_to_dict(self):
        stats = StoreStats(total_entries=5, total_size_bytes=1024)
        d = stats.to_dict()
        assert d["total_entries"] == 5
        assert d["total_size_bytes"] == 1024


# ============================================================
# 39.2 内存持久化存储测试
# ============================================================

class TestMemoryPersistentStore:
    """测试内存持久化存储"""

    def setup_method(self):
        self.store = MemoryPersistentStore()

    def test_put_and_get(self):
        assert self.store.put("key1", "value1") is True
        entry = self.store.get("key1")
        assert entry is not None
        assert entry.value == "value1"

    def test_put_complex_value(self):
        data = {"list": [1, 2, 3], "nested": {"a": "b"}}
        assert self.store.put("complex", data) is True
        entry = self.store.get("complex")
        assert entry.value == data

    def test_get_nonexistent(self):
        assert self.store.get("nope") is None

    def test_delete(self):
        self.store.put("key1", "value1")
        assert self.store.delete("key1") is True
        assert self.store.get("key1") is None

    def test_delete_nonexistent(self):
        assert self.store.delete("nope") is False

    def test_exists(self):
        self.store.put("key1", "value1")
        assert self.store.exists("key1") is True
        assert self.store.exists("nope") is False

    def test_keys(self):
        self.store.put("a", 1)
        self.store.put("b", 2)
        self.store.put("c", 3)
        keys = self.store.keys()
        assert set(keys) == {"a", "b", "c"}

    def test_count(self):
        self.store.put("a", 1)
        self.store.put("b", 2)
        assert self.store.count() == 2

    def test_namespaces(self):
        self.store.put("a", 1, namespace="ns1")
        self.store.put("b", 2, namespace="ns2")
        ns_list = self.store.namespaces()
        assert "ns1" in ns_list
        assert "ns2" in ns_list

    def test_namespace_isolation(self):
        self.store.put("key", "ns1_value", namespace="ns1")
        self.store.put("key", "ns2_value", namespace="ns2")
        assert self.store.get("key", "ns1").value == "ns1_value"
        assert self.store.get("key", "ns2").value == "ns2_value"

    def test_clear_namespace(self):
        self.store.put("a", 1, namespace="ns1")
        self.store.put("b", 2, namespace="ns2")
        count = self.store.clear(namespace="ns1")
        assert count == 1
        assert self.store.count("ns1") == 0
        assert self.store.count("ns2") == 1

    def test_clear_all(self):
        self.store.put("a", 1)
        self.store.put("b", 2)
        count = self.store.clear()
        assert count == 2
        assert self.store.count() == 0

    def test_ttl_expiry(self):
        self.store.put("temp", "data", ttl=0.001)
        time.sleep(0.01)
        assert self.store.get("temp") is None

    def test_metadata(self):
        self.store.put("key", "value", metadata={"tag": "test"})
        entry = self.store.get("key")
        assert entry.metadata == {"tag": "test"}

    def test_versioning(self):
        self.store.put("key", "v1")
        self.store.put("key", "v2")
        entry = self.store.get("key")
        assert entry.value == "v2"
        assert entry.version == 2

    def test_max_entries(self):
        config = StoreConfig(max_entries=2)
        store = MemoryPersistentStore(config)
        assert store.put("a", 1) is True
        assert store.put("b", 2) is True
        assert store.put("c", 3) is False  # 超出容量

    def test_flush(self):
        assert self.store.flush() is True

    def test_close(self):
        self.store.put("key", "value")
        self.store.close()
        assert self.store.get("key") is None
        assert self.store.status == StoreStatus.CLOSED

    def test_closed_store_operations(self):
        self.store.close()
        assert self.store.put("key", "value") is False
        assert self.store.get("key") is None
        assert self.store.delete("key") is False

    def test_stats(self):
        self.store.put("a", 1)
        self.store.put("b", 2)
        stats = self.store.get_stats()
        assert stats.total_entries == 2
        assert stats.total_size_bytes > 0
        assert stats.namespace_count >= 1

    def test_checksum_enabled(self):
        config = StoreConfig(enable_checksum=True)
        store = MemoryPersistentStore(config)
        store.put("key", "value")
        entry = store.get("key")
        assert len(entry.checksum) == 16

    def test_checksum_disabled(self):
        config = StoreConfig(enable_checksum=False)
        store = MemoryPersistentStore(config)
        store.put("key", "value")
        entry = store.get("key")
        assert entry.checksum == ""


# ============================================================
# 39.3 文件持久化存储测试
# ============================================================

class TestFilePersistentStore:
    """测试文件持久化存储"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = create_file_store(
            base_path=self.tmpdir,
            format=SerializationFormat.JSON,
            compression=CompressionType.NONE,
        )

    def teardown_method(self):
        self.store.close()
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_put_and_get(self):
        assert self.store.put("key1", "value1") is True
        entry = self.store.get("key1")
        assert entry is not None
        assert entry.value == "value1"

    def test_put_complex_value(self):
        data = {"list": [1, 2, 3], "nested": {"a": "b"}}
        assert self.store.put("complex", data) is True
        entry = self.store.get("complex")
        assert entry.value == data

    def test_get_nonexistent(self):
        assert self.store.get("nope") is None

    def test_delete(self):
        self.store.put("key1", "value1")
        assert self.store.delete("key1") is True
        assert self.store.get("key1") is None

    def test_exists(self):
        self.store.put("key1", "value1")
        assert self.store.exists("key1") is True
        assert self.store.exists("nope") is False

    def test_keys(self):
        self.store.put("a", 1)
        self.store.put("b", 2)
        keys = self.store.keys()
        assert set(keys) == {"a", "b"}

    def test_count(self):
        self.store.put("a", 1)
        self.store.put("b", 2)
        assert self.store.count() == 2

    def test_namespaces(self):
        self.store.put("a", 1, namespace="ns1")
        self.store.put("b", 2, namespace="ns2")
        ns_list = self.store.namespaces()
        assert "ns1" in ns_list
        assert "ns2" in ns_list

    def test_namespace_isolation(self):
        self.store.put("key", "ns1_value", namespace="ns1")
        self.store.put("key", "ns2_value", namespace="ns2")
        assert self.store.get("key", "ns1").value == "ns1_value"
        assert self.store.get("key", "ns2").value == "ns2_value"

    def test_clear_namespace(self):
        self.store.put("a", 1, namespace="ns1")
        self.store.put("b", 2, namespace="ns2")
        count = self.store.clear(namespace="ns1")
        assert count == 1
        assert self.store.count("ns1") == 0

    def test_clear_all(self):
        self.store.put("a", 1)
        self.store.put("b", 2)
        count = self.store.clear()
        assert count == 2
        assert self.store.count() == 0

    def test_metadata(self):
        self.store.put("key", "value", metadata={"tag": "test"})
        entry = self.store.get("key")
        assert entry.metadata == {"tag": "test"}

    def test_persistence_across_instances(self):
        """测试跨实例持久化"""
        self.store.put("key", "value")
        self.store.close()

        # 重新打开
        new_store = create_file_store(
            base_path=self.tmpdir,
            format=SerializationFormat.JSON,
        )
        entry = new_store.get("key")
        assert entry is not None
        assert entry.value == "value"
        new_store.close()

    def test_flush(self):
        assert self.store.flush() is True

    def test_close(self):
        self.store.put("key", "value")
        self.store.close()
        assert self.store.status == StoreStatus.CLOSED

    def test_stats(self):
        self.store.put("a", 1)
        self.store.put("b", 2)
        stats = self.store.get_stats()
        assert stats.total_entries == 2
        assert stats.total_size_bytes > 0

    def test_gzip_compression(self):
        store = create_file_store(
            base_path=self.tmpdir + "_gz",
            format=SerializationFormat.JSON,
            compression=CompressionType.GZIP,
        )
        try:
            data = {"text": "hello world " * 100}
            assert store.put("key", data) is True
            entry = store.get("key")
            assert entry.value == data
        finally:
            store.close()
            if os.path.exists(self.tmpdir + "_gz"):
                shutil.rmtree(self.tmpdir + "_gz", ignore_errors=True)


# ============================================================
# 39.4 快照管理器测试
# ============================================================

class TestSnapshotManager:
    """测试快照管理器"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.snapshot_dir = os.path.join(self.tmpdir, "snapshots")
        self.store = MemoryPersistentStore()
        self.manager = SnapshotManager(self.store, self.snapshot_dir)

    def teardown_method(self):
        self.store.close()
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_snapshot(self):
        self.store.put("a", 1)
        self.store.put("b", 2)
        info = self.manager.create_snapshot(description="test")
        assert info.entry_count == 2
        assert info.total_size > 0
        assert len(info.snapshot_id) == 8

    def test_restore_snapshot(self):
        self.store.put("a", 1)
        self.store.put("b", 2)
        info = self.manager.create_snapshot()

        # 清空存储
        self.store.clear()
        assert self.store.count() == 0

        # 恢复
        result = self.manager.restore_snapshot(info.snapshot_id)
        assert result.success is True
        assert result.recovered_entries == 2
        assert self.store.count() == 2

    def test_restore_to_different_store(self):
        self.store.put("a", 1)
        info = self.manager.create_snapshot()

        target = MemoryPersistentStore()
        result = self.manager.restore_snapshot(info.snapshot_id, target_store=target)
        assert result.success is True
        assert target.get("a").value == 1
        target.close()

    def test_restore_nonexistent(self):
        result = self.manager.restore_snapshot("nonexistent")
        assert result.success is False
        assert len(result.errors) > 0

    def test_list_snapshots(self):
        self.store.put("a", 1)
        self.manager.create_snapshot("first")
        self.manager.create_snapshot("second")
        snapshots = self.manager.list_snapshots()
        assert len(snapshots) == 2

    def test_delete_snapshot(self):
        self.store.put("a", 1)
        info = self.manager.create_snapshot()
        assert self.manager.delete_snapshot(info.snapshot_id) is True
        assert len(self.manager.list_snapshots()) == 0

    def test_delete_nonexistent_snapshot(self):
        assert self.manager.delete_snapshot("nonexistent") is False

    def test_get_snapshot_info(self):
        self.store.put("a", 1)
        info = self.manager.create_snapshot("test info")
        retrieved = self.manager.get_snapshot_info(info.snapshot_id)
        assert retrieved is not None
        assert retrieved.description == "test info"

    def test_empty_snapshot(self):
        info = self.manager.create_snapshot("empty")
        assert info.entry_count == 0

    def test_snapshot_with_namespaces(self):
        self.store.put("a", 1, namespace="ns1")
        self.store.put("b", 2, namespace="ns2")
        info = self.manager.create_snapshot(namespaces=["ns1"])
        assert info.entry_count == 1


# ============================================================
# 持久化存储管理器测试
# ============================================================

class TestPersistentStoreManager:
    """测试持久化存储管理器"""

    def test_create_with_default(self):
        store = MemoryPersistentStore()
        manager = PersistentStoreManager(store)
        assert manager.get("key") is None
        manager.close_all()

    def test_register_and_get(self):
        manager = PersistentStoreManager()
        store = MemoryPersistentStore()
        manager.register_store("mem", store)
        assert manager.get_store("mem") is store
        manager.close_all()

    def test_unregister(self):
        manager = PersistentStoreManager()
        store = MemoryPersistentStore()
        manager.register_store("mem", store)
        assert manager.unregister_store("mem") is True
        assert manager.get_store("mem") is None

    def test_convenience_operations(self):
        store = MemoryPersistentStore()
        manager = PersistentStoreManager(store)
        assert manager.put("key", "value") is True
        entry = manager.get("key")
        assert entry.value == "value"
        assert manager.exists("key") is True
        assert manager.count() == 1
        assert manager.delete("key") is True
        assert manager.count() == 0
        manager.close_all()

    def test_snapshot_manager(self):
        store = MemoryPersistentStore()
        store.put("a", 1)
        manager = PersistentStoreManager(store)
        sm = manager.get_snapshot_manager()
        assert sm is not None
        info = sm.create_snapshot("test")
        assert info.entry_count == 1
        manager.close_all()

    def test_close_all(self):
        manager = PersistentStoreManager()
        manager.register_store("s1", MemoryPersistentStore())
        manager.register_store("s2", MemoryPersistentStore())
        manager.close_all()
        assert manager.get_store() is None


# ============================================================
# 便捷函数测试
# ============================================================

class TestConvenienceFunctions:
    """测试便捷函数"""

    def test_create_memory_store(self):
        store = create_memory_store()
        assert isinstance(store, MemoryPersistentStore)
        store.close()

    def test_create_file_store(self):
        tmpdir = tempfile.mkdtemp()
        try:
            store = create_file_store(base_path=tmpdir)
            assert isinstance(store, FilePersistentStore)
            store.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_create_persistent_manager(self):
        manager = create_persistent_manager()
        assert isinstance(manager, PersistentStoreManager)
        manager.close_all()

    def test_create_memory_store_with_config(self):
        config = StoreConfig(max_entries=50, enable_checksum=False)
        store = create_memory_store(config)
        assert isinstance(store, MemoryPersistentStore)
        store.close()


# ============================================================
# 集成测试
# ============================================================

class TestIntegration:
    """集成测试"""

    def test_full_memory_workflow(self):
        """测试完整内存工作流"""
        store = create_memory_store()
        try:
            # 写入数据
            store.put("user:1", {"name": "Alice", "age": 30})
            store.put("user:2", {"name": "Bob", "age": 25}, namespace="users")
            store.put("config", {"theme": "dark"}, metadata={"source": "api"})

            # 读取验证
            assert store.get("user:1").value["name"] == "Alice"
            assert store.get("user:2", "users").value["name"] == "Bob"
            assert store.get("config").metadata["source"] == "api"

            # 统计
            stats = store.get_stats()
            assert stats.total_entries == 3

            # 清理
            store.clear()
            assert store.count() == 0
        finally:
            store.close()

    def test_full_file_workflow(self):
        """测试完整文件工作流"""
        tmpdir = tempfile.mkdtemp()
        try:
            store = create_file_store(base_path=tmpdir)
            store.put("data", {"items": list(range(100))})
            store.put("meta", {"version": "1.0"})

            # 验证持久化
            store.close()
            store2 = create_file_store(base_path=tmpdir)
            entry = store2.get("data")
            assert entry is not None
            assert len(entry.value["items"]) == 100
            store2.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_snapshot_and_restore_workflow(self):
        """测试快照与恢复工作流"""
        tmpdir = tempfile.mkdtemp()
        try:
            store = MemoryPersistentStore()
            snapshot_dir = os.path.join(tmpdir, "snaps")
            manager = SnapshotManager(store, snapshot_dir)

            # 写入数据
            store.put("a", {"value": 100})
            store.put("b", {"value": 200})

            # 创建快照
            info = manager.create_snapshot("before modification")

            # 修改数据
            store.put("a", {"value": 999})
            assert store.get("a").value["value"] == 999

            # 恢复快照
            result = manager.restore_snapshot(info.snapshot_id)
            assert result.success is True
            assert store.get("a").value["value"] == 100

            store.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_multi_store_manager_workflow(self):
        """测试多存储管理器工作流"""
        manager = PersistentStoreManager()
        mem_store = create_memory_store()
        tmpdir = tempfile.mkdtemp()
        try:
            file_store = create_file_store(base_path=tmpdir)

            manager.register_store("memory", mem_store)
            manager.register_store("file", file_store)

            # 写入内存
            manager.put("key", "memory_value")
            assert manager.get("key").value == "memory_value"

            # 切换到文件
            mem_store.put("key", "memory_value")
            file_store.put("key", "file_value")

            mem_entry = mem_store.get("key")
            file_entry = file_store.get("key")
            assert mem_entry.value == "memory_value"
            assert file_entry.value == "file_value"

        finally:
            manager.close_all()
            shutil.rmtree(tmpdir, ignore_errors=True)
