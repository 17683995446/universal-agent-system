"""
阶段40: Agent数据导出与导入系统 - 单元测试
"""

import json
import os
import pickle
import shutil
import tempfile

import pytest

from src.agent.data_portability import (
    ExportFormat,
    ConflictPolicy,
    ImportStatus,
    ExportResult,
    ImportResult,
    MigrationResult,
    ExportManifest,
    DataExporter,
    DataImporter,
    DataMigrator,
    create_exporter,
    create_importer,
    create_migrator,
)
from src.agent.persistent_store import (
    MemoryPersistentStore,
    create_memory_store,
    create_file_store,
)


class TestExportFormat:
    def test_values(self):
        assert ExportFormat.JSON == "json"
        assert ExportFormat.PICKLE == "pickle"
        assert ExportFormat.CSV == "csv"
        assert ExportFormat.ARCHIVE == "archive"


class TestConflictPolicy:
    def test_values(self):
        assert ConflictPolicy.SKIP == "skip"
        assert ConflictPolicy.OVERWRITE == "overwrite"
        assert ConflictPolicy.RENAME == "rename"
        assert ConflictPolicy.MERGE == "merge"


class TestImportStatus:
    def test_values(self):
        assert ImportStatus.SUCCESS == "success"
        assert ImportStatus.PARTIAL == "partial"
        assert ImportStatus.FAILED == "failed"


class TestExportResult:
    def test_success(self):
        r = ExportResult(success=True, format=ExportFormat.JSON, entry_count=10)
        assert r.success is True
        assert r.entry_count == 10

    def test_to_dict(self):
        r = ExportResult(success=True, format=ExportFormat.JSON, entry_count=5, total_size=1024)
        d = r.to_dict()
        assert d["format"] == "json"
        assert d["entry_count"] == 5


class TestImportResult:
    def test_success(self):
        r = ImportResult(status=ImportStatus.SUCCESS, imported_count=10)
        assert r.status == ImportStatus.SUCCESS

    def test_to_dict(self):
        r = ImportResult(status=ImportStatus.PARTIAL, imported_count=5, error_count=1)
        d = r.to_dict()
        assert d["status"] == "partial"


class TestMigrationResult:
    def test_to_dict(self):
        r = MigrationResult(success=True, migrated_count=10)
        d = r.to_dict()
        assert d["success"] is True


class TestExportManifest:
    def test_creation(self):
        m = ExportManifest(format=ExportFormat.ARCHIVE, source_type="MemoryPersistentStore")
        assert len(m.export_id) == 8
        assert m.source_type == "MemoryPersistentStore"

    def test_to_dict(self):
        m = ExportManifest(entry_count=10)
        d = m.to_dict()
        assert d["entry_count"] == 10


class TestDataExporter:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = create_memory_store()
        self.store.put("a", {"value": 1})
        self.store.put("b", {"value": 2}, namespace="ns1")
        self.exporter = DataExporter(self.store)

    def teardown_method(self):
        self.store.close()
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_export_to_json(self):
        path = os.path.join(self.tmpdir, "data.json")
        result = self.exporter.export_to_json(path)
        assert result.success is True
        assert result.entry_count == 2
        assert os.path.exists(path)
        data = json.loads(open(path).read())
        assert len(data) == 2

    def test_export_to_json_specific_ns(self):
        path = os.path.join(self.tmpdir, "ns1.json")
        result = self.exporter.export_to_json(path, namespaces=["ns1"])
        assert result.entry_count == 1

    def test_export_to_pickle(self):
        path = os.path.join(self.tmpdir, "data.pkl")
        result = self.exporter.export_to_pickle(path)
        assert result.success is True
        assert result.entry_count == 2
        data = pickle.loads(open(path, "rb").read())
        assert len(data) == 2

    def test_export_to_csv(self):
        path = os.path.join(self.tmpdir, "data.csv")
        result = self.exporter.export_to_csv(path)
        assert result.success is True
        assert result.entry_count == 2
        assert os.path.exists(path)

    def test_export_to_archive(self):
        path = os.path.join(self.tmpdir, "archive")
        result = self.exporter.export_to_archive(path)
        assert result.success is True
        assert result.entry_count == 2
        assert os.path.exists(os.path.join(path, "_manifest.json"))

    def test_export_empty_store(self):
        empty_store = create_memory_store()
        exporter = DataExporter(empty_store)
        path = os.path.join(self.tmpdir, "empty.json")
        result = exporter.export_to_json(path)
        assert result.success is True
        assert result.entry_count == 0
        empty_store.close()


class TestDataImporter:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = create_memory_store()
        self.importer = DataImporter(self.store)

    def teardown_method(self):
        self.store.close()
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_import_from_json(self):
        # 先导出
        src = create_memory_store()
        src.put("x", 100)
        src.put("y", 200)
        exporter = DataExporter(src)
        path = os.path.join(self.tmpdir, "data.json")
        exporter.export_to_json(path)
        src.close()

        # 再导入
        result = self.importer.import_from_json(path)
        assert result.status == ImportStatus.SUCCESS
        assert result.imported_count == 2
        assert self.store.get("x").value == 100
        assert self.store.get("y").value == 200

    def test_import_from_pickle(self):
        src = create_memory_store()
        src.put("a", "hello")
        exporter = DataExporter(src)
        path = os.path.join(self.tmpdir, "data.pkl")
        exporter.export_to_pickle(path)
        src.close()

        result = self.importer.import_from_pickle(path)
        assert result.imported_count == 1
        assert self.store.get("a").value == "hello"

    def test_import_from_csv(self):
        src = create_memory_store()
        src.put("key1", "val1")
        exporter = DataExporter(src)
        path = os.path.join(self.tmpdir, "data.csv")
        exporter.export_to_csv(path)
        src.close()

        result = self.importer.import_from_csv(path)
        assert result.imported_count == 1

    def test_import_conflict_skip(self):
        self.store.put("existing", "old")
        src = create_memory_store()
        src.put("existing", "new")
        exporter = DataExporter(src)
        path = os.path.join(self.tmpdir, "data.json")
        exporter.export_to_json(path)
        src.close()

        result = self.importer.import_from_json(path, conflict_policy=ConflictPolicy.SKIP)
        assert result.conflict_count == 1
        assert self.store.get("existing").value == "old"

    def test_import_conflict_overwrite(self):
        self.store.put("existing", "old")
        src = create_memory_store()
        src.put("existing", "new")
        exporter = DataExporter(src)
        path = os.path.join(self.tmpdir, "data.json")
        exporter.export_to_json(path)
        src.close()

        result = self.importer.import_from_json(path, conflict_policy=ConflictPolicy.OVERWRITE)
        assert result.conflict_count == 1
        assert result.imported_count == 1
        assert self.store.get("existing").value == "new"

    def test_import_conflict_rename(self):
        self.store.put("existing", "old")
        src = create_memory_store()
        src.put("existing", "new")
        exporter = DataExporter(src)
        path = os.path.join(self.tmpdir, "data.json")
        exporter.export_to_json(path)
        src.close()

        result = self.importer.import_from_json(path, conflict_policy=ConflictPolicy.RENAME)
        assert result.conflict_count == 1
        assert result.imported_count == 1
        assert self.store.get("existing").value == "old"

    def test_import_nonexistent_file(self):
        result = self.importer.import_from_json("/nonexistent/file.json")
        assert result.status == ImportStatus.FAILED

    def test_import_from_archive(self):
        src = create_memory_store()
        src.put("a", 1, namespace="ns1")
        exporter = DataExporter(src)
        path = os.path.join(self.tmpdir, "archive")
        exporter.export_to_archive(path)
        src.close()

        result = self.importer.import_from_archive(path)
        assert result.imported_count == 1

    def test_namespace_override(self):
        src = create_memory_store()
        src.put("key", "value", namespace="ns1")
        exporter = DataExporter(src)
        path = os.path.join(self.tmpdir, "data.json")
        exporter.export_to_json(path)
        src.close()

        result = self.importer.import_from_json(path, namespace_override="ns2")
        assert result.imported_count == 1
        assert self.store.get("key", "ns2") is not None


class TestDataMigrator:
    def test_migrate(self):
        source = create_memory_store()
        target = create_memory_store()
        source.put("a", 1)
        source.put("b", 2)
        source.put("c", 3, namespace="ns1")

        migrator = DataMigrator()
        result = migrator.migrate(source, target)
        assert result.success is True
        assert result.migrated_count == 3
        assert target.get("a").value == 1
        assert target.get("c", "ns1").value == 3

        source.close()
        target.close()

    def test_migrate_with_conflict_skip(self):
        source = create_memory_store()
        target = create_memory_store()
        source.put("a", "new")
        target.put("a", "old")

        migrator = DataMigrator()
        result = migrator.migrate(source, target, conflict_policy=ConflictPolicy.SKIP)
        assert result.migrated_count == 0
        assert target.get("a").value == "old"

        source.close()
        target.close()

    def test_migrate_specific_namespaces(self):
        source = create_memory_store()
        target = create_memory_store()
        source.put("a", 1, namespace="ns1")
        source.put("b", 2, namespace="ns2")

        migrator = DataMigrator()
        result = migrator.migrate(source, target, namespaces=["ns1"])
        assert result.migrated_count == 1
        assert target.get("a", "ns1").value == 1
        assert target.get("b", "ns2") is None

        source.close()
        target.close()

    def test_sync_incremental(self):
        source = create_memory_store()
        target = create_memory_store()
        source.put("a", 1)
        source.put("b", 2)

        migrator = DataMigrator()
        result = migrator.sync_incremental(source, target)
        assert result.migrated_count == 2

        # 更新源数据
        source.put("a", 999)
        source.put("c", 3)

        result2 = migrator.sync_incremental(source, target)
        assert result2.migrated_count == 2  # a updated + c new
        assert target.get("a").value == 999
        assert target.get("c").value == 3

        source.close()
        target.close()


class TestConvenienceFunctions:
    def test_create_exporter(self):
        store = create_memory_store()
        e = create_exporter(store)
        assert isinstance(e, DataExporter)
        store.close()

    def test_create_importer(self):
        store = create_memory_store()
        i = create_importer(store)
        assert isinstance(i, DataImporter)
        store.close()

    def test_create_migrator(self):
        m = create_migrator()
        assert isinstance(m, DataMigrator)


class TestIntegration:
    def test_full_export_import_cycle(self):
        tmpdir = tempfile.mkdtemp()
        try:
            # 创建源数据
            source = create_memory_store()
            source.put("user:1", {"name": "Alice", "age": 30})
            source.put("user:2", {"name": "Bob", "age": 25}, namespace="users")
            source.put("config", {"theme": "dark"})

            # 导出
            exporter = DataExporter(source)
            json_path = os.path.join(tmpdir, "export.json")
            result = exporter.export_to_json(json_path)
            assert result.success is True
            assert result.entry_count == 3

            # 导入到新存储
            target = create_memory_store()
            importer = DataImporter(target)
            import_result = importer.import_from_json(json_path)
            assert import_result.status == ImportStatus.SUCCESS
            assert import_result.imported_count == 3

            # 验证
            assert target.get("user:1").value["name"] == "Alice"
            assert target.get("user:2", "users").value["name"] == "Bob"

            source.close()
            target.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_cross_store_migration(self):
        tmpdir = tempfile.mkdtemp()
        try:
            # 内存 -> 文件
            mem = create_memory_store()
            mem.put("data", list(range(100)))

            file_store = create_file_store(base_path=os.path.join(tmpdir, "file"))
            migrator = DataMigrator()
            result = migrator.migrate(mem, file_store)
            assert result.success is True
            assert result.migrated_count == 1

            # 验证文件存储
            entry = file_store.get("data")
            assert entry is not None
            assert len(entry.value) == 100

            mem.close()
            file_store.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
