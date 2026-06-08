"""
阶段40: Agent数据导出与导入系统

遵循"图难于其易，为大于其细"的原则，实现数据导出、导入、跨存储迁移。
"""

import csv
import io
import json
import os
import pickle
import shutil
import threading
import time
import uuid
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
)
from src.agent.persistent_store import (
    PersistentStore,
    MemoryPersistentStore,
    FilePersistentStore,
    StoreEntry,
    StoreConfig,
)


# ============================================================
# 40.1 导出导入模型
# ============================================================

class ExportFormat(str, Enum):
    """导出格式"""
    JSON = "json"
    PICKLE = "pickle"
    CSV = "csv"
    ARCHIVE = "archive"  # 打包归档


class ConflictPolicy(str, Enum):
    """导入冲突策略"""
    SKIP = "skip"         # 跳过冲突
    OVERWRITE = "overwrite"  # 覆盖
    RENAME = "rename"     # 重命名
    MERGE = "merge"       # 合并


class ImportStatus(str, Enum):
    """导入状态"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class ExportResult:
    """导出结果"""
    success: bool
    format: ExportFormat
    entry_count: int = 0
    total_size: int = 0
    output_path: str = ""
    duration: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "format": self.format.value,
            "entry_count": self.entry_count,
            "total_size": self.total_size,
            "output_path": self.output_path,
            "duration": round(self.duration, 4),
            "errors": self.errors,
        }


@dataclass
class ImportResult:
    """导入结果"""
    status: ImportStatus
    imported_count: int = 0
    skipped_count: int = 0
    conflict_count: int = 0
    error_count: int = 0
    errors: List[str] = field(default_factory=list)
    duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "imported_count": self.imported_count,
            "skipped_count": self.skipped_count,
            "conflict_count": self.conflict_count,
            "error_count": self.error_count,
            "errors": self.errors,
            "duration": round(self.duration, 4),
        }


@dataclass
class MigrationResult:
    """迁移结果"""
    success: bool
    source_store: str = ""
    target_store: str = ""
    migrated_count: int = 0
    failed_count: int = 0
    errors: List[str] = field(default_factory=list)
    duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "source_store": self.source_store,
            "target_store": self.target_store,
            "migrated_count": self.migrated_count,
            "failed_count": self.failed_count,
            "errors": self.errors,
            "duration": round(self.duration, 4),
        }


@dataclass
class ExportManifest:
    """导出清单"""
    export_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: float = field(default_factory=time.time)
    format: ExportFormat = ExportFormat.JSON
    source_type: str = ""
    entry_count: int = 0
    namespaces: List[str] = field(default_factory=list)
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "export_id": self.export_id,
            "created_at": self.created_at,
            "format": self.format.value,
            "source_type": self.source_type,
            "entry_count": self.entry_count,
            "namespaces": self.namespaces,
            "version": self.version,
        }


# ============================================================
# 40.2 数据导出器
# ============================================================

class DataExporter:
    """数据导出器"""

    def __init__(self, store: PersistentStore):
        self._store = store
        self._serializer = SerializationManager()

    def export_to_json(
        self,
        output_path: str,
        namespaces: Optional[List[str]] = None,
        pretty: bool = True,
    ) -> ExportResult:
        """导出为JSON文件"""
        start = time.time()
        errors = []
        entries_data = []

        target_ns = namespaces or self._store.namespaces()
        for ns in target_ns:
            keys = self._store.keys(ns)
            for key in keys:
                entry = self._store.get(key, ns)
                if entry is None:
                    continue
                try:
                    entries_data.append({
                        "key": entry.key,
                        "namespace": entry.namespace,
                        "value": entry.value,
                        "metadata": entry.metadata,
                        "created_at": entry.created_at,
                        "updated_at": entry.updated_at,
                        "version": entry.version,
                    })
                except Exception as e:
                    errors.append(f"导出 {ns}/{key} 失败: {e}")

        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            indent = 2 if pretty else None
            content = json.dumps(entries_data, ensure_ascii=False, indent=indent, default=str)
            path.write_text(content, encoding="utf-8")
            total_size = len(content.encode("utf-8"))
        except Exception as e:
            return ExportResult(
                success=False, format=ExportFormat.JSON,
                errors=[f"写入文件失败: {e}"], duration=time.time() - start,
            )

        return ExportResult(
            success=True,
            format=ExportFormat.JSON,
            entry_count=len(entries_data),
            total_size=total_size,
            output_path=str(path.absolute()),
            duration=time.time() - start,
            errors=errors,
        )

    def export_to_pickle(
        self,
        output_path: str,
        namespaces: Optional[List[str]] = None,
    ) -> ExportResult:
        """导出为Pickle文件"""
        start = time.time()
        errors = []
        entries_data = []

        target_ns = namespaces or self._store.namespaces()
        for ns in target_ns:
            keys = self._store.keys(ns)
            for key in keys:
                entry = self._store.get(key, ns)
                if entry is None:
                    continue
                try:
                    entries_data.append({
                        "key": entry.key,
                        "namespace": entry.namespace,
                        "value": entry.value,
                        "metadata": entry.metadata,
                        "created_at": entry.created_at,
                        "updated_at": entry.updated_at,
                        "version": entry.version,
                    })
                except Exception as e:
                    errors.append(f"导出 {ns}/{key} 失败: {e}")

        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = pickle.dumps(entries_data, protocol=pickle.HIGHEST_PROTOCOL)
            path.write_bytes(data)
            total_size = len(data)
        except Exception as e:
            return ExportResult(
                success=False, format=ExportFormat.PICKLE,
                errors=[f"写入文件失败: {e}"], duration=time.time() - start,
            )

        return ExportResult(
            success=True,
            format=ExportFormat.PICKLE,
            entry_count=len(entries_data),
            total_size=total_size,
            output_path=str(path.absolute()),
            duration=time.time() - start,
            errors=errors,
        )

    def export_to_csv(
        self,
        output_path: str,
        namespaces: Optional[List[str]] = None,
        value_field: str = "value",
    ) -> ExportResult:
        """导出为CSV文件（仅导出简单键值对）"""
        start = time.time()
        errors = []
        rows = []

        target_ns = namespaces or self._store.namespaces()
        for ns in target_ns:
            keys = self._store.keys(ns)
            for key in keys:
                entry = self._store.get(key, ns)
                if entry is None:
                    continue
                try:
                    value_str = str(entry.value) if not isinstance(entry.value, str) else entry.value
                    rows.append({
                        "key": entry.key,
                        "namespace": entry.namespace,
                        "value": value_str,
                        "version": entry.version,
                        "created_at": entry.created_at,
                    })
                except Exception as e:
                    errors.append(f"导出 {ns}/{key} 失败: {e}")

        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
            total_size = path.stat().st_size
        except Exception as e:
            return ExportResult(
                success=False, format=ExportFormat.CSV,
                errors=[f"写入文件失败: {e}"], duration=time.time() - start,
            )

        return ExportResult(
            success=True,
            format=ExportFormat.CSV,
            entry_count=len(rows),
            total_size=total_size,
            output_path=str(path.absolute()),
            duration=time.time() - start,
            errors=errors,
        )

    def export_to_archive(
        self,
        output_dir: str,
        namespaces: Optional[List[str]] = None,
    ) -> ExportResult:
        """导出为归档目录（每个条目一个文件）"""
        start = time.time()
        errors = []
        entry_count = 0
        total_size = 0

        target_ns = namespaces or self._store.namespaces()
        archive_path = Path(output_dir)

        try:
            archive_path.mkdir(parents=True, exist_ok=True)

            manifest = ExportManifest(
                format=ExportFormat.ARCHIVE,
                source_type=type(self._store).__name__,
                namespaces=target_ns,
            )

            for ns in target_ns:
                keys = self._store.keys(ns)
                ns_dir = archive_path / self._safe_name(ns)
                ns_dir.mkdir(parents=True, exist_ok=True)

                for key in keys:
                    entry = self._store.get(key, ns)
                    if entry is None:
                        continue
                    try:
                        entry_file = ns_dir / f"{self._safe_name(key)}.json"
                        data = {
                            "key": entry.key,
                            "namespace": entry.namespace,
                            "value": entry.value,
                            "metadata": entry.metadata,
                            "created_at": entry.created_at,
                            "updated_at": entry.updated_at,
                            "version": entry.version,
                        }
                        content = json.dumps(data, ensure_ascii=False, indent=2, default=str)
                        entry_file.write_text(content, encoding="utf-8")
                        total_size += len(content.encode("utf-8"))
                        entry_count += 1
                    except Exception as e:
                        errors.append(f"导出 {ns}/{key} 失败: {e}")

            manifest.entry_count = entry_count
            manifest_file = archive_path / "_manifest.json"
            manifest_file.write_text(
                json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        except Exception as e:
            return ExportResult(
                success=False, format=ExportFormat.ARCHIVE,
                errors=[f"创建归档失败: {e}"], duration=time.time() - start,
            )

        return ExportResult(
            success=True,
            format=ExportFormat.ARCHIVE,
            entry_count=entry_count,
            total_size=total_size,
            output_path=str(archive_path.absolute()),
            duration=time.time() - start,
            errors=errors,
        )

    def _safe_name(self, name: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


# ============================================================
# 40.3 数据导入器
# ============================================================

class DataImporter:
    """数据导入器"""

    def __init__(self, store: PersistentStore):
        self._store = store
        self._conflict_policy = ConflictPolicy.SKIP

    def set_conflict_policy(self, policy: ConflictPolicy) -> None:
        """设置冲突策略"""
        self._conflict_policy = policy

    def import_from_json(
        self,
        input_path: str,
        conflict_policy: Optional[ConflictPolicy] = None,
        namespace_override: Optional[str] = None,
    ) -> ImportResult:
        """从JSON文件导入"""
        start = time.time()
        policy = conflict_policy or self._conflict_policy
        imported = 0
        skipped = 0
        conflicts = 0
        errors_list = []

        try:
            path = Path(input_path)
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return ImportResult(
                status=ImportStatus.FAILED,
                errors=[f"读取文件失败: {e}"],
                duration=time.time() - start,
            )

        for item in data:
            key = item.get("key", "")
            value = item.get("value")
            ns = namespace_override or item.get("namespace", "default")
            metadata = item.get("metadata", {})

            if not key:
                skipped += 1
                continue

            try:
                if self._store.exists(key, ns):
                    conflicts += 1
                    if policy == ConflictPolicy.SKIP:
                        skipped += 1
                        continue
                    elif policy == ConflictPolicy.OVERWRITE:
                        pass  # 继续覆盖
                    elif policy == ConflictPolicy.RENAME:
                        key = f"{key}_{uuid.uuid4().hex[:4]}"

                self._store.put(key, value, namespace=ns, metadata=metadata)
                imported += 1
            except Exception as e:
                errors_list.append(f"导入 {key} 失败: {e}")

        status = ImportStatus.SUCCESS if errors_list else ImportStatus.PARTIAL if errors_list else ImportStatus.SUCCESS
        if imported == 0 and errors_list:
            status = ImportStatus.FAILED

        return ImportResult(
            status=status,
            imported_count=imported,
            skipped_count=skipped,
            conflict_count=conflicts,
            error_count=len(errors_list),
            errors=errors_list,
            duration=time.time() - start,
        )

    def import_from_pickle(
        self,
        input_path: str,
        conflict_policy: Optional[ConflictPolicy] = None,
        namespace_override: Optional[str] = None,
    ) -> ImportResult:
        """从Pickle文件导入"""
        start = time.time()
        policy = conflict_policy or self._conflict_policy
        imported = 0
        skipped = 0
        conflicts = 0
        errors_list = []

        try:
            path = Path(input_path)
            data = pickle.loads(path.read_bytes())
        except Exception as e:
            return ImportResult(
                status=ImportStatus.FAILED,
                errors=[f"读取文件失败: {e}"],
                duration=time.time() - start,
            )

        for item in data:
            key = item.get("key", "")
            value = item.get("value")
            ns = namespace_override or item.get("namespace", "default")
            metadata = item.get("metadata", {})

            if not key:
                skipped += 1
                continue

            try:
                if self._store.exists(key, ns):
                    conflicts += 1
                    if policy == ConflictPolicy.SKIP:
                        skipped += 1
                        continue
                    elif policy == ConflictPolicy.OVERWRITE:
                        pass
                    elif policy == ConflictPolicy.RENAME:
                        key = f"{key}_{uuid.uuid4().hex[:4]}"

                self._store.put(key, value, namespace=ns, metadata=metadata)
                imported += 1
            except Exception as e:
                errors_list.append(f"导入 {key} 失败: {e}")

        status = ImportStatus.SUCCESS if not errors_list else (ImportStatus.PARTIAL if imported > 0 else ImportStatus.FAILED)

        return ImportResult(
            status=status,
            imported_count=imported,
            skipped_count=skipped,
            conflict_count=conflicts,
            error_count=len(errors_list),
            errors=errors_list,
            duration=time.time() - start,
        )

    def import_from_csv(
        self,
        input_path: str,
        conflict_policy: Optional[ConflictPolicy] = None,
        namespace_override: Optional[str] = None,
    ) -> ImportResult:
        """从CSV文件导入"""
        start = time.time()
        policy = conflict_policy or self._conflict_policy
        imported = 0
        skipped = 0
        conflicts = 0
        errors_list = []

        try:
            path = Path(input_path)
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception as e:
            return ImportResult(
                status=ImportStatus.FAILED,
                errors=[f"读取文件失败: {e}"],
                duration=time.time() - start,
            )

        for row in rows:
            key = row.get("key", "")
            value = row.get("value", "")
            ns = namespace_override or row.get("namespace", "default")

            if not key:
                skipped += 1
                continue

            try:
                if self._store.exists(key, ns):
                    conflicts += 1
                    if policy == ConflictPolicy.SKIP:
                        skipped += 1
                        continue
                    elif policy == ConflictPolicy.OVERWRITE:
                        pass
                    elif policy == ConflictPolicy.RENAME:
                        key = f"{key}_{uuid.uuid4().hex[:4]}"

                self._store.put(key, value, namespace=ns)
                imported += 1
            except Exception as e:
                errors_list.append(f"导入 {key} 失败: {e}")

        status = ImportStatus.SUCCESS if not errors_list else (ImportStatus.PARTIAL if imported > 0 else ImportStatus.FAILED)

        return ImportResult(
            status=status,
            imported_count=imported,
            skipped_count=skipped,
            conflict_count=conflicts,
            error_count=len(errors_list),
            errors=errors_list,
            duration=time.time() - start,
        )

    def import_from_archive(
        self,
        input_dir: str,
        conflict_policy: Optional[ConflictPolicy] = None,
    ) -> ImportResult:
        """从归档目录导入"""
        start = time.time()
        policy = conflict_policy or self._conflict_policy
        imported = 0
        skipped = 0
        conflicts = 0
        errors_list = []

        archive_path = Path(input_dir)
        if not archive_path.exists():
            return ImportResult(
                status=ImportStatus.FAILED,
                errors=["归档目录不存在"],
                duration=time.time() - start,
            )

        for ns_dir in archive_path.iterdir():
            if not ns_dir.is_dir() or ns_dir.name.startswith("_"):
                continue
            for entry_file in ns_dir.glob("*.json"):
                try:
                    data = json.loads(entry_file.read_text(encoding="utf-8"))
                    key = data.get("key", "")
                    value = data.get("value")
                    ns = data.get("namespace", ns_dir.name)
                    metadata = data.get("metadata", {})

                    if not key:
                        skipped += 1
                        continue

                    if self._store.exists(key, ns):
                        conflicts += 1
                        if policy == ConflictPolicy.SKIP:
                            skipped += 1
                            continue
                        elif policy == ConflictPolicy.RENAME:
                            key = f"{key}_{uuid.uuid4().hex[:4]}"

                    self._store.put(key, value, namespace=ns, metadata=metadata)
                    imported += 1
                except Exception as e:
                    errors_list.append(f"导入 {entry_file.name} 失败: {e}")

        status = ImportStatus.SUCCESS if not errors_list else (ImportStatus.PARTIAL if imported > 0 else ImportStatus.FAILED)

        return ImportResult(
            status=status,
            imported_count=imported,
            skipped_count=skipped,
            conflict_count=conflicts,
            error_count=len(errors_list),
            errors=errors_list,
            duration=time.time() - start,
        )


# ============================================================
# 40.4 批量迁移
# ============================================================

class DataMigrator:
    """数据迁移器"""

    def __init__(self):
        self._lock = threading.RLock()

    def migrate(
        self,
        source: PersistentStore,
        target: PersistentStore,
        namespaces: Optional[List[str]] = None,
        conflict_policy: ConflictPolicy = ConflictPolicy.OVERWRITE,
        batch_size: int = 100,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> MigrationResult:
        """迁移数据"""
        start = time.time()
        migrated = 0
        failed = 0
        errors_list = []

        target_ns = namespaces or source.namespaces()
        total_keys = sum(len(source.keys(ns)) for ns in target_ns)
        processed = 0

        for ns in target_ns:
            keys = source.keys(ns)
            for key in keys:
                entry = source.get(key, ns)
                if entry is None:
                    continue

                try:
                    if target.exists(key, ns):
                        if conflict_policy == ConflictPolicy.SKIP:
                            processed += 1
                            continue
                        elif conflict_policy == ConflictPolicy.RENAME:
                            key = f"{key}_{uuid.uuid4().hex[:4]}"

                    target.put(key, entry.value, namespace=ns,
                               metadata=entry.metadata)
                    migrated += 1
                except Exception as e:
                    failed += 1
                    errors_list.append(f"迁移 {ns}/{key} 失败: {e}")

                processed += 1
                if progress_callback and processed % batch_size == 0:
                    progress_callback(processed, total_keys)

        if progress_callback:
            progress_callback(total_keys, total_keys)

        return MigrationResult(
            success=failed == 0,
            source_store=type(source).__name__,
            target_store=type(target).__name__,
            migrated_count=migrated,
            failed_count=failed,
            errors=errors_list,
            duration=time.time() - start,
        )

    def sync_incremental(
        self,
        source: PersistentStore,
        target: PersistentStore,
        namespace: str = "default",
    ) -> MigrationResult:
        """增量同步"""
        start = time.time()
        migrated = 0
        failed = 0
        errors_list = []

        source_keys = set(source.keys(namespace))
        target_keys = set(target.keys(namespace))

        # 找出需要同步的键
        new_keys = source_keys - target_keys
        all_keys = source_keys  # 也检查更新

        for key in all_keys:
            source_entry = source.get(key, namespace)
            if source_entry is None:
                continue

            target_entry = target.get(key, namespace)

            # 新键或更新的键
            need_sync = (
                target_entry is None or
                source_entry.updated_at > target_entry.updated_at
            )

            if need_sync:
                try:
                    target.put(key, source_entry.value, namespace=namespace,
                               metadata=source_entry.metadata)
                    migrated += 1
                except Exception as e:
                    failed += 1
                    errors_list.append(f"同步 {key} 失败: {e}")

        return MigrationResult(
            success=failed == 0,
            source_store=type(source).__name__,
            target_store=type(target).__name__,
            migrated_count=migrated,
            failed_count=failed,
            errors=errors_list,
            duration=time.time() - start,
        )


# ============================================================
# 便捷函数
# ============================================================

def create_exporter(store: PersistentStore) -> DataExporter:
    """创建数据导出器"""
    return DataExporter(store)


def create_importer(store: PersistentStore) -> DataImporter:
    """创建数据导入器"""
    return DataImporter(store)


def create_migrator() -> DataMigrator:
    """创建数据迁移器"""
    return DataMigrator()
