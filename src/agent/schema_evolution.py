"""
阶段68: Agent数据模式演化系统

遵循"图难于其易，为大于其细"的原则，实现模式版本、字段变更、迁移、兼容性检查。
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

class FieldChangeType(str, Enum):
    ADD = "add"
    REMOVE = "remove"
    RENAME = "rename"
    TYPE_CHANGE = "type_change"
    DEFAULT_CHANGE = "default_change"

class CompatibilityLevel(str, Enum):
    FULL = "full"
    BACKWARD = "backward"
    FORWARD = "forward"
    NONE = "none"

@dataclass
class SchemaField:
    name: str
    field_type: str = "string"
    required: bool = True
    default: Any = None
    description: str = ""

@dataclass
class SchemaVersion:
    version: int
    fields: Dict[str, SchemaField] = field(default_factory=dict)
    created_at: str = ""

    def add_field(self, field: SchemaField) -> None:
        self.fields[field.name] = field

    def remove_field(self, name: str) -> bool:
        return self.fields.pop(name, None) is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "fields": {k: {"type": v.field_type, "required": v.required} for k, v in self.fields.items()},
        }

class SchemaEvolution:
    """模式演化"""
    def __init__(self, name: str):
        self.name = name
        self._versions: Dict[int, SchemaVersion] = {}
        self._current_version = 0

    def create_version(self, version: int) -> SchemaVersion:
        sv = SchemaVersion(version=version)
        self._versions[version] = sv
        self._current_version = version
        return sv

    def get_version(self, version: int) -> Optional[SchemaVersion]:
        return self._versions.get(version)

    def get_current(self) -> Optional[SchemaVersion]:
        return self._versions.get(self._current_version)

    def add_field(self, version: int, field: SchemaField) -> bool:
        sv = self._versions.get(version)
        if sv:
            sv.add_field(field)
            return True
        return False

    def compare_versions(self, v1: int, v2: int) -> List[Dict[str, Any]]:
        """比较两个版本的差异"""
        sv1 = self._versions.get(v1)
        sv2 = self._versions.get(v2)
        if not sv1 or not sv2:
            return []
        changes = []
        # 新增字段
        for name in sv2.fields:
            if name not in sv1.fields:
                changes.append({"type": FieldChangeType.ADD, "field": name})
        # 删除字段
        for name in sv1.fields:
            if name not in sv2.fields:
                changes.append({"type": FieldChangeType.REMOVE, "field": name})
        # 类型变更
        for name in sv1.fields:
            if name in sv2.fields:
                f1 = sv1.fields[name]
                f2 = sv2.fields[name]
                if f1.field_type != f2.field_type:
                    changes.append({"type": FieldChangeType.TYPE_CHANGE, "field": name,
                                    "from": f1.field_type, "to": f2.field_type})
        return changes

    def check_compatibility(self, reader_version: int, writer_version: int) -> CompatibilityLevel:
        """检查读写兼容性"""
        reader = self._versions.get(reader_version)
        writer = self._versions.get(writer_version)
        if not reader or not writer:
            return CompatibilityLevel.NONE

        reader_can_read_writer = all(
            f.name in writer.fields for f in reader.fields.values() if f.required
        )
        writer_can_write_reader = all(
            f.name in reader.fields for f in writer.fields.values() if f.required
        )

        if reader_can_read_writer and writer_can_write_reader:
            return CompatibilityLevel.FULL
        elif reader_can_read_writer:
            return CompatibilityLevel.BACKWARD
        elif writer_can_write_reader:
            return CompatibilityLevel.FORWARD
        return CompatibilityLevel.NONE

    def migrate_data(self, data: Dict[str, Any], from_version: int, to_version: int) -> Dict[str, Any]:
        """数据迁移"""
        source = self._versions.get(from_version)
        target = self._versions.get(to_version)
        if not source or not target:
            return data

        result = dict(data)
        # 添加新字段的默认值
        for name, field in target.fields.items():
            if name not in result and field.default is not None:
                result[name] = field.default
        # 删除已移除的字段
        for name in list(result.keys()):
            if name not in target.fields:
                del result[name]
        return result

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "versions": list(self._versions.keys()),
            "current_version": self._current_version,
        }

class SchemaRegistry:
    """模式注册表"""
    def __init__(self):
        self._schemas: Dict[str, SchemaEvolution] = {}

    def register(self, name: str) -> SchemaEvolution:
        se = SchemaEvolution(name)
        self._schemas[name] = se
        return se

    def get(self, name: str) -> Optional[SchemaEvolution]:
        return self._schemas.get(name)

    def list_schemas(self) -> List[str]:
        return list(self._schemas.keys())

def create_schema_registry() -> SchemaRegistry:
    return SchemaRegistry()
