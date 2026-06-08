"""阶段68: Agent数据模式演化系统 - 单元测试"""
import pytest
from src.agent.schema_evolution import (
    FieldChangeType, CompatibilityLevel, SchemaField, SchemaVersion,
    SchemaEvolution, SchemaRegistry, create_schema_registry,
)

class TestSchemaEvolution:
    def test_create_version(self):
        se = SchemaEvolution("user")
        v1 = se.create_version(1)
        assert v1.version == 1
        assert se.get_current().version == 1

    def test_add_field(self):
        se = SchemaEvolution("user")
        se.create_version(1)
        se.add_field(1, SchemaField("name", "string"))
        assert "name" in se.get_version(1).fields

    def test_compare_versions(self):
        se = SchemaEvolution("user")
        v1 = se.create_version(1)
        v1.add_field(SchemaField("name", "string"))
        v2 = se.create_version(2)
        v2.add_field(SchemaField("name", "string"))
        v2.add_field(SchemaField("age", "int"))
        changes = se.compare_versions(1, 2)
        assert len(changes) == 1
        assert changes[0]["type"] == FieldChangeType.ADD

    def test_type_change(self):
        se = SchemaEvolution("user")
        v1 = se.create_version(1)
        v1.add_field(SchemaField("age", "string"))
        v2 = se.create_version(2)
        v2.add_field(SchemaField("age", "int"))
        changes = se.compare_versions(1, 2)
        assert changes[0]["type"] == FieldChangeType.TYPE_CHANGE

    def test_compatibility_full(self):
        se = SchemaEvolution("user")
        v1 = se.create_version(1)
        v1.add_field(SchemaField("name", "string"))
        v2 = se.create_version(2)
        v2.add_field(SchemaField("name", "string"))
        assert se.check_compatibility(1, 2) == CompatibilityLevel.FULL

    def test_compatibility_none(self):
        se = SchemaEvolution("user")
        v1 = se.create_version(1)
        v1.add_field(SchemaField("name", "string", required=True))
        v2 = se.create_version(2)
        v2.add_field(SchemaField("age", "int", required=True))
        assert se.check_compatibility(1, 2) == CompatibilityLevel.NONE

    def test_migrate_data(self):
        se = SchemaEvolution("user")
        v1 = se.create_version(1)
        v1.add_field(SchemaField("name", "string"))
        v2 = se.create_version(2)
        v2.add_field(SchemaField("name", "string"))
        v2.add_field(SchemaField("age", "int", default=0))
        result = se.migrate_data({"name": "Alice"}, 1, 2)
        assert result["name"] == "Alice"
        assert result["age"] == 0

    def test_migrate_remove_field(self):
        se = SchemaEvolution("user")
        v1 = se.create_version(1)
        v1.add_field(SchemaField("name", "string"))
        v1.add_field(SchemaField("temp", "string"))
        v2 = se.create_version(2)
        v2.add_field(SchemaField("name", "string"))
        result = se.migrate_data({"name": "Alice", "temp": "x"}, 1, 2)
        assert "temp" not in result

class TestSchemaRegistry:
    def test_register(self):
        r = SchemaRegistry()
        se = r.register("user")
        assert isinstance(se, SchemaEvolution)

    def test_get(self):
        r = SchemaRegistry()
        r.register("user")
        assert r.get("user") is not None

    def test_create_schema_registry(self):
        r = create_schema_registry()
        assert isinstance(r, SchemaRegistry)
