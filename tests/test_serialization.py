"""
阶段38: Agent序列化系统 - 单元测试

遵循"图难于其易，为大于其细"的原则，全面测试序列化功能。
"""

import pytest
from datetime import datetime
from typing import Dict, List, Any

from src.agent.serialization import (
    SerializationFormat,
    CompressionType,
    SerializationResult,
    VersionInfo,
    JsonSerializer,
    PickleSerializer,
    MessagePackSerializer,
    CompressionLayer,
    SerializationManager,
    VersionedSerializer,
    create_json_serializer,
    create_pickle_serializer,
    create_msgpack_serializer,
    create_serialization_manager,
    serialize_to_json,
    deserialize_from_json,
)


# ============================================================
# 38.1 序列化模型测试
# ============================================================

class TestSerializationFormat:
    """测试序列化格式枚举"""

    def test_format_values(self):
        """测试格式枚举值"""
        assert SerializationFormat.JSON == "json"
        assert SerializationFormat.PICKLE == "pickle"
        assert SerializationFormat.MESSAGEPACK == "msgpack"
        assert SerializationFormat.RAW == "raw"


class TestCompressionType:
    """测试压缩类型枚举"""

    def test_compression_values(self):
        """测试压缩类型枚举值"""
        assert CompressionType.NONE == "none"
        assert CompressionType.GZIP == "gzip"
        assert CompressionType.LZ4 == "lz4"


class TestSerializationResult:
    """测试序列化结果"""

    def test_basic_creation(self):
        """测试基本创建"""
        result = SerializationResult(
            data=b"test data",
            format=SerializationFormat.JSON,
            compression=CompressionType.NONE,
            original_size=100,
            compressed_size=100,
        )
        assert result.data == b"test data"
        assert result.format == SerializationFormat.JSON
        assert result.compression == CompressionType.NONE
        assert result.original_size == 100
        assert result.compressed_size == 100
        assert result.version == "1.0"
        assert isinstance(result.timestamp, datetime)

    def test_compression_ratio(self):
        """测试压缩比率计算"""
        result = SerializationResult(
            data=b"test",
            format=SerializationFormat.JSON,
            compression=CompressionType.GZIP,
            original_size=100,
            compressed_size=50,
        )
        assert result.compression_ratio == 0.5

    def test_compression_ratio_zero(self):
        """测试零大小压缩比率"""
        result = SerializationResult(
            data=b"",
            format=SerializationFormat.JSON,
            compression=CompressionType.NONE,
            original_size=0,
            compressed_size=0,
        )
        assert result.compression_ratio == 1.0

    def test_to_dict(self):
        """测试转换为字典"""
        result = SerializationResult(
            data=b"test",
            format=SerializationFormat.JSON,
            compression=CompressionType.GZIP,
            original_size=100,
            compressed_size=50,
            version="2.0",
            metadata={"key": "value"},
        )
        d = result.to_dict()
        assert d["format"] == "json"
        assert d["compression"] == "gzip"
        assert d["original_size"] == 100
        assert d["compressed_size"] == 50
        assert d["compression_ratio"] == 0.5
        assert d["version"] == "2.0"
        assert d["metadata"] == {"key": "value"}


class TestVersionInfo:
    """测试版本信息"""

    def test_creation(self):
        """测试创建"""
        version = VersionInfo(major=1, minor=2, patch=3)
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_str_representation(self):
        """测试字符串表示"""
        version = VersionInfo(major=2, minor=5, patch=1)
        assert str(version) == "2.5.1"

    def test_to_bytes_and_from_bytes(self):
        """测试字节序列化"""
        version = VersionInfo(major=1, minor=2, patch=3)
        data = version.to_bytes()
        restored = VersionInfo.from_bytes(data)
        assert restored.major == 1
        assert restored.minor == 2
        assert restored.patch == 3


# ============================================================
# 38.2 JSON序列化测试
# ============================================================

class TestJsonSerializer:
    """测试JSON序列化器"""

    def test_basic_serialize_deserialize(self):
        """测试基本序列化和反序列化"""
        serializer = JsonSerializer()
        data = {"name": "test", "value": 123}
        serialized = serializer.serialize(data)
        assert isinstance(serialized, bytes)
        restored = serializer.deserialize(serialized)
        assert restored == data

    def test_serialize_nested_dict(self):
        """测试嵌套字典序列化"""
        serializer = JsonSerializer()
        data = {
            "user": {"name": "Alice", "age": 30},
            "items": [1, 2, 3],
        }
        serialized = serializer.serialize(data)
        restored = serializer.deserialize(serialized)
        assert restored == data

    def test_serialize_datetime(self):
        """测试日期时间序列化"""
        serializer = JsonSerializer()
        now = datetime(2024, 1, 15, 10, 30, 0)
        data = {"time": now}
        serialized = serializer.serialize(data)
        restored = serializer.deserialize(serialized)
        assert restored["time"] == now

    def test_serialize_bytes(self):
        """测试字节序列化"""
        serializer = JsonSerializer()
        data = {"content": b"hello world"}
        serialized = serializer.serialize(data)
        restored = serializer.deserialize(serialized)
        assert restored["content"] == b"hello world"

    def test_serialize_set(self):
        """测试集合序列化"""
        serializer = JsonSerializer()
        data = {"items": {1, 2, 3}}
        serialized = serializer.serialize(data)
        restored = serializer.deserialize(serialized)
        assert restored["items"] == {1, 2, 3}

    def test_serialize_list(self):
        """测试列表序列化"""
        serializer = JsonSerializer()
        data = [1, 2, 3, "hello", True, None]
        serialized = serializer.serialize(data)
        restored = serializer.deserialize(serialized)
        assert restored == data

    def test_indent_option(self):
        """测试缩进选项"""
        serializer = JsonSerializer(indent=2)
        data = {"key": "value"}
        serialized = serializer.serialize(data)
        assert b"\n" in serialized  # 应该有换行符

    def test_ensure_ascii_option(self):
        """测试ASCII选项"""
        serializer = JsonSerializer(ensure_ascii=False)
        data = {"text": "中文"}
        serialized = serializer.serialize(data)
        assert "中文".encode("utf-8") in serialized

    def test_custom_encoder(self):
        """测试自定义编码器"""
        class CustomType:
            def __init__(self, value):
                self.value = value

        serializer = JsonSerializer()
        serializer.register_encoder(CustomType, lambda obj: {"__type__": "CustomType", "value": obj.value})
        serializer.register_decoder("CustomType", lambda dct, _: CustomType(dct["value"]))

        data = {"custom": CustomType(42)}
        serialized = serializer.serialize(data)
        restored = serializer.deserialize(serialized)
        assert restored["custom"].value == 42


# ============================================================
# 38.3 二进制序列化测试
# ============================================================

class TestPickleSerializer:
    """测试Pickle序列化器"""

    def test_basic_serialize_deserialize(self):
        """测试基本序列化和反序列化"""
        serializer = PickleSerializer()
        data = {"name": "test", "value": 123}
        serialized = serializer.serialize(data)
        assert isinstance(serialized, bytes)
        restored = serializer.deserialize(serialized)
        assert restored == data

    def test_serialize_complex_object(self):
        """测试复杂对象序列化"""
        serializer = PickleSerializer()
        data = {
            "datetime": datetime.now(),
            "list": [1, 2, 3],
            "nested": {"a": 1, "b": 2},
        }
        serialized = serializer.serialize(data)
        restored = serializer.deserialize(serialized)
        assert restored["list"] == data["list"]
        assert restored["nested"] == data["nested"]

    def test_protocol_selection(self):
        """测试协议选择"""
        import pickle
        serializer = PickleSerializer(protocol=pickle.HIGHEST_PROTOCOL)
        data = {"test": "value"}
        serialized = serializer.serialize(data)
        restored = serializer.deserialize(serialized)
        assert restored == data


class TestMessagePackSerializer:
    """测试MessagePack序列化器"""

    def test_is_available(self):
        """测试可用性检查"""
        serializer = MessagePackSerializer()
        # 不强制要求安装msgpack
        assert isinstance(serializer.is_available(), bool)

    def test_serialize_deserialize_if_available(self):
        """测试序列化和反序列化（如果可用）"""
        serializer = MessagePackSerializer()
        if not serializer.is_available():
            pytest.skip("msgpack not installed")

        data = {"name": "test", "value": 123, "items": [1, 2, 3]}
        serialized = serializer.serialize(data)
        assert isinstance(serialized, bytes)
        restored = serializer.deserialize(serialized)
        assert restored == data

    def test_serialize_datetime_if_available(self):
        """测试日期时间序列化（如果可用）"""
        serializer = MessagePackSerializer()
        if not serializer.is_available():
            pytest.skip("msgpack not installed")

        now = datetime(2024, 1, 15, 10, 30, 0)
        data = {"time": now}
        serialized = serializer.serialize(data)
        restored = serializer.deserialize(serialized)
        assert restored["time"] == now

    def test_raises_if_not_available(self):
        """测试未安装时抛出异常"""
        serializer = MessagePackSerializer()
        if serializer.is_available():
            pytest.skip("msgpack is installed")

        with pytest.raises(RuntimeError, match="msgpack is not installed"):
            serializer.serialize({"test": "value"})


# ============================================================
# 38.4 压缩层测试
# ============================================================

class TestCompressionLayer:
    """测试压缩层"""

    def test_none_compression(self):
        """测试无压缩"""
        layer = CompressionLayer(CompressionType.NONE)
        data = b"hello world" * 100
        compressed = layer.compress(data)
        assert compressed == data
        decompressed = layer.decompress(compressed)
        assert decompressed == data

    def test_gzip_compression(self):
        """测试GZIP压缩"""
        layer = CompressionLayer(CompressionType.GZIP)
        data = b"hello world" * 100
        compressed = layer.compress(data)
        assert len(compressed) < len(data)
        decompressed = layer.decompress(compressed)
        assert decompressed == data

    def test_compression_type_property(self):
        """测试压缩类型属性"""
        layer = CompressionLayer(CompressionType.GZIP)
        assert layer.compression_type == CompressionType.GZIP


# ============================================================
# 序列化管理器测试
# ============================================================

class TestSerializationManager:
    """测试序列化管理器"""

    def test_default_serialization(self):
        """测试默认序列化"""
        manager = SerializationManager()
        data = {"name": "test", "value": 123}
        result = manager.serialize(data)
        assert isinstance(result, SerializationResult)
        assert result.format == SerializationFormat.JSON
        assert result.compression == CompressionType.NONE
        assert result.original_size > 0

    def test_json_serialization(self):
        """测试JSON序列化"""
        manager = SerializationManager()
        data = {"name": "test", "value": 123}
        result = manager.serialize(data, format=SerializationFormat.JSON)
        assert result.format == SerializationFormat.JSON
        restored = manager.deserialize(result)
        assert restored == data

    def test_pickle_serialization(self):
        """测试Pickle序列化"""
        manager = SerializationManager()
        data = {"name": "test", "datetime": datetime.now()}
        result = manager.serialize(data, format=SerializationFormat.PICKLE)
        assert result.format == SerializationFormat.PICKLE
        restored = manager.deserialize(result)
        assert restored["name"] == data["name"]

    def test_gzip_compression(self):
        """测试GZIP压缩"""
        manager = SerializationManager()
        data = {"text": "hello world" * 100}
        result = manager.serialize(
            data,
            format=SerializationFormat.JSON,
            compression=CompressionType.GZIP,
        )
        assert result.compression == CompressionType.GZIP
        assert result.compressed_size < result.original_size
        restored = manager.deserialize(result)
        assert restored == data

    def test_set_default_format(self):
        """测试设置默认格式"""
        manager = SerializationManager()
        manager.set_default_format(SerializationFormat.PICKLE)
        data = {"test": "value"}
        result = manager.serialize(data)
        assert result.format == SerializationFormat.PICKLE

    def test_set_default_compression(self):
        """测试设置默认压缩"""
        manager = SerializationManager()
        manager.set_default_compression(CompressionType.GZIP)
        data = {"test": "value" * 100}
        result = manager.serialize(data)
        assert result.compression == CompressionType.GZIP

    def test_register_custom_serializer(self):
        """测试注册自定义序列化器"""
        class CustomSerializer:
            def serialize(self, obj):
                return str(obj).encode()

            def deserialize(self, data, target_type=None):
                return data.decode()

        manager = SerializationManager()
        manager.register_serializer(SerializationFormat.RAW, CustomSerializer())
        data = "custom data"
        result = manager.serialize(data, format=SerializationFormat.RAW)
        assert result.format == SerializationFormat.RAW


# ============================================================
# 版本化序列化器测试
# ============================================================

class TestVersionedSerializer:
    """测试版本化序列化器"""

    def test_creation(self):
        """测试创建"""
        version = VersionInfo(1, 0, 0)
        serializer = VersionedSerializer(version)
        assert serializer._version == version

    def test_serialize_with_version(self):
        """测试带版本序列化"""
        version = VersionInfo(1, 2, 3)
        vserializer = VersionedSerializer(version)
        manager = SerializationManager()
        data = {"name": "test"}
        serialized = vserializer.serialize_with_version(data, manager)
        # 格式: [版本6字节][数据长度4字节][数据]
        assert len(serialized) > 10

    def test_deserialize_with_version(self):
        """测试带版本反序列化"""
        version = VersionInfo(1, 0, 0)
        vserializer = VersionedSerializer(version)
        manager = SerializationManager()
        data = {"name": "test", "value": 123}
        serialized = vserializer.serialize_with_version(data, manager)
        restored = vserializer.deserialize_with_version(serialized, manager)
        assert restored == data

    def test_register_migration(self):
        """测试注册迁移函数"""
        version = VersionInfo(1, 0, 0)
        serializer = VersionedSerializer(version)

        def migrator(data, version):
            data["migrated"] = True
            return data

        serializer.register_migration("0.9.0", migrator)
        assert "0.9.0" in serializer._migrations


# ============================================================
# 便捷函数测试
# ============================================================

class TestConvenienceFunctions:
    """测试便捷函数"""

    def test_create_json_serializer(self):
        """测试创建JSON序列化器"""
        serializer = create_json_serializer(indent=2)
        assert isinstance(serializer, JsonSerializer)

    def test_create_pickle_serializer(self):
        """测试创建Pickle序列化器"""
        serializer = create_pickle_serializer()
        assert isinstance(serializer, PickleSerializer)

    def test_create_msgpack_serializer(self):
        """测试创建MessagePack序列化器"""
        serializer = create_msgpack_serializer()
        assert isinstance(serializer, MessagePackSerializer)

    def test_create_serialization_manager(self):
        """测试创建序列化管理器"""
        manager = create_serialization_manager()
        assert isinstance(manager, SerializationManager)

    def test_serialize_to_json(self):
        """测试快捷JSON序列化"""
        data = {"key": "value"}
        serialized = serialize_to_json(data)
        assert isinstance(serialized, bytes)
        restored = deserialize_from_json(serialized)
        assert restored == data

    def test_deserialize_from_json(self):
        """测试快捷JSON反序列化"""
        data = {"key": "value"}
        serialized = serialize_to_json(data)
        restored = deserialize_from_json(serialized)
        assert restored == data


# ============================================================
# 集成测试
# ============================================================

class TestIntegration:
    """集成测试"""

    def test_full_workflow_json(self):
        """测试完整JSON工作流"""
        manager = SerializationManager()
        original = {
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ],
            "timestamp": datetime.now(),
            "metadata": {"version": "1.0"},
        }

        # 序列化
        result = manager.serialize(original, compression=CompressionType.GZIP)
        assert result.compression == CompressionType.GZIP
        assert result.compression_ratio < 1.0

        # 反序列化
        restored = manager.deserialize(result)
        assert restored["users"] == original["users"]
        assert restored["metadata"] == original["metadata"]

    def test_full_workflow_pickle(self):
        """测试完整Pickle工作流"""
        manager = SerializationManager()
        original = {
            "data": list(range(1000)),
            "nested": {"a": {"b": {"c": "deep"}}},
        }

        result = manager.serialize(original, format=SerializationFormat.PICKLE)
        restored = manager.deserialize(result)
        assert restored == original

    def test_format_comparison(self):
        """测试不同格式比较"""
        manager = SerializationManager()
        data = {"numbers": list(range(100)), "text": "hello" * 50}

        json_result = manager.serialize(data, format=SerializationFormat.JSON)
        pickle_result = manager.serialize(data, format=SerializationFormat.PICKLE)

        # Pickle通常更紧凑
        assert pickle_result.compressed_size <= json_result.compressed_size

        # 两者都能正确还原
        assert manager.deserialize(json_result) == data
        assert manager.deserialize(pickle_result) == data

    def test_compression_comparison(self):
        """测试不同压缩比较"""
        manager = SerializationManager()
        data = {"text": "repeated text " * 1000}

        none_result = manager.serialize(data, compression=CompressionType.NONE)
        gzip_result = manager.serialize(data, compression=CompressionType.GZIP)

        # GZIP应该压缩得更好
        assert gzip_result.compressed_size < none_result.compressed_size

        # 两者都能正确还原
        assert manager.deserialize(none_result) == data
        assert manager.deserialize(gzip_result) == data

    def test_complex_nested_structure(self):
        """测试复杂嵌套结构"""
        manager = SerializationManager()
        original = {
            "level1": {
                "level2": {
                    "level3": {
                        "data": [1, 2, 3],
                        "meta": {"created": datetime.now()},
                    }
                }
            },
            "list_of_dicts": [
                {"id": i, "value": f"item_{i}"}
                for i in range(10)
            ],
        }

        result = manager.serialize(original)
        restored = manager.deserialize(result)
        assert restored["level1"]["level2"]["level3"]["data"] == [1, 2, 3]
        assert len(restored["list_of_dicts"]) == 10
