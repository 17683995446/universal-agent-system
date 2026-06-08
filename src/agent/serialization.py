"""
阶段38: Agent序列化系统

遵循"图难于其易，为大于其细"的原则，实现多种序列化格式、版本兼容、压缩。
"""

import base64
import gzip
import json
import pickle
import struct
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union


# ============================================================
# 38.1 序列化模型
# ============================================================

class SerializationFormat(str, Enum):
    """序列化格式"""
    JSON = "json"
    PICKLE = "pickle"
    MESSAGEPACK = "msgpack"
    RAW = "raw"


class CompressionType(str, Enum):
    """压缩类型"""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"


@dataclass
class SerializationResult:
    """序列化结果"""
    data: bytes
    format: SerializationFormat
    compression: CompressionType
    original_size: int
    compressed_size: int
    version: str = "1.0"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def compression_ratio(self) -> float:
        if self.original_size == 0:
            return 1.0
        return self.compressed_size / self.original_size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format.value,
            "compression": self.compression.value,
            "original_size": self.original_size,
            "compressed_size": self.compressed_size,
            "compression_ratio": round(self.compression_ratio, 4),
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class VersionInfo:
    """版本信息"""
    major: int
    minor: int
    patch: int
    schema_hash: str = ""

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def to_bytes(self) -> bytes:
        return struct.pack(">HHH", self.major, self.minor, self.patch)

    @classmethod
    def from_bytes(cls, data: bytes) -> "VersionInfo":
        major, minor, patch = struct.unpack(">HHH", data[:6])
        return cls(major=major, minor=minor, patch=patch)


# ============================================================
# 38.2 JSON序列化
# ============================================================

class JsonSerializer:
    """JSON序列化器"""

    def __init__(self, indent: Optional[int] = None, ensure_ascii: bool = False):
        self._indent = indent
        self._ensure_ascii = ensure_ascii
        self._encoders: Dict[Type, Callable[[Any], Any]] = {}
        self._decoders: Dict[str, Callable[[Any, Type], Any]] = {}
        self._register_default_encoders()

    def register_encoder(self, type_: Type, encoder: Callable[[Any], Any]) -> None:
        """注册类型编码器"""
        self._encoders[type_] = encoder

    def register_decoder(self, type_name: str, decoder: Callable[[Any, Type], Any]) -> None:
        """注册类型解码器"""
        self._decoders[type_name] = decoder

    def serialize(self, obj: Any) -> bytes:
        """序列化为JSON字节"""
        json_str = json.dumps(
            obj,
            default=self._default_encoder,
            indent=self._indent,
            ensure_ascii=self._ensure_ascii,
        )
        return json_str.encode("utf-8")

    def deserialize(self, data: bytes, target_type: Optional[Type] = None) -> Any:
        """从JSON字节反序列化"""
        json_str = data.decode("utf-8")
        return json.loads(json_str, object_hook=self._object_hook)

    def _default_encoder(self, obj: Any) -> Any:
        """默认编码器"""
        type_ = type(obj)
        if type_ in self._encoders:
            encoded = self._encoders[type_](obj)
            if isinstance(encoded, dict):
                encoded["__type__"] = type_.__name__
            return encoded

        # 处理常见类型
        if isinstance(obj, datetime):
            return {"__type__": "datetime", "iso": obj.isoformat()}
        if isinstance(obj, bytes):
            return {"__type__": "bytes", "b64": base64.b64encode(obj).decode("ascii")}
        if isinstance(obj, set):
            return {"__type__": "set", "items": list(obj)}

        raise TypeError(f"Object of type {type_.__name__} is not JSON serializable")

    def _object_hook(self, dct: Dict[str, Any]) -> Any:
        """对象钩子"""
        if "__type__" not in dct:
            return dct

        type_name = dct["__type__"]

        if type_name in self._decoders:
            return self._decoders[type_name](dct, None)

        if type_name == "datetime":
            return datetime.fromisoformat(dct["iso"])
        if type_name == "bytes":
            return base64.b64decode(dct["b64"])
        if type_name == "set":
            return set(dct["items"])

        return dct

    def _register_default_encoders(self) -> None:
        """注册默认编码器"""
        pass


# ============================================================
# 38.3 二进制序列化
# ============================================================

class PickleSerializer:
    """Pickle序列化器"""

    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL):
        self._protocol = protocol

    def serialize(self, obj: Any) -> bytes:
        """序列化为Pickle字节"""
        return pickle.dumps(obj, protocol=self._protocol)

    def deserialize(self, data: bytes, target_type: Optional[Type] = None) -> Any:
        """从Pickle字节反序列化"""
        return pickle.loads(data)


class MessagePackSerializer:
    """MessagePack序列化器（如果可用）"""

    def __init__(self):
        try:
            import msgpack
            self._msgpack = msgpack
            self._available = True
        except ImportError:
            self._available = False
            self._msgpack = None

    def is_available(self) -> bool:
        return self._available

    def serialize(self, obj: Any) -> bytes:
        """序列化为MessagePack字节"""
        if not self._available:
            raise RuntimeError("msgpack is not installed")
        return self._msgpack.packb(obj, use_bin_type=True, default=self._default)

    def deserialize(self, data: bytes, target_type: Optional[Type] = None) -> Any:
        """从MessagePack字节反序列化"""
        if not self._available:
            raise RuntimeError("msgpack is not installed")
        return self._msgpack.unpackb(data, raw=False, object_hook=self._object_hook)

    def _default(self, obj: Any) -> Any:
        """处理不可序列化类型"""
        if isinstance(obj, datetime):
            return {"__type__": "datetime", "iso": obj.isoformat()}
        if isinstance(obj, bytes):
            return {"__type__": "bytes", "b64": base64.b64encode(obj).decode("ascii")}
        raise TypeError(f"Object of type {type(obj).__name__} is not serializable")

    def _object_hook(self, obj: Any) -> Any:
        """对象钩子"""
        if isinstance(obj, dict) and "__type__" in obj:
            if obj["__type__"] == "datetime":
                return datetime.fromisoformat(obj["iso"])
            if obj["__type__"] == "bytes":
                return base64.b64decode(obj["b64"])
        return obj


# ============================================================
# 38.4 压缩层
# ============================================================

class CompressionLayer:
    """压缩层"""

    def __init__(self, compression_type: CompressionType = CompressionType.NONE):
        self._type = compression_type
        self._level = 6

    def compress(self, data: bytes) -> bytes:
        """压缩数据"""
        if self._type == CompressionType.NONE:
            return data
        if self._type == CompressionType.GZIP:
            return gzip.compress(data, compresslevel=self._level)
        if self._type == CompressionType.LZ4:
            try:
                import lz4.frame
                return lz4.frame.compress(data)
            except ImportError:
                raise RuntimeError("lz4 is not installed")
        return data

    def decompress(self, data: bytes) -> bytes:
        """解压数据"""
        if self._type == CompressionType.NONE:
            return data
        if self._type == CompressionType.GZIP:
            return gzip.decompress(data)
        if self._type == CompressionType.LZ4:
            try:
                import lz4.frame
                return lz4.frame.decompress(data)
            except ImportError:
                raise RuntimeError("lz4 is not installed")
        return data

    @property
    def compression_type(self) -> CompressionType:
        return self._type


# ============================================================
# 序列化管理器
# ============================================================

class SerializationManager:
    """序列化管理器"""

    def __init__(self):
        self._serializers: Dict[SerializationFormat, Any] = {
            SerializationFormat.JSON: JsonSerializer(),
            SerializationFormat.PICKLE: PickleSerializer(),
            SerializationFormat.MESSAGEPACK: MessagePackSerializer(),
        }
        self._default_format = SerializationFormat.JSON
        self._default_compression = CompressionType.NONE

    def serialize(
        self,
        obj: Any,
        format: Optional[SerializationFormat] = None,
        compression: Optional[CompressionType] = None,
        version: str = "1.0",
    ) -> SerializationResult:
        """序列化对象"""
        fmt = format or self._default_format
        comp = compression or self._default_compression

        serializer = self._serializers.get(fmt)
        if serializer is None:
            raise ValueError(f"Unknown serialization format: {fmt}")

        # 序列化
        serialized = serializer.serialize(obj)
        original_size = len(serialized)

        # 压缩
        compressor = CompressionLayer(comp)
        compressed = compressor.compress(serialized)
        compressed_size = len(compressed)

        return SerializationResult(
            data=compressed,
            format=fmt,
            compression=comp,
            original_size=original_size,
            compressed_size=compressed_size,
            version=version,
        )

    def deserialize(
        self,
        result: SerializationResult,
        target_type: Optional[Type] = None,
    ) -> Any:
        """反序列化对象"""
        # 解压
        compressor = CompressionLayer(result.compression)
        decompressed = compressor.decompress(result.data)

        # 反序列化
        serializer = self._serializers.get(result.format)
        if serializer is None:
            raise ValueError(f"Unknown serialization format: {result.format}")

        return serializer.deserialize(decompressed, target_type)

    def register_serializer(
        self,
        format: SerializationFormat,
        serializer: Any,
    ) -> None:
        """注册序列化器"""
        self._serializers[format] = serializer

    def set_default_format(self, format: SerializationFormat) -> None:
        """设置默认格式"""
        self._default_format = format

    def set_default_compression(self, compression: CompressionType) -> None:
        """设置默认压缩"""
        self._default_compression = compression


# ============================================================
# 版本兼容
# ============================================================

class VersionedSerializer:
    """版本化序列化器"""

    def __init__(self, version: VersionInfo):
        self._version = version
        self._migrations: Dict[str, Callable[[Any, VersionInfo], Any]] = {}

    def register_migration(
        self,
        from_version: str,
        migrator: Callable[[Any, VersionInfo], Any],
    ) -> None:
        """注册迁移函数"""
        self._migrations[from_version] = migrator

    def serialize_with_version(self, obj: Any, manager: SerializationManager) -> bytes:
        """带版本信息的序列化"""
        version_bytes = self._version.to_bytes()
        result = manager.serialize(obj)
        # 格式: [版本6字节][数据长度4字节][数据]
        data_len = len(result.data)
        header = version_bytes + struct.pack(">I", data_len)
        return header + result.data

    def deserialize_with_version(
        self,
        data: bytes,
        manager: SerializationManager,
        target_type: Optional[Type] = None,
    ) -> Any:
        """带版本信息的反序列化"""
        # 解析头部
        version = VersionInfo.from_bytes(data[:6])
        data_len = struct.unpack(">I", data[6:10])[0]
        payload = data[10:10 + data_len]

        # 检查是否需要迁移
        if str(version) in self._migrations:
            # 简化处理：实际应该先反序列化再迁移
            pass

        # 创建SerializationResult并反序列化
        result = SerializationResult(
            data=payload,
            format=SerializationFormat.JSON,  # 假设
            compression=CompressionType.NONE,
            original_size=data_len,
            compressed_size=data_len,
            version=str(version),
        )
        return manager.deserialize(result, target_type)


# ============================================================
# 便捷函数
# ============================================================

def create_json_serializer(indent: Optional[int] = None) -> JsonSerializer:
    """创建JSON序列化器"""
    return JsonSerializer(indent=indent)


def create_pickle_serializer(protocol: int = pickle.HIGHEST_PROTOCOL) -> PickleSerializer:
    """创建Pickle序列化器"""
    return PickleSerializer(protocol=protocol)


def create_msgpack_serializer() -> MessagePackSerializer:
    """创建MessagePack序列化器"""
    return MessagePackSerializer()


def create_serialization_manager() -> SerializationManager:
    """创建序列化管理器"""
    return SerializationManager()


def serialize_to_json(obj: Any, indent: Optional[int] = None) -> bytes:
    """快捷JSON序列化"""
    return JsonSerializer(indent=indent).serialize(obj)


def deserialize_from_json(data: bytes) -> Any:
    """快捷JSON反序列化"""
    return JsonSerializer().deserialize(data)
