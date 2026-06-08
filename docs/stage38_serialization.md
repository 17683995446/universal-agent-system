# 阶段38: Agent序列化系统

> 遵循"图难于其易，为大于其细。天下难事，必作于易；天下大事，必作于细。"的原则，实现多种序列化格式、版本兼容、压缩。

## 38.1 序列化模型

### 核心枚举

```python
class SerializationFormat(str, Enum):
    JSON = "json"           # JSON格式，人类可读
    PICKLE = "pickle"       # Python原生二进制格式
    MESSAGEPACK = "msgpack" # MessagePack二进制格式
    RAW = "raw"             # 原始字节

class CompressionType(str, Enum):
    NONE = "none"   # 无压缩
    GZIP = "gzip"   # GZIP压缩
    LZ4 = "lz4"     # LZ4压缩（快速）
```

### SerializationResult - 序列化结果

```python
@dataclass
class SerializationResult:
    data: bytes                  # 序列化后的数据
    format: SerializationFormat  # 序列化格式
    compression: CompressionType # 压缩类型
    original_size: int           # 原始大小
    compressed_size: int         # 压缩后大小
    version: str = "1.0"         # 数据版本
    timestamp: datetime          # 序列化时间
    metadata: Dict[str, Any]     # 元数据

    @property
    def compression_ratio(self) -> float:
        """压缩比率"""
        return self.compressed_size / self.original_size
```

### VersionInfo - 版本信息

```python
@dataclass
class VersionInfo:
    major: int      # 主版本
    minor: int      # 次版本
    patch: int      # 补丁版本
    schema_hash: str

    def to_bytes(self) -> bytes:
        """序列化为6字节"""
        return struct.pack(">HHH", self.major, self.minor, self.patch)

    @classmethod
    def from_bytes(cls, data: bytes) -> "VersionInfo":
        """从字节反序列化"""
        major, minor, patch = struct.unpack(">HHH", data[:6])
        return cls(major=major, minor=minor, patch=patch)
```

## 38.2 JSON序列化

### JsonSerializer - JSON序列化器

```python
serializer = JsonSerializer(indent=2, ensure_ascii=False)

# 基本序列化
data = {"name": "test", "value": 123}
serialized = serializer.serialize(data)  # bytes
restored = serializer.deserialize(serialized)

# 支持特殊类型
serializer.serialize({
    "time": datetime.now(),    # 自动编码
    "content": b"bytes",       # Base64编码
    "items": {1, 2, 3},        # 集合编码
})

# 自定义编码器
serializer.register_encoder(CustomType, lambda obj: {"value": obj.value})
serializer.register_decoder("CustomType", lambda dct, _: CustomType(dct["value"]))
```

### 内置类型支持

| 类型 | 编码方式 | 示例 |
|------|----------|------|
| datetime | ISO格式 | `{"__type__": "datetime", "iso": "2024-01-15T10:30:00"}` |
| bytes | Base64 | `{"__type__": "bytes", "b64": "aGVsbG8="}` |
| set | 转列表 | `{"__type__": "set", "items": [1, 2, 3]}` |

## 38.3 二进制序列化

### PickleSerializer - Pickle序列化器

```python
serializer = PickleSerializer(protocol=pickle.HIGHEST_PROTOCOL)

# 序列化任意Python对象
data = {"complex": [1, 2, 3], "nested": {"a": 1}}
serialized = serializer.serialize(data)
restored = serializer.deserialize(serialized)
```

### MessagePackSerializer - MessagePack序列化器

```python
serializer = MessagePackSerializer()

if serializer.is_available():
    data = {"name": "test", "items": [1, 2, 3]}
    serialized = serializer.serialize(data)
    restored = serializer.deserialize(serialized)
```

> **注意**: MessagePack需要安装 `msgpack` 包

## 38.4 压缩层

### CompressionLayer - 压缩层

```python
# 无压缩
layer = CompressionLayer(CompressionType.NONE)
compressed = layer.compress(data)  # 原样返回

# GZIP压缩
layer = CompressionLayer(CompressionType.GZIP)
compressed = layer.compress(data)  # GZIP压缩

# LZ4压缩（需要安装lz4）
layer = CompressionLayer(CompressionType.LZ4)
compressed = layer.compress(data)  # LZ4压缩
```

### 压缩效果对比

```python
manager = SerializationManager()
data = {"text": "repeated text " * 1000}

# 无压缩
none_result = manager.serialize(data, compression=CompressionType.NONE)
print(f"无压缩: {none_result.compressed_size} bytes")

# GZIP压缩
gzip_result = manager.serialize(data, compression=CompressionType.GZIP)
print(f"GZIP: {gzip_result.compressed_size} bytes")
print(f"压缩率: {gzip_result.compression_ratio:.2%}")
```

## 序列化管理器

### SerializationManager - 统一管理

```python
manager = SerializationManager()

# 序列化（使用默认格式JSON）
result = manager.serialize({"key": "value"})

# 指定格式
result = manager.serialize(data, format=SerializationFormat.PICKLE)

# 指定压缩
result = manager.serialize(data, compression=CompressionType.GZIP)

# 反序列化
restored = manager.deserialize(result)

# 设置默认
manager.set_default_format(SerializationFormat.PICKLE)
manager.set_default_compression(CompressionType.GZIP)
```

### 完整工作流示例

```python
from src.agent.serialization import (
    SerializationManager,
    SerializationFormat,
    CompressionType,
)

# 创建管理器
manager = SerializationManager()

# 准备数据
data = {
    "users": [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ],
    "metadata": {"version": "1.0"},
}

# 序列化并压缩
result = manager.serialize(
    data,
    format=SerializationFormat.JSON,
    compression=CompressionType.GZIP,
)

print(f"原始大小: {result.original_size}")
print(f"压缩后: {result.compressed_size}")
print(f"压缩率: {result.compression_ratio:.2%}")

# 反序列化
restored = manager.deserialize(result)
assert restored == data
```

## 版本化序列化

### VersionedSerializer - 带版本控制

```python
# 定义版本
version = VersionInfo(major=1, minor=0, patch=0)
serializer = VersionedSerializer(version)

# 序列化（带版本头）
serialized = serializer.serialize_with_version(data, manager)
# 格式: [版本6字节][数据长度4字节][数据]

# 反序列化（自动识别版本）
restored = serializer.deserialize_with_version(serialized, manager)

# 注册迁移函数
def migrate_v1_to_v2(data, version):
    data["new_field"] = "default"
    return data

serializer.register_migration("1.0.0", migrate_v1_to_v2)
```

## 便捷函数

```python
# 快速JSON序列化
serialized = serialize_to_json({"key": "value"})
restored = deserialize_from_json(serialized)

# 创建特定序列化器
json_ser = create_json_serializer(indent=2)
pickle_ser = create_pickle_serializer()
msgpack_ser = create_msgpack_serializer()

# 创建管理器
manager = create_serialization_manager()
```

## 性能对比

```python
import time

manager = SerializationManager()
data = {"numbers": list(range(1000)), "text": "hello" * 100}

# JSON
start = time.time()
for _ in range(1000):
    result = manager.serialize(data, format=SerializationFormat.JSON)
    manager.deserialize(result)
print(f"JSON: {time.time() - start:.3f}s")

# Pickle
start = time.time()
for _ in range(1000):
    result = manager.serialize(data, format=SerializationFormat.PICKLE)
    manager.deserialize(result)
print(f"Pickle: {time.time() - start:.3f}s")
```

## 测试覆盖

阶段38包含50个测试用例：

- **序列化模型测试**: 9个测试
- **JSON序列化测试**: 9个测试
- **二进制序列化测试**: 7个测试
- **压缩层测试**: 3个测试
- **管理器测试**: 7个测试
- **版本化序列化测试**: 4个测试
- **便捷函数测试**: 6个测试
- **集成测试**: 5个测试

运行测试:
```bash
pytest tests/test_serialization.py -v
```

## 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    SerializationManager                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ JsonSerializer│ │PickleSerializer│ │MessagePackSerializer│  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    CompressionLayer                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   NONE      │  │    GZIP     │  │        LZ4          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  SerializationResult                         │
│  ├─ data: bytes                                              │
│  ├─ format: SerializationFormat                              │
│  ├─ compression: CompressionType                             │
│  ├─ original_size: int                                       │
│  ├─ compressed_size: int                                     │
│  └─ version: str                                             │
└─────────────────────────────────────────────────────────────┘
```

## 下一步

阶段38完成，系统已具备：
- ✅ 多种序列化格式（JSON、Pickle、MessagePack）
- ✅ 压缩支持（GZIP、LZ4）
- ✅ 版本控制与迁移
- ✅ 统一的管理接口

下一阶段（阶段39）将实现 **Agent数据持久化存储**，基于序列化系统构建可靠的持久化机制。
