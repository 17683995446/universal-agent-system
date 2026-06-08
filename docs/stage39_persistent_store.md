# 阶段39: Agent数据持久化存储系统

> 遵循"图难于其易，为大于其细。天下难事，必作于易；天下大事，必作于细。"的原则，在阶段38序列化系统基础上，实现二进制持久化存储、快照管理、数据恢复。

## 39.1 持久化模型

### 核心枚举

```python
class PersistentStoreType(str, Enum):
    MEMORY = "memory"           # 内存存储
    FILE_BINARY = "file_binary" # 文件二进制存储
    FILE_JSON = "file_json"     # 文件JSON存储

class StoreStatus(str, Enum):
    ACTIVE = "active"       # 活跃状态
    READ_ONLY = "read_only" # 只读状态
    CLOSED = "closed"       # 已关闭
    ERROR = "error"         # 错误状态
```

### StoreEntry - 持久化存储条目

```python
@dataclass
class StoreEntry:
    key: str
    value: Any
    namespace: str = "default"
    created_at: float
    updated_at: float
    version: int = 1
    checksum: str = ""       # SHA256校验和（前16位）
    size_bytes: int = 0
    metadata: Dict[str, Any]
    ttl: Optional[float]     # 生存时间（秒）
```

### StoreConfig - 存储配置

```python
@dataclass
class StoreConfig:
    store_type: PersistentStoreType = PersistentStoreType.MEMORY
    base_path: str = "./data/persistent"
    serialization_format: SerializationFormat = SerializationFormat.JSON
    compression: CompressionType = CompressionType.NONE
    max_entries: int = 100000
    auto_flush_interval: float = 30.0
    enable_checksum: bool = True
    enable_versioning: bool = True
    max_versions_per_key: int = 10
```

### SnapshotInfo / RecoveryResult / StoreStats

```python
@dataclass
class SnapshotInfo:
    snapshot_id: str       # 快照ID（8位UUID）
    created_at: float
    entry_count: int
    total_size: int
    description: str

@dataclass
class RecoveryResult:
    success: bool
    recovered_entries: int
    failed_entries: int
    errors: List[str]

@dataclass
class StoreStats:
    total_entries: int
    total_size_bytes: int
    namespace_count: int
```

## 39.2 内存持久化存储

### MemoryPersistentStore

```python
from src.agent.persistent_store import create_memory_store, StoreConfig

# 基本使用
store = create_memory_store()
store.put("key", "value")
entry = store.get("key")
print(entry.value)  # "value"

# 带配置
config = StoreConfig(max_entries=1000, enable_checksum=True)
store = create_memory_store(config)

# 命名空间隔离
store.put("key", "ns1_value", namespace="users")
store.put("key", "ns2_value", namespace="config")
# 不同命名空间互不影响

# TTL过期
store.put("temp", "data", ttl=60)  # 60秒后过期

# 统计
stats = store.get_stats()
print(f"总条目: {stats.total_entries}")
print(f"总大小: {stats.total_size_bytes}")
```

### 特性

| 特性 | 说明 |
|------|------|
| 命名空间隔离 | 不同namespace数据完全隔离 |
| TTL过期 | 自动检测并清理过期条目 |
| 版本控制 | 每次更新自动递增版本号 |
| 校验和 | SHA256前16位校验 |
| 容量限制 | max_entries控制最大条目数 |
| 线程安全 | RLock保护所有操作 |

## 39.3 文件持久化存储

### FilePersistentStore

```python
from src.agent.persistent_store import create_file_store
from src.agent.serialization import SerializationFormat, CompressionType

# JSON格式
store = create_file_store(base_path="./data/store")

# 带GZIP压缩
store = create_file_store(
    base_path="./data/compressed",
    format=SerializationFormat.JSON,
    compression=CompressionType.GZIP,
)

# 基本操作
store.put("config", {"theme": "dark", "lang": "zh"})
entry = store.get("config")
print(entry.value)  # {"theme": "dark", "lang": "zh"}

# 跨实例持久化
store.close()
store2 = create_file_store(base_path="./data/store")
entry = store2.get("config")  # 数据仍然存在
```

### 存储结构

```
data/persistent/
├── default/                    # 命名空间目录
│   ├── config.bin              # 数据文件（二进制序列化）
│   ├── config.meta.json        # 元数据文件
│   └── user_1.bin
├── users/
│   ├── alice.bin
│   └── alice.meta.json
```

### 元数据格式

```json
{
    "key": "config",
    "namespace": "default",
    "created_at": 1716000000.0,
    "updated_at": 1716000100.0,
    "version": 3,
    "checksum": "a1b2c3d4e5f6g7h8",
    "size_bytes": 128,
    "metadata": {"source": "api"},
    "ttl": null
}
```

## 39.4 快照管理器

### SnapshotManager

```python
from src.agent.persistent_store import (
    create_memory_store,
    PersistentStoreManager,
)

store = create_memory_store()
manager = PersistentStoreManager(store)

# 写入数据
store.put("user:1", {"name": "Alice", "score": 100})
store.put("user:2", {"name": "Bob", "score": 85})

# 创建快照
sm = manager.get_snapshot_manager()
snapshot = sm.create_snapshot(description="备份前修改")

# 修改数据
store.put("user:1", {"name": "Alice", "score": 99})

# 恢复快照
result = sm.restore_snapshot(snapshot.snapshot_id)
print(f"恢复成功: {result.success}")
print(f"恢复条目: {result.recovered_entries}")

# 列出所有快照
snapshots = sm.list_snapshots()
for s in snapshots:
    print(f"{s.snapshot_id}: {s.description} ({s.entry_count}条目)")

# 删除快照
sm.delete_snapshot(snapshot.snapshot_id)
```

### 快照存储结构

```
data/snapshots/
└── a1b2c3d4/                  # 快照ID
    ├── manifest.json           # 清单文件
    ├── default/                # 命名空间
    │   ├── user_1.snap.gz      # GZIP压缩数据
    │   └── user_1.meta.json    # 元数据
    └── users/
        ├── alice.snap.gz
        └── alice.meta.json
```

### 恢复到不同存储

```python
# 恢复到新的内存存储
target = create_memory_store()
result = sm.restore_snapshot(snapshot.snapshot_id, target_store=target)

# 恢复到文件存储
target = create_file_store(base_path="./data/restored")
result = sm.restore_snapshot(snapshot.snapshot_id, target_store=target)
```

## 持久化存储管理器

### PersistentStoreManager

```python
from src.agent.persistent_store import (
    PersistentStoreManager,
    create_memory_store,
    create_file_store,
)

manager = PersistentStoreManager()

# 注册多个存储
mem = create_memory_store()
file = create_file_store(base_path="./data/store")
manager.register_store("memory", mem)
manager.register_store("file", file)

# 便捷操作（使用默认存储）
manager.put("key", "value")
entry = manager.get("key")

# 获取快照管理器
sm = manager.get_snapshot_manager("memory")
sm.create_snapshot("backup")

# 关闭所有
manager.close_all()
```

## 便捷函数

```python
# 创建内存存储
store = create_memory_store()

# 创建文件存储
store = create_file_store(base_path="./data")

# 创建管理器
manager = create_persistent_manager(default_store=store)
```

## 测试覆盖

阶段39包含79个测试用例：

- **持久化模型测试**: 16个测试
- **内存持久化存储测试**: 21个测试
- **文件持久化存储测试**: 17个测试
- **快照管理器测试**: 10个测试
- **管理器测试**: 6个测试
- **便捷函数测试**: 4个测试
- **集成测试**: 5个测试

运行测试:
```bash
PYTHONPATH=/workspace pytest tests/test_persistent_store.py -v
```

## 架构图

```
┌──────────────────────────────────────────────────────────────┐
│                   PersistentStoreManager                      │
├──────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐                  │
│  │MemoryPersistent  │  │FilePersistent    │                  │
│  │    Store          │  │    Store          │                  │
│  │  ┌────────────┐  │  │  ┌────────────┐  │                  │
│  │  │ Dict Store │  │  │  │ File System │  │                  │
│  │  │ + Lock     │  │  │  │ + Serializer│  │                  │
│  │  │ + TTL      │  │  │  │ + Cache    │  │                  │
│  │  └────────────┘  │  │  └────────────┘  │                  │
│  └──────────────────┘  └──────────────────┘                  │
├──────────────────────────────────────────────────────────────┤
│                    SnapshotManager                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐    │
│  │  Create    │  │  Restore   │  │  List / Delete      │    │
│  │  Snapshot  │  │  Snapshot  │  │  Snapshots          │    │
│  └────────────┘  └────────────┘  └────────────────────┘    │
├──────────────────────────────────────────────────────────────┤
│              依赖: 阶段38 SerializationManager                │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐    │
│  │   JSON     │  │  Pickle    │  │  GZIP Compression  │    │
│  └────────────┘  └────────────┘  └────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

## 下一步

阶段39完成，系统已具备：
- ✅ 内存持久化存储（命名空间、TTL、版本控制、校验和）
- ✅ 文件持久化存储（基于序列化系统、GZIP压缩）
- ✅ 快照创建与恢复
- ✅ 统一的持久化存储管理器

下一阶段（阶段40）将实现 **Agent数据导出与导入系统**，支持跨系统数据迁移。
