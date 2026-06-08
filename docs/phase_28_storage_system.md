# 阶段28: Agent存储与持久化 (Agent Storage & Persistence)

## 概述

阶段28实现了Agent存储与持久化系统，支持多种存储后端。系统提供统一的存储接口、内存存储、文件存储、TTL过期机制、命名空间隔离等功能。

## 核心组件

### 1. StorageBackend (存储后端抽象接口)

定义所有存储后端必须实现的接口：

```python
from src.agent import StorageBackend

class MyStorage(StorageBackend):
    def get(self, key, namespace="default"): ...
    def put(self, entry): ...
    def delete(self, key, namespace="default"): ...
    def exists(self, key, namespace="default"): ...
    def keys(self, namespace="default"): ...
    def count(self, namespace="default"): ...
    def clear(self, namespace=None): ...
    def close(self): ...
```

### 2. StorageEntry (存储条目)

表示一个存储的数据项：

```python
from src.agent import StorageEntry

entry = StorageEntry(
    key="user_001",
    value={"name": "张三", "age": 30},
    namespace="users",
    ttl=3600,  # 1小时过期
    metadata={"source": "api"},
)

# 检查过期
entry.is_expired()  # False

# 更新时间戳
entry.touch()
```

### 3. MemoryStorage (内存存储)

数据存储在内存中，速度快但重启后丢失：

```python
from src.agent import MemoryStorage, StorageEntry

storage = MemoryStorage()

# 存储数据
entry = StorageEntry(key="key_1", value="value_1")
storage.put(entry)

# 获取数据
result = storage.get("key_1")
print(result.value)  # "value_1"

# 检查存在
storage.exists("key_1")  # True

# 获取所有键
keys = storage.keys()  # ["key_1"]

# 获取数量
count = storage.count()  # 1

# 删除
storage.delete("key_1")

# 清除命名空间
storage.clear(namespace="default")

# 清除所有
storage.clear()
```

### 4. FileStorage (文件存储)

数据持久化到文件系统：

```python
from src.agent import FileStorage, StorageEntry

storage = FileStorage(base_path="./data/storage")

# 存储数据（自动创建目录和JSON文件）
entry = StorageEntry(key="config", value={"debug": True})
storage.put(entry)

# 获取数据
result = storage.get("config")
print(result.value)  # {"debug": True}

# 支持复杂数据类型
complex_entry = StorageEntry(
    key="data",
    value={"list": [1, 2, 3], "nested": {"key": "value"}},
)
storage.put(complex_entry)
```

### 5. StorageManager (存储管理器)

统一管理多个存储后端：

```python
from src.agent import StorageManager, MemoryStorage, FileStorage

manager = StorageManager()

# 注册后端
manager.register_backend("memory", MemoryStorage())
manager.register_backend("file", FileStorage("./data/storage"))

# 设置默认后端
manager.set_default("memory")

# 便捷操作（使用默认后端）
manager.put("key_1", "value_1", namespace="default", ttl=60)
value = manager.get_value("key_1")  # "value_1"
value = manager.get_value("nonexistent", default="default")  # "default"

# 获取完整条目
entry = manager.get("key_1")

# 检查、删除、计数
manager.exists("key_1")  # True
manager.delete("key_1")
manager.count()  # 0
manager.keys()  # []

# 跨后端复制
manager.copy_between("key_1", "memory", "file")

# 同步到目标后端
count = manager.sync_to("file")

# 获取统计
stats = manager.get_stats()

# 关闭所有后端
manager.close_all()
```

## 特性

### TTL过期机制

```python
from src.agent import StorageEntry, MemoryStorage

storage = MemoryStorage()

# 设置5秒过期
entry = StorageEntry(key="temp", value="data", ttl=5)
storage.put(entry)

# 5秒后自动失效
import time
time.sleep(6)
storage.get("temp")  # None
```

### 命名空间隔离

```python
from src.agent import MemoryStorage, StorageEntry

storage = MemoryStorage()

# 不同命名空间
storage.put(StorageEntry(key="key_1", value="v1", namespace="users"))
storage.put(StorageEntry(key="key_1", value="v2", namespace="sessions"))

storage.get("key_1", "users").value  # "v1"
storage.get("key_1", "sessions").value  # "v2"

# 清除特定命名空间
storage.clear(namespace="users")
```

## 最佳实践

1. **内存存储**：适合缓存、临时数据
2. **文件存储**：适合持久化、配置数据
3. **命名空间**：按模块/功能划分命名空间
4. **TTL**：为临时数据设置合理的过期时间
5. **多后端**：使用内存缓存 + 文件持久化的组合策略

## 线程安全

- MemoryStorage 使用 threading.RLock
- FileStorage 使用 threading.RLock
- StorageManager 使用 threading.RLock

## 测试覆盖

- 63个测试用例，覆盖所有核心功能
- 包括条目、内存存储、文件存储、管理器、异常、导出测试
- 文件存储使用临时目录，测试后自动清理

## 总结

阶段28成功实现了Agent存储与持久化系统，提供了：
- 统一的存储抽象接口
- 内存和文件两种存储后端
- TTL过期机制
- 命名空间隔离
- 多后端管理和数据同步
- 线程安全的实现