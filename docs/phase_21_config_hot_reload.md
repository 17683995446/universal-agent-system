# 阶段21：Agent配置热更新

## 概述

实现运行时配置热更新、配置验证、变更检测和回滚功能。

**原则遵循**："图难于其易，为大于其细。天下难事，必作于易；天下大事，必作于细。"

---

## 核心设计

### 配置变更类型

```
ADDED    → 新增配置项
REMOVED  → 删除配置项
MODIFIED → 修改配置项
```

### 需要重启的配置项

| 配置项 | 说明 |
|--------|------|
| agent_id | Agent标识 |
| llm_config.model | LLM模型 |
| llm_config.base_url | LLM服务地址 |
| memory_config.vector_db_url | 向量数据库地址 |

---

## 实现内容

### 21.1 ConfigChangeType枚举

**文件**：`/workspace/src/agent/config_manager.py`

```python
class ConfigChangeType(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"
```

### 21.2 ConfigChange变更记录

```python
@dataclass
class ConfigChange:
    key: str                          # 配置键路径
    change_type: ConfigChangeType
    old_value: Any = None
    new_value: Any = None
    requires_restart: bool = False    # 是否需要重启
```

### 21.3 ConfigUpdateResult更新结果

```python
@dataclass
class ConfigUpdateResult:
    success: bool
    changes: list[ConfigChange]
    errors: list[str]
    timestamp: float
    rollback_available: bool

    @property
    def has_changes -> bool
    @property
    def requires_restart -> bool
```

### 21.4 ConfigValidator验证器

```python
class ConfigValidator:
    @classmethod
    def validate(config) -> list[str]      # 返回错误列表
    @classmethod
    def requires_restart(key) -> bool      # 检查是否需要重启
```

### 21.5 ConfigManager配置管理器

```python
class ConfigManager:
    def __init__(self, initial_config, max_history=10)

    @property
    def current_config -> AgentConfig      # 当前配置（深拷贝）
    @property
    def update_count -> int

    def update(new_config, validate=True) -> ConfigUpdateResult
    def update_partial(updates: dict, validate=True) -> ConfigUpdateResult
    def rollback() -> ConfigUpdateResult
    def can_rollback() -> bool
    def get_history(limit) -> list[dict]
    def get_stats() -> dict
```

### 21.6 Agent集成

**修改文件**：`/workspace/src/agent/agent.py`

```python
class Agent:
    def _init_config_manager(self):
        self._config_manager = ConfigManager(self._config)

    def update_config(new_config, validate=True) -> ConfigUpdateResult
    def update_config_partial(updates: dict, validate=True) -> ConfigUpdateResult
    def rollback_config() -> ConfigUpdateResult

    @property
    def config_manager -> Optional[ConfigManager]
```

**配置变更事件**：
```python
AgentEventType.CONFIG_CHANGED

# 更新时发射
event_bus.emit(
    AgentEventType.CONFIG_CHANGED,
    data={
        "changes": [...],
        "requires_restart": True/False,
    }
)
```

---

## 测试结果

**文件**：`/workspace/tests/test_config_manager.py`

| 测试类 | 测试数 | 结果 |
|--------|--------|------|
| TestConfigChangeType | 1 | ✅ |
| TestConfigChange | 2 | ✅ |
| TestConfigUpdateResult | 3 | ✅ |
| TestConfigValidator | 5 | ✅ |
| TestConfigManager | 12 | ✅ |
| TestAgentConfigIntegration | 4 | ✅ |
| TestExportConfigManager | 3 | ✅ |

**总计**：29 passed ✅

**全系统测试**：800 passed（新增29个），2 failed（阶段15遗留）。

---

## 使用示例

```python
from src.agent import Agent, AgentConfig

# 创建Agent
config = AgentConfig(name="MyAgent", max_turns=10)
agent = Agent(config=config)

# 完整更新配置
new_config = AgentConfig(name="UpdatedAgent", max_turns=20)
result = agent.update_config(new_config)

if result.success:
    print(f"Updated: {len(result.changes)} changes")
    if result.requires_restart:
        print("Warning: Restart required for some changes")

# 部分更新
result = agent.update_config_partial({
    "max_turns": 30,
    "enable_memory": False,
})

# 回滚配置
if agent.config_manager.can_rollback():
    result = agent.rollback_config()
    print("Rolled back successfully")

# 查看历史
history = agent.config_manager.get_history(limit=5)
for entry in history:
    print(f"{entry['timestamp']}: {entry['config_summary']['name']}")
```

---

## 架构关系

```
Agent
  ├── _config_manager: ConfigManager
  │       ├── current_config (深拷贝)
  │       ├── _history: list[(config, timestamp)]
  │       ├── update() → ConfigUpdateResult
  │       ├── update_partial() → ConfigUpdateResult
  │       ├── rollback() → ConfigUpdateResult
  │       └── _detect_changes() → list[ConfigChange]
  │
  ├── update_config() → 调用 _config_manager.update()
  │       └── emit CONFIG_CHANGED 事件
  ├── update_config_partial() → 调用 _config_manager.update_partial()
  └── rollback_config() → 调用 _config_manager.rollback()

ConfigValidator
  ├── validate() → list[str] (错误列表)
  └── requires_restart(key) → bool
```

---

## 下阶段规划

**阶段22**：Agent性能监控

- 实现性能指标收集
- 添加性能报告生成
- 实现性能告警

---

## 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `/workspace/src/agent/config_manager.py` | 新建 | 配置管理核心 |
| `/workspace/src/agent/agent.py` | 修改 | 集成热更新方法 |
| `/workspace/src/agent/event_bus.py` | 修改 | 添加CONFIG_CHANGED事件 |
| `/workspace/src/agent/__init__.py` | 修改 | 导出配置管理类 |
| `/workspace/tests/test_config_manager.py` | 新建 | 29个测试 |

---

**阶段21完成** ✅
