# 阶段18：Agent生命周期管理

## 概述

实现Agent完整生命周期管理，包括状态机、资源清理、快照保存/恢复功能。

**原则遵循**："图难于其易，为大于其细。天下难事，必作于易；天下大事，必作于细。"

---

## 核心设计

### Agent状态机

```
                    ┌──────────┐
         ┌─────────│  PAUSED  │
         │         └────┬─────┘
         │              │ resume
    pause│              ▼
         │    ┌──────────────────┐
         └───▶│      IDLE        │◀────────┐
              └────┬─────────────┘         │
                   │ chat()                 │ restart
                   ▼                        │
              ┌──────────────────┐          │
              │    THINKING      │──────────┤
              └────┬─────────────┘  stop    │
                   │ execute_tool()         │
                   ▼                        │
              ┌──────────────────┐          │
              │    EXECUTING     │──────────┘
              └──────────────────┘
                   │    │
                   ▼    ▼
              ┌──────────────────┐
              │      ERROR       │
              └──────────────────┘
                   │
                   ▼
              ┌──────────────────┐
              │     STOPPED      │
              └──────────────────┘
```

### 状态转换规则

| 当前状态 | 可转换到 |
|----------|----------|
| IDLE | THINKING, PAUSED, ERROR, STOPPED |
| THINKING | EXECUTING, IDLE, ERROR, STOPPED |
| EXECUTING | THINKING, IDLE, ERROR, STOPPED |
| PAUSED | IDLE, THINKING, ERROR, STOPPED |
| ERROR | IDLE, STOPPED |
| STOPPED | IDLE (重启) |

---

## 实现内容

### 18.1 AgentState扩展

**文件**：`/workspace/src/agent/agent_types.py`

```python
class AgentState(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"

    @classmethod
    def can_transition(cls, from_state, to_state) -> bool

    @property
    def is_active(self) -> bool  # IDLE/THINKING/EXECUTING

    @property
    def is_terminal(self) -> bool  # STOPPED
```

### 18.2 AgentSnapshot快照类

```python
@dataclass
class AgentSnapshot:
    version: str = "1.0"
    snapshot_id: str = ""
    created_at: float
    agent_id: str = ""
    agent_name: str = ""
    state: str = "idle"
    config_dict: dict
    conversation_history: list
    stats_dict: dict
    task_context: dict
    compression_stats: dict
    metadata: dict

    def to_dict() -> dict
    def to_json() -> str
    @classmethod
    def from_dict(cls, data) -> "AgentSnapshot"
    @classmethod
    def from_json(cls, json_str) -> "AgentSnapshot"
    def is_valid() -> bool
    def get_summary() -> dict
```

### 18.3 AgentStats序列化支持

```python
@dataclass
class AgentStats:
    ...
    @classmethod
    def from_dict(cls, data) -> "AgentStats"
```

### 18.4 Agent生命周期方法

**文件**：`/workspace/src/agent/agent.py`

#### 状态管理
```python
def _transition_state(self, new_state: AgentState) -> bool
    """状态转换（带验证）"""

def reset(self) -> None
    """重置Agent（强制到IDLE）"""

def stop(self) -> None
    """停止Agent"""

def pause(self) -> bool
    """暂停Agent"""

def resume(self) -> bool
    """恢复Agent"""

def restart(self) -> bool
    """从STOPPED重启到IDLE"""
```

#### 资源清理
```python
def cleanup(self) -> None
    """优雅关闭，释放所有资源"""

def _cleanup_components(self) -> None
    """清理各组件：
    - MemoryCoordinator.clear_all()
    - TaskContextManager.clear_task()
    - ToolExecutor.clear_history()
    - ContextBuilder.reset_stats()
    - ContextCompressor.reset_stats()
    """
```

#### 快照管理
```python
def create_snapshot(self) -> AgentSnapshot
    """创建当前状态快照"""

def save_snapshot(self, filepath=None) -> str
    """保存快照到文件（默认/worskspace/snapshots/）"""

def restore_from_snapshot(self, snapshot: AgentSnapshot) -> bool
    """从快照对象恢复"""

def load_snapshot(self, filepath: str) -> bool
    """从文件加载并恢复"""
```

---

## 测试结果

**文件**：`/workspace/tests/test_agent_lifecycle.py`

| 测试类 | 测试数 | 结果 |
|--------|--------|------|
| TestAgentStateTransitions | 4 | ✅ |
| TestAgentSnapshot | 6 | ✅ |
| TestAgentStatsSerialization | 2 | ✅ |
| TestAgentLifecycle | 9 | ✅ |
| TestAgentSnapshotExport | 1 | ✅ |

**总计**：22 passed ✅

**全系统测试**：696 passed（新增22个），2 failed（阶段15遗留）。

---

## 使用示例

```python
from src.agent import Agent, AgentConfig, AgentState

# 创建Agent
config = AgentConfig(agent_id="my_agent", name="MyAgent")
agent = Agent(config=config)

# 状态管理
assert agent.state == AgentState.IDLE

agent.pause()
assert agent.state == AgentState.PAUSED

agent.resume()
assert agent.state == AgentState.IDLE

agent.stop()
assert agent.state == AgentState.STOPPED

agent.restart()
assert agent.state == AgentState.IDLE

# 创建快照
snapshot = agent.create_snapshot()
print(snapshot.get_summary())

# 保存快照
filepath = agent.save_snapshot()  # 默认保存到 /workspace/snapshots/

# 加载快照
new_agent = Agent(config=config)
new_agent.load_snapshot(filepath)

# 资源清理
agent.cleanup()  # 优雅关闭
```

---

## 架构关系

```
Agent
  ├── _state: AgentState
  │     ├── IDLE/THINKING/EXECUTING/PAUSED/ERROR/STOPPED
  │     ├── can_transition(from, to) → bool
  │     ├── is_active → bool
  │     └── is_terminal → bool
  │
  ├── _transition_state(new_state) → bool
  ├── reset() / stop() / pause() / resume() / restart()
  ├── cleanup() → 优雅关闭
  │     └── _cleanup_components()
  │           ├── MemoryCoordinator.clear_all()
  │           ├── TaskContextManager.clear_task()
  │           ├── ToolExecutor.clear_history()
  │           ├── ContextBuilder.reset_stats()
  │           └── ContextCompressor.reset_stats()
  │
  ├── create_snapshot() → AgentSnapshot
  ├── save_snapshot(filepath) → str
  ├── restore_from_snapshot(snapshot) → bool
  └── load_snapshot(filepath) → bool

AgentSnapshot
  ├── to_dict() / to_json()
  ├── from_dict() / from_json()
  ├── is_valid() → bool
  └── get_summary() → dict
```

---

## 下阶段规划

**阶段19**：错误处理与恢复机制

- 实现分级错误处理策略
- 添加自动重试和降级机制
- 实现错误日志和诊断

---

## 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `/workspace/src/agent/agent_types.py` | 修改 | AgentState扩展 + AgentSnapshot |
| `/workspace/src/agent/agent.py` | 修改 | 生命周期方法 |
| `/workspace/src/agent/__init__.py` | 修改 | 导出AgentSnapshot |
| `/workspace/tests/test_agent_lifecycle.py` | 新建 | 22个测试 |

---

**阶段18完成** ✅
