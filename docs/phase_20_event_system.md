# 阶段20：Agent事件系统

## 概述

实现事件总线（EventBus）、核心事件类型定义、发布/订阅机制，为Agent系统提供松耦合的事件驱动能力。

**原则遵循**："图难于其易，为大于其细。天下难事，必作于易；天下大事，必作于细。"

---

## 核心设计

### 事件类型分类（26种）

| 分类 | 事件 | 数量 |
|------|------|------|
| 生命周期 | STATE_CHANGED, AGENT_STARTED/STOPPED/RESET/PAUSED/RESUMED | 6 |
| 对话 | TURN_STARTED/COMPLETED/ERROR, USER_INPUT, ASSISTANT_REPLY | 6 |
| 工具 | TOOL_CALLED, TOOL_RESULT, TOOL_ERROR | 3 |
| 记忆 | MEMORY_STORED, MEMORY_RECALLED, MEMORY_ERROR | 3 |
| 上下文 | CONTEXT_BUILT, CONTEXT_COMPRESSED, PURIFICATION | 3 |
| 错误 | ERROR_OCCURRED, ERROR_RECOVERED, RETRY_ATTEMPT | 3 |
| 快照 | SNAPSHOT_CREATED, SNAPSHOT_RESTORED | 2 |

---

## 实现内容

### 20.1 AgentEventType枚举

**文件**：`/workspace/src/agent/event_bus.py`

```python
class AgentEventType(str, Enum):
    STATE_CHANGED = "state_changed"
    TURN_STARTED = "turn_started"
    TURN_COMPLETED = "turn_completed"
    # ... 共26种事件类型
```

### 20.2 AgentEvent事件数据

```python
@dataclass
class AgentEvent:
    event_type: AgentEventType
    agent_id: str = ""
    timestamp: float
    data: dict[str, Any]
    source: str = ""
    turn_id: str = ""

    def to_dict() -> dict
    def get(key, default) -> Any
    @classmethod
    def create(event_type, agent_id, data, source, turn_id) -> "AgentEvent"
```

### 20.3 EventBus事件总线

```python
class EventBus:
    # 订阅
    def subscribe(event_type, handler, once=False)
    def subscribe_all(handler, once=False)         # 通配符
    def unsubscribe(event_type, handler) -> bool
    def unsubscribe_all(handler) -> bool
    def clear_subscribers(event_type=None)

    # 发布
    def publish(event: AgentEvent) -> int          # 返回触发数
    def emit(event_type, agent_id, data, ...) -> int  # 便捷方法

    # 查询
    @property
    def subscriber_count -> int
    @property
    def event_history -> list[AgentEvent]
    def get_history(event_type, limit) -> list[dict]
    def get_stats() -> dict
```

**特性**：
- 线程安全（threading.Lock）
- 一次性订阅（once=True）
- 通配符订阅（subscribe_all）
- 处理器错误隔离（不影响其他处理器）
- 事件历史记录（可配置最大长度）
- 死锁修复（get_stats内部计算subscriber_count）

### 20.4 Agent集成

**修改文件**：`/workspace/src/agent/agent.py`

```python
class Agent:
    def _init_event_bus(self):
        self._event_bus = EventBus()

    def _transition_state(self, new_state):
        # 状态变更时发射 STATE_CHANGED 事件
        self._event_bus.emit(AgentEventType.STATE_CHANGED, ...)

    def chat(self, user_input):
        # 发射 TURN_STARTED 事件
        self._event_bus.emit(AgentEventType.TURN_STARTED, ...)

        try:
            result = self._process_turn(...)
            # 发射 TURN_COMPLETED 事件
            self._event_bus.emit(AgentEventType.TURN_COMPLETED, ...)
        except Exception as e:
            # 发射 TURN_ERROR 事件
            self._event_bus.emit(AgentEventType.TURN_ERROR, ...)

    @property
    def event_bus(self) -> Optional[EventBus]
```

---

## 测试结果

**文件**：`/workspace/tests/test_event_system.py`

| 测试类 | 测试数 | 结果 |
|--------|--------|------|
| TestAgentEventType | 3 | ✅ |
| TestAgentEvent | 4 | ✅ |
| TestEventBus | 17 | ✅ |
| TestAgentEventIntegration | 3 | ✅ |
| TestExportEventBus | 3 | ✅ |

**总计**：31 passed ✅

**全系统测试**：771 passed（新增31个），2 failed（阶段15遗留）。

---

## 使用示例

```python
from src.agent import EventBus, AgentEventType, AgentEvent

# 创建事件总线
bus = EventBus()

# 订阅状态变更
def on_state_changed(event: AgentEvent):
    print(f"State: {event.data['old_state']} -> {event.data['new_state']}")

bus.subscribe(AgentEventType.STATE_CHANGED, on_state_changed)

# 订阅所有事件
def on_any_event(event: AgentEvent):
    print(f"[{event.event_type.value}] {event.source}")

bus.subscribe_all(on_any_event)

# 一次性订阅
bus.subscribe(AgentEventType.TURN_COMPLETED, on_turn_done, once=True)

# 发布事件
bus.emit(
    AgentEventType.TURN_STARTED,
    agent_id="agent_1",
    data={"user_input": "hello"},
    turn_id="turn_001",
)

# 查看统计
print(bus.get_stats())
```

---

## 下阶段规划

**阶段21**：Agent配置热更新

- 实现运行时配置变更
- 配置变更事件通知
- 配置验证和回滚

---

## 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `/workspace/src/agent/event_bus.py` | 新建 | 事件系统核心 |
| `/workspace/src/agent/agent.py` | 修改 | 集成事件发射 |
| `/workspace/src/agent/__init__.py` | 修改 | 导出事件类 |
| `/workspace/tests/test_event_system.py` | 新建 | 31个测试 |

---

**阶段20完成** ✅
