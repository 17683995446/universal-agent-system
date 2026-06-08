# 阶段12：Agent主循环层（Agent Main Loop Layer）

> 图难于其易，为大于其细。天下难事，必作于易；天下大事，必作于细。

## 12.1 概述

Agent主循环层是整个系统的核心控制器，整合前11阶段的所有模块，提供完整的对话能力。本层负责：
- 对话生命周期管理
- 工具调用与结果回传
- 对话历史维护
- 运行统计追踪

### 设计原则

1. **模块化整合**：各模块通过依赖注入组合，而非硬编码耦合
2. **状态驱动**：清晰的Agent状态机（IDLE → THINKING → EXECUTING → IDLE）
3. **容错设计**：LLM调用失败、工具执行失败均有降级处理
4. **可观测性**：完整的统计信息（Token、耗时、成功率）

## 12.2 模块结构

```
src/agent/
├── __init__.py              # 模块导出
├── agent_types.py           # 状态枚举、结果数据结构
├── agent_config.py          # Agent配置
├── turn_manager.py          # 单轮对话管理
└── agent.py                 # Agent核心（主循环）
```

## 12.3 数据结构

### 12.3.1 枚举类型

| 枚举 | 值 | 说明 |
|------|-----|------|
| `AgentState` | idle/thinking/executing/paused/error/stopped | Agent运行状态 |
| `TurnStatus` | pending/processing/completed/tool_calls/error | 单轮对话状态 |

### 12.3.2 TurnResult

单轮对话结果：

```python
@dataclass
class TurnResult:
    turn_id: str                    # 回合ID
    user_input: str                 # 用户输入
    assistant_reply: str            # 助手回复
    status: TurnStatus              # 状态
    tool_calls: list[dict]          # 工具调用
    tool_results: list[dict]        # 工具执行结果
    tokens_used: dict[str, int]     # Token使用量
    duration_ms: float              # 耗时
```

**便捷属性**：
- `has_tool_calls` → 是否需要工具调用
- `is_completed` → 是否已完成
- `is_error` → 是否出错

### 12.3.3 AgentStats

运行统计：

```python
@dataclass
class AgentStats:
    total_turns: int                # 总回合数
    total_tokens: int               # 总Token消耗
    tool_calls_count: int           # 工具调用次数
    errors_count: int               # 错误次数
    avg_turn_duration_ms: float     # 平均回合耗时
    start_time: float               # 启动时间
```

**关键方法**：`record_turn(duration_ms, tokens, tool_calls, error)` → 记录一轮对话。

### 12.3.4 AgentConfig

Agent配置，整合各模块配置：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| agent_id | "default" | Agent唯一标识 |
| name | "Agent" | Agent名称 |
| system_prompt | "You are a helpful assistant." | 系统提示词 |
| max_turns | 0 | 最大回合数（0=无限制） |
| enable_memory | True | 是否启用记忆 |
| enable_tools | True | 是否启用工具 |
| max_tool_calls_per_turn | 10 | 每轮最大工具调用次数 |

## 12.4 TurnManager

### 核心流程

```
process_turn() → _build_messages() → _call_llm()
                                    ↓
                            有tool_calls? ──Yes──→ _handle_tool_calls() → 再次调用LLM
                                    ↓ No
                            返回纯文本结果
```

### 消息构建

构建LLM消息列表：
1. 系统提示（如配置）
2. 历史消息（用户+助手+工具）
3. 当前用户输入

### 工具调用处理

1. 执行工具调用（通过ToolExecutor）
2. 构建assistant消息（含tool_calls）
3. 构建tool消息（执行结果）
4. 追加到消息列表
5. 再次调用LLM获取最终回复

## 12.5 Agent

### 核心方法

| 方法 | 说明 |
|------|------|
| `chat(user_input)` | 对话入口，处理单轮对话 |
| `register_tool(tool)` | 注册单个工具 |
| `register_tools(tools)` | 批量注册工具 |
| `get_tools_schema()` | 获取OpenAI tools schema |
| `get_history()` | 获取对话历史 |
| `clear_history()` | 清空对话历史 |
| `reset()` | 重置Agent状态 |
| `stop()` | 停止Agent |
| `get_summary()` | 获取Agent摘要 |

### 状态流转

```
IDLE ──chat()──→ THINKING ──工具调用──→ EXECUTING ──完成──→ IDLE
  ↑                                              ↓
  └────────────────── 错误 ──────────────────────┘
```

### 回合数限制

当 `max_turns > 0` 且 `total_turns >= max_turns` 时，返回错误结果。

## 12.6 测试覆盖

共 **48个测试**，覆盖：

| 测试类 | 数量 | 覆盖范围 |
|--------|------|----------|
| TestAgentState | 2 | 枚举值、str枚举特性 |
| TestTurnStatus | 1 | 枚举值 |
| TestTurnResult | 9 | 创建、属性判断、序列化 |
| TestAgentStats | 7 | 记录调用、统计计算、序列化 |
| TestConversationEntry | 2 | 创建、序列化 |
| TestAgentConfig | 5 | 默认值、自定义、字典转换 |
| TestTurnManager | 6 | 消息构建、简单对话、工具调用、错误处理 |
| TestAgent | 16 | 创建、工具注册、对话、历史管理、生命周期 |
| TestAgentIntegration | 2 | 完整对话流程、带工具调用的对话 |

## 12.7 与其他层的关系

```
┌─────────────────────────────────────────────────────────┐
│              Agent主循环层 (Phase 12)                    │
│  Agent.chat() → TurnManager.process_turn() → TurnResult │
└──────────────┬──────────────────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │ 模型推理层 (Phase 10) │  LLMClient.chat()
    └──────────────────────┘
               │
    ┌──────────▼──────────┐
    │ 工具调用层 (Phase 11) │  ToolExecutor.execute()
    └──────────────────────┘
               │
    ┌──────────▼──────────┐
    │ 上下文装配层 (Phase 8) │  ContextAssembly（待集成）
    └──────────────────────┘
               │
    ┌──────────▼──────────┐
    │ 三层记忆层 (Phase 6)  │  MemoryManager（待集成）
    └──────────────────────┘
```

## 12.8 关键经验

### 12.8.1 依赖注入

Agent通过构造函数接收各模块实例，便于测试时注入Mock：
```python
def __init__(self, config: Optional[AgentConfig] = None):
    self._llm_client = LLMClient(config=self._config.llm_config)
    self._tool_registry = ToolRegistry()
    self._tool_executor = ToolExecutor(registry=self._tool_registry)
    self._turn_manager = TurnManager(
        llm_client=self._llm_client,
        tool_executor=self._tool_executor,
    )
```

### 12.8.2 状态管理

Agent状态与TurnStatus分离：
- `AgentState` — Agent生命周期状态（idle/thinking/executing/stopped）
- `TurnStatus` — 单轮对话结果状态（completed/tool_calls/error）

### 12.8.3 历史格式

对话历史使用OpenAI格式，便于直接传递给LLM：
```python
{"role": "user", "content": "Hello"}
{"role": "assistant", "content": "Hi", "tool_calls": [...]}
{"role": "tool", "tool_call_id": "call_1", "content": "Result"}
```

## 12.9 累计进度

| 阶段 | 模块 | 测试数 |
|------|------|--------|
| 1-5 | 核心+消息+Token | 150 |
| 6 | 三层记忆 | 88 |
| 7 | 任务快照 | 69 |
| 8 | 上下文装配 | 37 |
| 9 | 净化管道 | 35 |
| 10 | 模型推理 | 77 |
| 11 | 工具调用 | 81 |
| **12** | **Agent主循环** | **48** |
| **合计** | | **585** |