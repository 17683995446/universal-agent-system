# 阶段16：净化管道集成层

## 概述

将阶段9的净化管道（PurificationPipeline）集成到Agent系统，实现输入净化和输出净化的统一管理。

**原则遵循**："图难于其易，为大于其细。天下难事，必作于易；天下大事，必作于细。"

---

## 实现内容

### 16.1 ContentPurifier类

**文件**：`/workspace/src/agent/content_purifier.py`

统一净化入口，管理输入净化和输出净化：

```python
class ContentPurifier:
    def __init__(
        self,
        pipeline: Optional[PurificationPipeline] = None,
        enable_input_purification: bool = True,
        enable_output_purification: bool = False,
        input_max_level: PurificationLevel = PurificationLevel.STRUCTURED,
        output_max_level: PurificationLevel = PurificationLevel.RULE_BASED,
        llm_fn: Optional[Callable[[str], str]] = None,
    )
```

**核心方法**：
- `purify_input(text, original_tokens, max_level)` → PurificationResult
- `purify_output(text, original_tokens, max_level)` → PurificationResult
- `purify_user_input(user_input)` → str（便捷方法）
- `purify_memory_context(memory_context)` → str
- `purify_assistant_reply(reply)` → str
- `purify_tool_result(result)` → str
- `get_stats()` → dict

**属性**：
- `enable_input_purification`：是否启用输入净化
- `enable_output_purification`：是否启用输出净化
- `input_max_level`：输入最大净化级别
- `output_max_level`：输出最大净化级别

---

### 16.2 TurnManager集成

**文件**：`/workspace/src/agent/turn_manager.py`

修改内容：

1. **新增参数**：
```python
def __init__(
    self,
    ...
    content_purifier: Optional["ContentPurifier"] = None,
)
```

2. **process_turn流程集成**：
```python
def process_turn(...):
    # 输入净化（在漂移检测前）
    purified_input = self._purify_input(user_input)
    
    # 使用净化后的输入进行漂移检测
    drift_result = self._task_context.check_drift_periodic(purified_input)
    
    # 构建消息（使用净化后的输入）
    messages = self._build_messages(system_prompt, history, purified_input)
    
    # 调用LLM
    response = self._call_llm(messages, tools_schema)
    
    # 输出净化
    purified_reply = self._purify_output(response.content)
    
    return TurnResult(..., assistant_reply=purified_reply, ...)
```

3. **辅助方法**：
```python
def _purify_input(self, text: str) -> str:
    if self._content_purifier:
        return self._content_purifier.purify_user_input(text)
    return text

def _purify_output(self, text: str) -> str:
    if self._content_purifier:
        return self._content_purifier.purify_assistant_reply(text)
    return text
```

---

### 16.3 Agent集成

**文件**：`/workspace/src/agent/agent.py`

修改内容：

1. **新增字段**：
```python
self._content_purifier: Optional[ContentPurifier] = None
```

2. **初始化方法**：
```python
def _init_content_purifier(self) -> None:
    from src.modules.purification.pipeline import PurificationPipeline
    
    pipeline = PurificationPipeline()
    self._content_purifier = ContentPurifier(
        pipeline=pipeline,
        enable_input_purification=self._config.enable_input_purification,
        enable_output_purification=self._config.enable_output_purification,
    )
```

3. **属性**：
```python
@property
def content_purifier(self) -> Optional[ContentPurifier]:
    return self._content_purifier
```

4. **TurnManager传递**：
```python
def _init_turn_manager(self) -> None:
    self._turn_manager = TurnManager(
        ...
        content_purifier=self._content_purifier,
        ...
    )
```

---

### 16.4 AgentConfig扩展

**文件**：`/workspace/src/agent/agent_config.py`

新增配置字段：
```python
# 内容净化配置
enable_input_purification: bool = True
enable_output_purification: bool = False
```

序列化/反序列化支持已添加。

---

### 16.5 导出更新

**文件**：`/workspace/src/agent/__init__.py`

```python
from src.agent.content_purifier import ContentPurifier

__all__ = [
    ...
    "ContentPurifier",
    ...
]
```

---

## 测试结果

**文件**：`/workspace/tests/test_purification_integration.py`

| 测试类 | 测试数 | 结果 |
|--------|--------|------|
| TestContentPurifierBasic | 9 | ✅ 全部通过 |
| TestContentPurifierWithPipeline | 3 | ✅ 全部通过 |
| TestAgentPurificationIntegration | 6 | ✅ 全部通过 |
| TestTurnManagerPurificationFlow | 4 | ✅ 全部通过 |
| TestExportContentPurifier | 1 | ✅ 全部通过 |

**总计**：23 passed，全阶段测试通过。

**全系统测试**：628 passed（新增23个），2 failed（阶段15遗留问题）。

---

## 架构关系图

```
Agent
  │
  ├── AgentConfig
  │     └── enable_input_purification: bool
  │     └── enable_output_purification: bool
  │
  ├── ContentPurifier (新增)
  │     ├── PurificationPipeline (阶段9)
  │     │     ├── RuleDenoiser (L1)
  │     │     ├── StructureConverter (L2)
  │     │     └── Summarizer (L3)
  │     │
  │     ├── enable_input_purification
  │     ├── enable_output_purification
  │     ├── input_max_level
  │     └── output_max_level
  │
  └── TurnManager
        ├── content_purifier: ContentPurifier
        ├── _purify_input()
        └── _purify_output()
```

---

## 净化流程

### 输入净化流程

```
用户输入 → ContentPurifier.purify_user_input()
         → PurificationPipeline.purify_to_level()
         → [L1规则去噪] → [L2结构化] → [L3摘要]
         → purified_input
         → TurnManager.process_turn()
```

### 输出净化流程

```
LLM响应 → ContentPurifier.purify_assistant_reply()
        → PurificationPipeline.purify_to_level()
        → [L1规则去噪]（默认）
        → purified_reply
        → TurnResult.assistant_reply
```

---

## 配置示例

```python
# 默认配置（输入净化启用，输出净化禁用）
config = AgentConfig()

# 自定义配置
config = AgentConfig(
    enable_input_purification=True,
    enable_output_purification=True,
)

# 禁用净化
config = AgentConfig(
    enable_input_purification=False,
    enable_output_purification=False,
)
```

---

## 下阶段规划

**阶段17**：上下文压缩策略

- 实现渐进式压缩策略
- 根据Token预算动态调整压缩级别
- 集成到ContextBuilder

---

## 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `/workspace/src/agent/content_purifier.py` | 新建 | ContentPurifier类 |
| `/workspace/src/agent/agent.py` | 修改 | 集成净化器 |
| `/workspace/src/agent/agent_config.py` | 修改 | 新增净化配置 |
| `/workspace/src/agent/turn_manager.py` | 修改 | 净化流程集成 |
| `/workspace/src/agent/__init__.py` | 修改 | 导出ContentPurifier |
| `/workspace/tests/test_purification_integration.py` | 新建 | 23个测试 |

---

**阶段16完成** ✅