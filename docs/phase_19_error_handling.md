# 阶段19：错误处理与恢复机制

## 概述

实现分级错误处理策略、自动重试机制、结构化错误记录与诊断功能。

**原则遵循**："图难于其易，为大于其细。天下难事，必作于易；天下大事，必作于细。"

---

## 核心设计

### 错误分级体系

```
WARNING (0)     → 仅记录日志，不影响流程
DEGRADED (1)    → 部分功能降级，主流程继续
RECOVERABLE (2) → 可通过重试恢复
FATAL (3)       → 致命错误，需人工介入
```

### 异常类型映射

| 异常类型 | 分级 | 处理策略 |
|----------|------|----------|
| `ConfigurationError` | FATAL | 立即终止 |
| `LLMClientError` | RECOVERABLE | 自动重试 |
| `TokenLimitExceeded` | RECOVERABLE | 重试/压缩 |
| `IntegrityCheckFailed` | RECOVERABLE | 重试/降级 |
| `PurificationError` | RECOVERABLE | 重试 |
| `MemoryStorageError` | DEGRADED | 静默降级 |
| `MemoryRecallError` | DEGRADED | 静默降级 |
| `TaskDriftDetected` | DEGRADED | 标记漂移 |

---

## 实现内容

### 19.1 ErrorCategory枚举

**文件**：`/workspace/src/agent/error_handler.py`

```python
class ErrorCategory(IntEnum):
    WARNING = 0
    DEGRADED = 1
    RECOVERABLE = 2
    FATAL = 3
```

### 19.2 classify_error函数

```python
def classify_error(error: Exception) -> ErrorCategory:
    """对异常进行自动分级"""
```

### 19.3 ErrorRecord错误记录

```python
@dataclass
class ErrorRecord:
    error_id: str
    category: ErrorCategory
    error_type: str
    message: str
    context: dict
    timestamp: float
    traceback_str: str
    turn_id: str
    component: str
    recovered: bool
    recovery_method: str
```

### 19.4 RetryPolicy重试策略

```python
@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    retryable_categories: set[ErrorCategory] = {RECOVERABLE}

    def should_retry(error, attempt) -> bool
    def get_delay(attempt) -> float  # 指数退避
```

### 19.5 ErrorHandler统一处理器

```python
class ErrorHandler:
    def record_error(error, component, turn_id) -> ErrorRecord
    def handle_error(error, component, turn_id) -> ErrorRecord
    def execute_with_retry(fn, component, turn_id) -> Any
    def mark_recovered(error_id, method) -> bool
    def get_stats() -> dict
    def get_recent_errors(limit) -> list
    def get_unrecovered_errors() -> list
    def get_diagnosis() -> dict  # 诊断报告+建议
    def clear_history()
```

### 19.6 Agent集成

**修改文件**：`/workspace/src/agent/agent.py`

```python
class Agent:
    def _init_error_handler(self):
        self._error_handler = ErrorHandler()

    def chat(self, user_input):
        try:
            ...
        except Exception as e:
            error_record = self._error_handler.handle_error(
                error=e, component="Agent.chat", turn_id=turn_id
            )
            return TurnResult(
                status=TurnStatus.ERROR,
                metadata={"error": str(e), "error_category": error_record.category.name}
            )

    @property
    def error_handler(self) -> Optional[ErrorHandler]
```

---

## 测试结果

**文件**：`/workspace/tests/test_error_handling.py`

| 测试类 | 测试数 | 结果 |
|--------|--------|------|
| TestErrorCategory | 2 | ✅ |
| TestClassifyError | 9 | ✅ |
| TestErrorRecord | 3 | ✅ |
| TestRetryPolicy | 6 | ✅ |
| TestErrorHandler | 14 | ✅ |
| TestAgentErrorIntegration | 2 | ✅ |
| TestExportErrorHandler | 4 | ✅ |

**总计**：44 passed ✅

**全系统测试**：740 passed（新增44个），2 failed（阶段15遗留）。

---

## 使用示例

```python
from src.agent import ErrorHandler, RetryPolicy, ErrorCategory
from src.core.exceptions import LLMClientError

# 创建错误处理器
handler = ErrorHandler(
    retry_policy=RetryPolicy(max_attempts=3, base_delay=0.5)
)

# 带重试执行
try:
    result = handler.execute_with_retry(
        fn=llm_call,
        component="LLM",
        turn_id="turn_001",
    )
except Exception as e:
    print(f"All retries failed: {e}")

# 手动记录错误
error = MemoryStorageError(operation="store", error="disk full")
record = handler.record_error(error, component="Memory")

# 标记恢复
handler.mark_recovered(record.error_id, "fallback_storage")

# 获取诊断
diagnosis = handler.get_diagnosis()
print(diagnosis["status"])  # "healthy" or "degraded"
print(diagnosis["recommendations"])
```

---

## 下阶段规划

**阶段20**：Agent事件系统

- 实现事件总线（EventBus）
- 定义核心事件类型（状态变更、错误、快照等）
- 实现事件订阅/发布机制

---

## 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `/workspace/src/agent/error_handler.py` | 新建 | 错误处理核心 |
| `/workspace/src/agent/agent.py` | 修改 | 集成ErrorHandler |
| `/workspace/src/agent/__init__.py` | 修改 | 导出错误处理类 |
| `/workspace/tests/test_error_handling.py` | 新建 | 44个测试 |

---

**阶段19完成** ✅
