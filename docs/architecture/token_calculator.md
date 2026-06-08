# Token计算器技术文档

## 概述

Token计算器是"无上限上下文Agent系统"的核心基础设施组件，负责高精度的Token计算、预算分配和超限检测。

## 设计目标

- **高精度**：误差 < 1%
- **高性能**：< 10ms/1K tokens
- **多模型支持**：GPT系列、Claude系列等主流模型
- **易扩展**：支持新增模型和Tokenizer后端

## 架构设计

### 类图

```
BaseTokenizer (ABC)
    ├── TiktokenTokenizer    # OpenAI GPT系列
    ├── TransformersTokenizer # HuggingFace模型
    └── SimpleTokenizer      # 兜底方案

TokenizerFactory
    └── 创建和管理Tokenizer实例

TokenCalculator
    ├── calculate()          # 单条计算
    ├── calculate_batch()    # 批量计算
    ├── calculate_messages() # 消息列表计算
    ├── allocate_budget()    # 预算分配
    └── check_limit()        # 超限检测

TokenCounter (静态工具类)
    └── 便捷静态方法
```

### 模块关系

```
token_manager/
├── tokenizer.py    # Tokenizer实现
├── calculator.py   # Token计算器
└── partitioner.py  # 弹性分区（后续阶段）
```

## 核心功能

### 1. Token计算

```python
from src.modules.token_manager.calculator import TokenCalculator

calc = TokenCalculator(model="gpt-4")

# 单条计算
tokens = calc.calculate("Hello world")

# 批量计算
tokens_list = calc.calculate_batch(["Hello", "World"])

# 消息计算
from src.core.types import Message, Role
messages = [
    Message(role=Role.USER, content="Hello"),
]
total = calc.calculate_messages(messages)
```

### 2. Token预算分配

```python
# 默认分配
budget = calc.allocate_budget()
print(f"系统区: {budget.system_zone}")
print(f"会话区: {budget.session_zone}")
print(f"预留区: {budget.reserve_zone}")

# 自定义比例
budget = calc.allocate_budget(
    system_ratio=0.25,
    session_ratio=0.55,
    reserve_ratio=0.20
)
```

### 3. 超限检测

```python
# 检查但不抛出异常
result = calc.check_limit(current_tokens=7000)
print(f"是否预警: {result['is_warning']}")
print(f"是否超限: {result['is_exceeded']}")
print(f"剩余: {result['remaining']}")

# 超限时抛出异常
calc.check_limit(current_tokens=8000, raise_on_exceed=True)
# 抛出 TokenLimitExceeded
```

## 支持的模型

| 模型 | 窗口大小 | Tokenizer |
|------|----------|-----------|
| gpt-4 | 8192 | tiktoken (cl100k_base) |
| gpt-4-turbo | 128000 | tiktoken (cl100k_base) |
| gpt-4o | 128000 | tiktoken (cl100k_base) |
| gpt-3.5-turbo | 4096 | tiktoken (cl100k_base) |
| gpt-3.5-turbo-16k | 16384 | tiktoken (cl100k_base) |
| claude-3-* | 200000 | transformers |

## 常量配置

Token计算器依赖以下常量（定义于 `src/core/constants.py`）：

```python
# 窗口分区比例
SYSTEM_ZONE_DEFAULT_RATIO = 0.20  # 系统指令区
SESSION_ZONE_RATIO = 0.60          # 有效会话区
RESERVE_ZONE_RATIO = 0.20          # 预留推理区

# 阈值
TOKEN_WARNING_THRESHOLD = 0.85     # 预警阈值
TOKEN_HARD_LIMIT_RATIO = 0.95      # 严格限制
```

## 精度保证

### 误差来源分析

1. **Tokenizer差异**：不同模型的Tokenizer编码方式不同
2. **消息格式开销**：角色前缀、格式标记等额外Token
3. **估算误差**：字符数估算Token数时的近似误差

### 精度保证措施

1. **使用官方Tokenizer**：优先使用tiktoken、transformers等官方实现
2. **包含格式开销**：消息计算包含角色和格式开销
3. **预留安全余量**：20%预留区确保输出不超限

## 性能优化

### 缓存策略

```python
# Tokenizer实例缓存
TokenizerFactory._cache: dict[str, BaseTokenizer]

# TokenCounter计算器缓存
TokenCounter._calculators: dict[str, TokenCalculator]
```

### 性能基准

| 操作 | 性能指标 |
|------|----------|
| 单条计算(1K字符) | < 10ms |
| 批量计算(100条) | < 1s |
| 内存占用 | < 50MB |

## 测试覆盖

### 单元测试

```bash
pytest tests/unit/test_token_calculator.py -v
```

测试类别：
- 基础功能测试
- 精度测试
- Token预算测试
- 超限检测测试
- 性能测试
- 边界情况测试

### 基准测试

```bash
python tests/benchmark/benchmark_tokenizer.py
```

测试项目：
- 不同长度文本计算
- 批量计算性能
- 内存使用测试

## 使用示例

### 快速开始

```python
from src.modules.token_manager.calculator import quick_count, get_budget

# 快速计算
tokens = quick_count("Hello world", model="gpt-4")

# 获取预算
budget = get_budget("gpt-4")
```

### 完整流程

```python
from src.modules.token_manager.calculator import TokenCalculator
from src.core.types import Message, Role

calc = TokenCalculator("gpt-4")

# 1. 计算系统消息
system_msg = Message(role=Role.SYSTEM, content="You are a helpful assistant.")
system_tokens = calc.calculate_message(system_msg)

# 2. 计算用户消息
user_msg = Message(role=Role.USER, content="Hello!")
user_tokens = calc.calculate_message(user_msg)

# 3. 计算总Token
total = system_tokens + user_tokens

# 4. 检查限制
result = calc.check_limit(total)
if result['is_warning']:
    print("Token使用接近上限")
if result['is_exceeded']:
    print("Token超限，需要裁剪")
```

## 扩展指南

### 添加新模型

1. 在 `constants.py` 中添加窗口大小：
```python
MODEL_CONTEXT_WINDOWS = {
    "new-model": 32768,
}
```

2. 在 `tokenizer.py` 中添加Tokenizer映射：
```python
MODEL_ENCODING_MAP = {
    "new-model": "encoding_name",
}
```

### 添加新Tokenizer后端

1. 继承 `BaseTokenizer`：
```python
class NewTokenizer(BaseTokenizer):
    def encode(self, text: str) -> list[int]:
        # 实现编码逻辑
        pass
    
    def decode(self, tokens: list[int]) -> str:
        # 实现解码逻辑
        pass
```

2. 在 `TokenizerFactory` 中注册：
```python
elif backend == TokenizerBackend.NEW:
    tokenizer = NewTokenizer(model)
```

## 故障排查

### Tokenizer未找到

```
TokenizerNotFound: Tokenizer not found for model: xxx
```

**解决方案**：
1. 检查模型名称是否正确
2. 安装对应的Tokenizer库：`pip install tiktoken` 或 `pip install transformers`
3. 使用兜底方案：SimpleTokenizer

### Token超限

```
TokenLimitExceeded: Token limit exceeded: 9000 > 7782
```

**解决方案**：
1. 使用 `check_limit(raise_on_exceed=False)` 获取详细信息
2. 调用净化压缩模块减少Token数
3. 使用分片聚合模块处理超长文本

## API参考

### TokenCalculator

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| calculate | text: str | int | 计算单条文本 |
| calculate_batch | texts: list[str] | list[int] | 批量计算 |
| calculate_message | message: Message | int | 计算消息 |
| calculate_messages | messages: list[Message] | int | 计算消息列表 |
| allocate_budget | ... | TokenBudget | 分配预算 |
| check_limit | current: int, raise_on_exceed: bool | dict | 检查限制 |

### TokenCounter（静态方法）

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| count | text: str, model: str | int | 快速计算 |
| count_messages | messages: list[Message], model: str | int | 快速计算消息 |
| check | text: str, model: str | dict | 快速检查 |

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0.0 | 2026-05-25 | 初始版本，支持GPT系列模型 |

## 作者

AI Assistant

## 许可证

MIT License
