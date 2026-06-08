# 阶段17：上下文压缩策略

## 概述

实现渐进式上下文压缩策略，根据Token预算压力动态调整压缩级别，集成到ContextBuilder的构建流程中。

**原则遵循**："图难于其易，为大于其细。天下难事，必作于易；天下大事，必作于细。"

---

## 核心设计

### 渐进式压缩策略

根据Token使用率（压力级别）自动选择压缩级别：

| 压力级别 | 使用率范围 | 压缩级别 | 说明 |
|----------|-----------|----------|------|
| none | < 60% | NONE | 不压缩 |
| light | 60%-80% | NONE | 轻度压力，暂不压缩 |
| medium | 80%-95% | RULE_BASED (L1) | 规则去噪 |
| heavy | 95%-100% | STRUCTURED (L2) | 结构化压缩 |
| critical | > 100% | SUMMARIZED (L3) | 摘要压缩 |

### 四种策略模式

| 策略 | 说明 | 阈值特点 |
|------|------|----------|
| NONE | 不压缩 | 全部设为1.0 |
| CONSERVATIVE | 保守 | 阈值后移（80%/90%/97%） |
| BALANCED | 均衡（默认） | 标准阈值（60%/80%/95%） |
| AGGRESSIVE | 激进 | 阈值前移（40%/60%/80%） |

---

## 实现内容

### 17.1 CompressionStrategy枚举

**文件**：`/workspace/src/agent/context_compressor.py`

```python
class CompressionStrategy(IntEnum):
    NONE = 0             # 不压缩
    CONSERVATIVE = 1     # 保守策略
    BALANCED = 2         # 均衡策略（默认）
    AGGRESSIVE = 3       # 激进策略
```

### 17.2 CompressionConfig配置

```python
@dataclass
class CompressionConfig:
    strategy: CompressionStrategy = CompressionStrategy.BALANCED
    enabled: bool = True
    light_threshold: float = 0.60
    medium_threshold: float = 0.80
    heavy_threshold: float = 0.95
    critical_threshold: float = 1.0
    min_tokens_to_compress: int = 200
    preserve_recent_n: int = 2
    integrity_threshold: float = 0.7
```

**关键方法**：
- `get_thresholds()` → 根据策略返回四个阈值
- `to_dict()` / `from_dict()` → 序列化/反序列化

### 17.3 CompressionResult结果

```python
@dataclass
class CompressionResult:
    original_tokens: int
    compressed_tokens: int
    compression_level: PurificationLevel
    messages_compressed: int
    messages_preserved: int
    pressure_level: str
    compression_ratio: float
    details: list[dict[str, Any]]
```

**属性**：
- `tokens_saved` → 节省的Token数
- `compression_percentage` → 压缩百分比
- `was_compressed` → 是否发生压缩

### 17.4 ContextCompressor核心类

```python
class ContextCompressor:
    def __init__(
        self,
        config: Optional[CompressionConfig] = None,
        pipeline: Optional[PurificationPipeline] = None,
    )
```

**核心方法**：

| 方法 | 说明 |
|------|------|
| `assess_pressure(used_tokens, window_size)` | 评估Token压力级别 |
| `determine_compression_level(pressure_level)` | 根据压力确定压缩级别 |
| `compress_messages(messages, used_tokens, window_size)` | 压缩消息列表 |
| `compress_text(text, target_level, original_tokens)` | 压缩单段文本 |
| `get_stats()` / `reset_stats()` | 统计管理 |

**压缩流程**：
1. 评估Token压力级别
2. 确定压缩级别
3. 分离可压缩消息和保留消息（最近N条）
4. 对可压缩消息执行净化
5. 合并返回

### 17.5 ContextBuilder集成

**修改文件**：`/workspace/src/agent/context_builder.py`

```python
class ContextBuilder:
    def __init__(self, ..., compressor=None):
        self._compressor = compressor

    def build(self, ...):
        # 新增：压缩历史消息
        effective_history = self._maybe_compress_history(history, window_size)
        session_messages = self._build_session_messages(effective_history, user_input)
        ...
```

**新增方法**：
- `compressor` 属性
- `last_compression_result` 属性
- `set_compressor(compressor)` 方法
- `_maybe_compress_history(history, window_size)` 内部方法

### 17.6 AgentConfig扩展

```python
@dataclass
class AgentConfig:
    # 新增
    enable_context_compression: bool = True
    compression_strategy: str = "balanced"
```

### 17.7 Agent集成

```python
class Agent:
    def _init_context_compressor(self):
        # 解析策略 → 创建CompressionConfig → 创建ContextCompressor
        # 注入到ContextBuilder

    @property
    def context_compressor(self) -> Optional[ContextCompressor]
```

---

## 架构关系

```
Agent
  ├── AgentConfig
  │     ├── enable_context_compression: bool
  │     └── compression_strategy: str
  │
  ├── ContextCompressor (新增)
  │     ├── CompressionConfig
  │     │     ├── strategy: CompressionStrategy
  │     │     ├── thresholds: (light, medium, heavy, critical)
  │     │     └── preserve_recent_n: int
  │     │
  │     ├── assess_pressure() → (level, ratio)
  │     ├── determine_compression_level() → PurificationLevel
  │     ├── compress_messages() → (messages, CompressionResult)
  │     └── compress_text() → PurificationResult
  │
  └── ContextBuilder
        ├── compressor: ContextCompressor
        ├── _maybe_compress_history()
        └── last_compression_result
```

---

## 测试结果

**文件**：`/workspace/tests/test_context_compression.py`

| 测试类 | 测试数 | 结果 |
|--------|--------|------|
| TestCompressionStrategy | 3 | ✅ |
| TestCompressionConfig | 10 | ✅ |
| TestCompressionResult | 3 | ✅ |
| TestContextCompressor | 14 | ✅ |
| TestContextBuilderCompression | 6 | ✅ |
| TestAgentCompressionIntegration | 6 | ✅ |
| TestExportCompression | 3 | ✅ |

**总计**：46 passed ✅

**全系统测试**：674 passed（新增46个），2 failed（阶段15遗留）。

---

## 配置示例

```python
# 默认（均衡策略）
config = AgentConfig()

# 保守策略（仅在极限压力时压缩）
config = AgentConfig(compression_strategy="conservative")

# 激进策略（轻度压力就开始压缩）
config = AgentConfig(compression_strategy="aggressive")

# 禁用压缩
config = AgentConfig(enable_context_compression=False)

# 自定义压缩配置
from src.agent.context_compressor import CompressionConfig, CompressionStrategy
compression_config = CompressionConfig(
    strategy=CompressionStrategy.BALANCED,
    preserve_recent_n=5,
    min_tokens_to_compress=500,
)
```

---

## 下阶段规划

**阶段18**：Agent生命周期管理

- 实现Agent状态机（IDLE/THINKING/EXECUTING/ERROR/STOPPED）
- 添加优雅关闭和资源清理
- 实现Agent快照和恢复

---

## 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `/workspace/src/agent/context_compressor.py` | 新建 | 压缩策略核心 |
| `/workspace/src/agent/context_builder.py` | 修改 | 集成压缩 |
| `/workspace/src/agent/agent.py` | 修改 | 初始化压缩器 |
| `/workspace/src/agent/agent_config.py` | 修改 | 压缩配置字段 |
| `/workspace/src/agent/__init__.py` | 修改 | 导出压缩类 |
| `/workspace/tests/test_context_compression.py` | 新建 | 46个测试 |

---

**阶段17完成** ✅