# 阶段30: Agent日志与追踪系统 (Agent Logging & Tracing)

## 概述

阶段30实现了Agent日志与追踪系统，支持结构化日志记录和分布式调用链追踪。系统提供6级日志、Span追踪、Trace链管理、自动关联日志与追踪等功能。

## 核心组件

### 1. LogLevel (日志级别)

6个级别，按优先级从低到高：TRACE → DEBUG → INFO → WARN → ERROR → FATAL

### 2. LogRecord (日志记录)

结构化日志条目，包含级别、消息、追踪上下文、错误信息等。

### 3. Span (追踪跨度)

表示一个操作的时间跨度，支持标签、嵌套日志、持续时间计算。

### 4. Trace (追踪链)

表示一个完整的请求追踪，包含多个Span，自动检测完成状态。

### 5. StructuredLogger (结构化日志器)

分级日志记录，支持追踪上下文关联、日志处理器、记录过滤。

### 6. Tracer (分布式追踪器)

管理Trace和Span的创建与生命周期，支持父子Span嵌套。

### 7. TraceManager (追踪管理器)

统一管理日志器和追踪器，自动关联日志与追踪。

## 使用示例

```python
from src.agent import TraceManager, SpanStatus

manager = TraceManager(service_name="my_agent")

# 开始追踪
trace = manager.start_trace(name="process_request")

# 创建嵌套Span
parent = manager.start_span("validate_input", trace_id=trace.trace_id)
manager.finish_span(parent.span_id)

child = manager.start_span("execute_task", trace_id=trace.trace_id, parent_id=parent.span_id)
manager.finish_span(child.span_id)

# 结束追踪
manager.finish_trace(trace.trace_id)

# 获取统计
stats = manager.get_stats()
```

## 测试覆盖

- 59个测试用例全部通过
- 覆盖日志级别、记录、Span、Trace、日志器、追踪器、管理器、导出

## 总结

阶段30成功实现了Agent日志与追踪系统，提供了：
- 6级结构化日志
- 分布式Span追踪
- Trace链管理
- 日志与追踪自动关联
- 线程安全的实现