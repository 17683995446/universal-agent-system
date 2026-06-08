# 阶段25: Agent工作流引擎 (Agent Workflow Engine)

## 概述

阶段25实现了Agent工作流引擎，支持可视化编排复杂任务流程。系统提供节点定义、流程控制、条件分支、执行跟踪等功能，为构建复杂的Agent业务流程提供基础架构。

## 核心组件

### 1. WorkflowNodeType (节点类型枚举)

定义工作流节点的类型：

- `START`: 开始节点，流程入口
- `END`: 结束节点，流程出口
- `TASK`: 任务节点，执行业务逻辑
- `CONDITION`: 条件节点，分支判断
- `LOOP`: 循环节点，重复执行
- `PARALLEL`: 并行节点，并发执行
- `WAIT`: 等待节点，延时等待
- `SUBFLOW`: 子流程节点，嵌套流程

```python
from src.agent import WorkflowNodeType

# 创建任务节点
node = WorkflowNode(
    node_id="task_1",
    node_type=WorkflowNodeType.TASK,
    name="数据处理",
)
```

### 2. WorkflowStatus (工作流状态枚举)

定义工作流实例的状态：

- `PENDING`: 等待执行
- `RUNNING`: 执行中
- `PAUSED`: 暂停
- `COMPLETED`: 完成
- `FAILED`: 失败
- `CANCELLED`: 取消

### 3. NodeExecutionStatus (节点执行状态枚举)

定义节点的执行状态：

- `PENDING`: 等待执行
- `RUNNING`: 执行中
- `COMPLETED`: 完成
- `FAILED`: 失败
- `SKIPPED`: 跳过

### 4. WorkflowContext (工作流上下文)

存储工作流执行过程中的变量和状态：

```python
from src.agent import WorkflowContext

context = WorkflowContext()

# 设置变量
context.set_variable("user_id", "12345")
context.set_variable("order_amount", 100.0)

# 获取变量
user_id = context.get_variable("user_id")
amount = context.get_variable("order_amount", 0.0)

# 设置节点结果
context.set_node_result("validate_node", {"valid": True})

# 获取执行路径
print(context.execution_path)  # ['start', 'validate', 'process']
```

### 5. WorkflowNode (工作流节点)

表示工作流中的一个步骤：

```python
from src.agent import WorkflowNode, WorkflowNodeType

# 创建任务节点
task_node = WorkflowNode(
    node_id="process_order",
    node_type=WorkflowNodeType.TASK,
    name="处理订单",
    config={"task": {"type": "order_processing"}},
    retry_count=3,
    timeout=30.0,
)

# 添加后续节点
task_node.add_next("send_notification")
```

### 6. NodeExecutionRecord (节点执行记录)

记录节点的执行历史和结果：

```python
record = NodeExecutionRecord(node_id="task_1")
record.mark_running()
# ... 执行任务 ...
record.mark_completed({"result": "success"})

# 获取执行时长
duration = record.get_duration()
```

### 7. WorkflowDefinition (工作流定义)

定义工作流的结构和节点：

```python
from src.agent import WorkflowDefinition

definition = WorkflowDefinition(
    workflow_id="order_process",
    name="订单处理流程",
    version="1.0.0",
    description="处理用户订单的完整流程",
)

# 添加节点
start = WorkflowNode("start", WorkflowNodeType.START, "开始")
task = WorkflowNode("validate", WorkflowNodeType.TASK, "验证订单")
end = WorkflowNode("end", WorkflowNodeType.END, "结束")

definition.add_node(start)
definition.add_node(task)
definition.add_node(end)

# 连接节点
definition.connect("start", "validate")
definition.connect("validate", "end")

# 验证定义
valid, errors = definition.validate()
```

### 8. WorkflowInstance (工作流实例)

表示一个正在执行或已执行的工作流：

```python
instance = WorkflowInstance(
    instance_id="inst_001",
    workflow_id="order_process",
)

# 获取执行摘要
summary = instance.get_execution_summary()
print(f"状态: {summary['status']}")
print(f"完成节点: {summary['completed_nodes']}/{summary['total_nodes']}")
```

### 9. WorkflowEngine (工作流执行引擎)

核心执行引擎：

```python
from src.agent import WorkflowEngine

engine = WorkflowEngine()

# 注册工作流定义
engine.register_definition(definition)

# 创建实例
instance = engine.create_instance(
    workflow_id="order_process",
    initial_variables={"order_id": "12345"},
)

# 启动执行
engine.start_instance(instance.instance_id)

# 获取状态
status = engine.get_instance_status(instance.instance_id)

# 取消执行
engine.cancel_instance(instance.instance_id)

# 获取引擎统计
stats = engine.get_engine_stats()
```

### 10. WorkflowBuilder (工作流构建器)

提供流畅的API构建工作流：

```python
from src.agent import WorkflowBuilder, WorkflowNodeType

builder = WorkflowBuilder("order_workflow", "订单处理")

definition = (
    builder
    .description("处理用户订单")
    .version("1.0.0")
    .add_start("start", "开始")
    .add_task("validate", "验证订单", "validation")
    .add_condition("check_stock", "检查库存", "stock > 0")
    .add_task("process", "处理订单", "processing")
    .add_task("out_of_stock", "库存不足", "notification")
    .add_task("notify", "发送通知", "notification")
    .add_end("end", "结束")
    .connect("start", "validate")
    .connect("validate", "check_stock")
    .connect("check_stock", "process")      # true 分支
    .connect("check_stock", "out_of_stock") # false 分支
    .connect("process", "notify")
    .connect("out_of_stock", "notify")
    .connect("notify", "end")
    .build()
)

# 注册并执行
engine.register_definition(definition)
```

## 使用场景

### 场景1: 订单处理流程

```python
# 构建订单处理工作流
builder = WorkflowBuilder("order_flow", "订单处理")
definition = (
    builder
    .add_start("start", "开始")
    .add_task("validate", "验证订单")
    .add_condition("check_payment", "检查支付", "paid == true")
    .add_task("fulfill", "履行订单")
    .add_task("cancel", "取消订单")
    .add_task("notify", "通知用户")
    .add_end("end", "结束")
    .connect("start", "validate")
    .connect("validate", "check_payment")
    .connect("check_payment", "fulfill")
    .connect("check_payment", "cancel")
    .connect("fulfill", "notify")
    .connect("cancel", "notify")
    .connect("notify", "end")
    .build()
)

# 执行
engine = WorkflowEngine()
engine.register_definition(definition)
instance = engine.create_instance("order_flow", {"order_id": "123", "paid": True})
engine.start_instance(instance.instance_id)
```

### 场景2: 审批流程

```python
# 多级审批工作流
builder = WorkflowBuilder("approval_flow", "审批流程")
definition = (
    builder
    .add_start("start", "提交申请")
    .add_task("manager_review", "经理审批")
    .add_condition("manager_approved", "经理通过", "approved == true")
    .add_task("director_review", "总监审批")
    .add_task("reject", "拒绝")
    .add_condition("director_approved", "总监通过", "approved == true")
    .add_task("execute", "执行")
    .add_end("end", "结束")
    .connect("start", "manager_review")
    .connect("manager_review", "manager_approved")
    .connect("manager_approved", "director_review")
    .connect("manager_approved", "reject")
    .connect("director_review", "director_approved")
    .connect("director_approved", "execute")
    .connect("director_approved", "reject")
    .connect("execute", "end")
    .connect("reject", "end")
    .build()
)
```

### 场景3: 定时任务

```python
# 带等待节点的工作流
builder = WorkflowBuilder("scheduled_task", "定时任务")
definition = (
    builder
    .add_start("start", "开始")
    .add_task("prepare", "准备数据")
    .add_wait("wait", "等待", 3600)  # 等待1小时
    .add_task("process", "处理数据")
    .add_end("end", "结束")
    .connect("start", "prepare")
    .connect("prepare", "wait")
    .connect("wait", "process")
    .connect("process", "end")
    .build()
)
```

## 条件表达式

支持简单的条件表达式：

```python
# 等于判断
"status == 'approved'"
"count == 10"

# 使用变量
context.set_variable("user_type", "vip")
# 条件: "user_type == 'vip'"

# 自定义条件评估器
def custom_evaluator(condition: str, context: WorkflowContext) -> bool:
    # 自定义逻辑
    return True

engine.set_condition_evaluator(custom_evaluator)
```

## 节点处理器

可以注册自定义节点处理器：

```python
def custom_task_handler(node: WorkflowNode, context: WorkflowContext) -> Any:
    task_type = node.config.get("task", {}).get("type")
    if task_type == "send_email":
        # 发送邮件逻辑
        return {"sent": True}
    return {"executed": True}

engine.register_node_handler(WorkflowNodeType.TASK, custom_task_handler)
```

## 执行跟踪

```python
# 获取执行路径
instance = engine.get_instance(instance_id)
print(f"执行路径: {instance.context.execution_path}")

# 获取节点结果
result = instance.context.get_node_result("validate")

# 获取执行记录
record = instance.execution_records["validate"]
print(f"状态: {record.status.value}")
print(f"耗时: {record.get_duration()}秒")
```

## 线程安全

WorkflowEngine 使用 threading.RLock 保证线程安全：
- 工作流定义注册是线程安全的
- 实例创建和查询是线程安全的
- 状态变更是线程安全的

## 最佳实践

1. **验证定义**: 注册前使用 `validate()` 检查工作流定义
2. **设置超时**: 为任务节点设置合理的超时时间
3. **处理失败**: 检查实例状态，处理失败情况
4. **使用构建器**: 使用 WorkflowBuilder 简化工作流创建
5. **监控执行**: 定期获取引擎统计信息

## 测试覆盖

- 62个测试用例，覆盖所有核心功能
- 包括枚举测试、节点测试、引擎测试、构建器测试
- 验证条件分支、执行跟踪、导出功能

## 总结

阶段25成功实现了Agent工作流引擎，提供了：
- 丰富的节点类型（开始、结束、任务、条件、等待等）
- 流畅的构建器API
- 完整的执行引擎
- 条件分支支持
- 执行跟踪和监控

为构建复杂的Agent业务流程自动化奠定了坚实基础。
