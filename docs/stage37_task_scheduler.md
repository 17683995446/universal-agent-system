# 阶段37: Agent任务调度系统

> 遵循"图难于其易，为大于其细。天下难事，必作于易；天下大事，必作于细。"的原则

## 概述

阶段37实现了完整的Agent任务调度系统，提供即时/延迟任务提交、Cron周期调度、延迟任务队列、超时控制和失败重试等功能。

## 核心组件

### 1. 任务调度模型

- **TaskStatus**: PENDING / RUNNING / SUCCESS / FAILED / CANCELLED / RETRYING / TIMEOUT
- **TriggerType**: IMMEDIATE / DELAYED / CRON / INTERVAL / ONCE
- **TaskResult**: 任务执行结果（状态、耗时、错误、重试次数）
- **TaskDefinition**: 任务定义（函数、参数、触发器、重试策略）
- **SchedulerStats**: 调度器统计

### 2. TaskScheduler - 任务调度器

```python
scheduler = create_task_scheduler(max_workers=4)

# 即时执行
scheduler.submit(my_func, args=(1, 2))

# 延迟执行
scheduler.submit(my_func, delay_seconds=5.0)

# 间隔任务
scheduler.submit_interval(my_func, interval_seconds=10.0)

# 取消任务
scheduler.cancel(task_id)

# 超时控制
scheduler.submit(slow_func, timeout_seconds=3.0)

scheduler.start()
scheduler.stop()
```

### 3. CronExpression - Cron表达式解析

支持标准5字段Cron表达式：`分 时 日 月 星期`

```python
cron = CronExpression("*/5 * * * *")  # 每5分钟
cron = CronExpression("0 9 * * 1-5")  # 工作日9点
cron = CronExpression("30 2 1 * *")    # 每月1号2:30

assert cron.matches(datetime(2024, 1, 1, 12, 5))  # True
next_run = cron.next_run()
```

### 4. CronScheduler - Cron调度器

```python
cron = create_cron_scheduler()
cron.add_job("cleanup", "0 2 * * *", cleanup_func)
cron.add_job("report", "*/30 * * * *", generate_report)
cron.start()
cron.stop()
```

### 5. DelayedTaskQueue - 延迟任务队列

```python
queue = create_delayed_queue(max_retries=3, retry_delay=1.0)

# 入队（可延迟）
queue.enqueue(send_email, delay_seconds=60.0, args=(user,))

# 带重试
queue.enqueue(fragile_task, max_retries=5)

queue.start()
queue.stop()
```

## 特性

| 特性 | 说明 |
|------|------|
| 即时/延迟执行 | 支持立即执行和延迟执行 |
| Cron周期调度 | 标准5字段Cron表达式 |
| 固定间隔调度 | 按固定间隔重复执行 |
| 超时控制 | 任务执行超时自动终止 |
| 失败重试 | 可配置重试次数和延迟 |
| 任务取消 | 支持取消等待中的任务 |
| 并发控制 | 限制最大并发工作线程 |
| 回调通知 | 完成/失败回调 |

## 测试覆盖

共36个测试用例，覆盖模型、Cron解析、调度器、延迟队列、集成测试。

## 文件清单

| 文件 | 说明 |
|------|------|
| `src/agent/task_scheduler.py` | 任务调度实现 |
| `tests/test_task_scheduler.py` | 单元测试 |
| `docs/stage37_task_scheduler.md` | 本文档 |

## 与前一阶段的集成

- 任务调度可与阶段36缓存结合，缓存任务结果
- 可与阶段35消息队列结合，异步投递任务
- 可与阶段34限流器结合，控制任务提交速率

## 下一步（阶段38）

下一阶段将实现 **Agent序列化系统**，提供多种序列化格式、版本兼容、压缩等功能。

## 总结

1. ✅ 定义了任务调度模型（状态、触发器、结果、定义、统计）
2. ✅ 实现了任务调度器（即时/延迟/间隔执行、超时、取消、回调）
3. ✅ 实现了Cron表达式解析器（5字段、范围、步长、列表）
4. ✅ 实现了Cron调度器（周期调度、任务管理）
5. ✅ 实现了延迟任务队列（延迟投递、失败重试、结果追踪）
6. ✅ 编写了36个单元测试，全部通过
7. ✅ 集成到系统导出
