"""
阶段37测试: Agent任务调度系统
"""

import time
import threading
import pytest
from datetime import datetime

from src.agent.task_scheduler import (
    TaskStatus,
    TriggerType,
    TaskResult,
    TaskDefinition,
    ScheduledTask,
    SchedulerStats,
    CronExpression,
    TaskScheduler,
    CronScheduler,
    DelayedTaskQueue,
    create_task_scheduler,
    create_cron_scheduler,
    create_delayed_queue,
)


# ============================================================
# 模型测试
# ============================================================

class TestTaskStatus:
    def test_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.SUCCESS.value == "success"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"
        assert TaskStatus.TIMEOUT.value == "timeout"


class TestTriggerType:
    def test_values(self):
        assert TriggerType.IMMEDIATE.value == "immediate"
        assert TriggerType.DELAYED.value == "delayed"
        assert TriggerType.CRON.value == "cron"
        assert TriggerType.INTERVAL.value == "interval"


class TestTaskResult:
    def test_creation(self):
        result = TaskResult(
            task_id="t1", status=TaskStatus.SUCCESS,
            duration_seconds=1.5, attempt=2,
        )
        assert result.task_id == "t1"
        assert result.attempt == 2

    def test_to_dict(self):
        result = TaskResult(task_id="t1", status=TaskStatus.FAILED, error="oops")
        d = result.to_dict()
        assert d["status"] == "failed"
        assert d["error"] == "oops"


class TestSchedulerStats:
    def test_to_dict(self):
        stats = SchedulerStats(total_submitted=10, total_completed=8, total_failed=2)
        d = stats.to_dict()
        assert d["total_submitted"] == 10
        assert d["total_completed"] == 8


# ============================================================
# CronExpression 测试
# ============================================================

class TestCronExpression:
    def test_every_minute(self):
        cron = CronExpression("* * * * *")
        assert cron.matches(datetime(2024, 1, 1, 12, 30))

    def test_specific_minute(self):
        cron = CronExpression("30 * * * *")
        assert cron.matches(datetime(2024, 1, 1, 12, 30))
        assert not cron.matches(datetime(2024, 1, 1, 12, 31))

    def test_range(self):
        cron = CronExpression("0-5 * * * *")
        assert cron.matches(datetime(2024, 1, 1, 12, 3))
        assert not cron.matches(datetime(2024, 1, 1, 12, 10))

    def test_step(self):
        cron = CronExpression("*/15 * * * *")
        assert cron.matches(datetime(2024, 1, 1, 12, 0))
        assert cron.matches(datetime(2024, 1, 1, 12, 15))
        assert cron.matches(datetime(2024, 1, 1, 12, 30))
        assert cron.matches(datetime(2024, 1, 1, 12, 45))
        assert not cron.matches(datetime(2024, 1, 1, 12, 7))

    def test_list(self):
        cron = CronExpression("0,30 * * * *")
        assert cron.matches(datetime(2024, 1, 1, 12, 0))
        assert cron.matches(datetime(2024, 1, 1, 12, 30))
        assert not cron.matches(datetime(2024, 1, 1, 12, 15))

    def test_hour(self):
        cron = CronExpression("0 9 * * *")
        assert cron.matches(datetime(2024, 1, 1, 9, 0))
        assert not cron.matches(datetime(2024, 1, 1, 10, 0))

    def test_next_run(self):
        cron = CronExpression("0 * * * *")
        next_run = cron.next_run(datetime(2024, 1, 1, 12, 30))
        assert next_run.hour == 13
        assert next_run.minute == 0

    def test_invalid_expression(self):
        with pytest.raises(ValueError):
            CronExpression("invalid")


# ============================================================
# TaskScheduler 测试
# ============================================================

class TestTaskScheduler:
    def setup_method(self):
        self.scheduler = create_task_scheduler(max_workers=4)

    def teardown_method(self):
        self.scheduler.stop()

    def test_submit_immediate(self):
        results = []
        def task():
            results.append("done")

        tid = self.scheduler.submit(task)
        self.scheduler.start()
        time.sleep(0.2)
        self.scheduler.stop()

        assert len(results) == 1
        stats = self.scheduler.stats
        assert stats.total_submitted == 1
        assert stats.total_completed == 1

    def test_submit_delayed(self):
        results = []
        def task():
            results.append("done")

        tid = self.scheduler.submit(task, delay_seconds=0.2)
        self.scheduler.start()
        time.sleep(0.1)
        assert len(results) == 0  # 还没执行
        time.sleep(0.3)
        self.scheduler.stop()
        assert len(results) == 1

    def test_submit_with_args(self):
        results = []
        def task(a, b):
            results.append(a + b)

        self.scheduler.submit(task, args=(3, 4))
        self.scheduler.start()
        time.sleep(0.2)
        self.scheduler.stop()

        assert results == [7]

    def test_submit_with_kwargs(self):
        results = []
        def task(name, value):
            results.append((name, value))

        self.scheduler.submit(task, kwargs={"name": "test", "value": 42})
        self.scheduler.start()
        time.sleep(0.2)
        self.scheduler.stop()

        assert results == [("test", 42)]

    def test_cancel(self):
        results = []
        def task():
            results.append("done")

        tid = self.scheduler.submit(task, delay_seconds=1.0)
        assert self.scheduler.cancel(tid) is True
        self.scheduler.start()
        time.sleep(0.2)
        self.scheduler.stop()

        assert len(results) == 0
        assert self.scheduler.stats.total_cancelled == 1

    def test_timeout(self):
        def slow_task():
            time.sleep(5.0)

        self.scheduler.submit(slow_task, timeout_seconds=0.1)
        self.scheduler.start()
        time.sleep(0.5)
        self.scheduler.stop()

        stats = self.scheduler.stats
        assert stats.total_timeout == 1

    def test_stats(self):
        self.scheduler.submit(lambda: None)
        self.scheduler.submit(lambda: None, delay_seconds=1.0)
        self.scheduler.start()
        time.sleep(0.2)
        self.scheduler.stop()

        stats = self.scheduler.stats
        assert stats.total_submitted == 2

    def test_on_complete_callback(self):
        completed = []
        def task():
            pass

        self.scheduler.set_on_complete(lambda r: completed.append(r.task_id))
        self.scheduler.submit(task)
        self.scheduler.start()
        time.sleep(0.2)
        self.scheduler.stop()

        assert len(completed) == 1


# ============================================================
# CronScheduler 测试
# ============================================================

class TestCronScheduler:
    def setup_method(self):
        self.cron = create_cron_scheduler()

    def teardown_method(self):
        self.cron.stop()

    def test_add_remove_job(self):
        jid = self.cron.add_job("test", "* * * * *", lambda: None)
        assert self.cron.remove_job(jid) is True
        assert self.cron.remove_job(jid) is False

    def test_get_jobs(self):
        self.cron.add_job("job1", "0 * * * *", lambda: None)
        self.cron.add_job("job2", "*/5 * * * *", lambda: None)
        jobs = self.cron.get_jobs()
        assert len(jobs) == 2

    def test_cron_execution(self):
        results = []
        def task():
            results.append(datetime.now())

        # 每秒执行（用秒级近似）
        self.cron.add_job("frequent", "* * * * *", task)
        self.cron.start()
        time.sleep(2.5)
        self.cron.stop()

        # 至少执行了1次
        assert len(results) >= 1

    def test_stats(self):
        self.cron.add_job("test", "* * * * *", lambda: None)
        self.cron.start()
        time.sleep(2.0)
        self.cron.stop()

        stats = self.cron.stats
        assert stats.total_completed >= 1


# ============================================================
# DelayedTaskQueue 测试
# ============================================================

class TestDelayedTaskQueue:
    def setup_method(self):
        self.queue = create_delayed_queue(max_retries=2, retry_delay=0.1)

    def teardown_method(self):
        self.queue.stop()

    def test_enqueue_and_execute(self):
        results = []
        def task():
            results.append("done")

        self.queue.enqueue(task)
        self.queue.start()
        time.sleep(0.2)
        self.queue.stop()

        assert len(results) == 1
        assert self.queue.stats.total_completed == 1

    def test_delayed_execution(self):
        results = []
        def task():
            results.append("done")

        self.queue.enqueue(task, delay_seconds=0.2)
        self.queue.start()
        time.sleep(0.1)
        assert len(results) == 0
        time.sleep(0.3)
        self.queue.stop()

        assert len(results) == 1

    def test_retry_on_failure(self):
        attempt_count = 0
        def failing_task():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise RuntimeError("fail")

        self.queue.enqueue(failing_task, max_retries=3)
        self.queue.start()
        time.sleep(1.0)
        self.queue.stop()

        assert attempt_count == 3  # 1次 + 2次重试
        assert self.queue.stats.total_completed == 1
        assert self.queue.stats.total_retried == 2

    def test_max_retries_exceeded(self):
        attempt_count = 0
        def always_fail():
            nonlocal attempt_count
            attempt_count += 1
            raise RuntimeError("always fail")

        self.queue.enqueue(always_fail, max_retries=2)
        self.queue.start()
        time.sleep(1.0)
        self.queue.stop()

        assert attempt_count == 3  # 1次 + 2次重试
        assert self.queue.stats.total_failed == 1

    def test_with_args(self):
        results = []
        def task(a, b):
            results.append(a * b)

        self.queue.enqueue(task, args=(6, 7))
        self.queue.start()
        time.sleep(0.2)
        self.queue.stop()

        assert results == [42]

    def test_stats(self):
        self.queue.enqueue(lambda: None)
        self.queue.enqueue(lambda: None, delay_seconds=10.0)
        self.queue.start()
        time.sleep(0.2)
        self.queue.stop()

        stats = self.queue.stats
        assert stats.total_submitted == 2
        assert stats.total_completed == 1


# ============================================================
# 便捷函数测试
# ============================================================

class TestConvenienceFunctions:
    def test_create_task_scheduler(self):
        s = create_task_scheduler(8)
        assert isinstance(s, TaskScheduler)

    def test_create_cron_scheduler(self):
        c = create_cron_scheduler()
        assert isinstance(c, CronScheduler)

    def test_create_delayed_queue(self):
        q = create_delayed_queue(max_retries=5)
        assert isinstance(q, DelayedTaskQueue)


# ============================================================
# 集成测试
# ============================================================

class TestIntegration:
    def test_scheduler_with_delayed_queue(self):
        """调度器与延迟队列联合"""
        scheduler = create_task_scheduler(max_workers=2)
        queue = create_delayed_queue(max_retries=2, retry_delay=0.05)

        results = []

        def task(name):
            results.append(name)

        # 通过调度器提交
        scheduler.submit(task, args=("scheduler",))
        # 通过延迟队列提交
        queue.enqueue(task, args=("queue",))

        scheduler.start()
        queue.start()
        time.sleep(0.3)
        scheduler.stop()
        queue.stop()

        assert "scheduler" in results
        assert "queue" in results

    def test_cron_with_callback(self):
        """Cron调度器带回调"""
        cron = create_cron_scheduler()
        completed = []

        cron.set_on_complete(lambda r: completed.append(r.task_id))
        cron.add_job("test", "* * * * *", lambda: None)
        cron.start()
        time.sleep(2.0)
        cron.stop()

        assert len(completed) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
