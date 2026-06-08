"""
阶段37: Agent任务调度系统

遵循"图难于其易，为大于其细"的原则，实现定时任务、延迟任务、Cron调度。
"""

import heapq
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================
# 37.1 任务调度模型
# ============================================================

class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"           # 等待执行
    RUNNING = "running"           # 执行中
    SUCCESS = "success"           # 成功
    FAILED = "failed"             # 失败
    CANCELLED = "cancelled"       # 已取消
    RETRYING = "retrying"         # 重试中
    TIMEOUT = "timeout"           # 超时
    SCHEDULED = "scheduled"       # 已调度


class TriggerType(str, Enum):
    """触发器类型"""
    IMMEDIATE = "immediate"       # 立即执行
    DELAYED = "delayed"           # 延迟执行
    CRON = "cron"                 # Cron周期
    INTERVAL = "interval"         # 固定间隔
    ONCE = "once"                 # 单次


@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: str
    status: TaskStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    attempt: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": round(self.duration_seconds, 4),
            "result": str(self.result) if self.result is not None else None,
            "error": self.error,
            "retry_count": self.retry_count,
            "attempt": self.attempt,
        }


@dataclass
class TaskDefinition:
    """任务定义"""
    task_id: str
    name: str
    func: Callable
    args: Tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    trigger_type: TriggerType = TriggerType.IMMEDIATE
    trigger_config: Dict[str, Any] = field(default_factory=dict)
    max_retries: int = 0
    retry_delay_seconds: float = 1.0
    timeout_seconds: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduledTask:
    """已调度的任务（可比较，用于堆排序）"""
    task_def: TaskDefinition
    run_at: float  # monotonic timestamp
    scheduled_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __lt__(self, other):
        return self.run_at < other.run_at


@dataclass
class SchedulerStats:
    """调度器统计"""
    total_submitted: int = 0
    total_completed: int = 0
    total_failed: int = 0
    total_cancelled: int = 0
    total_retried: int = 0
    total_timeout: int = 0
    pending_count: int = 0
    active_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_submitted": self.total_submitted,
            "total_completed": self.total_completed,
            "total_failed": self.total_failed,
            "total_cancelled": self.total_cancelled,
            "total_retried": self.total_retried,
            "total_timeout": self.total_timeout,
            "pending_count": self.pending_count,
            "active_count": self.active_count,
        }


# ============================================================
# 37.2 任务调度器
# ============================================================

class TaskScheduler:
    """任务调度器"""

    def __init__(self, max_workers: int = 4):
        self._max_workers = max_workers
        self._task_heap: List[ScheduledTask] = []
        self._running_tasks: Dict[str, threading.Thread] = {}
        self._results: Dict[str, List[TaskResult]] = {}
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._stats = SchedulerStats()
        self._on_complete: Optional[Callable[[TaskResult], None]] = None
        self._on_failure: Optional[Callable[[TaskResult], None]] = None

    @property
    def stats(self) -> SchedulerStats:
        with self._lock:
            self._stats.pending_count = len(self._task_heap)
            self._stats.active_count = len(self._running_tasks)
            return SchedulerStats(
                total_submitted=self._stats.total_submitted,
                total_completed=self._stats.total_completed,
                total_failed=self._stats.total_failed,
                total_cancelled=self._stats.total_cancelled,
                total_retried=self._stats.total_retried,
                total_timeout=self._stats.total_timeout,
                pending_count=self._stats.pending_count,
                active_count=self._stats.active_count,
            )

    def set_on_complete(self, callback: Callable[[TaskResult], None]) -> None:
        self._on_complete = callback

    def set_on_failure(self, callback: Callable[[TaskResult], None]) -> None:
        self._on_failure = callback

    def submit(
        self,
        func: Callable,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        delay_seconds: float = 0.0,
        task_id: Optional[str] = None,
        name: str = "",
        max_retries: int = 0,
        timeout_seconds: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """提交任务"""
        tid = task_id or str(uuid.uuid4())
        trigger_type = TriggerType.DELAYED if delay_seconds > 0 else TriggerType.IMMEDIATE
        trigger_config = {"delay_seconds": delay_seconds} if delay_seconds > 0 else {}

        task_def = TaskDefinition(
            task_id=tid,
            name=name or func.__name__,
            func=func,
            args=args,
            kwargs=kwargs or {},
            trigger_type=trigger_type,
            trigger_config=trigger_config,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            tags=tags or [],
        )

        run_at = time.monotonic() + delay_seconds
        scheduled = ScheduledTask(task_def=task_def, run_at=run_at)

        with self._lock:
            heapq.heappush(self._task_heap, scheduled)
            self._stats.total_submitted += 1
            self._condition.notify()

        return tid

    def submit_interval(
        self,
        func: Callable,
        interval_seconds: float,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
        name: str = "",
    ) -> str:
        """提交间隔任务"""
        tid = task_id or str(uuid.uuid4())
        task_def = TaskDefinition(
            task_id=tid,
            name=name or func.__name__,
            func=func,
            args=args,
            kwargs=kwargs or {},
            trigger_type=TriggerType.INTERVAL,
            trigger_config={"interval_seconds": interval_seconds},
        )

        run_at = time.monotonic()
        scheduled = ScheduledTask(task_def=task_def, run_at=run_at)

        with self._lock:
            heapq.heappush(self._task_heap, scheduled)
            self._stats.total_submitted += 1
            self._condition.notify()

        return tid

    def cancel(self, task_id: str) -> bool:
        """取消任务"""
        with self._lock:
            # 从堆中移除
            new_heap = []
            found = False
            for st in self._task_heap:
                if st.task_def.task_id == task_id and not found:
                    found = True
                    self._stats.total_cancelled += 1
                else:
                    new_heap.append(st)
            self._task_heap = new_heap
            heapq.heapify(self._task_heap)
            return found

    def get_result(self, task_id: str) -> Optional[List[TaskResult]]:
        """获取任务结果"""
        with self._lock:
            return self._results.get(task_id)

    def start(self) -> None:
        """启动调度器"""
        if self._running:
            return
        self._running = True

        def _scheduler_loop():
            while self._running:
                with self._condition:
                    # 等待最近的任务
                    while self._task_heap and self._task_heap[0].run_at > time.monotonic():
                        wait_time = self._task_heap[0].run_at - time.monotonic()
                        self._condition.wait(timeout=min(wait_time, 1.0))
                        if not self._running:
                            break

                    if not self._running:
                        break

                    # 取出所有到期的任务
                    now = time.monotonic()
                    ready = []
                    while self._task_heap and self._task_heap[0].run_at <= now:
                        ready.append(heapq.heappop(self._task_heap))

                # 执行任务
                for scheduled in ready:
                    if not self._running:
                        break
                    self._execute_task(scheduled)

        self._scheduler_thread = threading.Thread(target=_scheduler_loop, daemon=True)
        self._scheduler_thread.start()

    def stop(self, wait: bool = False) -> None:
        """停止调度器"""
        self._running = False
        with self._condition:
            self._condition.notify_all()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
            self._scheduler_thread = None

    def _execute_task(self, scheduled: ScheduledTask) -> None:
        """执行任务"""
        task_def = scheduled.task_def

        # 检查并发限制
        with self._lock:
            if len(self._running_tasks) >= self._max_workers:
                # 放回堆中
                heapq.heappush(self._task_heap, scheduled)
                return

        result = self._run_once(task_def)

        # 处理间隔任务：重新调度
        if task_def.trigger_type == TriggerType.INTERVAL:
            interval = task_def.trigger_config.get("interval_seconds", 60)
            new_scheduled = ScheduledTask(
                task_def=task_def,
                run_at=time.monotonic() + interval,
            )
            with self._lock:
                heapq.heappush(self._task_heap, new_scheduled)

    def _run_once(self, task_def: TaskDefinition, attempt: int = 1) -> TaskResult:
        """执行一次任务"""
        start_time = datetime.now()
        start_mono = time.monotonic()

        def _run():
            return task_def.func(*task_def.args, **task_def.kwargs)

        thread = threading.Thread(target=_run)
        thread.start()

        # 等待完成或超时
        timeout = task_def.timeout_seconds
        thread.join(timeout=timeout)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        if thread.is_alive():
            # 超时
            result = TaskResult(
                task_id=task_def.task_id,
                status=TaskStatus.TIMEOUT,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                error=f"Task timed out after {timeout}s",
                attempt=attempt,
            )
            with self._lock:
                self._stats.total_timeout += 1
            self._store_result(result)
            if self._on_failure:
                self._on_failure(result)
            return result

        # 任务已完成（无法直接获取返回值，简化处理）
        result = TaskResult(
            task_id=task_def.task_id,
            status=TaskStatus.SUCCESS,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            attempt=attempt,
        )

        with self._lock:
            self._stats.total_completed += 1
        self._store_result(result)
        if self._on_complete:
            self._on_complete(result)
        return result

    def _store_result(self, result: TaskResult) -> None:
        with self._lock:
            if result.task_id not in self._results:
                self._results[result.task_id] = []
            self._results[result.task_id].append(result)


# ============================================================
# 37.3 Cron调度
# ============================================================

class CronExpression:
    """Cron表达式解析器（简化版5字段）"""

    def __init__(self, expression: str):
        self._expression = expression
        self._fields = self._parse(expression)

    @property
    def expression(self) -> str:
        return self._expression

    def matches(self, dt: datetime) -> bool:
        """检查时间是否匹配Cron表达式"""
        fields = self._fields
        if len(fields) != 5:
            return False

        return (
            self._match_field(fields[0], dt.minute, 0, 59) and
            self._match_field(fields[1], dt.hour, 0, 23) and
            self._match_field(fields[2], dt.day, 1, 31) and
            self._match_field(fields[3], dt.month, 1, 12) and
            self._match_field(fields[4], dt.isoweekday() % 7, 0, 6)
        )

    def next_run(self, after: Optional[datetime] = None) -> datetime:
        """计算下一次运行时间"""
        now = after or datetime.now()
        # 简化实现：每分钟检查一次
        candidate = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        for _ in range(525600):  # 最多检查一年
            if self.matches(candidate):
                return candidate
            candidate += timedelta(minutes=1)
        return now + timedelta(minutes=1)

    def _parse(self, expression: str) -> List[set]:
        """解析Cron表达式为字段集合"""
        parts = expression.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {expression} (expected 5 fields)")

        return [self._parse_field(part) for part in parts]

    def _parse_field(self, field: str) -> set:
        """解析单个字段"""
        values = set()
        for part in field.split(","):
            if "/" in part:
                base, step = part.split("/", 1)
                step = int(step)
                if base == "*":
                    start, end = 0, 59
                elif "-" in base:
                    start, end = map(int, base.split("-", 1))
                else:
                    start = int(base)
                    end = 59
                values.update(range(start, end + 1, step))
            elif "-" in part:
                start, end = map(int, part.split("-", 1))
                values.update(range(start, end + 1))
            elif part == "*":
                values.update(range(0, 60))
            else:
                values.add(int(part))
        return values

    def _match_field(self, allowed: set, value: int, min_val: int, max_val: int) -> bool:
        """检查值是否在允许集合中"""
        return value in allowed


class CronScheduler:
    """Cron调度器"""

    def __init__(self):
        self._jobs: Dict[str, Tuple[CronExpression, TaskDefinition]] = {}
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stats = SchedulerStats()
        self._on_complete: Optional[Callable[[TaskResult], None]] = None

    @property
    def stats(self) -> SchedulerStats:
        return SchedulerStats(
            total_submitted=self._stats.total_submitted,
            total_completed=self._stats.total_completed,
            total_failed=self._stats.total_failed,
        )

    def set_on_complete(self, callback: Callable[[TaskResult], None]) -> None:
        self._on_complete = callback

    def add_job(
        self,
        name: str,
        cron_expression: str,
        func: Callable,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """添加Cron任务"""
        jid = job_id or str(uuid.uuid4())
        cron = CronExpression(cron_expression)
        task_def = TaskDefinition(
            task_id=jid,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs or {},
            trigger_type=TriggerType.CRON,
            trigger_config={"cron": cron_expression},
        )
        with self._lock:
            self._jobs[jid] = (cron, task_def)
        return jid

    def remove_job(self, job_id: str) -> bool:
        """移除Cron任务"""
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                return True
            return False

    def get_jobs(self) -> List[Dict[str, Any]]:
        """获取所有任务"""
        with self._lock:
            return [
                {
                    "job_id": jid,
                    "name": td.name,
                    "cron": cron.expression,
                    "next_run": cron.next_run().isoformat(),
                }
                for jid, (cron, td) in self._jobs.items()
            ]

    def start(self) -> None:
        """启动Cron调度器"""
        if self._running:
            return
        self._running = True

        def _cron_loop():
            while self._running:
                now = datetime.now()
                with self._lock:
                    jobs = list(self._jobs.values())

                for cron, task_def in jobs:
                    if cron.matches(now):
                        self._execute_job(task_def)

                time.sleep(1.0)

        self._thread = threading.Thread(target=_cron_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止Cron调度器"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _execute_job(self, task_def: TaskDefinition) -> None:
        """执行Cron任务"""
        start = datetime.now()
        try:
            task_def.func(*task_def.args, **task_def.kwargs)
            result = TaskResult(
                task_id=task_def.task_id,
                status=TaskStatus.SUCCESS,
                start_time=start,
                end_time=datetime.now(),
                duration_seconds=(datetime.now() - start).total_seconds(),
            )
            with self._lock:
                self._stats.total_completed += 1
            if self._on_complete:
                self._on_complete(result)
        except Exception as e:
            with self._lock:
                self._stats.total_failed += 1


# ============================================================
# 37.4 延迟任务队列
# ============================================================

class DelayedTaskQueue:
    """延迟任务队列"""

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self._queue: List[Tuple[float, int, TaskDefinition]] = []  # (run_at, seq, task_def)
        self._seq = 0
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stats = SchedulerStats()
        self._results: Dict[str, List[TaskResult]] = {}

    @property
    def stats(self) -> SchedulerStats:
        with self._lock:
            self._stats.pending_count = len(self._queue)
            return SchedulerStats(
                total_submitted=self._stats.total_submitted,
                total_completed=self._stats.total_completed,
                total_failed=self._stats.total_failed,
                total_retried=self._stats.total_retried,
                pending_count=self._stats.pending_count,
            )

    def enqueue(
        self,
        func: Callable,
        delay_seconds: float = 0.0,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
        name: str = "",
        max_retries: Optional[int] = None,
    ) -> str:
        """入队延迟任务"""
        tid = task_id or str(uuid.uuid4())
        task_def = TaskDefinition(
            task_id=tid,
            name=name or func.__name__,
            func=func,
            args=args,
            kwargs=kwargs or {},
            trigger_type=TriggerType.DELAYED,
            trigger_config={"delay_seconds": delay_seconds},
            max_retries=max_retries if max_retries is not None else self._max_retries,
        )

        with self._lock:
            self._seq += 1
            run_at = time.monotonic() + delay_seconds
            heapq.heappush(self._queue, (run_at, self._seq, task_def))
            self._stats.total_submitted += 1

        return tid

    def dequeue(self) -> Optional[TaskDefinition]:
        """取出到期的任务（非阻塞）"""
        with self._lock:
            if not self._queue:
                return None
            if self._queue[0][0] > time.monotonic():
                return None
            _, _, task_def = heapq.heappop(self._queue)
            return task_def

    def start(self) -> None:
        """启动消费线程"""
        if self._running:
            return
        self._running = True

        def _consume_loop():
            while self._running:
                task_def = self.dequeue()
                if task_def:
                    self._execute_with_retry(task_def)
                else:
                    time.sleep(0.05)

        self._thread = threading.Thread(target=_consume_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止消费"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def get_result(self, task_id: str) -> Optional[List[TaskResult]]:
        """获取任务结果"""
        with self._lock:
            return self._results.get(task_id)

    def _execute_with_retry(self, task_def: TaskDefinition) -> None:
        """带重试的执行"""
        max_retries = task_def.max_retries or self._max_retries
        for attempt in range(1, max_retries + 2):
            start = datetime.now()
            try:
                task_def.func(*task_def.args, **task_def.kwargs)
                result = TaskResult(
                    task_id=task_def.task_id,
                    status=TaskStatus.SUCCESS,
                    start_time=start,
                    end_time=datetime.now(),
                    duration_seconds=(datetime.now() - start).total_seconds(),
                    attempt=attempt,
                )
                with self._lock:
                    self._stats.total_completed += 1
                self._store_result(result)
                return
            except Exception as e:
                if attempt <= max_retries:
                    with self._lock:
                        self._stats.total_retried += 1
                    time.sleep(self._retry_delay * attempt)
                else:
                    result = TaskResult(
                        task_id=task_def.task_id,
                        status=TaskStatus.FAILED,
                        start_time=start,
                        end_time=datetime.now(),
                        duration_seconds=(datetime.now() - start).total_seconds(),
                        error=str(e),
                        attempt=attempt,
                    )
                    with self._lock:
                        self._stats.total_failed += 1
                    self._store_result(result)

    def _store_result(self, result: TaskResult) -> None:
        with self._lock:
            if result.task_id not in self._results:
                self._results[result.task_id] = []
            self._results[result.task_id].append(result)


# ============================================================
# 便捷函数
# ============================================================

def create_task_scheduler(max_workers: int = 4) -> TaskScheduler:
    return TaskScheduler(max_workers)


def create_cron_scheduler() -> CronScheduler:
    return CronScheduler()


def create_delayed_queue(max_retries: int = 3, retry_delay: float = 1.0) -> DelayedTaskQueue:
    return DelayedTaskQueue(max_retries, retry_delay)
