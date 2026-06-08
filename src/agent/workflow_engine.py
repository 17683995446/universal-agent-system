"""
阶段81: Agent自动化工作流编排系统

遵循"图难于其易，为大于其细"的原则，实现工作流定义、任务调度、依赖管理、状态追踪。
"""
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class Task:
    task_id: str
    name: str
    action: Callable[[], Any]
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "retry_count": self.retry_count,
        }


@dataclass
class Workflow:
    workflow_id: str
    name: str
    tasks: Dict[str, Task] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "status": self.status.value,
            "task_count": len(self.tasks),
        }


class DependencyResolver:
    """依赖解析器"""

    def __init__(self, tasks: Dict[str, Task]):
        self._tasks = tasks

    def get_ready_tasks(self) -> List[str]:
        """获取所有依赖已满足的任务"""
        ready = []
        for task_id, task in self._tasks.items():
            if task.status != TaskStatus.PENDING:
                continue
            if all(self._tasks[dep].status == TaskStatus.SUCCESS for dep in task.dependencies):
                ready.append(task_id)
        return ready

    def detect_cycles(self) -> Optional[List[str]]:
        """检测循环依赖，返回循环路径或None"""
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> Optional[List[str]]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            for dep in self._tasks.get(node, Task("", "", lambda: None)).dependencies:
                if dep not in self._tasks:
                    continue
                if dep not in visited:
                    result = dfs(dep)
                    if result:
                        return result
                elif dep in rec_stack:
                    cycle_start = path.index(dep)
                    return path[cycle_start:] + [dep]
            path.pop()
            rec_stack.remove(node)
            return None

        for task_id in self._tasks:
            if task_id not in visited:
                cycle = dfs(task_id)
                if cycle:
                    return cycle
        return None

    def get_execution_order(self) -> List[str]:
        """拓扑排序获取执行顺序"""
        in_degree = {tid: 0 for tid in self._tasks}
        for task in self._tasks.values():
            for dep in task.dependencies:
                if dep in self._tasks:
                    in_degree[task.task_id] += 1

        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        order = []
        while queue:
            tid = queue.pop(0)
            order.append(tid)
            for task in self._tasks.values():
                if tid in task.dependencies:
                    in_degree[task.task_id] -= 1
                    if in_degree[task.task_id] == 0:
                        queue.append(task.task_id)
        return order


class TaskExecutor:
    """任务执行器"""

    def execute(self, task: Task) -> None:
        if task.status != TaskStatus.PENDING:
            return
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        try:
            task.result = task.action()
            task.status = TaskStatus.SUCCESS
        except Exception as e:
            task.error = str(e)
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
            else:
                task.status = TaskStatus.FAILED
        task.ended_at = time.time()


class WorkflowEngine:
    """工作流引擎"""

    def __init__(self):
        self._workflows: Dict[str, Workflow] = {}
        self._executor = TaskExecutor()

    def create_workflow(self, name: str) -> Workflow:
        wf_id = str(uuid.uuid4())[:8]
        wf = Workflow(workflow_id=wf_id, name=name)
        self._workflows[wf_id] = wf
        return wf

    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        return self._workflows.get(workflow_id)

    def add_task(self, workflow_id: str, name: str, action: Callable[[], Any],
                 dependencies: Optional[List[str]] = None, max_retries: int = 0) -> Optional[Task]:
        wf = self._workflows.get(workflow_id)
        if not wf:
            return None
        task_id = f"{workflow_id}_{len(wf.tasks)}"
        task = Task(
            task_id=task_id,
            name=name,
            action=action,
            dependencies=dependencies or [],
            max_retries=max_retries,
        )
        wf.tasks[task_id] = task
        return task

    def run(self, workflow_id: str) -> bool:
        wf = self._workflows.get(workflow_id)
        if not wf:
            return False
        resolver = DependencyResolver(wf.tasks)
        cycle = resolver.detect_cycles()
        if cycle:
            wf.status = TaskStatus.FAILED
            return False

        wf.status = TaskStatus.RUNNING
        while True:
            ready = resolver.get_ready_tasks()
            if not ready:
                break
            for task_id in ready:
                task = wf.tasks[task_id]
                self._executor.execute(task)

        # 检查最终状态
        if any(t.status == TaskStatus.FAILED for t in wf.tasks.values()):
            wf.status = TaskStatus.FAILED
        else:
            wf.status = TaskStatus.SUCCESS
        wf.ended_at = time.time()
        return wf.status == TaskStatus.SUCCESS

    def cancel(self, workflow_id: str) -> bool:
        wf = self._workflows.get(workflow_id)
        if not wf:
            return False
        for task in wf.tasks.values():
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
        wf.status = TaskStatus.CANCELLED
        return True

    def get_task_results(self, workflow_id: str) -> Dict[str, Any]:
        wf = self._workflows.get(workflow_id)
        if not wf:
            return {}
        return {tid: t.result for tid, t in wf.tasks.items() if t.status == TaskStatus.SUCCESS}

    def get_stats(self, workflow_id: str) -> Dict[str, Any]:
        wf = self._workflows.get(workflow_id)
        if not wf:
            return {}
        statuses = {}
        for t in wf.tasks.values():
            statuses[t.status.value] = statuses.get(t.status.value, 0) + 1
        return {
            "workflow_id": workflow_id,
            "name": wf.name,
            "status": wf.status.value,
            "task_count": len(wf.tasks),
            "statuses": statuses,
        }


def create_workflow_engine() -> WorkflowEngine:
    return WorkflowEngine()
