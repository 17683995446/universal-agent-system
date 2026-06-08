"""阶段81: Agent自动化工作流编排系统 - 单元测试"""
import pytest
from src.agent.workflow_engine import (
    TaskStatus, Task, Workflow,
    DependencyResolver, TaskExecutor, WorkflowEngine, create_workflow_engine,
)


class TestDependencyResolver:
    def test_get_ready_tasks(self):
        t1 = Task("t1", "task1", lambda: 1)
        t2 = Task("t2", "task2", lambda: 2, dependencies=["t1"])
        t3 = Task("t3", "task3", lambda: 3, dependencies=["t1", "t2"])
        tasks = {"t1": t1, "t2": t2, "t3": t3}
        resolver = DependencyResolver(tasks)
        ready = resolver.get_ready_tasks()
        assert ready == ["t1"]
        t1.status = TaskStatus.SUCCESS
        ready = resolver.get_ready_tasks()
        assert "t2" in ready
        t2.status = TaskStatus.SUCCESS
        ready = resolver.get_ready_tasks()
        assert "t3" in ready

    def test_detect_cycles_no_cycle(self):
        t1 = Task("t1", "task1", lambda: 1)
        t2 = Task("t2", "task2", lambda: 2, dependencies=["t1"])
        resolver = DependencyResolver({"t1": t1, "t2": t2})
        assert resolver.detect_cycles() is None

    def test_detect_cycles_with_cycle(self):
        t1 = Task("t1", "task1", lambda: 1, dependencies=["t2"])
        t2 = Task("t2", "task2", lambda: 2, dependencies=["t1"])
        resolver = DependencyResolver({"t1": t1, "t2": t2})
        cycle = resolver.detect_cycles()
        assert cycle is not None
        assert "t1" in cycle and "t2" in cycle

    def test_get_execution_order(self):
        t1 = Task("t1", "task1", lambda: 1)
        t2 = Task("t2", "task2", lambda: 2, dependencies=["t1"])
        t3 = Task("t3", "task3", lambda: 3, dependencies=["t1"])
        resolver = DependencyResolver({"t1": t1, "t2": t2, "t3": t3})
        order = resolver.get_execution_order()
        assert order[0] == "t1"
        assert "t2" in order[1:]
        assert "t3" in order[1:]


class TestTaskExecutor:
    def test_execute_success(self):
        executor = TaskExecutor()
        task = Task("t1", "task1", lambda: 42)
        executor.execute(task)
        assert task.status == TaskStatus.SUCCESS
        assert task.result == 42

    def test_execute_failure(self):
        executor = TaskExecutor()
        task = Task("t1", "task1", lambda: (_ for _ in ()).throw(ValueError("error")))
        executor.execute(task)
        assert task.status == TaskStatus.FAILED
        assert "error" in task.error

    def test_execute_retry(self):
        executor = TaskExecutor()
        task = Task("t1", "task1", lambda: (_ for _ in ()).throw(ValueError("error")), max_retries=2)
        executor.execute(task)
        assert task.status == TaskStatus.PENDING  # 重试
        assert task.retry_count == 1


class TestWorkflowEngine:
    def test_create_workflow(self):
        engine = WorkflowEngine()
        wf = engine.create_workflow("test")
        assert wf.name == "test"
        assert wf.workflow_id in engine._workflows

    def test_add_task(self):
        engine = WorkflowEngine()
        wf = engine.create_workflow("test")
        task = engine.add_task(wf.workflow_id, "step1", lambda: 1)
        assert task is not None
        assert task.name == "step1"
        assert engine.add_task("bad_id", "step", lambda: 1) is None

    def test_run_linear(self):
        engine = WorkflowEngine()
        wf = engine.create_workflow("test")
        results = []
        engine.add_task(wf.workflow_id, "step1", lambda: results.append(1))
        t2 = engine.add_task(wf.workflow_id, "step2", lambda: results.append(2))
        t3 = engine.add_task(wf.workflow_id, "step3", lambda: results.append(3), dependencies=[t2.task_id])
        success = engine.run(wf.workflow_id)
        assert success
        assert len(results) == 3

    def test_run_with_failure(self):
        engine = WorkflowEngine()
        wf = engine.create_workflow("test")
        engine.add_task(wf.workflow_id, "step1", lambda: (_ for _ in ()).throw(ValueError("fail")))
        success = engine.run(wf.workflow_id)
        assert not success

    def test_run_cycle(self):
        engine = WorkflowEngine()
        wf = engine.create_workflow("test")
        t1 = engine.add_task(wf.workflow_id, "step1", lambda: 1)
        t2 = engine.add_task(wf.workflow_id, "step2", lambda: 2, dependencies=[t1.task_id])
        # 制造循环依赖
        t1.dependencies = [t2.task_id]
        success = engine.run(wf.workflow_id)
        assert not success

    def test_cancel(self):
        engine = WorkflowEngine()
        wf = engine.create_workflow("test")
        t1 = engine.add_task(wf.workflow_id, "step1", lambda: 1)
        assert engine.cancel(wf.workflow_id)
        assert t1.status == TaskStatus.CANCELLED
        assert wf.status == TaskStatus.CANCELLED

    def test_get_task_results(self):
        engine = WorkflowEngine()
        wf = engine.create_workflow("test")
        engine.add_task(wf.workflow_id, "step1", lambda: 42)
        engine.run(wf.workflow_id)
        results = engine.get_task_results(wf.workflow_id)
        assert len(results) == 1
        assert 42 in results.values()

    def test_get_stats(self):
        engine = WorkflowEngine()
        wf = engine.create_workflow("test")
        engine.add_task(wf.workflow_id, "step1", lambda: 1)
        engine.add_task(wf.workflow_id, "step2", lambda: 2)
        engine.run(wf.workflow_id)
        stats = engine.get_stats(wf.workflow_id)
        assert stats["task_count"] == 2
        assert stats["status"] == "success"

    def test_create_workflow_engine(self):
        engine = create_workflow_engine()
        assert isinstance(engine, WorkflowEngine)
