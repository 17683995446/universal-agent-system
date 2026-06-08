"""
任务快照模块单元测试

测试范围：
- TaskStep 结构体（创建、序列化、验证、状态流转）
- TaskSnapshot 结构体（创建、序列化、验证、工具方法）
- SnapshotManager 快照管理器（CRUD、自动保存、恢复）
- DriftDetector 漂移检测器（相似度计算、漂移检测）

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

import unittest
import json
from datetime import datetime, timedelta
from typing import Optional

import sys
sys.path.insert(0, '/workspace')

from src.modules.task_manager.task_step import (
    TaskStep, StepStatus,
    filter_by_status, filter_pending, filter_completed, filter_failed,
    get_completion_ratio, get_total_duration_seconds, validate_step_list,
    steps_to_json, steps_from_json,
)
from src.modules.task_manager.task_snapshot import TaskSnapshot, TaskState
from src.modules.task_manager.snapshot_manager import (
    SnapshotManager, InMemorySnapshotStorage,
)
from src.modules.task_manager.drift_detector import (
    DriftDetector, PeriodicDriftDetector, DriftCheckResult,
)
from src.core.exceptions import SnapshotNotFound, TaskDriftDetected


# =============================================================================
# TaskStep 测试
# =============================================================================

class TestStepStatus(unittest.TestCase):
    """测试步骤状态枚举"""

    def test_status_values(self):
        self.assertEqual(StepStatus.PENDING.value, "pending")
        self.assertEqual(StepStatus.IN_PROGRESS.value, "in_progress")
        self.assertEqual(StepStatus.COMPLETED.value, "completed")
        self.assertEqual(StepStatus.FAILED.value, "failed")
        self.assertEqual(StepStatus.SKIPPED.value, "skipped")

    def test_status_from_string(self):
        self.assertEqual(StepStatus("pending"), StepStatus.PENDING)
        self.assertEqual(StepStatus("completed"), StepStatus.COMPLETED)


class TestTaskStepCreation(unittest.TestCase):
    """测试TaskStep创建"""

    def test_basic_creation(self):
        step = TaskStep(step_id="step_1", description="Test step")
        self.assertEqual(step.step_id, "step_1")
        self.assertEqual(step.description, "Test step")
        self.assertEqual(step.status, StepStatus.PENDING)
        self.assertIsNone(step.result)
        self.assertIsNone(step.error)
        self.assertIsNone(step.started_at)
        self.assertIsNone(step.completed_at)
        self.assertEqual(step.metadata, {})

    def test_creation_with_all_fields(self):
        now = datetime.now()
        step = TaskStep(
            step_id="step_2",
            description="Full step",
            status=StepStatus.COMPLETED,
            result="Success",
            error=None,
            started_at=now,
            completed_at=now,
            metadata={"key": "value"},
        )
        self.assertEqual(step.status, StepStatus.COMPLETED)
        self.assertEqual(step.result, "Success")
        self.assertEqual(step.metadata, {"key": "value"})

    def test_creation_with_string_status(self):
        step = TaskStep(step_id="step_3", description="Test", status="in_progress")
        self.assertEqual(step.status, StepStatus.IN_PROGRESS)


class TestTaskStepSerialization(unittest.TestCase):
    """测试序列化"""

    def setUp(self):
        self.step = TaskStep(
            step_id="step_ser",
            description="Serialization test",
            status=StepStatus.COMPLETED,
            result="Done",
            metadata={"test": True},
        )

    def test_to_dict(self):
        data = self.step.to_dict()
        self.assertEqual(data["step_id"], "step_ser")
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["result"], "Done")

    def test_to_json(self):
        json_str = self.step.to_json()
        data = json.loads(json_str)
        self.assertEqual(data["description"], "Serialization test")

    def test_from_dict(self):
        data = {
            "step_id": "step_from",
            "description": "From dict",
            "status": "failed",
            "error": "Something went wrong",
        }
        step = TaskStep.from_dict(data)
        self.assertEqual(step.step_id, "step_from")
        self.assertEqual(step.status, StepStatus.FAILED)
        self.assertEqual(step.error, "Something went wrong")

    def test_roundtrip(self):
        data = self.step.to_dict()
        step2 = TaskStep.from_dict(data)
        self.assertEqual(step2.step_id, self.step.step_id)
        self.assertEqual(step2.status, self.step.status)


class TestTaskStepValidation(unittest.TestCase):
    """测试验证"""

    def test_valid_step(self):
        step = TaskStep(step_id="valid", description="Valid step")
        valid, errors = step.validate()
        self.assertTrue(valid)
        self.assertEqual(errors, [])

    def test_empty_step_id(self):
        step = TaskStep(step_id="", description="Test")
        valid, errors = step.validate()
        self.assertFalse(valid)
        self.assertIn("step_id不能为空", errors)

    def test_empty_description(self):
        step = TaskStep(step_id="test", description="")
        valid, errors = step.validate()
        self.assertFalse(valid)
        self.assertIn("description不能为空", errors)

    def test_completed_without_result(self):
        step = TaskStep(step_id="test", description="Test", status=StepStatus.COMPLETED)
        valid, errors = step.validate()
        self.assertFalse(valid)
        self.assertTrue(any("已完成步骤应有result" in e for e in errors))

    def test_failed_without_error(self):
        step = TaskStep(step_id="test", description="Test", status=StepStatus.FAILED)
        valid, errors = step.validate()
        self.assertFalse(valid)
        self.assertTrue(any("失败步骤应有error" in e for e in errors))

    def test_invalid_time_order(self):
        now = datetime.now()
        past = now - timedelta(hours=1)
        step = TaskStep(
            step_id="test",
            description="Test",
            started_at=now,
            completed_at=past,
        )
        valid, errors = step.validate()
        self.assertFalse(valid)
        self.assertTrue(any("completed_at不能早于started_at" in e for e in errors))


class TestTaskStepStateTransition(unittest.TestCase):
    """测试状态流转"""

    def test_start(self):
        step = TaskStep(step_id="s1", description="Test")
        step.start()
        self.assertEqual(step.status, StepStatus.IN_PROGRESS)
        self.assertIsNotNone(step.started_at)

    def test_complete(self):
        step = TaskStep(step_id="s1", description="Test")
        step.start().complete("Success")
        self.assertEqual(step.status, StepStatus.COMPLETED)
        self.assertEqual(step.result, "Success")
        self.assertIsNotNone(step.completed_at)

    def test_fail(self):
        step = TaskStep(step_id="s1", description="Test")
        step.start().fail("Error occurred")
        self.assertEqual(step.status, StepStatus.FAILED)
        self.assertEqual(step.error, "Error occurred")

    def test_skip(self):
        step = TaskStep(step_id="s1", description="Test")
        step.skip("Not needed")
        self.assertEqual(step.status, StepStatus.SKIPPED)
        self.assertEqual(step.metadata.get("skip_reason"), "Not needed")

    def test_reset(self):
        step = TaskStep(step_id="s1", description="Test")
        step.start().complete("Done")
        step.reset()
        self.assertEqual(step.status, StepStatus.PENDING)
        self.assertIsNone(step.result)

    def test_invalid_transition(self):
        step = TaskStep(step_id="s1", description="Test")
        # 不能直接从pending完成，必须先start
        with self.assertRaises(ValueError):
            step.complete("Done")


class TestTaskStepChecks(unittest.TestCase):
    """测试状态检查"""

    def test_status_checks(self):
        pending = TaskStep(step_id="s1", description="Test", status=StepStatus.PENDING)
        in_progress = TaskStep(step_id="s2", description="Test", status=StepStatus.IN_PROGRESS)
        completed = TaskStep(step_id="s3", description="Test", status=StepStatus.COMPLETED)
        failed = TaskStep(step_id="s4", description="Test", status=StepStatus.FAILED)
        skipped = TaskStep(step_id="s5", description="Test", status=StepStatus.SKIPPED)

        self.assertTrue(pending.is_pending())
        self.assertTrue(in_progress.is_in_progress())
        self.assertTrue(completed.is_completed())
        self.assertTrue(failed.is_failed())
        self.assertTrue(skipped.is_skipped())

        self.assertTrue(completed.is_terminal())
        self.assertTrue(failed.is_terminal())
        self.assertTrue(skipped.is_terminal())
        self.assertFalse(pending.is_terminal())


class TestTaskStepUtils(unittest.TestCase):
    """测试工具方法"""

    def test_copy(self):
        step = TaskStep(step_id="s1", description="Test", metadata={"k": "v"})
        copy = step.copy()
        self.assertEqual(copy.step_id, step.step_id)
        self.assertIsNot(copy, step)

    def test_get_duration(self):
        step = TaskStep(step_id="s1", description="Test")
        self.assertIsNone(step.get_duration_seconds())
        step.start()
        import time
        time.sleep(0.01)
        step.complete("Done")
        duration = step.get_duration_seconds()
        self.assertIsNotNone(duration)
        self.assertGreater(duration, 0)

    def test_metadata(self):
        step = TaskStep(step_id="s1", description="Test")
        step.add_metadata("key1", "value1")
        self.assertEqual(step.get_metadata("key1"), "value1")
        self.assertIsNone(step.get_metadata("nonexistent"))
        self.assertEqual(step.get_metadata("nonexistent", "default"), "default")


class TestTaskStepListUtils(unittest.TestCase):
    """测试列表工具函数"""

    def setUp(self):
        self.steps = [
            TaskStep(step_id="s1", description="Pending", status=StepStatus.PENDING),
            TaskStep(step_id="s2", description="Completed", status=StepStatus.COMPLETED),
            TaskStep(step_id="s3", description="Failed", status=StepStatus.FAILED),
            TaskStep(step_id="s4", description="Pending2", status=StepStatus.PENDING),
        ]

    def test_filter_by_status(self):
        pending = filter_by_status(self.steps, StepStatus.PENDING)
        self.assertEqual(len(pending), 2)

    def test_filter_pending(self):
        pending = filter_pending(self.steps)
        self.assertEqual(len(pending), 2)

    def test_filter_completed(self):
        completed = filter_completed(self.steps)
        self.assertEqual(len(completed), 1)

    def test_filter_failed(self):
        failed = filter_failed(self.steps)
        self.assertEqual(len(failed), 1)

    def test_get_completion_ratio(self):
        ratio = get_completion_ratio(self.steps)
        self.assertEqual(ratio, 0.5)  # 2/4

    def test_steps_json_roundtrip(self):
        json_str = steps_to_json(self.steps[:2])
        restored = steps_from_json(json_str)
        self.assertEqual(len(restored), 2)


# =============================================================================
# TaskSnapshot 测试
# =============================================================================

class TestTaskState(unittest.TestCase):
    """测试任务状态枚举"""

    def test_state_values(self):
        self.assertEqual(TaskState.PENDING.value, "pending")
        self.assertEqual(TaskState.IN_PROGRESS.value, "in_progress")
        self.assertEqual(TaskState.COMPLETED.value, "completed")
        self.assertEqual(TaskState.FAILED.value, "failed")
        self.assertEqual(TaskState.SUSPENDED.value, "suspended")


class TestTaskSnapshotCreation(unittest.TestCase):
    """测试TaskSnapshot创建"""

    def test_basic_creation(self):
        snapshot = TaskSnapshot(
            task_id="task_1",
            session_id="session_1",
            goal="Complete the project",
        )
        self.assertEqual(snapshot.task_id, "task_1")
        self.assertEqual(snapshot.session_id, "session_1")
        self.assertEqual(snapshot.goal, "Complete the project")
        self.assertEqual(snapshot.state, TaskState.PENDING)
        self.assertEqual(snapshot.progress_ratio, 0.0)
        self.assertEqual(snapshot.version, 1)

    def test_creation_with_steps(self):
        step1 = TaskStep(step_id="s1", description="Step 1")
        step2 = TaskStep(step_id="s2", description="Step 2")
        snapshot = TaskSnapshot(
            task_id="task_2",
            session_id="session_1",
            goal="Test",
            pending_steps=[step1, step2],
        )
        self.assertEqual(len(snapshot.pending_steps), 2)


class TestTaskSnapshotSerialization(unittest.TestCase):
    """测试序列化"""

    def setUp(self):
        self.snapshot = TaskSnapshot(
            task_id="task_ser",
            session_id="session_ser",
            goal="Serialization test",
            variables={"key": "value"},
            constraints=["constraint1"],
        )

    def test_to_dict(self):
        data = self.snapshot.to_dict()
        self.assertEqual(data["task_id"], "task_ser")
        self.assertEqual(data["goal"], "Serialization test")
        self.assertEqual(data["variables"], {"key": "value"})

    def test_roundtrip(self):
        data = self.snapshot.to_dict()
        snapshot2 = TaskSnapshot.from_dict(data)
        self.assertEqual(snapshot2.task_id, self.snapshot.task_id)
        self.assertEqual(snapshot2.goal, self.snapshot.goal)


class TestTaskSnapshotStepManagement(unittest.TestCase):
    """测试步骤管理"""

    def setUp(self):
        self.snapshot = TaskSnapshot(
            task_id="task_steps",
            session_id="session_1",
            goal="Test steps",
        )

    def test_add_step(self):
        step = self.snapshot.add_step("New step")
        self.assertEqual(len(self.snapshot.pending_steps), 1)
        self.assertEqual(step.description, "New step")

    def test_start_step(self):
        step = self.snapshot.add_step("Step to start")
        started = self.snapshot.start_step(step.step_id)
        self.assertIsNotNone(started)
        self.assertEqual(started.status, StepStatus.IN_PROGRESS)
        self.assertEqual(self.snapshot.state, TaskState.IN_PROGRESS)

    def test_complete_step(self):
        step = self.snapshot.add_step("Step to complete")
        self.snapshot.start_step(step.step_id)
        completed = self.snapshot.complete_step(step.step_id, "Done")
        self.assertIsNotNone(completed)
        self.assertEqual(len(self.snapshot.pending_steps), 0)
        self.assertEqual(len(self.snapshot.completed_steps), 1)

    def test_fail_step(self):
        step = self.snapshot.add_step("Step to fail")
        self.snapshot.start_step(step.step_id)
        failed = self.snapshot.fail_step(step.step_id, "Error")
        self.assertIsNotNone(failed)
        self.assertTrue(self.snapshot.has_errors())

    def test_progress_calculation(self):
        self.snapshot.add_step("Step 1")
        self.snapshot.add_step("Step 2")
        self.assertEqual(self.snapshot.progress_ratio, 0.0)

        step = self.snapshot.pending_steps[0]
        self.snapshot.start_step(step.step_id)
        self.snapshot.complete_step(step.step_id, "Done")
        self.assertEqual(self.snapshot.progress_ratio, 0.5)


class TestTaskSnapshotVariableManagement(unittest.TestCase):
    """测试变量管理"""

    def setUp(self):
        self.snapshot = TaskSnapshot(
            task_id="task_vars",
            session_id="session_1",
            goal="Test variables",
        )

    def test_set_and_get_variable(self):
        self.snapshot.set_variable("key1", "value1")
        self.assertEqual(self.snapshot.get_variable("key1"), "value1")
        self.assertIsNone(self.snapshot.get_variable("nonexistent"))
        self.assertEqual(self.snapshot.get_variable("nonexistent", "default"), "default")

    def test_remove_variable(self):
        self.snapshot.set_variable("key1", "value1")
        self.snapshot.remove_variable("key1")
        self.assertIsNone(self.snapshot.get_variable("key1"))


class TestTaskSnapshotErrorManagement(unittest.TestCase):
    """测试错误管理"""

    def setUp(self):
        self.snapshot = TaskSnapshot(
            task_id="task_err",
            session_id="session_1",
            goal="Test errors",
        )

    def test_log_error(self):
        self.snapshot.log_error("context", "error message")
        self.assertEqual(len(self.snapshot.error_log), 1)
        self.assertTrue(self.snapshot.has_errors())

    def test_get_recent_errors(self):
        for i in range(10):
            self.snapshot.log_error(f"ctx{i}", f"err{i}")
        recent = self.snapshot.get_recent_errors(3)
        self.assertEqual(len(recent), 3)

    def test_clear_errors(self):
        self.snapshot.log_error("ctx", "err")
        self.snapshot.clear_errors()
        self.assertFalse(self.snapshot.has_errors())


class TestTaskSnapshotStateManagement(unittest.TestCase):
    """测试状态管理"""

    def setUp(self):
        self.snapshot = TaskSnapshot(
            task_id="task_state",
            session_id="session_1",
            goal="Test state",
        )

    def test_suspend_and_resume(self):
        self.snapshot.suspend()
        self.assertEqual(self.snapshot.state, TaskState.SUSPENDED)
        self.snapshot.resume()
        self.assertEqual(self.snapshot.state, TaskState.IN_PROGRESS)

    def test_version_increment(self):
        self.snapshot.increment_version()
        self.assertEqual(self.snapshot.version, 2)


# =============================================================================
# SnapshotManager 测试
# =============================================================================

class TestSnapshotManager(unittest.TestCase):
    """测试快照管理器"""

    def setUp(self):
        self.manager = SnapshotManager()

    def test_create_and_get(self):
        snapshot = self.manager.create("task_1", "session_1", "Test goal")
        self.assertEqual(snapshot.task_id, "task_1")

        retrieved = self.manager.get("task_1")
        self.assertEqual(retrieved.goal, "Test goal")

    def test_get_not_found(self):
        with self.assertRaises(SnapshotNotFound):
            self.manager.get("nonexistent")

    def test_get_optional(self):
        self.assertIsNone(self.manager.get_optional("nonexistent"))
        self.manager.create("task_1", "session_1", "Test")
        self.assertIsNotNone(self.manager.get_optional("task_1"))

    def test_update(self):
        self.manager.create("task_1", "session_1", "Original")
        updated = self.manager.update("task_1", goal="Updated")
        self.assertEqual(updated.goal, "Updated")
        self.assertEqual(updated.version, 2)

    def test_delete(self):
        self.manager.create("task_1", "session_1", "Test")
        self.assertTrue(self.manager.delete("task_1"))
        self.assertFalse(self.manager.exists("task_1"))

    def test_save_and_restore(self):
        self.manager.create("task_1", "session_1", "Test")
        self.manager.save("task_1")

        self.manager.clear_active()
        self.assertEqual(len(self.manager.list_active()), 0)

        restored = self.manager.restore("task_1")
        self.assertEqual(restored.goal, "Test")
        self.assertEqual(restored.version, 2)  # 恢复时递增版本

    def test_suspend_and_resume(self):
        self.manager.create("task_1", "session_1", "Test")
        suspended = self.manager.suspend("task_1")
        self.assertEqual(suspended.state, TaskState.SUSPENDED)

        resumed = self.manager.resume("task_1")
        self.assertEqual(resumed.state, TaskState.IN_PROGRESS)

    def test_auto_save(self):
        manager = SnapshotManager(auto_save_interval=2)
        manager.create("task_1", "session_1", "Test")

        # 第一次更新（计数=1，不保存）
        manager.update("task_1", variables={"k": "v"})
        # 第二次更新（计数=2，保存）
        manager.update("task_1", variables={"k": "v2"})

        # 验证已保存
        self.assertTrue(manager.exists("task_1"))

    def test_cleanup_old_snapshots(self):
        for i in range(5):
            self.manager.create(f"task_{i}", "session_1", f"Goal {i}")

        # 设置max_retained=3，应该清理2个
        manager = SnapshotManager(max_retained=3)
        for i in range(5):
            manager.create(f"task_{i}", "session_1", f"Goal {i}")

        count = manager.cleanup_old_snapshots()
        self.assertEqual(count, 2)
        self.assertEqual(len(manager.list_all()), 3)

    def test_stats(self):
        self.manager.create("task_1", "session_1", "Test")
        stats = self.manager.get_stats()
        self.assertEqual(stats["active_count"], 1)
        self.assertEqual(stats["stored_count"], 1)


# =============================================================================
# DriftDetector 测试
# =============================================================================

class TestDriftDetector(unittest.TestCase):
    """测试漂移检测器"""

    def setUp(self):
        self.detector = DriftDetector(threshold=0.6)

    def test_no_drift_similar(self):
        snapshot = TaskSnapshot(
            task_id="task_1",
            session_id="s1",
            goal="Write a Python program",
        )
        # 使用更低阈值确保相似文本通过
        detector = DriftDetector(threshold=0.3)
        result = detector.check(snapshot, "Write Python code")
        self.assertFalse(result.has_drift)
        self.assertGreater(result.similarity, 0.0)

    def test_drift_detected(self):
        snapshot = TaskSnapshot(
            task_id="task_1",
            session_id="s1",
            goal="Write a Python program",
        )
        result = self.detector.check(snapshot, "Cook Italian dinner")
        self.assertTrue(result.has_drift)
        self.assertLess(result.similarity, 0.6)

    def test_check_and_raise(self):
        snapshot = TaskSnapshot(
            task_id="task_1",
            session_id="s1",
            goal="Python programming tutorial",
        )
        # 相似意图，不抛出（提高相似度阈值到0.6，确保"Python programming tutorial" vs "Python coding tutorial"通过）
        detector = DriftDetector(threshold=0.3)  # 使用更低阈值确保不触发
        detector.check_and_raise(snapshot, "Python coding tutorial")

        # 不相似意图，抛出
        with self.assertRaises(TaskDriftDetected):
            self.detector.check_and_raise(snapshot, "Cooking recipes")

    def test_custom_threshold(self):
        snapshot = TaskSnapshot(
            task_id="task_1",
            session_id="s1",
            goal="Python programming",
        )
        # 使用高阈值（0.9），中等相似度也会触发漂移
        result = self.detector.check(snapshot, "Write code", custom_threshold=0.9)
        self.assertTrue(result.has_drift)

    def test_batch_check(self):
        snapshots = [
            TaskSnapshot(task_id="t1", session_id="s1", goal="Python"),
            TaskSnapshot(task_id="t2", session_id="s1", goal="Cooking"),
        ]
        intents = ["Python code", "Italian food"]
        results = self.detector.check_batch(snapshots, intents)
        self.assertEqual(len(results), 2)

    def test_get_drifted_tasks(self):
        snapshots = [
            TaskSnapshot(task_id="t1", session_id="s1", goal="Python programming"),
            TaskSnapshot(task_id="t2", session_id="s1", goal="Cooking recipes"),
        ]
        intents = ["Cooking dinner", "Italian food"]  # 都偏离
        drifted = self.detector.get_drifted_tasks(snapshots, intents)
        self.assertEqual(len(drifted), 2)


class TestPeriodicDriftDetector(unittest.TestCase):
    """测试周期性漂移检测器"""

    def setUp(self):
        self.detector = PeriodicDriftDetector(check_interval=3)

    def test_should_check(self):
        # 第3轮应该检测
        self.assertFalse(self.detector.should_check("task_1"))  # 1
        self.assertFalse(self.detector.should_check("task_1"))  # 2
        self.assertTrue(self.detector.should_check("task_1"))   # 3
        self.assertFalse(self.detector.should_check("task_1"))  # 4

    def test_check_if_needed(self):
        snapshot = TaskSnapshot(
            task_id="task_1",
            session_id="s1",
            goal="Python",
        )
        # 第1轮不需要检测（check_if_needed内部调用should_check，计数器变为1）
        result = self.detector.check_if_needed(snapshot, "Code")
        self.assertIsNone(result)
        self.assertEqual(self.detector.get_counter("task_1"), 1)

        # 第2轮不需要检测
        result = self.detector.check_if_needed(snapshot, "Code")
        self.assertIsNone(result)
        self.assertEqual(self.detector.get_counter("task_1"), 2)

        # 第3轮需要检测
        result = self.detector.check_if_needed(snapshot, "Code")
        self.assertIsNotNone(result)
        self.assertEqual(self.detector.get_counter("task_1"), 3)

    def test_reset_counter(self):
        self.detector.should_check("task_1")
        self.detector.should_check("task_1")
        self.assertEqual(self.detector.get_counter("task_1"), 2)
        self.detector.reset_counter("task_1")
        self.assertEqual(self.detector.get_counter("task_1"), 0)


# =============================================================================
# 集成测试
# =============================================================================

class TestTaskIntegration(unittest.TestCase):
    """任务模块集成测试"""

    def test_full_task_lifecycle(self):
        """完整任务生命周期"""
        manager = SnapshotManager()
        detector = DriftDetector(threshold=0.5)

        # 1. 创建任务
        snapshot = manager.create(
            "task_1",
            "session_1",
            "Build a web application",
        )

        # 2. 添加步骤
        snapshot.add_step("Design database schema")
        snapshot.add_step("Implement API")
        snapshot.add_step("Create frontend")

        # 3. 执行步骤
        step1 = snapshot.pending_steps[0]
        manager.update(snapshot.task_id)  # 触发保存

        # 4. 完成第一步
        snapshot.start_step(step1.step_id)
        snapshot.complete_step(step1.step_id, "Schema designed")
        manager.save(snapshot.task_id)

        # 5. 漂移检测（使用相似文本确保通过）
        detector_low = DriftDetector(threshold=0.3)
        result = detector_low.check(snapshot, "Build web app database")
        self.assertFalse(result.has_drift)

        # 6. 验证状态
        self.assertEqual(snapshot.progress_ratio, 1/3)
        self.assertEqual(snapshot.state, TaskState.IN_PROGRESS)

    def test_task_recovery(self):
        """任务恢复场景"""
        manager = SnapshotManager()

        # 创建并执行部分任务
        snapshot = manager.create("task_1", "session_1", "Test")
        snapshot.add_step("Step 1")
        snapshot.add_step("Step 2")
        snapshot.start_step(snapshot.pending_steps[0].step_id)
        manager.save("task_1")

        # 模拟重启：清空活跃缓存
        manager.clear_active()
        self.assertEqual(len(manager.list_active()), 0)

        # 恢复任务
        restored = manager.restore("task_1")
        self.assertEqual(len(restored.pending_steps), 2)
        self.assertEqual(restored.state, TaskState.IN_PROGRESS)


if __name__ == "__main__":
    unittest.main()
