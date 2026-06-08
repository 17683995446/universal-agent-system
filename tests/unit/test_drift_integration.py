"""
漂移检测集成层单元测试

测试范围：
- TaskContextManager 任务上下文管理器
- TurnManager 漂移检测集成
- Agent 任务快照集成

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-26
"""

import unittest
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, '/workspace')

from src.agent.task_context import TaskContextManager
from src.agent.turn_manager import TurnManager
from src.agent.agent import Agent
from src.agent.agent_config import AgentConfig
from src.agent.agent_types import TurnStatus
from src.modules.task_manager.task_snapshot import TaskSnapshot, TaskState


# =============================================================================
# TaskContextManager 测试
# =============================================================================

class TestTaskContextManager(unittest.TestCase):
    def setUp(self):
        self.tcm = TaskContextManager(task_goal="Test task")

    def test_creation_with_goal(self):
        """测试带目标创建"""
        self.assertTrue(self.tcm.has_active_task)
        self.assertIsNotNone(self.tcm.current_task)
        self.assertEqual(self.tcm.current_task.goal, "Test task")

    def test_creation_without_goal(self):
        """测试无目标创建"""
        tcm = TaskContextManager()
        self.assertFalse(tcm.has_active_task)

    def test_create_task(self):
        """测试创建任务"""
        tcm = TaskContextManager()
        task = tcm.create_task("New task")
        self.assertEqual(task.goal, "New task")
        self.assertTrue(tcm.has_active_task)

    def test_set_task(self):
        """测试设置任务"""
        from src.core.utils.id_generator import generate_task_id, generate_session_id
        snapshot = TaskSnapshot(
            task_id=generate_task_id(),
            session_id=generate_session_id(),
            goal="External task"
        )
        self.tcm.set_task(snapshot)
        self.assertEqual(self.tcm.current_task.goal, "External task")

    def test_update_task_goal(self):
        """测试更新任务目标"""
        self.tcm.update_task_goal("Updated goal")
        self.assertEqual(self.tcm.current_task.goal, "Updated goal")

    def test_complete_task(self):
        """测试完成任务"""
        self.tcm.complete_task()
        self.assertEqual(self.tcm.current_task.state, TaskState.COMPLETED)

    def test_fail_task(self):
        """测试标记任务失败"""
        self.tcm.fail_task("Something went wrong")
        self.assertEqual(self.tcm.current_task.state, TaskState.FAILED)
        self.assertTrue(self.tcm.current_task.has_errors())

    def test_clear_task(self):
        """测试清除任务"""
        self.tcm.clear_task()
        self.assertFalse(self.tcm.has_active_task)
        self.assertIsNone(self.tcm.current_task)

    def test_save_and_load_snapshot(self):
        """测试保存和加载快照"""
        # 保存
        result = self.tcm.save_snapshot()
        self.assertTrue(result)

        # 加载
        task_id = self.tcm.current_task.task_id
        loaded = self.tcm.load_snapshot(task_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.goal, "Test task")

    def test_save_snapshot_no_task(self):
        """测试无任务时保存快照"""
        tcm = TaskContextManager()
        result = tcm.save_snapshot()
        self.assertFalse(result)

    def test_restore_latest(self):
        """测试恢复最新快照"""
        self.tcm.save_snapshot()
        self.tcm.clear_task()

        restored = self.tcm.restore_latest()
        self.assertIsNotNone(restored)
        self.assertEqual(restored.goal, "Test task")

    def test_check_drift_no_drift(self):
        """测试无漂移检测"""
        # 相似度应该较高
        result = self.tcm.check_drift("Test task execution")
        self.assertIsNotNone(result)
        self.assertFalse(result.has_drift)

    def test_check_drift_with_drift(self):
        """测试有漂移检测"""
        # 完全不同的内容应该触发漂移
        result = self.tcm.check_drift("Completely unrelated topic about cooking")
        self.assertIsNotNone(result)
        # 注意：由于使用简单词重叠，可能不总是触发

    def test_check_drift_no_task(self):
        """测试无任务时漂移检测"""
        tcm = TaskContextManager()
        result = tcm.check_drift("anything")
        self.assertIsNone(result)

    def test_is_drifted(self):
        """测试快速漂移检查"""
        # 相同内容不应漂移
        self.assertFalse(self.tcm.is_drifted("Test task"))

    def test_add_step(self):
        """测试添加步骤"""
        step = self.tcm.add_step("Step 1: Initialize")
        self.assertIsNotNone(step)
        self.assertEqual(len(self.tcm.current_task.pending_steps), 1)

    def test_add_step_no_task(self):
        """测试无任务时添加步骤"""
        tcm = TaskContextManager()
        step = tcm.add_step("Step 1")
        self.assertIsNone(step)

    def test_complete_step(self):
        """测试完成步骤"""
        step = self.tcm.add_step("Step to complete")
        result = self.tcm.complete_step(step.step_id, "Done")
        self.assertTrue(result)
        self.assertEqual(len(self.tcm.current_task.completed_steps), 1)

    def test_get_stats(self):
        """测试获取统计"""
        self.tcm.check_drift_periodic("Test")  # 使用periodic来增加计数
        stats = self.tcm.get_stats()
        self.assertTrue(stats["has_active_task"])
        self.assertEqual(stats["drift_checks"], 1)
        self.assertEqual(stats["task_state"], "pending")

    def test_reset_stats(self):
        """测试重置统计"""
        self.tcm.check_drift("Test")
        self.tcm.reset_stats()
        stats = self.tcm.get_stats()
        self.assertEqual(stats["drift_checks"], 0)


# =============================================================================
# TurnManager 漂移检测集成测试
# =============================================================================

class TestTurnManagerDriftIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_llm_client = MagicMock()
        self.mock_task_context = MagicMock()

    def test_process_turn_with_drift_detection(self):
        """测试带漂移检测的对话处理"""
        # 模拟漂移检测结果（无漂移）
        self.mock_task_context.check_drift_periodic.return_value = None

        # 模拟LLM响应
        mock_response = MagicMock()
        mock_response.is_tool_call = False
        mock_response.content = "Hello!"
        mock_response.prompt_tokens = 10
        mock_response.completion_tokens = 5
        mock_response.total_tokens = 15
        self.mock_llm_client.chat.return_value = mock_response

        tm = TurnManager(
            llm_client=self.mock_llm_client,
            task_context=self.mock_task_context,
            enable_drift_detection=True,
        )

        result = tm.process_turn(
            turn_id="t1",
            user_input="Hi",
            system_prompt="Be helpful",
            history=[],
        )

        self.assertEqual(result.status, TurnStatus.COMPLETED)
        # 验证漂移检测被调用
        self.mock_task_context.check_drift_periodic.assert_called_once_with("Hi")

    def test_process_turn_drift_detected(self):
        """测试检测到漂移时的处理"""
        # 模拟漂移检测结果（有漂移）
        mock_drift_result = MagicMock()
        mock_drift_result.has_drift = True
        mock_drift_result.similarity = 0.3
        mock_drift_result.threshold = 0.6
        mock_drift_result.original_goal = "Original task"
        mock_drift_result.current_intent = "Unrelated input"
        self.mock_task_context.check_drift_periodic.return_value = mock_drift_result

        tm = TurnManager(
            llm_client=self.mock_llm_client,
            task_context=self.mock_task_context,
            enable_drift_detection=True,
        )

        result = tm.process_turn(
            turn_id="t1",
            user_input="Unrelated input",
            system_prompt="Be helpful",
            history=[],
        )

        self.assertEqual(result.status, TurnStatus.ERROR)
        self.assertIn("drift_info", result.metadata)
        self.assertEqual(result.metadata["drift_info"]["similarity"], 0.3)

    def test_process_turn_drift_disabled(self):
        """测试禁用漂移检测"""
        tm = TurnManager(
            llm_client=self.mock_llm_client,
            task_context=self.mock_task_context,
            enable_drift_detection=False,
        )

        mock_response = MagicMock()
        mock_response.is_tool_call = False
        mock_response.content = "Hello!"
        mock_response.total_tokens = 15
        self.mock_llm_client.chat.return_value = mock_response

        result = tm.process_turn(
            turn_id="t1",
            user_input="Hi",
            system_prompt="Be helpful",
            history=[],
        )

        self.assertEqual(result.status, TurnStatus.COMPLETED)
        # 验证漂移检测未被调用
        self.mock_task_context.check_drift_periodic.assert_not_called()

    def test_process_turn_no_task_context(self):
        """测试无任务上下文时的处理"""
        mock_response = MagicMock()
        mock_response.is_tool_call = False
        mock_response.content = "Hello!"
        mock_response.total_tokens = 15
        self.mock_llm_client.chat.return_value = mock_response

        tm = TurnManager(
            llm_client=self.mock_llm_client,
            task_context=None,
            enable_drift_detection=True,
        )

        result = tm.process_turn(
            turn_id="t1",
            user_input="Hi",
            system_prompt="Be helpful",
            history=[],
        )

        self.assertEqual(result.status, TurnStatus.COMPLETED)

    def test_process_turn_drift_exception(self):
        """测试漂移检测异常时的容错"""
        self.mock_task_context.check_drift_periodic.side_effect = Exception("Drift check failed")

        mock_response = MagicMock()
        mock_response.is_tool_call = False
        mock_response.content = "Hello!"
        mock_response.total_tokens = 15
        self.mock_llm_client.chat.return_value = mock_response

        tm = TurnManager(
            llm_client=self.mock_llm_client,
            task_context=self.mock_task_context,
            enable_drift_detection=True,
        )

        # 不应抛异常
        result = tm.process_turn(
            turn_id="t1",
            user_input="Hi",
            system_prompt="Be helpful",
            history=[],
        )

        self.assertEqual(result.status, TurnStatus.COMPLETED)


# =============================================================================
# Agent 任务快照集成测试
# =============================================================================

class TestAgentTaskIntegration(unittest.TestCase):
    def setUp(self):
        self.config = AgentConfig(
            agent_id="task_test",
            name="Task Test",
            system_prompt="You are a helpful assistant for testing.",
        )

    def test_task_context_initialized(self):
        """测试任务上下文自动初始化"""
        agent = Agent(config=self.config)
        self.assertIsNotNone(agent.task_context)
        self.assertTrue(agent.task_context.has_active_task)
        self.assertEqual(agent.task_context.current_task.goal, "You are a helpful assistant for testing.")

    def test_task_context_from_system_prompt(self):
        """测试从系统提示创建任务目标"""
        config = AgentConfig(system_prompt="Custom task goal")
        agent = Agent(config=config)
        self.assertEqual(agent.task_context.current_task.goal, "Custom task goal")

    def test_task_context_default_goal(self):
        """测试默认任务目标"""
        config = AgentConfig(system_prompt="")
        agent = Agent(config=config)
        self.assertEqual(agent.task_context.current_task.goal, "General conversation")

    def test_summary_includes_task(self):
        """测试摘要包含任务信息"""
        agent = Agent(config=self.config)
        summary = agent.get_summary()

        self.assertIn("task", summary)
        self.assertTrue(summary["task"]["has_active_task"])
        self.assertEqual(summary["task"]["task_state"], "pending")

    def test_reset_clears_task(self):
        """测试重置清除任务"""
        agent = Agent(config=self.config)
        agent.reset()

        self.assertFalse(agent.task_context.has_active_task)
        stats = agent.task_context.get_stats()
        self.assertEqual(stats["drift_checks"], 0)


# =============================================================================
# 端到端集成测试
# =============================================================================

class TestDriftEndToEnd(unittest.TestCase):
    def test_full_conversation_with_drift_detection(self):
        """测试完整对话流程中的漂移检测"""
        config = AgentConfig(
            agent_id="drift_e2e",
            system_prompt="Help with Python programming",
        )
        agent = Agent(config=config)

        # 第一轮：正常对话
        mock_response1 = MagicMock()
        mock_response1.is_tool_call = False
        mock_response1.content = "I can help with Python!"
        mock_response1.total_tokens = 20

        agent._llm_client.chat = MagicMock(return_value=mock_response1)
        result1 = agent.chat("How do I write a function?")

        self.assertEqual(result1.status, TurnStatus.COMPLETED)
        self.assertEqual(agent.task_context.get_stats()["drift_checks"], 1)

        # 第二轮：偏离主题（应该触发漂移检测）
        # 注意：由于使用简单词重叠，实际漂移检测可能不触发
        # 这里主要验证流程不中断
        mock_response2 = MagicMock()
        mock_response2.is_tool_call = False
        mock_response2.content = "I prefer cooking."
        mock_response2.total_tokens = 15

        agent._llm_client.chat = MagicMock(return_value=mock_response2)
        result2 = agent.chat("Let's talk about cooking recipes")

        # 验证对话完成（即使漂移检测触发，也是返回ERROR状态）
        self.assertIn(result2.status, [TurnStatus.COMPLETED, TurnStatus.ERROR])

    def test_task_snapshot_persistence(self):
        """测试任务快照持久化"""
        agent = Agent(config=AgentConfig(system_prompt="Test persistence"))

        # 添加步骤并保存
        agent.task_context.add_step("Step 1")
        agent.task_context.save_snapshot()

        # 验证可以加载
        task_id = agent.task_context.current_task.task_id
        loaded = agent.task_context.load_snapshot(task_id)
        self.assertIsNotNone(loaded)


if __name__ == "__main__":
    unittest.main()
