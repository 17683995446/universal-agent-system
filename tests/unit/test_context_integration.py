"""
上下文装配集成层单元测试

测试范围：
- ContextBuilder 上下文构建器
- TurnManager 上下文装配集成
- Agent 上下文构建器集成

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-26
"""

import unittest
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, '/workspace')

from src.agent.context_builder import ContextBuilder
from src.agent.turn_manager import TurnManager
from src.agent.agent import Agent
from src.agent.agent_config import AgentConfig
from src.agent.agent_types import TurnStatus
from src.modules.context_manager.partitioner import PartitionStrategy


# =============================================================================
# ContextBuilder 测试
# =============================================================================

class TestContextBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = ContextBuilder(window_size=1000)

    def test_creation(self):
        """测试创建"""
        self.assertEqual(self.builder._window_size, 1000)
        self.assertIsNotNone(self.builder._assembler)
        self.assertIsNotNone(self.builder._partitioner)

    def test_build_basic(self):
        """测试基本构建"""
        result = self.builder.build(
            system_prompt="Be helpful",
            history=[],
            user_input="Hello",
        )

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.assembly)

    def test_build_with_history(self):
        """测试带历史构建"""
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]

        result = self.builder.build(
            system_prompt="Be helpful",
            history=history,
            user_input="How are you?",
        )

        self.assertIsNotNone(result)

    def test_build_with_memory_context(self):
        """测试带记忆上下文构建"""
        result = self.builder.build(
            system_prompt="Be helpful",
            history=[],
            user_input="Hello",
            memory_context="[Relevant Memories]\n1. Previous topic",
        )

        self.assertIsNotNone(result)

    def test_build_with_different_strategies(self):
        """测试不同策略构建"""
        strategies = [
            PartitionStrategy.DEFAULT,
            PartitionStrategy.SYSTEM_HEAVY,
            PartitionStrategy.SESSION_PRIORITY,
        ]

        for strategy in strategies:
            result = self.builder.build(
                system_prompt="Be helpful",
                history=[],
                user_input="Hello",
                strategy=strategy,
            )
            self.assertIsNotNone(result)

    def test_build_with_adaptive_budget(self):
        """测试自适应预算构建"""
        result = self.builder.build_with_adaptive_budget(
            system_prompt="Be helpful",
            history=[],
            user_input="Hello",
        )

        self.assertIsNotNone(result)

    def test_get_stats(self):
        """测试获取统计"""
        self.builder.build("System", [], "User")
        stats = self.builder.get_stats()

        self.assertEqual(stats["total_builds"], 1)
        self.assertIn("avg_tokens_per_build", stats)
        self.assertIn("truncation_rate", stats)

    def test_reset_stats(self):
        """测试重置统计"""
        self.builder.build("System", [], "User")
        self.builder.reset_stats()
        stats = self.builder.get_stats()

        self.assertEqual(stats["total_builds"], 0)

    def test_set_window_size(self):
        """测试设置窗口大小"""
        self.builder.set_window_size(2000)
        self.assertEqual(self.builder._window_size, 2000)

    def test_set_strategy(self):
        """测试设置策略"""
        self.builder.set_strategy(PartitionStrategy.SESSION_PRIORITY)
        self.assertEqual(self.builder._strategy, PartitionStrategy.SESSION_PRIORITY)

    def test_get_available_strategies(self):
        """测试获取可用策略"""
        strategies = self.builder.get_available_strategies()
        self.assertIn(PartitionStrategy.DEFAULT, strategies)


# =============================================================================
# TurnManager 上下文集成测试
# =============================================================================

class TestTurnManagerContextIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_llm_client = MagicMock()
        self.mock_context_builder = MagicMock()

    def test_build_messages_with_context_builder(self):
        """测试使用ContextBuilder构建消息"""
        # 模拟AssemblyResult
        mock_assembly = MagicMock()
        mock_assembly.to_openai_format.return_value = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]

        mock_result = MagicMock()
        mock_result.assembly = mock_assembly
        mock_result.truncated = False

        self.mock_context_builder.build.return_value = mock_result

        tm = TurnManager(
            llm_client=self.mock_llm_client,
            context_builder=self.mock_context_builder,
        )

        messages = tm._build_messages(
            system_prompt="Be helpful",
            history=[],
            user_input="Hello",
        )

        # 验证ContextBuilder被调用
        self.mock_context_builder.build.assert_called_once()
        self.assertEqual(len(messages), 2)

    def test_build_messages_without_context_builder(self):
        """测试无ContextBuilder时简单构建"""
        tm = TurnManager(llm_client=self.mock_llm_client)

        messages = tm._build_messages(
            system_prompt="Be helpful",
            history=[],
            user_input="Hello",
        )

        # 简单构建：system + user
        self.assertEqual(len(messages), 2)

    def test_build_messages_truncation_logged(self):
        """测试截断信息记录"""
        mock_assembly = MagicMock()
        mock_assembly.to_openai_format.return_value = [{"role": "user", "content": "Hi"}]

        mock_result = MagicMock()
        mock_result.assembly = mock_assembly
        mock_result.truncated = True
        mock_result.messages_removed = 5
        mock_result.tokens_saved = 100

        self.mock_context_builder.build.return_value = mock_result

        tm = TurnManager(
            llm_client=self.mock_llm_client,
            context_builder=self.mock_context_builder,
        )

        # 构建消息（截断信息会被记录到日志）
        messages = tm._build_messages("System", [], "User")
        self.assertIsNotNone(messages)

    def test_process_turn_with_context_builder(self):
        """测试带ContextBuilder的完整对话处理"""
        mock_assembly = MagicMock()
        mock_assembly.to_openai_format.return_value = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]

        mock_result = MagicMock()
        mock_result.assembly = mock_assembly
        mock_result.truncated = False

        self.mock_context_builder.build.return_value = mock_result

        mock_response = MagicMock()
        mock_response.is_tool_call = False
        mock_response.content = "Hello!"
        mock_response.total_tokens = 20

        self.mock_llm_client.chat.return_value = mock_response

        tm = TurnManager(
            llm_client=self.mock_llm_client,
            context_builder=self.mock_context_builder,
        )

        result = tm.process_turn(
            turn_id="t1",
            user_input="Hello",
            system_prompt="Be helpful",
            history=[],
        )

        self.assertEqual(result.status, TurnStatus.COMPLETED)


# =============================================================================
# Agent 上下文集成测试
# =============================================================================

class TestAgentContextIntegration(unittest.TestCase):
    def setUp(self):
        self.config = AgentConfig(
            agent_id="context_test",
            name="Context Test",
        )

    def test_context_builder_initialized(self):
        """测试上下文构建器自动初始化"""
        agent = Agent(config=self.config)
        self.assertIsNotNone(agent.context_builder)

    def test_context_builder_window_size(self):
        """测试窗口大小从LLM配置获取"""
        config = AgentConfig()
        config.llm_config.max_tokens = 2000

        agent = Agent(config=config)
        self.assertEqual(agent.context_builder._window_size, 2000)

    def test_context_builder_default_window(self):
        """测试默认窗口大小"""
        config = AgentConfig()
        config.llm_config.max_tokens = None

        agent = Agent(config=config)
        self.assertEqual(agent.context_builder._window_size, 4096)

    def test_chat_with_context_builder(self):
        """测试带上下文构建器的对话"""
        agent = Agent(config=self.config)

        mock_response = MagicMock()
        mock_response.is_tool_call = False
        mock_response.content = "Hello!"
        mock_response.total_tokens = 20

        agent._llm_client.chat = MagicMock(return_value=mock_response)
        result = agent.chat("Hello")

        self.assertEqual(result.status, TurnStatus.COMPLETED)

        # 验证上下文构建器统计更新
        stats = agent.context_builder.get_stats()
        self.assertEqual(stats["total_builds"], 1)


# =============================================================================
# 端到端集成测试
# =============================================================================

class TestContextEndToEnd(unittest.TestCase):
    def test_full_conversation_with_context_management(self):
        """测试完整对话流程中的上下文管理"""
        config = AgentConfig(
            agent_id="e2e_context",
            system_prompt="Help with conversation",  # 更相关的任务目标
        )
        agent = Agent(config=config)

        # 多轮对话
        mock_response = MagicMock()
        mock_response.is_tool_call = False
        mock_response.content = "OK"
        mock_response.total_tokens = 10

        agent._llm_client.chat = MagicMock(return_value=mock_response)

        for i in range(5):
            result = agent.chat(f"Help me with conversation message {i}")
            self.assertEqual(result.status, TurnStatus.COMPLETED)

        # 验证上下文构建器统计
        stats = agent.context_builder.get_stats()
        self.assertEqual(stats["total_builds"], 5)

    def test_context_truncation_with_long_history(self):
        """测试长历史时的上下文截断"""
        builder = ContextBuilder(window_size=100)  # 很小的窗口

        # 构建长历史
        history = []
        for i in range(20):
            history.append({"role": "user", "content": f"User message {i} with some content"})
            history.append({"role": "assistant", "content": f"Assistant reply {i} with response"})

        result = builder.build(
            system_prompt="Be helpful",
            history=history,
            user_input="Current message",
        )

        # 应该发生截断
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()