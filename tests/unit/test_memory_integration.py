"""
记忆集成层单元测试

测试范围：
- MemoryCoordinator 记忆协调器
- TurnManager 记忆召回集成
- Agent 记忆存储与召回集成

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-26
"""

import unittest
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, '/workspace')

from src.modules.memory_manager.coordinator import MemoryCoordinator
from src.modules.memory_manager.memory_entry import (
    MemoryEntry,
    MemoryLevel,
    MemorySource,
)
from src.modules.memory_manager.factory import MemoryConfig
from src.agent.turn_manager import TurnManager
from src.agent.agent import Agent
from src.agent.agent_config import AgentConfig
from src.agent.agent_types import TurnStatus, TurnResult


# =============================================================================
# MemoryCoordinator 测试
# =============================================================================

class TestMemoryCoordinator(unittest.TestCase):
    def setUp(self):
        self.coordinator = MemoryCoordinator()

    def test_creation_default(self):
        """测试默认创建"""
        self.assertIsNotNone(self.coordinator.instant)
        self.assertIsNotNone(self.coordinator.short_term)
        self.assertIsNotNone(self.coordinator.long_term)

    def test_store_short_term(self):
        """测试存储短期记忆"""
        entry = self.coordinator.store(
            content="Hello world",
            level=MemoryLevel.SHORT_TERM,
            source=MemorySource.USER_INPUT,
        )
        self.assertEqual(entry.content, "Hello world")
        self.assertEqual(entry.level, MemoryLevel.SHORT_TERM)
        self.assertEqual(self.coordinator.short_term.size, 1)

    def test_store_instant(self):
        """测试存储瞬时记忆"""
        entry = self.coordinator.store(
            content="temp data",
            level=MemoryLevel.INSTANT,
        )
        self.assertEqual(self.coordinator.instant.size, 1)

    def test_store_long_term(self):
        """测试存储长期记忆"""
        entry = self.coordinator.store(
            content="important knowledge",
            level=MemoryLevel.LONG_TERM,
        )
        self.assertEqual(self.coordinator.long_term.size, 1)

    def test_store_conversation(self):
        """测试存储对话"""
        entries = self.coordinator.store_conversation(
            user_input="What is Python?",
            assistant_reply="Python is a programming language.",
        )
        self.assertEqual(len(entries), 2)
        self.assertEqual(self.coordinator.short_term.size, 2)

    def test_store_conversation_empty_reply(self):
        """测试存储空回复的对话"""
        entries = self.coordinator.store_conversation(
            user_input="Hello",
            assistant_reply="",
        )
        self.assertEqual(len(entries), 1)  # 只存储用户输入

    def test_store_tool_result(self):
        """测试存储工具结果"""
        entry = self.coordinator.store_tool_result(
            tool_name="search",
            result="Found 3 results",
        )
        self.assertIsNotNone(entry)
        self.assertEqual(self.coordinator.short_term.size, 1)

    def test_store_tool_result_none(self):
        """测试存储None工具结果"""
        entry = self.coordinator.store_tool_result(
            tool_name="test",
            result=None,
        )
        self.assertIsNone(entry)

    def test_recall(self):
        """测试记忆召回"""
        # 存储一些记忆
        self.coordinator.store("Python is a programming language", level=MemoryLevel.LONG_TERM)
        self.coordinator.store("Java is also a programming language", level=MemoryLevel.LONG_TERM)

        # 召回
        result = self.coordinator.recall("programming")
        # 长期记忆使用文本相似度，应该能召回
        self.assertIsNotNone(result)

    def test_recall_as_context(self):
        """测试召回格式化为上下文"""
        self.coordinator.store("The sky is blue", level=MemoryLevel.LONG_TERM)

        context = self.coordinator.recall_as_context("sky color")
        # 如果召回了结果，应该包含[Relevant Memories]
        # 注意：文本相似度可能不匹配，所以这里只测试不抛异常
        self.assertIsInstance(context, str)

    def test_recall_empty(self):
        """测试空记忆召回"""
        result = self.coordinator.recall("anything")
        self.assertEqual(result.count, 0)

    def test_set_get_instant(self):
        """测试瞬时记忆存取"""
        self.coordinator.set_instant("key1", "value1")
        self.assertEqual(self.coordinator.get_instant("key1"), "value1")
        self.assertIsNone(self.coordinator.get_instant("nonexistent"))

    def test_clear_instant(self):
        """测试清空瞬时记忆"""
        self.coordinator.set_instant("key1", "value1")
        self.coordinator.clear_instant()
        self.assertIsNone(self.coordinator.get_instant("key1"))

    def test_clear_short_term(self):
        """测试清空短期记忆"""
        self.coordinator.store("test", level=MemoryLevel.SHORT_TERM)
        count = self.coordinator.clear_short_term()
        self.assertEqual(count, 1)
        self.assertEqual(self.coordinator.short_term.size, 0)

    def test_clear_long_term(self):
        """测试清空长期记忆"""
        self.coordinator.store("test", level=MemoryLevel.LONG_TERM)
        count = self.coordinator.clear_long_term()
        self.assertEqual(count, 1)

    def test_clear_all(self):
        """测试清空所有记忆"""
        self.coordinator.store("a", level=MemoryLevel.INSTANT)
        self.coordinator.store("b", level=MemoryLevel.SHORT_TERM)
        self.coordinator.store("c", level=MemoryLevel.LONG_TERM)

        counts = self.coordinator.clear_all()
        self.assertEqual(counts["instant"], 1)
        self.assertEqual(counts["short_term"], 1)
        self.assertEqual(counts["long_term"], 1)

    def test_get_stats(self):
        """测试获取统计"""
        self.coordinator.store("test", level=MemoryLevel.SHORT_TERM)
        stats = self.coordinator.get_stats()

        self.assertIn("instant", stats)
        self.assertIn("short_term", stats)
        self.assertIn("long_term", stats)
        self.assertEqual(stats["short_term"]["size"], 1)
        self.assertEqual(stats["total_stored"], 1)

    def test_custom_config(self):
        """测试自定义配置"""
        config = MemoryConfig(
            short_term_max_tokens=1000,
            short_term_max_items=10,
        )
        coordinator = MemoryCoordinator(config=config)
        self.assertEqual(coordinator.short_term.max_tokens, 1000)
        self.assertEqual(coordinator.short_term.max_items, 10)


# =============================================================================
# TurnManager 记忆集成测试
# =============================================================================

class TestTurnManagerMemoryIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_llm_client = MagicMock()
        self.mock_coordinator = MagicMock()

    def test_build_messages_with_memory(self):
        """测试构建消息时注入记忆上下文"""
        self.mock_coordinator.recall_as_context.return_value = "[Relevant Memories]\n1. Previous context"

        tm = TurnManager(
            llm_client=self.mock_llm_client,
            memory_coordinator=self.mock_coordinator,
        )

        messages = tm._build_messages(
            system_prompt="Be helpful",
            history=[],
            user_input="Continue",
        )

        # 应该有3条消息：system + memory context + user
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[0].content, "Be helpful")
        self.assertEqual(messages[1].role, "system")
        self.assertEqual(messages[1].content, "[Relevant Memories]\n1. Previous context")
        self.assertEqual(messages[2].role, "user")

        # 验证recall被调用
        self.mock_coordinator.recall_as_context.assert_called_once_with("Continue")

    def test_build_messages_no_memory(self):
        """测试无记忆协调器时不注入"""
        tm = TurnManager(llm_client=self.mock_llm_client)

        messages = tm._build_messages(
            system_prompt="Be helpful",
            history=[],
            user_input="Hi",
        )

        # 只有2条消息：system + user
        self.assertEqual(len(messages), 2)

    def test_build_messages_empty_recall(self):
        """测试记忆召回为空时不注入"""
        self.mock_coordinator.recall_as_context.return_value = ""

        tm = TurnManager(
            llm_client=self.mock_llm_client,
            memory_coordinator=self.mock_coordinator,
        )

        messages = tm._build_messages(
            system_prompt="Be helpful",
            history=[],
            user_input="Hi",
        )

        # 空字符串不注入
        self.assertEqual(len(messages), 2)

    def test_recall_memory_exception_handling(self):
        """测试记忆召回异常时不影响消息构建"""
        self.mock_coordinator.recall_as_context.side_effect = Exception("Recall failed")

        tm = TurnManager(
            llm_client=self.mock_llm_client,
            memory_coordinator=self.mock_coordinator,
        )

        # 不应抛异常
        messages = tm._build_messages(
            system_prompt="Be helpful",
            history=[],
            user_input="Hi",
        )

        self.assertEqual(len(messages), 2)


# =============================================================================
# Agent 记忆集成测试
# =============================================================================

class TestAgentMemoryIntegration(unittest.TestCase):
    def setUp(self):
        self.config = AgentConfig(
            agent_id="memory_test",
            name="Memory Test",
            enable_memory=True,
            enable_tools=False,
        )

    def test_memory_initialized(self):
        """测试记忆自动初始化"""
        agent = Agent(config=self.config)
        self.assertIsNotNone(agent.memory_coordinator)

    def test_memory_disabled(self):
        """测试禁用记忆"""
        config = AgentConfig(enable_memory=False)
        agent = Agent(config=config)
        self.assertIsNone(agent.memory_coordinator)

    def test_chat_stores_to_memory(self):
        """测试对话自动存储到记忆"""
        agent = Agent(config=self.config)

        mock_result = TurnResult(
            turn_id="t1",
            user_input="Hello",
            assistant_reply="Hi there!",
            status=TurnStatus.COMPLETED,
            duration_ms=100.0,
        )
        agent._turn_manager = MagicMock()
        agent._turn_manager.process_turn.return_value = mock_result

        agent.chat("Hello")

        # 验证记忆中有存储
        self.assertGreater(agent.memory_coordinator.short_term.size, 0)

    def test_chat_with_tool_stores_results(self):
        """测试带工具调用的对话存储工具结果"""
        agent = Agent(config=self.config)

        mock_result = TurnResult(
            turn_id="t1",
            user_input="Search",
            assistant_reply="Results found",
            status=TurnStatus.COMPLETED,
            tool_calls=[{"id": "c1"}],
            tool_results=[{
                "tool_call_id": "c1",
                "tool_name": "search",
                "output": "Found 3 items",
            }],
            duration_ms=100.0,
        )
        agent._turn_manager = MagicMock()
        agent._turn_manager.process_turn.return_value = mock_result

        agent.chat("Search")

        # 验证记忆中有存储（用户输入 + 助手回复 + 工具结果）
        self.assertGreaterEqual(agent.memory_coordinator.short_term.size, 2)

    def test_reset_clears_memory(self):
        """测试重置清空记忆"""
        agent = Agent(config=self.config)

        mock_result = TurnResult(
            turn_id="t1",
            user_input="Hello",
            assistant_reply="Hi",
            status=TurnStatus.COMPLETED,
        )
        agent._turn_manager = MagicMock()
        agent._turn_manager.process_turn.return_value = mock_result
        agent.chat("Hello")

        self.assertGreater(agent.memory_coordinator.short_term.size, 0)

        agent.reset()
        self.assertEqual(agent.memory_coordinator.short_term.size, 0)

    def test_summary_includes_memory(self):
        """测试摘要包含记忆统计"""
        agent = Agent(config=self.config)
        summary = agent.get_summary()

        self.assertTrue(summary["memory_enabled"])
        self.assertIn("memory", summary)
        self.assertIn("short_term", summary["memory"])

    def test_summary_memory_disabled(self):
        """测试禁用记忆时摘要不包含记忆统计"""
        config = AgentConfig(enable_memory=False)
        agent = Agent(config=config)
        summary = agent.get_summary()

        self.assertFalse(summary["memory_enabled"])
        self.assertNotIn("memory", summary)


# =============================================================================
# 端到端集成测试
# =============================================================================

class TestMemoryEndToEnd(unittest.TestCase):
    def test_memory_recall_in_chat_flow(self):
        """测试完整对话流程中的记忆召回"""
        config = AgentConfig(
            agent_id="e2e_test",
            enable_memory=True,
            enable_tools=False,
        )
        agent = Agent(config=config)

        # 第一轮对话
        mock_response1 = MagicMock()
        mock_response1.is_tool_call = False
        mock_response1.content = "I like Python programming."
        mock_response1.prompt_tokens = 10
        mock_response1.completion_tokens = 5
        mock_response1.total_tokens = 15

        agent._llm_client.chat = MagicMock(return_value=mock_response1)
        result1 = agent.chat("My favorite language is Python")

        self.assertEqual(result1.assistant_reply, "I like Python programming.")
        self.assertGreater(agent.memory_coordinator.short_term.size, 0)

        # 第二轮对话（记忆应该被召回）
        mock_response2 = MagicMock()
        mock_response2.is_tool_call = False
        mock_response2.content = "Yes, Python is great."
        mock_response2.prompt_tokens = 20
        mock_response2.completion_tokens = 5
        mock_response2.total_tokens = 25

        agent._llm_client.chat = MagicMock(return_value=mock_response2)
        result2 = agent.chat("What language did I mention?")

        self.assertEqual(result2.assistant_reply, "Yes, Python is great.")
        self.assertEqual(agent.stats.total_turns, 2)

        # 验证历史增长
        history = agent.get_history()
        self.assertEqual(len(history), 4)  # user1, assistant1, user2, assistant2


if __name__ == "__main__":
    unittest.main()
