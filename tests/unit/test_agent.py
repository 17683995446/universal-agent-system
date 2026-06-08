"""
Agent核心模块单元测试

测试范围：
- AgentState/TurnStatus 枚举
- TurnResult 回合结果
- AgentStats 统计
- AgentConfig 配置
- TurnManager 回合管理
- Agent 核心Agent

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-26
"""

import unittest
from unittest.mock import MagicMock, patch
import time

import sys
sys.path.insert(0, '/workspace')

from src.agent.agent_types import AgentState, TurnStatus, TurnResult, AgentStats, ConversationEntry
from src.agent.agent_config import AgentConfig
from src.agent.turn_manager import TurnManager
from src.agent.agent import Agent


# =============================================================================
# AgentState 枚举测试
# =============================================================================

class TestAgentState(unittest.TestCase):
    def test_values(self):
        self.assertEqual(AgentState.IDLE, "idle")
        self.assertEqual(AgentState.THINKING, "thinking")
        self.assertEqual(AgentState.EXECUTING, "executing")
        self.assertEqual(AgentState.PAUSED, "paused")
        self.assertEqual(AgentState.ERROR, "error")
        self.assertEqual(AgentState.STOPPED, "stopped")

    def test_is_str_enum(self):
        self.assertIsInstance(AgentState.IDLE, str)
        self.assertEqual(AgentState.IDLE.value, "idle")


# =============================================================================
# TurnStatus 枚举测试
# =============================================================================

class TestTurnStatus(unittest.TestCase):
    def test_values(self):
        self.assertEqual(TurnStatus.PENDING, "pending")
        self.assertEqual(TurnStatus.PROCESSING, "processing")
        self.assertEqual(TurnStatus.COMPLETED, "completed")
        self.assertEqual(TurnStatus.TOOL_CALLS, "tool_calls")
        self.assertEqual(TurnStatus.ERROR, "error")


# =============================================================================
# TurnResult 测试
# =============================================================================

class TestTurnResult(unittest.TestCase):
    def test_basic_creation(self):
        result = TurnResult(turn_id="t1")
        self.assertEqual(result.turn_id, "t1")
        self.assertEqual(result.user_input, "")
        self.assertEqual(result.assistant_reply, "")
        self.assertEqual(result.status, TurnStatus.COMPLETED)
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.tool_results, [])

    def test_has_tool_calls_true(self):
        result = TurnResult(
            turn_id="t1",
            status=TurnStatus.TOOL_CALLS,
            tool_calls=[{"id": "c1"}],
        )
        self.assertTrue(result.has_tool_calls)

    def test_has_tool_calls_false_wrong_status(self):
        result = TurnResult(
            turn_id="t1",
            status=TurnStatus.COMPLETED,
            tool_calls=[{"id": "c1"}],
        )
        self.assertFalse(result.has_tool_calls)

    def test_has_tool_calls_false_empty(self):
        result = TurnResult(
            turn_id="t1",
            status=TurnStatus.TOOL_CALLS,
            tool_calls=[],
        )
        self.assertFalse(result.has_tool_calls)

    def test_is_completed_true(self):
        result = TurnResult(turn_id="t1", status=TurnStatus.COMPLETED)
        self.assertTrue(result.is_completed)

    def test_is_completed_false(self):
        result = TurnResult(turn_id="t1", status=TurnStatus.TOOL_CALLS)
        self.assertFalse(result.is_completed)

    def test_is_error_true(self):
        result = TurnResult(turn_id="t1", status=TurnStatus.ERROR)
        self.assertTrue(result.is_error)

    def test_is_error_false(self):
        result = TurnResult(turn_id="t1", status=TurnStatus.COMPLETED)
        self.assertFalse(result.is_error)

    def test_to_dict(self):
        result = TurnResult(
            turn_id="t1",
            user_input="Hello",
            assistant_reply="Hi",
            status=TurnStatus.COMPLETED,
            tokens_used={"total": 100},
            duration_ms=500.5,
        )
        d = result.to_dict()
        self.assertEqual(d["turn_id"], "t1")
        self.assertEqual(d["user_input"], "Hello")
        self.assertEqual(d["assistant_reply"], "Hi")
        self.assertEqual(d["status"], "completed")
        self.assertEqual(d["tokens_used"], {"total": 100})
        self.assertEqual(d["duration_ms"], 500.5)


# =============================================================================
# AgentStats 测试
# =============================================================================

class TestAgentStats(unittest.TestCase):
    def test_default_values(self):
        stats = AgentStats()
        self.assertEqual(stats.total_turns, 0)
        self.assertEqual(stats.total_messages, 0)
        self.assertEqual(stats.total_tokens, 0)
        self.assertEqual(stats.tool_calls_count, 0)
        self.assertEqual(stats.errors_count, 0)
        self.assertEqual(stats.avg_turn_duration_ms, 0.0)

    def test_record_turn_success(self):
        stats = AgentStats()
        stats.record_turn(duration_ms=100.0, tokens=50, tool_calls=1, error=False)

        self.assertEqual(stats.total_turns, 1)
        self.assertEqual(stats.total_tokens, 50)
        self.assertEqual(stats.tool_calls_count, 1)
        self.assertEqual(stats.errors_count, 0)
        self.assertEqual(stats.avg_turn_duration_ms, 100.0)

    def test_record_turn_error(self):
        stats = AgentStats()
        stats.record_turn(duration_ms=100.0, tokens=50, error=True)

        self.assertEqual(stats.errors_count, 1)

    def test_record_multiple_turns(self):
        stats = AgentStats()
        stats.record_turn(duration_ms=100.0, tokens=50)
        stats.record_turn(duration_ms=200.0, tokens=100)

        self.assertEqual(stats.total_turns, 2)
        self.assertEqual(stats.avg_turn_duration_ms, 150.0)

    def test_uptime_seconds(self):
        stats = AgentStats()
        stats.start_time = time.time() - 10  # 10秒前启动
        self.assertAlmostEqual(stats.uptime_seconds, 10, delta=1)

    def test_uptime_zero_when_not_started(self):
        stats = AgentStats()
        self.assertEqual(stats.uptime_seconds, 0.0)

    def test_to_dict(self):
        stats = AgentStats()
        stats.record_turn(duration_ms=100.0, tokens=50)
        d = stats.to_dict()
        self.assertEqual(d["total_turns"], 1)
        self.assertEqual(d["total_tokens"], 50)
        self.assertEqual(d["avg_turn_duration_ms"], 100.0)


# =============================================================================
# ConversationEntry 测试
# =============================================================================

class TestConversationEntry(unittest.TestCase):
    def test_basic_creation(self):
        entry = ConversationEntry(
            turn_id="t1",
            role="user",
            content="Hello",
        )
        self.assertEqual(entry.turn_id, "t1")
        self.assertEqual(entry.role, "user")
        self.assertEqual(entry.content, "Hello")

    def test_to_dict(self):
        entry = ConversationEntry(turn_id="t1", role="user", content="Hello")
        d = entry.to_dict()
        self.assertEqual(d["turn_id"], "t1")
        self.assertEqual(d["role"], "user")
        self.assertEqual(d["content"], "Hello")


# =============================================================================
# AgentConfig 测试
# =============================================================================

class TestAgentConfig(unittest.TestCase):
    def test_default_values(self):
        config = AgentConfig()
        self.assertEqual(config.agent_id, "default")
        self.assertEqual(config.name, "Agent")
        self.assertEqual(config.system_prompt, "You are a helpful assistant.")
        self.assertEqual(config.max_turns, 0)
        self.assertTrue(config.enable_memory)
        self.assertTrue(config.enable_tools)
        self.assertEqual(config.max_tool_calls_per_turn, 10)

    def test_custom_values(self):
        config = AgentConfig(
            agent_id="test_agent",
            name="Test Agent",
            system_prompt="Custom prompt",
            max_turns=100,
            enable_tools=False,
        )
        self.assertEqual(config.agent_id, "test_agent")
        self.assertEqual(config.name, "Test Agent")
        self.assertEqual(config.system_prompt, "Custom prompt")
        self.assertEqual(config.max_turns, 100)
        self.assertFalse(config.enable_tools)

    def test_to_dict(self):
        config = AgentConfig(agent_id="test", name="Test")
        d = config.to_dict()
        self.assertEqual(d["agent_id"], "test")
        self.assertEqual(d["name"], "Test")
        self.assertEqual(d["system_prompt"], "You are a helpful assistant.")

    def test_from_dict(self):
        data = {
            "agent_id": "from_dict",
            "name": "From Dict",
            "system_prompt": "Custom",
            "max_turns": 50,
        }
        config = AgentConfig.from_dict(data)
        self.assertEqual(config.agent_id, "from_dict")
        self.assertEqual(config.name, "From Dict")
        self.assertEqual(config.system_prompt, "Custom")
        self.assertEqual(config.max_turns, 50)


# =============================================================================
# TurnManager 测试
# =============================================================================

class TestTurnManager(unittest.TestCase):
    def setUp(self):
        self.mock_llm_client = MagicMock()
        self.mock_tool_executor = MagicMock()
        self.turn_manager = TurnManager(
            llm_client=self.mock_llm_client,
            tool_executor=self.mock_tool_executor,
            max_tool_calls=5,
        )

    def test_process_turn_simple(self):
        """测试简单对话（无工具调用）"""
        # 模拟LLM响应
        mock_response = MagicMock()
        mock_response.is_tool_call = False
        mock_response.content = "Hello!"
        mock_response.prompt_tokens = 10
        mock_response.completion_tokens = 5
        mock_response.total_tokens = 15

        self.mock_llm_client.chat.return_value = mock_response

        result = self.turn_manager.process_turn(
            turn_id="t1",
            user_input="Hi",
            system_prompt="Be helpful",
            history=[],
        )

        self.assertEqual(result.status, TurnStatus.COMPLETED)
        self.assertEqual(result.assistant_reply, "Hello!")
        self.assertEqual(result.tokens_used["total"], 15)

    def test_process_turn_with_tools(self):
        """测试带工具调用的对话"""
        # 第一次LLM调用返回tool_calls
        mock_response1 = MagicMock()
        mock_response1.is_tool_call = True
        mock_response1.content = ""
        mock_response1.tool_calls = [{"id": "c1", "function": {"name": "test", "arguments": "{}"}}]
        mock_response1.prompt_tokens = 20
        mock_response1.completion_tokens = 10
        mock_response1.total_tokens = 30

        # 工具执行结果
        mock_tool_result = MagicMock()
        mock_tool_result.to_dict.return_value = {"tool_call_id": "c1", "output": "Tool output"}
        self.mock_tool_executor.execute_from_llm_response.return_value = [mock_tool_result]
        self.mock_tool_executor.results_to_tool_messages.return_value = [
            {"role": "tool", "tool_call_id": "c1", "content": "Tool output"}
        ]

        # 第二次LLM调用返回最终结果
        mock_response2 = MagicMock()
        mock_response2.content = "Final answer"
        mock_response2.prompt_tokens = 30
        mock_response2.completion_tokens = 10
        mock_response2.total_tokens = 40

        self.mock_llm_client.chat.side_effect = [mock_response1, mock_response2]

        result = self.turn_manager.process_turn(
            turn_id="t1",
            user_input="Use tool",
            system_prompt="Be helpful",
            history=[],
            tools_schema=[{"type": "function", "function": {"name": "test"}}],
        )

        self.assertEqual(result.status, TurnStatus.COMPLETED)
        self.assertEqual(result.assistant_reply, "Final answer")
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tokens_used["total"], 70)  # 30 + 40

    def test_process_turn_llm_error(self):
        """测试LLM调用失败"""
        self.mock_llm_client.chat.side_effect = Exception("API error")

        result = self.turn_manager.process_turn(
            turn_id="t1",
            user_input="Hi",
            system_prompt="Be helpful",
            history=[],
        )

        self.assertEqual(result.status, TurnStatus.ERROR)
        self.assertIn("API error", result.metadata["error"])

    def test_build_messages(self):
        """测试消息构建"""
        messages = self.turn_manager._build_messages(
            system_prompt="System",
            history=[{"role": "user", "content": "Previous"}],
            user_input="Current",
        )

        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[0].content, "System")
        self.assertEqual(messages[1].role, "user")
        self.assertEqual(messages[1].content, "Previous")
        self.assertEqual(messages[2].role, "user")
        self.assertEqual(messages[2].content, "Current")

    def test_build_messages_empty_system(self):
        """测试空系统提示"""
        messages = self.turn_manager._build_messages(
            system_prompt="",
            history=[],
            user_input="Hi",
        )

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "user")


# =============================================================================
# Agent 测试
# =============================================================================

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.config = AgentConfig(
            agent_id="test",
            name="Test Agent",
            enable_tools=True,
        )

    def test_creation(self):
        agent = Agent(config=self.config)
        self.assertEqual(agent.config.agent_id, "test")
        self.assertEqual(agent.state, AgentState.IDLE)
        self.assertEqual(agent.stats.total_turns, 0)

    def test_creation_default_config(self):
        agent = Agent()
        self.assertEqual(agent.config.agent_id, "default")
        self.assertEqual(agent.config.name, "Agent")

    def test_register_tool(self):
        agent = Agent(config=self.config)
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"

        agent.register_tool(mock_tool)
        self.assertTrue(agent.tool_registry.has("test_tool"))

    def test_register_tool_disabled(self):
        config = AgentConfig(enable_tools=False)
        agent = Agent(config=config)

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"

        # 不应抛异常，只是记录警告
        agent.register_tool(mock_tool)

    def test_register_tools(self):
        agent = Agent(config=self.config)
        mock_tools = [MagicMock(name=f"tool_{i}") for i in range(3)]
        for i, t in enumerate(mock_tools):
            t.name = f"tool_{i}"

        count = agent.register_tools(mock_tools)
        self.assertEqual(count, 3)

    def test_get_tools_schema(self):
        agent = Agent(config=self.config)
        mock_tool = MagicMock()
        mock_tool.name = "test"
        mock_tool.to_openai_schema.return_value = {"type": "function"}

        agent.register_tool(mock_tool)
        schema = agent.get_tools_schema()

        self.assertEqual(len(schema), 1)
        self.assertEqual(schema[0], {"type": "function"})

    def test_get_tools_schema_disabled(self):
        config = AgentConfig(enable_tools=False)
        agent = Agent(config=config)
        schema = agent.get_tools_schema()
        self.assertIsNone(schema)

    def test_chat_simple(self):
        """测试简单对话"""
        agent = Agent(config=self.config)

        # Mock turn_manager
        mock_result = TurnResult(
            turn_id="t1",
            user_input="Hello",
            assistant_reply="Hi there",
            status=TurnStatus.COMPLETED,
            duration_ms=100.0,
        )
        agent._turn_manager = MagicMock()
        agent._turn_manager.process_turn.return_value = mock_result

        result = agent.chat("Hello")

        self.assertEqual(result.assistant_reply, "Hi there")
        self.assertEqual(agent.stats.total_turns, 1)
        self.assertEqual(agent.state, AgentState.IDLE)

    def test_chat_stopped(self):
        """测试已停止的Agent"""
        agent = Agent(config=self.config)
        agent.stop()

        result = agent.chat("Hello")
        self.assertEqual(result.status, TurnStatus.ERROR)
        self.assertIn("stopped", result.metadata["error"])

    def test_chat_max_turns(self):
        """测试回合数限制"""
        config = AgentConfig(max_turns=2)
        agent = Agent(config=config)

        # 模拟已达到限制
        agent._stats.total_turns = 2

        result = agent.chat("Hello")
        self.assertEqual(result.status, TurnStatus.ERROR)
        self.assertIn("Max turns", result.metadata["error"])

    def test_get_history(self):
        agent = Agent(config=self.config)
        agent._conversation_history = [{"role": "user", "content": "Hi"}]

        history = agent.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["content"], "Hi")

    def test_clear_history(self):
        agent = Agent(config=self.config)
        agent._conversation_history = [{"role": "user", "content": "Hi"}]

        agent.clear_history()
        self.assertEqual(len(agent._conversation_history), 0)

    def test_get_conversation_summary(self):
        agent = Agent(config=self.config)
        agent._conversation_history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "tool", "content": "Result"},
        ]

        summary = agent.get_conversation_summary()
        self.assertEqual(summary["total_messages"], 3)
        self.assertEqual(summary["user_messages"], 1)
        self.assertEqual(summary["assistant_messages"], 1)
        self.assertEqual(summary["tool_calls"], 1)

    def test_reset(self):
        agent = Agent(config=self.config)
        agent._conversation_history = [{"role": "user", "content": "Hi"}]
        agent._stats.total_turns = 5

        agent.reset()

        self.assertEqual(len(agent._conversation_history), 0)
        self.assertEqual(agent.stats.total_turns, 0)
        self.assertEqual(agent.state, AgentState.IDLE)

    def test_stop(self):
        agent = Agent(config=self.config)
        agent.stop()
        self.assertEqual(agent.state, AgentState.STOPPED)

    def test_get_summary(self):
        agent = Agent(config=self.config)
        summary = agent.get_summary()

        self.assertEqual(summary["agent_id"], "test")
        self.assertEqual(summary["name"], "Test Agent")
        self.assertEqual(summary["state"], "idle")
        self.assertIn("stats", summary)
        self.assertIn("conversation", summary)


# =============================================================================
# 集成测试
# =============================================================================

class TestAgentIntegration(unittest.TestCase):
    def test_full_conversation_flow(self):
        """完整对话流程"""
        config = AgentConfig(
            agent_id="integration_test",
            name="Integration Test",
            system_prompt="You are a helpful assistant.",
        )

        agent = Agent(config=config)

        # Mock LLM responses
        mock_response1 = MagicMock()
        mock_response1.is_tool_call = False
        mock_response1.content = "Hello! How can I help you?"
        mock_response1.prompt_tokens = 10
        mock_response1.completion_tokens = 5
        mock_response1.total_tokens = 15

        mock_response2 = MagicMock()
        mock_response2.is_tool_call = False
        mock_response2.content = "I can help with that."
        mock_response2.prompt_tokens = 15
        mock_response2.completion_tokens = 5
        mock_response2.total_tokens = 20

        # 使用patch替换chat方法
        agent._llm_client.chat = MagicMock(side_effect=[mock_response1, mock_response2])

        # 第一轮对话
        result1 = agent.chat("Hi")
        self.assertEqual(result1.assistant_reply, "Hello! How can I help you?")
        self.assertEqual(agent.stats.total_turns, 1)

        # 第二轮对话
        result2 = agent.chat("I need help")
        self.assertEqual(result2.assistant_reply, "I can help with that.")
        self.assertEqual(agent.stats.total_turns, 2)

        # 验证历史
        history = agent.get_history()
        self.assertEqual(len(history), 4)  # user1, assistant1, user2, assistant2

    def test_conversation_with_tools(self):
        """带工具调用的对话"""
        from src.modules.tool_manager.tool_types import ToolDefinition, ToolParameter

        config = AgentConfig(enable_tools=True)
        agent = Agent(config=config)

        # 注册工具
        def search_handler(query: str):
            return f"Results for: {query}"

        tool = ToolDefinition(
            name="search",
            description="Search the web",
            parameters=[ToolParameter(name="query", type="string")],
            handler=search_handler,
        )
        agent.register_tool(tool)

        # Mock LLM responses
        mock_response1 = MagicMock()
        mock_response1.is_tool_call = True
        mock_response1.content = ""
        mock_response1.tool_calls = [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "search", "arguments": '{"query": "python"}'},
        }]
        mock_response1.prompt_tokens = 20
        mock_response1.completion_tokens = 10
        mock_response1.total_tokens = 30

        mock_response2 = MagicMock()
        mock_response2.is_tool_call = False
        mock_response2.content = "Here are the search results."
        mock_response2.prompt_tokens = 40
        mock_response2.completion_tokens = 10
        mock_response2.total_tokens = 50

        # 使用patch替换chat方法
        agent._llm_client.chat = MagicMock(side_effect=[mock_response1, mock_response2])

        # 对话
        result = agent.chat("Search for python")

        self.assertEqual(result.status, TurnStatus.COMPLETED)
        self.assertEqual(result.assistant_reply, "Here are the search results.")
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(agent.stats.tool_calls_count, 1)


if __name__ == "__main__":
    unittest.main()