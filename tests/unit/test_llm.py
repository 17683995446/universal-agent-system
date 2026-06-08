"""
LLM推理层单元测试

测试范围：
- LLMMessage 消息序列化
- LLMRequest 请求构建
- LLMResponse 响应属性
- LLMConfig 配置与环境变量
- LLMCallStats 调用统计
- LLMClient 客户端调用与重试
- ResponseParser 响应解析

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import json
import time

import sys
sys.path.insert(0, '/workspace')

from src.infrastructure.llm_types import (
    MessageRole,
    FinishReason,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMConfig,
    LLMCallStats,
)
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.response_parser import ResponseParser
from src.core.exceptions import LLMClientError


# =============================================================================
# 辅助函数
# =============================================================================

def make_mock_openai_response(
    content="Hello!",
    model="gpt-4",
    finish_reason="stop",
    prompt_tokens=10,
    completion_tokens=5,
    total_tokens=15,
    tool_calls=None,
):
    """构建模拟的OpenAI ChatCompletion响应对象"""
    # 模拟 message
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls

    # 模拟 choice
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    # 模拟 usage
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = total_tokens

    # 模拟 response
    response = MagicMock()
    response.choices = [choice]
    response.model = model
    response.usage = usage

    return response


def make_mock_tool_call(id="call_123", name="get_weather", arguments='{"city": "Beijing"}'):
    """构建模拟的工具调用对象"""
    func = MagicMock()
    func.name = name
    func.arguments = arguments

    tc = MagicMock()
    tc.id = id
    tc.type = "function"
    tc.function = func

    return tc


# =============================================================================
# MessageRole 枚举测试
# =============================================================================

class TestMessageRole(unittest.TestCase):
    def test_values(self):
        self.assertEqual(MessageRole.SYSTEM, "system")
        self.assertEqual(MessageRole.USER, "user")
        self.assertEqual(MessageRole.ASSISTANT, "assistant")
        self.assertEqual(MessageRole.TOOL, "tool")

    def test_is_str_enum(self):
        self.assertIsInstance(MessageRole.SYSTEM, str)
        self.assertEqual(MessageRole.USER.value, "user")
        self.assertEqual(MessageRole.USER, "user")  # str enum supports ==


# =============================================================================
# FinishReason 枚举测试
# =============================================================================

class TestFinishReason(unittest.TestCase):
    def test_values(self):
        self.assertEqual(FinishReason.STOP, "stop")
        self.assertEqual(FinishReason.LENGTH, "length")
        self.assertEqual(FinishReason.TOOL_CALLS, "tool_calls")
        self.assertEqual(FinishReason.CONTENT_FILTER, "content_filter")


# =============================================================================
# LLMMessage 测试
# =============================================================================

class TestLLMMessage(unittest.TestCase):
    def test_basic_creation(self):
        msg = LLMMessage(role="user", content="Hello")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello")
        self.assertIsNone(msg.name)
        self.assertIsNone(msg.tool_calls)
        self.assertIsNone(msg.tool_call_id)

    def test_to_dict_basic(self):
        msg = LLMMessage(role="system", content="You are helpful.")
        d = msg.to_dict()
        self.assertEqual(d, {"role": "system", "content": "You are helpful."})

    def test_to_dict_with_optional_fields(self):
        msg = LLMMessage(
            role="assistant",
            content="",
            name="bot",
            tool_calls=[{"id": "c1", "type": "function"}],
            tool_call_id="tc_1",
        )
        d = msg.to_dict()
        self.assertEqual(d["role"], "assistant")
        self.assertEqual(d["content"], "")
        self.assertEqual(d["name"], "bot")
        self.assertEqual(d["tool_calls"], [{"id": "c1", "type": "function"}])
        self.assertEqual(d["tool_call_id"], "tc_1")

    def test_to_dict_excludes_none(self):
        msg = LLMMessage(role="user", content="Hi")
        d = msg.to_dict()
        self.assertNotIn("name", d)
        self.assertNotIn("tool_calls", d)
        self.assertNotIn("tool_call_id", d)

    def test_to_dict_empty_tool_calls_excluded(self):
        msg = LLMMessage(role="assistant", content="OK", tool_calls=None)
        d = msg.to_dict()
        self.assertNotIn("tool_calls", d)


# =============================================================================
# LLMRequest 测试
# =============================================================================

class TestLLMRequest(unittest.TestCase):
    def test_default_values(self):
        req = LLMRequest(model="gpt-4")
        self.assertEqual(req.model, "gpt-4")
        self.assertEqual(req.messages, [])
        self.assertEqual(req.temperature, 0.7)
        self.assertEqual(req.max_tokens, 4096)
        self.assertEqual(req.top_p, 1.0)
        self.assertFalse(req.stream)
        self.assertIsNone(req.stop)
        self.assertIsNone(req.tools)
        self.assertEqual(req.metadata, {})

    def test_to_openai_kwargs_basic(self):
        req = LLMRequest(model="gpt-4", messages=[
            LLMMessage(role="user", content="Hi"),
        ])
        kwargs = req.to_openai_kwargs()
        self.assertEqual(kwargs["model"], "gpt-4")
        self.assertEqual(kwargs["temperature"], 0.7)
        self.assertEqual(kwargs["max_tokens"], 4096)
        self.assertEqual(kwargs["top_p"], 1.0)
        self.assertFalse(kwargs["stream"])
        self.assertEqual(len(kwargs["messages"]), 1)
        self.assertEqual(kwargs["messages"][0]["role"], "user")
        self.assertEqual(kwargs["messages"][0]["content"], "Hi")

    def test_to_openai_kwargs_with_stop(self):
        req = LLMRequest(model="gpt-4", stop=["\n", "END"])
        kwargs = req.to_openai_kwargs()
        self.assertEqual(kwargs["stop"], ["\n", "END"])

    def test_to_openai_kwargs_stop_excluded_when_none(self):
        req = LLMRequest(model="gpt-4")
        kwargs = req.to_openai_kwargs()
        self.assertNotIn("stop", kwargs)

    def test_to_openai_kwargs_with_tools(self):
        tools = [{"type": "function", "function": {"name": "search"}}]
        req = LLMRequest(model="gpt-4", tools=tools)
        kwargs = req.to_openai_kwargs()
        self.assertEqual(kwargs["tools"], tools)

    def test_to_openai_kwargs_tools_excluded_when_none(self):
        req = LLMRequest(model="gpt-4")
        kwargs = req.to_openai_kwargs()
        self.assertNotIn("tools", kwargs)

    def test_to_openai_kwargs_messages_serialized(self):
        req = LLMRequest(model="gpt-3.5-turbo", messages=[
            LLMMessage(role="system", content="Be helpful"),
            LLMMessage(role="user", content="Hello"),
            LLMMessage(role="assistant", content="Hi there"),
        ])
        kwargs = req.to_openai_kwargs()
        self.assertEqual(len(kwargs["messages"]), 3)
        self.assertEqual(kwargs["messages"][0]["role"], "system")
        self.assertEqual(kwargs["messages"][2]["content"], "Hi there")


# =============================================================================
# LLMResponse 测试
# =============================================================================

class TestLLMResponse(unittest.TestCase):
    def test_default_values(self):
        resp = LLMResponse()
        self.assertEqual(resp.content, "")
        self.assertEqual(resp.model, "")
        self.assertEqual(resp.finish_reason, "")
        self.assertEqual(resp.prompt_tokens, 0)
        self.assertEqual(resp.completion_tokens, 0)
        self.assertEqual(resp.total_tokens, 0)
        self.assertIsNone(resp.tool_calls)
        self.assertEqual(resp.metadata, {})

    def test_is_tool_call_true(self):
        resp = LLMResponse(
            finish_reason="tool_calls",
            tool_calls=[{"id": "c1"}],
        )
        self.assertTrue(resp.is_tool_call)

    def test_is_tool_call_false_wrong_reason(self):
        resp = LLMResponse(
            finish_reason="stop",
            tool_calls=[{"id": "c1"}],
        )
        self.assertFalse(resp.is_tool_call)

    def test_is_tool_call_false_no_tool_calls(self):
        resp = LLMResponse(
            finish_reason="tool_calls",
            tool_calls=None,
        )
        self.assertFalse(resp.is_tool_call)

    def test_is_stopped_true(self):
        resp = LLMResponse(finish_reason="stop")
        self.assertTrue(resp.is_stopped)

    def test_is_stopped_false(self):
        resp = LLMResponse(finish_reason="length")
        self.assertFalse(resp.is_stopped)

    def test_is_length_limited_true(self):
        resp = LLMResponse(finish_reason="length")
        self.assertTrue(resp.is_length_limited)

    def test_is_length_limited_false(self):
        resp = LLMResponse(finish_reason="stop")
        self.assertFalse(resp.is_length_limited)

    def test_to_dict(self):
        resp = LLMResponse(
            content="Hello",
            model="gpt-4",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            tool_calls=[{"id": "c1"}],
        )
        d = resp.to_dict()
        self.assertEqual(d["content"], "Hello")
        self.assertEqual(d["model"], "gpt-4")
        self.assertEqual(d["finish_reason"], "stop")
        self.assertEqual(d["prompt_tokens"], 10)
        self.assertEqual(d["completion_tokens"], 5)
        self.assertEqual(d["total_tokens"], 15)
        self.assertEqual(d["tool_calls"], [{"id": "c1"}])


# =============================================================================
# LLMConfig 测试
# =============================================================================

class TestLLMConfig(unittest.TestCase):
    def test_default_values(self):
        config = LLMConfig()
        self.assertEqual(config.model, "gpt-4")
        self.assertIsNone(config.api_key)
        self.assertIsNone(config.base_url)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.max_tokens, 4096)
        self.assertEqual(config.timeout_connect, 10.0)
        self.assertEqual(config.timeout_read, 60.0)
        self.assertEqual(config.retry_max_attempts, 3)
        self.assertEqual(config.retry_backoff_base, 2.0)
        self.assertEqual(config.retry_max_backoff, 30.0)

    def test_custom_values(self):
        config = LLMConfig(
            model="gpt-3.5-turbo",
            api_key="sk-test",
            base_url="https://api.example.com/v1",
            temperature=0.5,
            max_tokens=2048,
            retry_max_attempts=5,
        )
        self.assertEqual(config.model, "gpt-3.5-turbo")
        self.assertEqual(config.api_key, "sk-test")
        self.assertEqual(config.base_url, "https://api.example.com/v1")
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.max_tokens, 2048)
        self.assertEqual(config.retry_max_attempts, 5)

    def test_to_dict(self):
        config = LLMConfig(model="gpt-4", base_url="https://api.example.com")
        d = config.to_dict()
        self.assertEqual(d["model"], "gpt-4")
        self.assertEqual(d["base_url"], "https://api.example.com")
        self.assertIn("temperature", d)
        self.assertIn("max_tokens", d)

    def test_to_dict_excludes_base_url_when_none(self):
        config = LLMConfig()
        d = config.to_dict()
        self.assertNotIn("base_url", d)

    def test_from_env(self):
        env_vars = {
            "DEFAULT_MODEL": "gpt-3.5-turbo",
            "OPENAI_API_KEY": "sk-env-test",
            "OPENAI_BASE_URL": "https://env.example.com/v1",
            "LLM_TEMPERATURE": "0.3",
            "LLM_MAX_TOKENS": "1024",
            "API_TIMEOUT_CONNECT": "5.0",
            "API_TIMEOUT_READ": "30.0",
            "API_RETRY_MAX_ATTEMPTS": "5",
            "API_RETRY_BACKOFF_BASE": "3.0",
        }
        with patch.dict("os.environ", env_vars, clear=False):
            config = LLMConfig.from_env()
        self.assertEqual(config.model, "gpt-3.5-turbo")
        self.assertEqual(config.api_key, "sk-env-test")
        self.assertEqual(config.base_url, "https://env.example.com/v1")
        self.assertEqual(config.temperature, 0.3)
        self.assertEqual(config.max_tokens, 1024)
        self.assertEqual(config.timeout_connect, 5.0)
        self.assertEqual(config.timeout_read, 30.0)
        self.assertEqual(config.retry_max_attempts, 5)
        self.assertEqual(config.retry_backoff_base, 3.0)

    def test_from_env_defaults(self):
        """测试环境变量缺失时使用默认值"""
        clean_env = {}
        with patch.dict("os.environ", clean_env, clear=True):
            config = LLMConfig.from_env()
        self.assertEqual(config.model, "gpt-4")
        self.assertIsNone(config.api_key)
        self.assertIsNone(config.base_url)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.max_tokens, 4096)
        self.assertEqual(config.timeout_connect, 10.0)
        self.assertEqual(config.timeout_read, 60.0)
        self.assertEqual(config.retry_max_attempts, 3)
        self.assertEqual(config.retry_backoff_base, 2.0)


# =============================================================================
# LLMCallStats 测试
# =============================================================================

class TestLLMCallStats(unittest.TestCase):
    def test_default_values(self):
        stats = LLMCallStats()
        self.assertEqual(stats.model, "")
        self.assertEqual(stats.start_time, 0.0)
        self.assertEqual(stats.end_time, 0.0)
        self.assertEqual(stats.duration_ms, 0.0)
        self.assertEqual(stats.prompt_tokens, 0)
        self.assertEqual(stats.completion_tokens, 0)
        self.assertEqual(stats.total_tokens, 0)
        self.assertEqual(stats.attempt, 1)
        self.assertTrue(stats.success)
        self.assertIsNone(stats.error)

    def test_custom_values(self):
        stats = LLMCallStats(
            model="gpt-4",
            duration_ms=500.0,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            attempt=2,
            success=False,
            error="timeout",
        )
        self.assertEqual(stats.model, "gpt-4")
        self.assertEqual(stats.duration_ms, 500.0)
        self.assertEqual(stats.attempt, 2)
        self.assertFalse(stats.success)
        self.assertEqual(stats.error, "timeout")

    def test_to_dict(self):
        stats = LLMCallStats(
            model="gpt-4",
            duration_ms=123.456,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            attempt=1,
            success=True,
        )
        d = stats.to_dict()
        self.assertEqual(d["model"], "gpt-4")
        self.assertEqual(d["duration_ms"], 123.46)  # round to 2 decimals
        self.assertEqual(d["prompt_tokens"], 10)
        self.assertEqual(d["completion_tokens"], 5)
        self.assertEqual(d["total_tokens"], 15)
        self.assertEqual(d["attempt"], 1)
        self.assertTrue(d["success"])
        self.assertIsNone(d["error"])


# =============================================================================
# LLMClient 测试
# =============================================================================

class TestLLMClient(unittest.TestCase):
    def test_creation_with_config(self):
        config = LLMConfig(model="gpt-3.5-turbo", api_key="sk-test")
        client = LLMClient(config=config)
        self.assertEqual(client._config.model, "gpt-3.5-turbo")
        self.assertEqual(client._config.api_key, "sk-test")

    def test_creation_without_config(self):
        """测试无配置时从环境变量加载"""
        with patch.dict("os.environ", {}, clear=True):
            client = LLMClient()
        self.assertEqual(client._config.model, "gpt-4")

    def test_get_client_lazy_load(self):
        """测试客户端懒加载"""
        config = LLMConfig(api_key="sk-test")
        client = LLMClient(config=config)
        self.assertIsNone(client._client)

        mock_openai_module = MagicMock()
        mock_openai_cls = MagicMock()
        mock_openai_module.OpenAI = mock_openai_cls
        mock_instance = MagicMock()
        mock_openai_cls.return_value = mock_instance

        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            result = client._get_client()
            self.assertEqual(result, mock_instance)
            mock_openai_cls.assert_called_once_with(api_key="sk-test")

    def test_get_client_with_base_url(self):
        """测试带base_url的客户端创建"""
        config = LLMConfig(api_key="sk-test", base_url="https://api.example.com/v1")
        client = LLMClient(config=config)

        mock_openai_module = MagicMock()
        mock_openai_cls = MagicMock()
        mock_openai_module.OpenAI = mock_openai_cls

        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            client._get_client()
            mock_openai_cls.assert_called_once_with(
                api_key="sk-test",
                base_url="https://api.example.com/v1",
            )

    def test_get_client_openai_not_installed(self):
        """测试openai包未安装时抛出异常"""
        config = LLMConfig()
        client = LLMClient(config=config)

        with patch.dict("sys.modules", {"openai": None}):
            with self.assertRaises(LLMClientError) as ctx:
                client._get_client()
            self.assertIn("openai package not installed", str(ctx.exception))

    def test_chat_success(self):
        """测试成功调用"""
        config = LLMConfig(model="gpt-4", api_key="sk-test")
        client = LLMClient(config=config)

        mock_response = make_mock_openai_response(
            content="Hello! How can I help?",
            model="gpt-4",
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
        )

        with patch.object(client, "_get_client") as mock_get_client:
            mock_api = MagicMock()
            mock_api.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_api

            messages = [LLMMessage(role="user", content="Hi")]
            result = client.chat(messages)

        self.assertEqual(result.content, "Hello! How can I help?")
        self.assertEqual(result.model, "gpt-4")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.prompt_tokens, 20)
        self.assertEqual(result.completion_tokens, 10)
        self.assertEqual(result.total_tokens, 30)

    def test_chat_uses_config_defaults(self):
        """测试chat使用配置默认值"""
        config = LLMConfig(
            model="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=1024,
            api_key="sk-test",
        )
        client = LLMClient(config=config)

        mock_response = make_mock_openai_response()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_api = MagicMock()
            mock_api.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_api

            messages = [LLMMessage(role="user", content="Hi")]
            client.chat(messages)

            # 验证传递给API的参数
            call_kwargs = mock_api.chat.completions.create.call_args[1]
            self.assertEqual(call_kwargs["model"], "gpt-3.5-turbo")
            self.assertEqual(call_kwargs["temperature"], 0.5)
            self.assertEqual(call_kwargs["max_tokens"], 1024)

    def test_chat_overrides_config(self):
        """测试chat参数覆盖配置"""
        config = LLMConfig(
            model="gpt-4",
            temperature=0.7,
            max_tokens=4096,
            api_key="sk-test",
        )
        client = LLMClient(config=config)

        mock_response = make_mock_openai_response()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_api = MagicMock()
            mock_api.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_api

            messages = [LLMMessage(role="user", content="Hi")]
            client.chat(messages, model="gpt-3.5-turbo", temperature=0.1, max_tokens=512)

            call_kwargs = mock_api.chat.completions.create.call_args[1]
            self.assertEqual(call_kwargs["model"], "gpt-3.5-turbo")
            self.assertEqual(call_kwargs["temperature"], 0.1)
            self.assertEqual(call_kwargs["max_tokens"], 512)

    def test_chat_retry_on_failure(self):
        """测试失败重试"""
        config = LLMConfig(
            model="gpt-4",
            api_key="sk-test",
            retry_max_attempts=3,
            retry_backoff_base=0.1,  # 快速退避用于测试
            retry_max_backoff=1.0,
        )
        client = LLMClient(config=config)

        mock_response = make_mock_openai_response(content="Success after retry")

        with patch.object(client, "_get_client") as mock_get_client, \
             patch("src.infrastructure.llm_client.time.sleep") as mock_sleep:
            mock_api = MagicMock()
            # 前两次失败，第三次成功
            mock_api.chat.completions.create.side_effect = [
                Exception("API error 1"),
                Exception("API error 2"),
                mock_response,
            ]
            mock_get_client.return_value = mock_api

            messages = [LLMMessage(role="user", content="Hi")]
            result = client.chat(messages)

        self.assertEqual(result.content, "Success after retry")
        self.assertEqual(mock_api.chat.completions.create.call_count, 3)
        # 验证退避等待
        self.assertEqual(mock_sleep.call_count, 2)

    def test_chat_all_retries_exhausted(self):
        """测试所有重试耗尽后抛出异常"""
        config = LLMConfig(
            model="gpt-4",
            api_key="sk-test",
            retry_max_attempts=2,
            retry_backoff_base=0.1,
            retry_max_backoff=1.0,
        )
        client = LLMClient(config=config)

        with patch.object(client, "_get_client") as mock_get_client, \
             patch("src.infrastructure.llm_client.time.sleep") as mock_sleep:
            mock_api = MagicMock()
            mock_api.chat.completions.create.side_effect = Exception("Persistent error")
            mock_get_client.return_value = mock_api

            messages = [LLMMessage(role="user", content="Hi")]
            with self.assertRaises(LLMClientError) as ctx:
                client.chat(messages)

        self.assertIn("All 2 attempts failed", str(ctx.exception))
        self.assertEqual(mock_api.chat.completions.create.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)  # 最后一次不等待

    def test_chat_with_tool_calls(self):
        """测试工具调用响应"""
        config = LLMConfig(api_key="sk-test")
        client = LLMClient(config=config)

        tc = make_mock_tool_call()
        mock_response = make_mock_openai_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[tc],
        )

        with patch.object(client, "_get_client") as mock_get_client:
            mock_api = MagicMock()
            mock_api.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_api

            messages = [LLMMessage(role="user", content="What's the weather?")]
            result = client.chat(messages)

        self.assertEqual(result.content, "")
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertTrue(result.is_tool_call)
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0]["id"], "call_123")
        self.assertEqual(result.tool_calls[0]["function"]["name"], "get_weather")

    def test_chat_simple(self):
        """测试简单聊天便捷方法"""
        config = LLMConfig(api_key="sk-test")
        client = LLMClient(config=config)

        mock_response = make_mock_openai_response(content="Simple reply")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_api = MagicMock()
            mock_api.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_api

            result = client.chat_simple("Be helpful", "Hello")

        self.assertEqual(result, "Simple reply")

        # 验证消息格式
        call_kwargs = mock_api.chat.completions.create.call_args[1]
        msgs = call_kwargs["messages"]
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], "Be helpful")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertEqual(msgs[1]["content"], "Hello")

    def test_calculate_backoff(self):
        """测试指数退避计算"""
        config = LLMConfig(retry_backoff_base=2.0, retry_max_backoff=30.0)
        client = LLMClient(config=config)

        self.assertAlmostEqual(client._calculate_backoff(1), 2.0)
        self.assertAlmostEqual(client._calculate_backoff(2), 4.0)
        self.assertAlmostEqual(client._calculate_backoff(3), 8.0)
        self.assertAlmostEqual(client._calculate_backoff(5), 30.0)  # capped

    def test_calculate_backoff_custom_base(self):
        """测试自定义退避基数"""
        config = LLMConfig(retry_backoff_base=3.0, retry_max_backoff=20.0)
        client = LLMClient(config=config)

        self.assertAlmostEqual(client._calculate_backoff(1), 3.0)
        self.assertAlmostEqual(client._calculate_backoff(2), 9.0)
        self.assertAlmostEqual(client._calculate_backoff(3), 20.0)  # capped

    def test_get_stats_empty(self):
        """测试空统计"""
        config = LLMConfig()
        client = LLMClient(config=config)
        stats = client.get_stats()
        self.assertEqual(stats, {"total_calls": 0})

    def test_get_stats_after_calls(self):
        """测试调用后统计"""
        config = LLMConfig(api_key="sk-test", retry_backoff_base=0.01)
        client = LLMClient(config=config)

        mock_response = make_mock_openai_response(
            prompt_tokens=10, completion_tokens=5, total_tokens=15,
        )

        with patch.object(client, "_get_client") as mock_get_client:
            mock_api = MagicMock()
            mock_api.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_api

            messages = [LLMMessage(role="user", content="Hi")]
            client.chat(messages)

        stats = client.get_stats()
        self.assertEqual(stats["total_calls"], 1)
        self.assertEqual(stats["success_calls"], 1)
        self.assertEqual(stats["failed_calls"], 0)
        self.assertEqual(stats["success_rate"], 1.0)
        self.assertEqual(stats["total_tokens"], 15)
        self.assertIsNotNone(stats["last_call"])
        self.assertTrue(stats["last_call"]["success"])

    def test_get_last_call_stats(self):
        """测试获取最近一次调用统计"""
        config = LLMConfig(api_key="sk-test", retry_backoff_base=0.01)
        client = LLMClient(config=config)

        self.assertIsNone(client.get_last_call_stats())

        mock_response = make_mock_openai_response()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_api = MagicMock()
            mock_api.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_api

            client.chat([LLMMessage(role="user", content="Hi")])

        last = client.get_last_call_stats()
        self.assertIsNotNone(last)
        self.assertEqual(last.model, "gpt-4")
        self.assertTrue(last.success)

    def test_clear_history(self):
        """测试清空调用历史"""
        config = LLMConfig(api_key="sk-test", retry_backoff_base=0.01)
        client = LLMClient(config=config)

        mock_response = make_mock_openai_response()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_api = MagicMock()
            mock_api.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_api

            client.chat([LLMMessage(role="user", content="Hi")])
            client.chat([LLMMessage(role="user", content="Hi again")])

        count = client.clear_history()
        self.assertEqual(count, 2)
        self.assertEqual(client.get_stats()["total_calls"], 0)

    def test_parse_response_with_usage(self):
        """测试解析带Token使用的响应"""
        config = LLMConfig()
        client = LLMClient(config=config)

        mock_response = make_mock_openai_response(
            content="Test",
            model="gpt-4",
            finish_reason="stop",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        result = client._parse_response(mock_response)
        self.assertEqual(result.content, "Test")
        self.assertEqual(result.model, "gpt-4")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.prompt_tokens, 100)
        self.assertEqual(result.completion_tokens, 50)
        self.assertEqual(result.total_tokens, 150)

    def test_parse_response_without_usage(self):
        """测试解析无Token使用的响应"""
        config = LLMConfig()
        client = LLMClient(config=config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "No usage"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "test-model"
        mock_response.usage = None

        result = client._parse_response(mock_response)
        self.assertEqual(result.content, "No usage")
        self.assertEqual(result.prompt_tokens, 0)
        self.assertEqual(result.completion_tokens, 0)
        self.assertEqual(result.total_tokens, 0)

    def test_parse_response_null_content(self):
        """测试content为None的响应"""
        config = LLMConfig()
        client = LLMClient(config=config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 0
        mock_response.usage.completion_tokens = 0
        mock_response.usage.total_tokens = 0

        result = client._parse_response(mock_response)
        self.assertEqual(result.content, "")


# =============================================================================
# ResponseParser 测试
# =============================================================================

class TestResponseParser(unittest.TestCase):
    def test_parse_openai_basic(self):
        """测试解析基本OpenAI响应"""
        mock_response = make_mock_openai_response(
            content="Hello!",
            model="gpt-4",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        result = ResponseParser.parse_openai(mock_response)
        self.assertEqual(result.content, "Hello!")
        self.assertEqual(result.model, "gpt-4")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.prompt_tokens, 10)
        self.assertEqual(result.completion_tokens, 5)
        self.assertEqual(result.total_tokens, 15)

    def test_parse_openai_with_tool_calls(self):
        """测试解析带工具调用的OpenAI响应"""
        tc = make_mock_tool_call()
        mock_response = make_mock_openai_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[tc],
        )
        result = ResponseParser.parse_openai(mock_response)
        self.assertTrue(result.is_tool_call)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0]["id"], "call_123")
        self.assertEqual(result.tool_calls[0]["function"]["name"], "get_weather")
        self.assertEqual(result.tool_calls[0]["function"]["arguments"], '{"city": "Beijing"}')

    def test_parse_openai_no_usage(self):
        """测试解析无usage的响应"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "No usage info"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = None

        result = ResponseParser.parse_openai(mock_response)
        self.assertEqual(result.content, "No usage info")
        self.assertEqual(result.prompt_tokens, 0)
        self.assertEqual(result.total_tokens, 0)

    def test_parse_dict_basic(self):
        """测试解析字典格式"""
        data = {
            "content": "Hello from dict",
            "model": "gpt-3.5-turbo",
            "finish_reason": "stop",
            "prompt_tokens": 8,
            "completion_tokens": 4,
            "total_tokens": 12,
        }
        result = ResponseParser.parse_dict(data)
        self.assertEqual(result.content, "Hello from dict")
        self.assertEqual(result.model, "gpt-3.5-turbo")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.prompt_tokens, 8)
        self.assertEqual(result.total_tokens, 12)

    def test_parse_dict_minimal(self):
        """测试解析最小字典"""
        result = ResponseParser.parse_dict({})
        self.assertEqual(result.content, "")
        self.assertEqual(result.model, "")
        self.assertEqual(result.finish_reason, "stop")  # 默认值
        self.assertEqual(result.prompt_tokens, 0)

    def test_parse_dict_with_tool_calls(self):
        """测试解析带工具调用的字典"""
        data = {
            "content": "",
            "finish_reason": "tool_calls",
            "tool_calls": [{"id": "c1", "name": "search"}],
        }
        result = ResponseParser.parse_dict(data)
        self.assertTrue(result.is_tool_call)
        self.assertEqual(result.tool_calls, [{"id": "c1", "name": "search"}])

    def test_extract_tool_calls_success(self):
        """测试提取工具调用"""
        response = LLMResponse(
            finish_reason="tool_calls",
            tool_calls=[{"id": "c1"}, {"id": "c2"}],
        )
        result = ResponseParser.extract_tool_calls(response)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], "c1")

    def test_extract_tool_calls_not_tool_call(self):
        """测试非工具调用响应"""
        response = LLMResponse(finish_reason="stop")
        result = ResponseParser.extract_tool_calls(response)
        self.assertEqual(result, [])

    def test_extract_tool_calls_none(self):
        """测试tool_calls为None"""
        response = LLMResponse(finish_reason="tool_calls", tool_calls=None)
        result = ResponseParser.extract_tool_calls(response)
        self.assertEqual(result, [])

    def test_parse_tool_arguments_valid_json(self):
        """测试解析有效的工具参数"""
        tool_calls = [
            {
                "id": "c1",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Beijing", "unit": "celsius"}',
                },
            },
            {
                "id": "c2",
                "function": {
                    "name": "search",
                    "arguments": '{"query": "Python tutorial"}',
                },
            },
        ]
        result = ResponseParser.parse_tool_arguments(tool_calls)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], "c1")
        self.assertEqual(result[0]["name"], "get_weather")
        self.assertEqual(result[0]["arguments"]["city"], "Beijing")
        self.assertEqual(result[0]["arguments"]["unit"], "celsius")
        self.assertEqual(result[1]["name"], "search")
        self.assertEqual(result[1]["arguments"]["query"], "Python tutorial")

    def test_parse_tool_arguments_invalid_json(self):
        """测试解析无效的JSON参数"""
        tool_calls = [
            {
                "id": "c1",
                "function": {
                    "name": "broken_func",
                    "arguments": "not valid json{{{",
                },
            },
        ]
        result = ResponseParser.parse_tool_arguments(tool_calls)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "broken_func")
        self.assertIn("raw_arguments", result[0]["arguments"])
        self.assertEqual(result[0]["arguments"]["raw_arguments"], "not valid json{{{")

    def test_parse_tool_arguments_empty(self):
        """测试空工具调用列表"""
        result = ResponseParser.parse_tool_arguments([])
        self.assertEqual(result, [])

    def test_parse_tool_arguments_missing_fields(self):
        """测试缺少字段的工具调用"""
        tool_calls = [{"id": "c1"}]  # 缺少function字段
        result = ResponseParser.parse_tool_arguments(tool_calls)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "c1")
        self.assertEqual(result[0]["name"], "")
        self.assertEqual(result[0]["arguments"], {})

    def test_extract_json_from_content_json_block(self):
        """测试从```json块中提取JSON"""
        content = 'Here is the result:\n```json\n{"key": "value", "num": 42}\n```\nDone.'
        result = ResponseParser.extract_json_from_content(content)
        self.assertIsNotNone(result)
        self.assertEqual(result["key"], "value")
        self.assertEqual(result["num"], 42)

    def test_extract_json_from_content_plain_braces(self):
        """测试从纯{...}块中提取JSON"""
        content = 'The answer is {"answer": 42, "correct": true} end.'
        result = ResponseParser.extract_json_from_content(content)
        self.assertIsNotNone(result)
        self.assertEqual(result["answer"], 42)
        self.assertTrue(result["correct"])

    def test_extract_json_from_content_no_json(self):
        """测试无JSON内容"""
        content = "This is just plain text without any JSON."
        result = ResponseParser.extract_json_from_content(content)
        self.assertIsNone(result)

    def test_extract_json_from_content_invalid_json(self):
        """测试无效JSON内容"""
        content = '```json\n{invalid json content}\n```'
        result = ResponseParser.extract_json_from_content(content)
        self.assertIsNone(result)

    def test_extract_json_from_content_nested(self):
        """测试嵌套JSON"""
        content = 'Result:\n```json\n{"outer": {"inner": "value"}}\n```'
        result = ResponseParser.extract_json_from_content(content)
        self.assertIsNotNone(result)
        self.assertEqual(result["outer"]["inner"], "value")

    def test_extract_json_from_content_code_block_without_json_tag(self):
        """测试无json标签的代码块"""
        content = '```\n{"key": "value"}\n```'
        result = ResponseParser.extract_json_from_content(content)
        self.assertIsNotNone(result)
        self.assertEqual(result["key"], "value")


# =============================================================================
# 集成测试
# =============================================================================

class TestLLMIntegration(unittest.TestCase):
    """LLM模块集成测试"""

    def test_full_request_response_cycle(self):
        """测试完整的请求-响应周期"""
        # 构建请求
        request = LLMRequest(
            model="gpt-4",
            messages=[
                LLMMessage(role="system", content="Be concise"),
                LLMMessage(role="user", content="What is 2+2?"),
            ],
            temperature=0.0,
            max_tokens=100,
        )

        # 验证序列化
        kwargs = request.to_openai_kwargs()
        self.assertEqual(kwargs["model"], "gpt-4")
        self.assertEqual(len(kwargs["messages"]), 2)
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertEqual(kwargs["max_tokens"], 100)

        # 模拟响应
        mock_resp = make_mock_openai_response(
            content="4",
            model="gpt-4",
            finish_reason="stop",
            prompt_tokens=15,
            completion_tokens=1,
            total_tokens=16,
        )

        # 解析响应
        response = ResponseParser.parse_openai(mock_resp)
        self.assertEqual(response.content, "4")
        self.assertTrue(response.is_stopped)
        self.assertFalse(response.is_tool_call)

    def test_tool_call_full_cycle(self):
        """测试工具调用完整周期"""
        # 构建工具调用请求
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                },
            },
        }]
        request = LLMRequest(
            model="gpt-4",
            messages=[LLMMessage(role="user", content="Weather in Beijing?")],
            tools=tools,
        )

        # 验证tools包含在kwargs中
        kwargs = request.to_openai_kwargs()
        self.assertEqual(kwargs["tools"], tools)

        # 模拟工具调用响应
        tc = make_mock_tool_call()
        mock_resp = make_mock_openai_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[tc],
        )

        # 解析
        response = ResponseParser.parse_openai(mock_resp)
        self.assertTrue(response.is_tool_call)

        # 提取工具调用
        extracted = ResponseParser.extract_tool_calls(response)
        self.assertEqual(len(extracted), 1)

        # 解析参数
        parsed = ResponseParser.parse_tool_arguments(extracted)
        self.assertEqual(parsed[0]["name"], "get_weather")
        self.assertEqual(parsed[0]["arguments"]["city"], "Beijing")

    def test_retry_then_success_with_stats(self):
        """测试重试后成功并验证统计"""
        config = LLMConfig(
            model="gpt-4",
            api_key="sk-test",
            retry_max_attempts=3,
            retry_backoff_base=0.01,
            retry_max_backoff=1.0,
        )
        client = LLMClient(config=config)

        mock_ok = make_mock_openai_response(
            content="OK",
            prompt_tokens=5,
            completion_tokens=2,
            total_tokens=7,
        )

        with patch.object(client, "_get_client") as mock_get_client, \
             patch("src.infrastructure.llm_client.time.sleep"):
            mock_api = MagicMock()
            mock_api.chat.completions.create.side_effect = [
                Exception("fail 1"),
                mock_ok,
            ]
            mock_get_client.return_value = mock_api

            result = client.chat([LLMMessage(role="user", content="Hi")])

        self.assertEqual(result.content, "OK")

        stats = client.get_stats()
        self.assertEqual(stats["total_calls"], 2)
        self.assertEqual(stats["success_calls"], 1)
        self.assertEqual(stats["failed_calls"], 1)
        self.assertAlmostEqual(stats["success_rate"], 0.5)
        self.assertEqual(stats["total_tokens"], 7)

    def test_config_from_env_creates_working_client(self):
        """测试从环境变量创建的配置能正常工作"""
        env = {
            "DEFAULT_MODEL": "gpt-3.5-turbo",
            "OPENAI_API_KEY": "sk-env",
            "LLM_TEMPERATURE": "0.2",
            "LLM_MAX_TOKENS": "512",
            "API_RETRY_MAX_ATTEMPTS": "1",
        }
        with patch.dict("os.environ", env, clear=True):
            config = LLMConfig.from_env()

        client = LLMClient(config=config)
        self.assertEqual(client._config.model, "gpt-3.5-turbo")
        self.assertEqual(client._config.temperature, 0.2)
        self.assertEqual(client._config.max_tokens, 512)
        self.assertEqual(client._config.retry_max_attempts, 1)

        mock_resp = make_mock_openai_response(content="Env test")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_api = MagicMock()
            mock_api.chat.completions.create.return_value = mock_resp
            mock_get_client.return_value = mock_api

            result = client.chat([LLMMessage(role="user", content="Test")])

        self.assertEqual(result.content, "Env test")


if __name__ == "__main__":
    unittest.main()
