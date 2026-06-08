"""
工具管理模块单元测试

测试范围：
- ToolParameter 参数定义
- ToolDefinition 工具定义与schema生成
- ToolResult 执行结果
- ToolCallRequest 调用请求
- ToolCallStats 调用统计
- ToolRegistry 注册器
- ToolExecutor 执行器

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-26
"""

import unittest
import json
import time

import sys
sys.path.insert(0, '/workspace')

from src.modules.tool_manager.tool_types import (
    ToolStatus,
    ToolCategory,
    ToolParameter,
    ToolDefinition,
    ToolResult,
    ToolCallRequest,
    ToolCallStats,
    create_simple_tool,
    tools_to_openai_schema,
)
from src.modules.tool_manager.tool_registry import (
    ToolRegistry,
    ToolNotFound,
    ToolRegistrationError,
)
from src.modules.tool_manager.tool_executor import (
    ToolExecutor,
    ToolExecutionError,
    ToolTimeoutError,
    execute_tool,
)


# =============================================================================
# ToolStatus 枚举测试
# =============================================================================

class TestToolStatus(unittest.TestCase):
    def test_values(self):
        self.assertEqual(ToolStatus.PENDING, "pending")
        self.assertEqual(ToolStatus.RUNNING, "running")
        self.assertEqual(ToolStatus.SUCCESS, "success")
        self.assertEqual(ToolStatus.FAILED, "failed")
        self.assertEqual(ToolStatus.TIMEOUT, "timeout")
        self.assertEqual(ToolStatus.CANCELLED, "cancelled")

    def test_is_str_enum(self):
        self.assertIsInstance(ToolStatus.SUCCESS, str)
        self.assertEqual(ToolStatus.SUCCESS.value, "success")


# =============================================================================
# ToolCategory 枚举测试
# =============================================================================

class TestToolCategory(unittest.TestCase):
    def test_values(self):
        self.assertEqual(ToolCategory.SEARCH, "search")
        self.assertEqual(ToolCategory.COMPUTATION, "computation")
        self.assertEqual(ToolCategory.ACTION, "action")
        self.assertEqual(ToolCategory.QUERY, "query")
        self.assertEqual(ToolCategory.UTILITY, "utility")


# =============================================================================
# ToolParameter 测试
# =============================================================================

class TestToolParameter(unittest.TestCase):
    def test_basic_creation(self):
        param = ToolParameter(name="city", type="string", description="City name")
        self.assertEqual(param.name, "city")
        self.assertEqual(param.type, "string")
        self.assertEqual(param.description, "City name")
        self.assertTrue(param.required)
        self.assertIsNone(param.default)
        self.assertIsNone(param.enum)

    def test_optional_param(self):
        param = ToolParameter(name="unit", type="string", required=False, default="celsius")
        self.assertFalse(param.required)
        self.assertEqual(param.default, "celsius")

    def test_enum_param(self):
        param = ToolParameter(
            name="format",
            type="string",
            enum=["json", "xml", "text"],
        )
        self.assertEqual(param.enum, ["json", "xml", "text"])

    def test_to_schema_basic(self):
        param = ToolParameter(name="query", type="string", description="Search query")
        schema = param.to_schema()
        self.assertEqual(schema["type"], "string")
        self.assertEqual(schema["description"], "Search query")

    def test_to_schema_with_enum(self):
        param = ToolParameter(name="format", type="string", enum=["json", "xml"])
        schema = param.to_schema()
        self.assertEqual(schema["enum"], ["json", "xml"])

    def test_to_schema_with_default(self):
        param = ToolParameter(name="limit", type="integer", default=10)
        schema = param.to_schema()
        self.assertEqual(schema["default"], 10)

    def test_to_schema_excludes_none_default(self):
        param = ToolParameter(name="query", type="string")
        schema = param.to_schema()
        self.assertNotIn("default", schema)


# =============================================================================
# ToolDefinition 测试
# =============================================================================

class TestToolDefinition(unittest.TestCase):
    def test_basic_creation(self):
        tool = ToolDefinition(
            name="get_weather",
            description="Get weather for a city",
        )
        self.assertEqual(tool.name, "get_weather")
        self.assertEqual(tool.description, "Get weather for a city")
        self.assertEqual(tool.parameters, [])
        self.assertEqual(tool.category, ToolCategory.UTILITY)
        self.assertEqual(tool.timeout, 30.0)
        self.assertIsNone(tool.handler)

    def test_with_parameters(self):
        tool = ToolDefinition(
            name="search",
            description="Search the web",
            parameters=[
                ToolParameter(name="query", type="string", required=True),
                ToolParameter(name="limit", type="integer", required=False),
            ],
        )
        self.assertEqual(len(tool.parameters), 2)
        self.assertEqual(tool.parameters[0].name, "query")
        self.assertEqual(tool.parameters[1].name, "limit")

    def test_to_openai_schema(self):
        tool = ToolDefinition(
            name="get_weather",
            description="Get weather information",
            parameters=[
                ToolParameter(name="city", type="string", description="City name"),
                ToolParameter(name="unit", type="string", required=False),
            ],
        )
        schema = tool.to_openai_schema()

        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "get_weather")
        self.assertEqual(schema["function"]["description"], "Get weather information")

        params = schema["function"]["parameters"]
        self.assertEqual(params["type"], "object")
        self.assertIn("city", params["properties"])
        self.assertIn("unit", params["properties"])
        self.assertEqual(params["required"], ["city"])

    def test_validate_arguments_success(self):
        tool = ToolDefinition(
            name="calc",
            description="Calculate",
            parameters=[
                ToolParameter(name="x", type="number", required=True),
                ToolParameter(name="y", type="number", required=True),
            ],
        )
        errors = tool.validate_arguments({"x": 1, "y": 2})
        self.assertEqual(errors, [])

    def test_validate_arguments_missing_required(self):
        tool = ToolDefinition(
            name="calc",
            description="Calculate",
            parameters=[
                ToolParameter(name="x", type="number", required=True),
                ToolParameter(name="y", type="number", required=True),
            ],
        )
        errors = tool.validate_arguments({"x": 1})
        self.assertEqual(len(errors), 1)
        self.assertIn("缺少必需参数: y", errors)

    def test_validate_arguments_missing_but_has_default(self):
        tool = ToolDefinition(
            name="calc",
            description="Calculate",
            parameters=[
                ToolParameter(name="x", type="number", required=True),
                ToolParameter(name="y", type="number", required=True, default=0),
            ],
        )
        errors = tool.validate_arguments({"x": 1})
        self.assertEqual(errors, [])  # y有默认值，不报错

    def test_validate_arguments_wrong_type(self):
        tool = ToolDefinition(
            name="calc",
            description="Calculate",
            parameters=[
                ToolParameter(name="x", type="number", required=True),
            ],
        )
        errors = tool.validate_arguments({"x": "not a number"})
        self.assertEqual(len(errors), 1)
        self.assertIn("应为number类型", errors[0])

    def test_validate_arguments_enum(self):
        tool = ToolDefinition(
            name="format",
            description="Format output",
            parameters=[
                ToolParameter(name="type", type="string", enum=["json", "xml"]),
            ],
        )
        errors = tool.validate_arguments({"type": "json"})
        self.assertEqual(errors, [])

        errors = tool.validate_arguments({"type": "yaml"})
        self.assertEqual(len(errors), 1)
        self.assertIn("不在允许范围内", errors[0])

    def test_validate_arguments_unknown_param_allowed(self):
        tool = ToolDefinition(
            name="test",
            description="Test",
            parameters=[
                ToolParameter(name="x", type="string"),
            ],
        )
        errors = tool.validate_arguments({"x": "val", "extra": "ignored"})
        self.assertEqual(errors, [])  # 未知参数允许但不校验

    def test_get_default_arguments(self):
        tool = ToolDefinition(
            name="search",
            description="Search",
            parameters=[
                ToolParameter(name="limit", type="integer", default=10),
                ToolParameter(name="offset", type="integer", default=0),
            ],
        )
        defaults = tool.get_default_arguments()
        self.assertEqual(defaults, {"limit": 10, "offset": 0})

    def test_get_default_arguments_empty(self):
        tool = ToolDefinition(name="test", description="Test")
        defaults = tool.get_default_arguments()
        self.assertEqual(defaults, {})


# =============================================================================
# ToolResult 测试
# =============================================================================

class TestToolResult(unittest.TestCase):
    def test_basic_creation(self):
        result = ToolResult(
            tool_call_id="call_123",
            tool_name="get_weather",
        )
        self.assertEqual(result.tool_call_id, "call_123")
        self.assertEqual(result.tool_name, "get_weather")
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertIsNone(result.output)
        self.assertIsNone(result.error)

    def test_success_result(self):
        result = ToolResult(
            tool_call_id="c1",
            tool_name="search",
            status=ToolStatus.SUCCESS,
            output={"results": ["a", "b"]},
        )
        self.assertTrue(result.is_success)
        self.assertFalse(result.is_failed)

    def test_failed_result(self):
        result = ToolResult(
            tool_call_id="c1",
            tool_name="search",
            status=ToolStatus.FAILED,
            error="API error",
        )
        self.assertFalse(result.is_success)
        self.assertTrue(result.is_failed)

    def test_timeout_result(self):
        result = ToolResult(
            tool_call_id="c1",
            tool_name="search",
            status=ToolStatus.TIMEOUT,
        )
        self.assertTrue(result.is_failed)

    def test_to_tool_message_content_string(self):
        result = ToolResult(
            tool_call_id="c1",
            tool_name="test",
            output="Hello world",
        )
        content = result.to_tool_message_content()
        self.assertEqual(content, "Hello world")

    def test_to_tool_message_content_dict(self):
        result = ToolResult(
            tool_call_id="c1",
            tool_name="test",
            output={"key": "value"},
        )
        content = result.to_tool_message_content()
        self.assertEqual(json.loads(content), {"key": "value"})

    def test_to_tool_message_content_list(self):
        result = ToolResult(
            tool_call_id="c1",
            tool_name="test",
            output=[1, 2, 3],
        )
        content = result.to_tool_message_content()
        self.assertEqual(json.loads(content), [1, 2, 3])

    def test_to_tool_message_content_error(self):
        result = ToolResult(
            tool_call_id="c1",
            tool_name="test",
            status=ToolStatus.FAILED,
            error="Something went wrong",
        )
        content = result.to_tool_message_content()
        self.assertEqual(content, "Error: Something went wrong")

    def test_to_dict(self):
        result = ToolResult(
            tool_call_id="c1",
            tool_name="test",
            status=ToolStatus.SUCCESS,
            output="OK",
            duration_ms=100.5,
            arguments={"x": 1},
        )
        d = result.to_dict()
        self.assertEqual(d["tool_call_id"], "c1")
        self.assertEqual(d["tool_name"], "test")
        self.assertEqual(d["status"], "success")
        self.assertEqual(d["output"], "OK")
        self.assertEqual(d["duration_ms"], 100.5)
        self.assertEqual(d["arguments"], {"x": 1})


# =============================================================================
# ToolCallRequest 测试
# =============================================================================

class TestToolCallRequest(unittest.TestCase):
    def test_basic_creation(self):
        request = ToolCallRequest(
            id="call_123",
            name="get_weather",
            arguments={"city": "Beijing"},
        )
        self.assertEqual(request.id, "call_123")
        self.assertEqual(request.name, "get_weather")
        self.assertEqual(request.arguments, {"city": "Beijing"})
        self.assertEqual(request.arguments_raw, "")

    def test_from_openai_tool_call(self):
        tool_call = {
            "id": "call_abc",
            "type": "function",
            "function": {
                "name": "search",
                "arguments": '{"query": "test", "limit": 10}',
            },
        }
        request = ToolCallRequest.from_openai_tool_call(tool_call)
        self.assertEqual(request.id, "call_abc")
        self.assertEqual(request.name, "search")
        self.assertEqual(request.arguments, {"query": "test", "limit": 10})
        self.assertEqual(request.arguments_raw, '{"query": "test", "limit": 10}')

    def test_from_openai_tool_call_invalid_json(self):
        tool_call = {
            "id": "call_xyz",
            "function": {
                "name": "test",
                "arguments": "not valid json",
            },
        }
        request = ToolCallRequest.from_openai_tool_call(tool_call)
        self.assertEqual(request.arguments, {"raw_arguments": "not valid json"})

    def test_to_dict(self):
        request = ToolCallRequest(
            id="c1",
            name="test",
            arguments={"x": 1},
            arguments_raw='{"x": 1}',
        )
        d = request.to_dict()
        self.assertEqual(d["id"], "c1")
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["arguments"], {"x": 1})


# =============================================================================
# ToolCallStats 测试
# =============================================================================

class TestToolCallStats(unittest.TestCase):
    def test_default_values(self):
        stats = ToolCallStats()
        self.assertEqual(stats.tool_name, "")
        self.assertEqual(stats.call_count, 0)
        self.assertEqual(stats.success_count, 0)
        self.assertEqual(stats.failed_count, 0)
        self.assertEqual(stats.total_duration_ms, 0.0)
        self.assertEqual(stats.avg_duration_ms, 0.0)

    def test_record_call_success(self):
        stats = ToolCallStats(tool_name="test")
        stats.record_call(100.0, True)
        self.assertEqual(stats.call_count, 1)
        self.assertEqual(stats.success_count, 1)
        self.assertEqual(stats.failed_count, 0)
        self.assertEqual(stats.total_duration_ms, 100.0)
        self.assertEqual(stats.avg_duration_ms, 100.0)

    def test_record_call_failed(self):
        stats = ToolCallStats(tool_name="test")
        stats.record_call(50.0, False)
        self.assertEqual(stats.call_count, 1)
        self.assertEqual(stats.success_count, 0)
        self.assertEqual(stats.failed_count, 1)

    def test_record_multiple_calls(self):
        stats = ToolCallStats(tool_name="test")
        stats.record_call(100.0, True)
        stats.record_call(200.0, True)
        stats.record_call(50.0, False)

        self.assertEqual(stats.call_count, 3)
        self.assertEqual(stats.success_count, 2)
        self.assertEqual(stats.failed_count, 1)
        self.assertEqual(stats.total_duration_ms, 350.0)
        self.assertAlmostEqual(stats.avg_duration_ms, 116.67, places=1)

    def test_success_rate(self):
        stats = ToolCallStats(tool_name="test")
        self.assertEqual(stats.success_rate, 0.0)

        stats.record_call(100.0, True)
        self.assertEqual(stats.success_rate, 1.0)

        stats.record_call(100.0, False)
        self.assertEqual(stats.success_rate, 0.5)

    def test_to_dict(self):
        stats = ToolCallStats(tool_name="test")
        stats.record_call(100.0, True)
        d = stats.to_dict()
        self.assertEqual(d["tool_name"], "test")
        self.assertEqual(d["call_count"], 1)
        self.assertEqual(d["success_rate"], 1.0)


# =============================================================================
# create_simple_tool 测试
# =============================================================================

class TestCreateSimpleTool(unittest.TestCase):
    def test_create_basic(self):
        def dummy_handler(city: str):
            return f"Weather in {city}"

        tool = create_simple_tool(
            "get_weather",
            "Get weather",
            dummy_handler,
            city="string",
            unit="string",
        )

        self.assertEqual(tool.name, "get_weather")
        self.assertEqual(tool.description, "Get weather")
        self.assertEqual(len(tool.parameters), 2)
        self.assertEqual(tool.parameters[0].name, "city")
        self.assertEqual(tool.parameters[0].type, "string")
        self.assertEqual(tool.parameters[1].name, "unit")
        self.assertEqual(tool.handler, dummy_handler)


# =============================================================================
# tools_to_openai_schema 测试
# =============================================================================

class TestToolsToOpenaiSchema(unittest.TestCase):
    def test_empty_list(self):
        schemas = tools_to_openai_schema([])
        self.assertEqual(schemas, [])

    def test_multiple_tools(self):
        tools = [
            ToolDefinition(name="t1", description="Tool 1"),
            ToolDefinition(name="t2", description="Tool 2"),
        ]
        schemas = tools_to_openai_schema(tools)
        self.assertEqual(len(schemas), 2)
        self.assertEqual(schemas[0]["function"]["name"], "t1")
        self.assertEqual(schemas[1]["function"]["name"], "t2")


# =============================================================================
# ToolRegistry 测试
# =============================================================================

class TestToolRegistry(unittest.TestCase):
    def test_register(self):
        registry = ToolRegistry()
        tool = ToolDefinition(name="test", description="Test tool")
        registry.register(tool)
        self.assertTrue(registry.has("test"))

    def test_register_duplicate_warning(self):
        registry = ToolRegistry()
        tool1 = ToolDefinition(name="test", description="Version 1")
        tool2 = ToolDefinition(name="test", description="Version 2")

        registry.register(tool1)
        registry.register(tool2)  # 应替换

        self.assertEqual(registry.get("test").description, "Version 2")

    def test_register_empty_name_raises(self):
        registry = ToolRegistry()
        tool = ToolDefinition(name="", description="Invalid")
        with self.assertRaises(ToolRegistrationError):
            registry.register(tool)

    def test_unregister(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="test", description="Test"))
        result = registry.unregister("test")
        self.assertTrue(result)
        self.assertFalse(registry.has("test"))

    def test_unregister_nonexistent(self):
        registry = ToolRegistry()
        result = registry.unregister("nonexistent")
        self.assertFalse(result)

    def test_register_batch(self):
        registry = ToolRegistry()
        tools = [
            ToolDefinition(name="t1", description="T1"),
            ToolDefinition(name="t2", description="T2"),
            ToolDefinition(name="", description="Invalid"),  # 会失败
        ]
        count = registry.register_batch(tools)
        self.assertEqual(count, 2)

    def test_get(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="test", description="Test"))
        tool = registry.get("test")
        self.assertEqual(tool.name, "test")

    def test_get_not_found_raises(self):
        registry = ToolRegistry()
        with self.assertRaises(ToolNotFound):
            registry.get("nonexistent")

    def test_get_optional(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="test", description="Test"))

        tool = registry.get_optional("test")
        self.assertIsNotNone(tool)

        tool = registry.get_optional("nonexistent")
        self.assertIsNone(tool)

    def test_list_all(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="t1", description="T1"))
        registry.register(ToolDefinition(name="t2", description="T2"))

        tools = registry.list_all()
        self.assertEqual(len(tools), 2)

    def test_list_names(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="t1", description="T1"))
        registry.register(ToolDefinition(name="t2", description="T2"))

        names = registry.list_names()
        self.assertEqual(set(names), {"t1", "t2"})

    def test_list_by_category(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="search", description="Search", category=ToolCategory.SEARCH))
        registry.register(ToolDefinition(name="calc", description="Calc", category=ToolCategory.COMPUTATION))
        registry.register(ToolDefinition(name="query", description="Query", category=ToolCategory.SEARCH))

        search_tools = registry.list_by_category(ToolCategory.SEARCH)
        self.assertEqual(len(search_tools), 2)

    def test_search(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="web_search", description="Search the web"))
        registry.register(ToolDefinition(name="file_search", description="Search files"))
        registry.register(ToolDefinition(name="calculator", description="Calculate math"))

        results = registry.search("search")
        self.assertEqual(len(results), 2)

        results = registry.search("calc")
        self.assertEqual(len(results), 1)

    def test_get_openai_schema(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="t1", description="T1"))
        registry.register(ToolDefinition(name="t2", description="T2"))

        schemas = registry.get_openai_schema()
        self.assertEqual(len(schemas), 2)

    def test_get_openai_schema_for_tools(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="t1", description="T1"))
        registry.register(ToolDefinition(name="t2", description="T2"))
        registry.register(ToolDefinition(name="t3", description="T3"))

        schemas = registry.get_openai_schema_for_tools(["t1", "t3"])
        self.assertEqual(len(schemas), 2)
        self.assertEqual(schemas[0]["function"]["name"], "t1")
        self.assertEqual(schemas[1]["function"]["name"], "t3")

    def test_record_call(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="test", description="Test"))

        registry.record_call("test", 100.0, True)
        stats = registry.get_stats("test")
        self.assertEqual(stats.call_count, 1)
        self.assertEqual(stats.success_count, 1)

    def test_get_summary(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="t1", description="T1", category=ToolCategory.SEARCH))
        registry.register(ToolDefinition(name="t2", description="T2", category=ToolCategory.COMPUTATION))

        summary = registry.get_summary()
        self.assertEqual(summary["total_tools"], 2)
        self.assertEqual(summary["categories"]["search"], 1)
        self.assertEqual(summary["categories"]["computation"], 1)

    def test_validate_tool_call(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="test",
            description="Test",
            parameters=[ToolParameter(name="x", type="string", required=True)],
        ))

        errors = registry.validate_tool_call("test", {"x": "value"})
        self.assertEqual(errors, [])

        errors = registry.validate_tool_call("test", {})
        self.assertEqual(len(errors), 1)

        errors = registry.validate_tool_call("nonexistent", {})
        self.assertEqual(len(errors), 1)

    def test_create_with_tools(self):
        tools = [
            ToolDefinition(name="t1", description="T1"),
            ToolDefinition(name="t2", description="T2"),
        ]
        registry = ToolRegistry.create_with_tools(tools)
        self.assertEqual(len(registry.list_all()), 2)


# =============================================================================
# ToolExecutor 测试
# =============================================================================

class TestToolExecutor(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.executor = ToolExecutor(registry=self.registry)

    def test_execute_tool_not_found(self):
        request = ToolCallRequest(id="c1", name="nonexistent", arguments={})
        result = self.executor.execute(request)
        self.assertEqual(result.status, ToolStatus.FAILED)
        self.assertIn("Tool not found", result.error)

    def test_execute_validation_failed(self):
        tool = ToolDefinition(
            name="test",
            description="Test",
            parameters=[ToolParameter(name="x", type="string", required=True)],
        )
        self.registry.register(tool)

        request = ToolCallRequest(id="c1", name="test", arguments={})
        result = self.executor.execute(request)
        self.assertEqual(result.status, ToolStatus.FAILED)
        self.assertIn("Validation failed", result.error)

    def test_execute_no_handler(self):
        tool = ToolDefinition(name="test", description="Test")
        self.registry.register(tool)

        request = ToolCallRequest(id="c1", name="test", arguments={})
        result = self.executor.execute(request)
        self.assertEqual(result.status, ToolStatus.FAILED)
        self.assertIn("No handler registered", result.error)

    def test_execute_success(self):
        def handler(query: str):
            return f"Result for: {query}"

        tool = ToolDefinition(
            name="search",
            description="Search",
            parameters=[ToolParameter(name="query", type="string")],
            handler=handler,
        )
        self.registry.register(tool)

        request = ToolCallRequest(id="c1", name="search", arguments={"query": "test"})
        result = self.executor.execute(request)

        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(result.output, "Result for: test")
        self.assertTrue(result.is_success)

    def test_execute_with_default_arguments(self):
        def handler(query: str, limit: int = 10):
            return f"Query: {query}, Limit: {limit}"

        tool = ToolDefinition(
            name="search",
            description="Search",
            parameters=[
                ToolParameter(name="query", type="string"),
                ToolParameter(name="limit", type="integer", default=10),
            ],
            handler=handler,
        )
        self.registry.register(tool)

        # 只提供query，limit使用默认值
        request = ToolCallRequest(id="c1", name="search", arguments={"query": "test"})
        result = self.executor.execute(request)

        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(result.output, "Query: test, Limit: 10")
        self.assertEqual(result.arguments["limit"], 10)

    def test_execute_handler_exception(self):
        def handler(x: int):
            raise ValueError("Handler error")

        tool = ToolDefinition(
            name="test",
            description="Test",
            parameters=[ToolParameter(name="x", type="integer")],
            handler=handler,
        )
        self.registry.register(tool)

        request = ToolCallRequest(id="c1", name="test", arguments={"x": 1})
        result = self.executor.execute(request)

        self.assertEqual(result.status, ToolStatus.FAILED)
        self.assertIn("Handler error", result.error)

    def test_execute_timeout(self):
        def slow_handler():
            time.sleep(5)
            return "Done"

        tool = ToolDefinition(
            name="slow",
            description="Slow tool",
            timeout=0.1,  # 100ms
            handler=slow_handler,
        )
        self.registry.register(tool)

        request = ToolCallRequest(id="c1", name="slow", arguments={})
        result = self.executor.execute(request)

        self.assertEqual(result.status, ToolStatus.TIMEOUT)

    def test_execute_custom_timeout(self):
        def handler():
            return "OK"

        tool = ToolDefinition(name="test", description="Test", handler=handler)
        self.registry.register(tool)

        request = ToolCallRequest(id="c1", name="test", arguments={})
        result = self.executor.execute(request, timeout=1.0)

        self.assertEqual(result.status, ToolStatus.SUCCESS)

    def test_execute_batch_sequential(self):
        def handler(x: int):
            return x * 2

        tool = ToolDefinition(
            name="double",
            description="Double",
            parameters=[ToolParameter(name="x", type="integer")],
            handler=handler,
        )
        self.registry.register(tool)

        requests = [
            ToolCallRequest(id="c1", name="double", arguments={"x": 1}),
            ToolCallRequest(id="c2", name="double", arguments={"x": 2}),
            ToolCallRequest(id="c3", name="double", arguments={"x": 3}),
        ]
        results = self.executor.execute_batch(requests, parallel=False)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].output, 2)
        self.assertEqual(results[1].output, 4)
        self.assertEqual(results[2].output, 6)

    def test_execute_batch_parallel(self):
        def handler(x: int):
            return x * 2

        tool = ToolDefinition(
            name="double",
            description="Double",
            parameters=[ToolParameter(name="x", type="integer")],
            handler=handler,
        )
        self.registry.register(tool)

        requests = [
            ToolCallRequest(id="c1", name="double", arguments={"x": 1}),
            ToolCallRequest(id="c2", name="double", arguments={"x": 2}),
        ]
        results = self.executor.execute_batch(requests, parallel=True)

        self.assertEqual(len(results), 2)
        # 结果按原始顺序排列
        self.assertEqual(results[0].tool_call_id, "c1")
        self.assertEqual(results[1].tool_call_id, "c2")

    def test_execute_from_llm_response(self):
        def handler(query: str):
            return f"Found: {query}"

        tool = ToolDefinition(
            name="search",
            description="Search",
            parameters=[ToolParameter(name="query", type="string")],
            handler=handler,
        )
        self.registry.register(tool)

        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": '{"query": "python"}',
                },
            },
        ]
        results = self.executor.execute_from_llm_response(tool_calls)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].output, "Found: python")

    def test_get_history(self):
        def handler():
            return "OK"

        tool = ToolDefinition(name="test", description="Test", handler=handler)
        self.registry.register(tool)

        self.executor.execute(ToolCallRequest(id="c1", name="test", arguments={}))
        self.executor.execute(ToolCallRequest(id="c2", name="test", arguments={}))

        history = self.executor.get_history()
        self.assertEqual(len(history), 2)

    def test_get_last_result(self):
        def handler():
            return "OK"

        tool = ToolDefinition(name="test", description="Test", handler=handler)
        self.registry.register(tool)

        self.executor.execute(ToolCallRequest(id="c1", name="test", arguments={}))
        last = self.executor.get_last_result()
        self.assertIsNotNone(last)
        self.assertEqual(last.tool_call_id, "c1")

    def test_clear_history(self):
        def handler():
            return "OK"

        tool = ToolDefinition(name="test", description="Test", handler=handler)
        self.registry.register(tool)

        self.executor.execute(ToolCallRequest(id="c1", name="test", arguments={}))
        count = self.executor.clear_history()
        self.assertEqual(count, 1)
        self.assertEqual(len(self.executor.get_history()), 0)

    def test_get_stats(self):
        def handler():
            return "OK"

        tool = ToolDefinition(name="test", description="Test", handler=handler)
        self.registry.register(tool)

        self.executor.execute(ToolCallRequest(id="c1", name="test", arguments={}))
        self.executor.execute(ToolCallRequest(id="c2", name="test", arguments={}))

        stats = self.executor.get_stats()
        self.assertEqual(stats["total_calls"], 2)
        self.assertEqual(stats["success_calls"], 2)

    def test_results_to_tool_messages(self):
        results = [
            ToolResult(tool_call_id="c1", tool_name="test", output="Result 1"),
            ToolResult(tool_call_id="c2", tool_name="test", status=ToolStatus.FAILED, error="Error"),
        ]
        messages = self.executor.results_to_tool_messages(results)

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "tool")
        self.assertEqual(messages[0]["tool_call_id"], "c1")
        self.assertEqual(messages[0]["content"], "Result 1")
        self.assertEqual(messages[1]["content"], "Error: Error")


# =============================================================================
# execute_tool 便捷函数测试
# =============================================================================

class TestExecuteTool(unittest.TestCase):
    def test_execute_tool_success(self):
        registry = ToolRegistry()
        def handler(x: int):
            return x * 2

        registry.register(ToolDefinition(
            name="double",
            description="Double",
            parameters=[ToolParameter(name="x", type="integer")],
            handler=handler,
        ))

        result = execute_tool("double", {"x": 5}, registry=registry)
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(result.output, 10)


# =============================================================================
# 集成测试
# =============================================================================

class TestToolIntegration(unittest.TestCase):
    def test_full_workflow(self):
        """完整工具调用流程"""
        # 1. 创建注册器
        registry = ToolRegistry()

        # 2. 定义工具
        def search_handler(query: str, limit: int = 5):
            return {"results": [f"Result {i} for {query}" for i in range(limit)]}

        search_tool = ToolDefinition(
            name="web_search",
            description="Search the web",
            category=ToolCategory.SEARCH,
            parameters=[
                ToolParameter(name="query", type="string", description="Search query"),
                ToolParameter(name="limit", type="integer", description="Max results", default=5),
            ],
            handler=search_handler,
        )

        # 3. 注册工具
        registry.register(search_tool)

        # 4. 获取OpenAI schema
        schema = registry.get_openai_schema()
        self.assertEqual(len(schema), 1)
        self.assertEqual(schema[0]["function"]["name"], "web_search")

        # 5. 模拟LLM返回tool_call
        tool_call = {
            "id": "call_search_001",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": '{"query": "Python tutorial"}',
            },
        }

        # 6. 执行工具
        executor = ToolExecutor(registry=registry)
        results = executor.execute_from_llm_response([tool_call])

        # 7. 验证结果
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, ToolStatus.SUCCESS)
        self.assertEqual(len(results[0].output["results"]), 5)

        # 8. 转换为tool消息
        messages = executor.results_to_tool_messages(results)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "tool")

        # 9. 检查统计
        summary = registry.get_summary()
        self.assertEqual(summary["total_calls"], 1)
        self.assertEqual(summary["success_calls"], 1)

    def test_multi_tool_workflow(self):
        """多工具协作流程"""
        registry = ToolRegistry()

        # 定义多个工具
        def add_handler(a: int, b: int):
            return a + b

        def multiply_handler(a: int, b: int):
            return a * b

        registry.register(ToolDefinition(
            name="add",
            description="Add two numbers",
            category=ToolCategory.COMPUTATION,
            parameters=[
                ToolParameter(name="a", type="integer"),
                ToolParameter(name="b", type="integer"),
            ],
            handler=add_handler,
        ))

        registry.register(ToolDefinition(
            name="multiply",
            description="Multiply two numbers",
            category=ToolCategory.COMPUTATION,
            parameters=[
                ToolParameter(name="a", type="integer"),
                ToolParameter(name="b", type="integer"),
            ],
            handler=multiply_handler,
        ))

        # 执行多个工具调用
        executor = ToolExecutor(registry=registry)
        tool_calls = [
            {"id": "c1", "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'}},
            {"id": "c2", "function": {"name": "multiply", "arguments": '{"a": 2, "b": 3}'}},
        ]
        results = executor.execute_from_llm_response(tool_calls)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].output, 5)  # 2+3
        self.assertEqual(results[1].output, 6)  # 2*3

        # 检查分类统计
        calc_tools = registry.list_by_category(ToolCategory.COMPUTATION)
        self.assertEqual(len(calc_tools), 2)


if __name__ == "__main__":
    unittest.main()