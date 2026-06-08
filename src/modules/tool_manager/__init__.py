"""
无上限上下文Agent系统 - 工具管理模块

提供工具定义、注册、执行等完整能力。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-26
"""

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
    get_global_registry,
    register_global_tool,
    get_global_tool,
)
from src.modules.tool_manager.tool_executor import (
    ToolExecutor,
    ToolExecutionError,
    ToolTimeoutError,
    execute_tool,
)

__all__ = [
    # 枚举
    "ToolStatus",
    "ToolCategory",
    # 数据结构
    "ToolParameter",
    "ToolDefinition",
    "ToolResult",
    "ToolCallRequest",
    "ToolCallStats",
    # 注册器
    "ToolRegistry",
    "ToolNotFound",
    "ToolRegistrationError",
    "get_global_registry",
    "register_global_tool",
    "get_global_tool",
    # 执行器
    "ToolExecutor",
    "ToolExecutionError",
    "ToolTimeoutError",
    "execute_tool",
    # 工具函数
    "create_simple_tool",
    "tools_to_openai_schema",
]