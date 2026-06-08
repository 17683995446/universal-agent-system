"""
无上限上下文Agent系统 - 工具注册器

管理工具的注册、查找、校验。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-26
"""

from typing import Optional, Any
import logging

from src.modules.tool_manager.tool_types import (
    ToolDefinition,
    ToolCategory,
    ToolStatus,
    ToolCallStats,
)
from src.core.exceptions import BaseAgentException

logger = logging.getLogger(__name__)


class ToolNotFound(BaseAgentException):
    """工具未找到异常"""

    def __init__(self, tool_name: str):
        super().__init__(
            f"Tool not found: {tool_name}",
            {"tool_name": tool_name}
        )
        self.tool_name = tool_name


class ToolRegistrationError(BaseAgentException):
    """工具注册错误"""

    def __init__(self, tool_name: str, error: str):
        super().__init__(
            f"Tool registration failed for {tool_name}: {error}",
            {"tool_name": tool_name, "error": error}
        )


class ToolRegistry:
    """
    工具注册器
    
    功能：
    - 注册/注销工具
    - 按名称/类别查找工具
    - 获取OpenAI tools schema
    - 统计工具调用情况
    
    Attributes:
        tools: 工具字典（name -> ToolDefinition）
        stats: 统计字典（name -> ToolCallStats）
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._stats: dict[str, ToolCallStats] = {}

    # =========================================================================
    # 注册与注销
    # =========================================================================

    def register(self, tool: ToolDefinition) -> None:
        """
        注册工具
        
        Args:
            tool: 工具定义
        
        Raises:
            ToolRegistrationError: 注册失败（名称冲突等）
        """
        if not tool.name:
            raise ToolRegistrationError("", "Tool name cannot be empty")

        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already registered, will be replaced")

        self._tools[tool.name] = tool
        self._stats[tool.name] = ToolCallStats(tool_name=tool.name)
        logger.info(f"Registered tool: {tool.name} (category: {tool.category.value})")

    def unregister(self, tool_name: str) -> bool:
        """
        注销工具
        
        Args:
            tool_name: 工具名称
        
        Returns:
            是否成功注销
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            del self._stats[tool_name]
            logger.info(f"Unregistered tool: {tool_name}")
            return True
        return False

    def register_batch(self, tools: list[ToolDefinition]) -> int:
        """
        批量注册工具
        
        Args:
            tools: 工具定义列表
        
        Returns:
            成功注册的数量
        """
        count = 0
        for tool in tools:
            try:
                self.register(tool)
                count += 1
            except ToolRegistrationError as e:
                logger.error(f"Failed to register tool: {e}")
        return count

    # =========================================================================
    # 查找与获取
    # =========================================================================

    def get(self, tool_name: str) -> ToolDefinition:
        """
        获取工具定义
        
        Args:
            tool_name: 工具名称
        
        Returns:
            ToolDefinition
        
        Raises:
            ToolNotFound: 工具不存在
        """
        if tool_name not in self._tools:
            raise ToolNotFound(tool_name)
        return self._tools[tool_name]

    def get_optional(self, tool_name: str) -> Optional[ToolDefinition]:
        """
        获取工具定义（不抛异常）
        
        Args:
            tool_name: 工具名称
        
        Returns:
            ToolDefinition或None
        """
        return self._tools.get(tool_name)

    def has(self, tool_name: str) -> bool:
        """检查工具是否存在"""
        return tool_name in self._tools

    def list_all(self) -> list[ToolDefinition]:
        """获取所有工具"""
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        """获取所有工具名称"""
        return list(self._tools.keys())

    def list_by_category(self, category: ToolCategory) -> list[ToolDefinition]:
        """按类别获取工具"""
        return [t for t in self._tools.values() if t.category == category]

    def search(self, keyword: str) -> list[ToolDefinition]:
        """
        搜索工具（按名称或描述）
        
        Args:
            keyword: 搜索关键词
        
        Returns:
            匹配的工具列表
        """
        keyword_lower = keyword.lower()
        return [
            t for t in self._tools.values()
            if keyword_lower in t.name.lower() or keyword_lower in t.description.lower()
        ]

    # =========================================================================
    # Schema生成
    # =========================================================================

    def get_openai_schema(self) -> list[dict[str, Any]]:
        """
        获取所有工具的OpenAI schema
        
        Returns:
            OpenAI tools格式列表
        """
        return [t.to_openai_schema() for t in self._tools.values()]

    def get_openai_schema_for_tools(self, tool_names: list[str]) -> list[dict[str, Any]]:
        """
        获取指定工具的OpenAI schema
        
        Args:
            tool_names: 工具名称列表
        
        Returns:
            OpenAI tools格式列表
        """
        schemas = []
        for name in tool_names:
            if name in self._tools:
                schemas.append(self._tools[name].to_openai_schema())
        return schemas

    # =========================================================================
    # 统计
    # =========================================================================

    def record_call(self, tool_name: str, duration_ms: float, success: bool) -> None:
        """记录一次工具调用"""
        if tool_name in self._stats:
            self._stats[tool_name].record_call(duration_ms, success)

    def get_stats(self, tool_name: str) -> Optional[ToolCallStats]:
        """获取单个工具的统计"""
        return self._stats.get(tool_name)

    def get_all_stats(self) -> dict[str, Any]:
        """获取所有工具的统计"""
        return {
            "total_tools": len(self._tools),
            "tools": {name: stats.to_dict() for name, stats in self._stats.items()},
        }

    def get_summary(self) -> dict[str, Any]:
        """获取注册器摘要"""
        total_calls = sum(s.call_count for s in self._stats.values())
        total_success = sum(s.success_count for s in self._stats.values())
        total_failed = sum(s.failed_count for s in self._stats.values())

        return {
            "total_tools": len(self._tools),
            "total_calls": total_calls,
            "success_calls": total_success,
            "failed_calls": total_failed,
            "success_rate": total_success / total_calls if total_calls > 0 else 0,
            "categories": {
                cat.value: len(self.list_by_category(cat))
                for cat in ToolCategory
            },
        }

    def clear_stats(self) -> None:
        """清空统计"""
        for stats in self._stats.values():
            stats.call_count = 0
            stats.success_count = 0
            stats.failed_count = 0
            stats.total_duration_ms = 0.0
            stats.avg_duration_ms = 0.0

    # =========================================================================
    # 校验
    # =========================================================================

    def validate_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> list[str]:
        """
        校验工具调用参数
        
        Args:
            tool_name: 工具名称
            arguments: 参数字典
        
        Returns:
            错误信息列表，空列表表示校验通过
        """
        tool = self.get_optional(tool_name)
        if tool is None:
            return [f"Tool not found: {tool_name}"]

        return tool.validate_arguments(arguments)

    # =========================================================================
    # 工厂方法
    # =========================================================================

    @classmethod
    def create_with_tools(cls, tools: list[ToolDefinition]) -> "ToolRegistry":
        """
        创建并批量注册工具
        
        Args:
            tools: 工具定义列表
        
        Returns:
            ToolRegistry实例
        """
        registry = cls()
        registry.register_batch(tools)
        return registry


# =============================================================================
# 全局注册器（可选）
# =============================================================================

_global_registry: Optional[ToolRegistry] = None


def get_global_registry() -> ToolRegistry:
    """获取全局工具注册器"""
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_global_tool(tool: ToolDefinition) -> None:
    """注册到全局注册器"""
    get_global_registry().register(tool)


def get_global_tool(tool_name: str) -> ToolDefinition:
    """从全局注册器获取工具"""
    return get_global_registry().get(tool_name)