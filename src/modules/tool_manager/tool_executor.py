"""
无上限上下文Agent系统 - 工具执行器

执行工具调用，处理超时、错误、结果回传。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-26
"""

import time
import logging
import threading
import concurrent.futures
from typing import Optional, Any, Callable

from src.modules.tool_manager.tool_types import (
    ToolDefinition,
    ToolCallRequest,
    ToolResult,
    ToolStatus,
    ToolCallStats,
)
from src.modules.tool_manager.tool_registry import ToolRegistry, ToolNotFound
from src.core.exceptions import BaseAgentException

logger = logging.getLogger(__name__)


class ToolExecutionError(BaseAgentException):
    """工具执行错误"""

    def __init__(self, tool_name: str, error: str):
        super().__init__(
            f"Tool execution failed: {tool_name} - {error}",
            {"tool_name": tool_name, "error": error}
        )
        self.tool_name = tool_name


class ToolTimeoutError(BaseAgentException):
    """工具执行超时"""

    def __init__(self, tool_name: str, timeout: float):
        super().__init__(
            f"Tool execution timeout: {tool_name} (timeout={timeout}s)",
            {"tool_name": tool_name, "timeout": timeout}
        )
        self.tool_name = tool_name
        self.timeout = timeout


class ToolExecutor:
    """
    工具执行器
    
    功能：
    - 执行单个/批量工具调用
    - 超时控制
    - 错误处理与回退
    - 结果统计
    
    Attributes:
        registry: 工具注册器
        default_timeout: 默认超时时间
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        default_timeout: float = 30.0,
    ):
        """
        Args:
            registry: 工具注册器（None则使用全局注册器）
            default_timeout: 默认超时时间（秒）
        """
        self._registry = registry or ToolRegistry()
        self._default_timeout = default_timeout
        self._call_history: list[ToolResult] = []

    # =========================================================================
    # 单个执行
    # =========================================================================

    def execute(
        self,
        request: ToolCallRequest,
        timeout: Optional[float] = None,
    ) -> ToolResult:
        """
        执行单个工具调用
        
        Args:
            request: 工具调用请求
            timeout: 超时时间（None则使用工具默认值）
        
        Returns:
            ToolResult 执行结果
        """
        tool_name = request.name
        tool_call_id = request.id

        # 获取工具定义
        tool = self._registry.get_optional(tool_name)
        if tool is None:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                status=ToolStatus.FAILED,
                error=f"Tool not found: {tool_name}",
                arguments=request.arguments,
            )

        # 确定超时时间
        actual_timeout = timeout if timeout is not None else tool.timeout

        # 校验参数
        validation_errors = tool.validate_arguments(request.arguments)
        if validation_errors:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                status=ToolStatus.FAILED,
                error=f"Validation failed: {validation_errors}",
                arguments=request.arguments,
            )

        # 合入默认参数
        final_args = {**tool.get_default_arguments(), **request.arguments}

        # 执行
        start_time = time.time()
        result = ToolResult(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=final_args,
        )

        try:
            output = self._execute_with_timeout(
                tool=tool,
                arguments=final_args,
                timeout=actual_timeout,
            )
            result.status = ToolStatus.SUCCESS
            result.output = output

        except ToolTimeoutError as e:
            result.status = ToolStatus.TIMEOUT
            result.error = str(e)

        except ToolExecutionError as e:
            result.status = ToolStatus.FAILED
            result.error = str(e)

        except Exception as e:
            result.status = ToolStatus.FAILED
            result.error = f"Unexpected error: {e}"
            logger.exception(f"Tool execution error: {tool_name}")

        # 记录耗时
        end_time = time.time()
        result.duration_ms = (end_time - start_time) * 1000

        # 更新统计
        self._registry.record_call(tool_name, result.duration_ms, result.is_success)

        # 保存历史
        self._call_history.append(result)

        return result

    def _execute_with_timeout(
        self,
        tool: ToolDefinition,
        arguments: dict[str, Any],
        timeout: float,
    ) -> Any:
        """
        带超时控制的执行
        
        Args:
            tool: 工具定义
            arguments: 参数
            timeout: 超时时间
        
        Returns:
            执行结果
        
        Raises:
            ToolTimeoutError: 执行超时
            ToolExecutionError: 执行失败
        """
        handler = tool.handler
        if handler is None:
            raise ToolExecutionError(tool.name, "No handler registered for this tool")

        # 使用线程池实现超时
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(handler, **arguments)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise ToolTimeoutError(tool.name, timeout)
            except Exception as e:
                raise ToolExecutionError(tool.name, str(e))

    # =========================================================================
    # 批量执行
    # =========================================================================

    def execute_batch(
        self,
        requests: list[ToolCallRequest],
        timeout: Optional[float] = None,
        parallel: bool = True,
    ) -> list[ToolResult]:
        """
        执行批量工具调用
        
        Args:
            requests: 工具调用请求列表
            timeout: 单个调用的超时时间
            parallel: 是否并行执行
        
        Returns:
            ToolResult列表
        """
        if parallel:
            return self._execute_parallel(requests, timeout)
        else:
            return self._execute_sequential(requests, timeout)

    def _execute_parallel(
        self,
        requests: list[ToolCallRequest],
        timeout: Optional[float],
    ) -> list[ToolResult]:
        """并行执行"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(requests)) as executor:
            futures = [
                executor.submit(self.execute, req, timeout)
                for req in requests
            ]
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
            # 按原始顺序排序
            return sorted(results, key=lambda r: requests.index(
                next(req for req in requests if req.id == r.tool_call_id)
            ))

    def _execute_sequential(
        self,
        requests: list[ToolCallRequest],
        timeout: Optional[float],
    ) -> list[ToolResult]:
        """顺序执行"""
        return [self.execute(req, timeout) for req in requests]

    # =========================================================================
    # 从LLM响应执行
    # =========================================================================

    def execute_from_llm_response(
        self,
        tool_calls: list[dict[str, Any]],
        timeout: Optional[float] = None,
    ) -> list[ToolResult]:
        """
        从LLM响应的tool_calls执行
        
        Args:
            tool_calls: OpenAI格式的tool_calls列表
            timeout: 超时时间
        
        Returns:
            ToolResult列表
        """
        requests = [
            ToolCallRequest.from_openai_tool_call(tc)
            for tc in tool_calls
        ]
        return self.execute_batch(requests, timeout)

    # =========================================================================
    # 统计与历史
    # =========================================================================

    def get_history(self) -> list[ToolResult]:
        """获取执行历史"""
        return self._call_history.copy()

    def get_last_result(self) -> Optional[ToolResult]:
        """获取最近一次执行结果"""
        return self._call_history[-1] if self._call_history else None

    def clear_history(self) -> int:
        """清空执行历史"""
        count = len(self._call_history)
        self._call_history.clear()
        return count

    def get_stats(self) -> dict[str, Any]:
        """获取执行统计"""
        if not self._call_history:
            return {"total_calls": 0}

        total = len(self._call_history)
        success = sum(1 for r in self._call_history if r.is_success)
        failed = total - success
        total_duration = sum(r.duration_ms for r in self._call_history)
        avg_duration = total_duration / total

        return {
            "total_calls": total,
            "success_calls": success,
            "failed_calls": failed,
            "success_rate": success / total,
            "total_duration_ms": round(total_duration, 2),
            "avg_duration_ms": round(avg_duration, 2),
        }

    # =========================================================================
    # 结果转换为LLM消息
    # =========================================================================

    def results_to_tool_messages(
        self,
        results: list[ToolResult],
    ) -> list[dict[str, Any]]:
        """
        将执行结果转换为tool消息格式
        
        用于回传给LLM继续推理。
        
        Args:
            results: 执行结果列表
        
        Returns:
            OpenAI格式的tool消息列表
        """
        messages = []
        for result in results:
            messages.append({
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": result.to_tool_message_content(),
            })
        return messages


# =============================================================================
# 便捷函数
# =============================================================================

def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    registry: Optional[ToolRegistry] = None,
    timeout: Optional[float] = None,
) -> ToolResult:
    """
    便捷执行单个工具
    
    Args:
        tool_name: 工具名称
        arguments: 参数
        registry: 注册器
        timeout: 超时
    
    Returns:
        ToolResult
    """
    executor = ToolExecutor(registry=registry)
    request = ToolCallRequest(
        id=f"call_{tool_name}_{time.time_ns()}",
        name=tool_name,
        arguments=arguments,
    )
    return executor.execute(request, timeout=timeout)