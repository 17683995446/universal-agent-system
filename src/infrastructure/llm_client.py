"""
无上限上下文Agent系统 - LLM客户端

封装LLM调用，支持重试、超时、多模型。
基于OpenAI SDK实现，遵循第一性原理（不使用LangChain）。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

import time
import logging
from typing import Optional, Any, Callable

from src.infrastructure.llm_types import (
    LLMRequest, LLMResponse, LLMConfig, LLMMessage,
    LLMCallStats,
)
from src.core.exceptions import LLMClientError

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM客户端

    特点：
    - 基于OpenAI SDK
    - 指数退避重试
    - 连接/读取超时
    - 调用统计
    - 支持自定义端点（base_url）
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Args:
            config: LLM配置（None则从环境变量加载）
        """
        self._config = config or LLMConfig.from_env()
        self._client = None
        self._call_history: list[LLMCallStats] = []

    def _get_client(self):
        """懒加载OpenAI客户端"""
        if self._client is None:
            try:
                from openai import OpenAI
                kwargs: dict[str, Any] = {}
                if self._config.api_key:
                    kwargs["api_key"] = self._config.api_key
                if self._config.base_url:
                    kwargs["base_url"] = self._config.base_url
                self._client = OpenAI(**kwargs)
            except ImportError:
                raise LLMClientError(
                    self._config.model,
                    "openai package not installed. Install with: pip install openai",
                )
        return self._client

    def chat(
        self,
        messages: list[LLMMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        发送聊天请求（带重试）

        Args:
            messages: 消息列表
            model: 模型名称（None则使用配置默认值）
            temperature: 温度
            max_tokens: 最大输出Token
            **kwargs: 其他参数（传递给API）

        Returns:
            LLMResponse 统一响应

        Raises:
            LLMClientError: 调用失败（重试耗尽后）
        """
        request = LLMRequest(
            model=model or self._config.model,
            messages=messages,
            temperature=temperature if temperature is not None else self._config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self._config.max_tokens,
            **kwargs,
        )

        last_error = None
        for attempt in range(1, self._config.retry_max_attempts + 1):
            stats = LLMCallStats(
                model=request.model,
                attempt=attempt,
            )
            stats.start_time = time.time()

            try:
                response = self._call_once(request, stats)
                stats.success = True
                self._call_history.append(stats)
                return response

            except Exception as e:
                stats.success = False
                stats.error = str(e)
                last_error = e
                stats.end_time = time.time()
                stats.duration_ms = (stats.end_time - stats.start_time) * 1000
                self._call_history.append(stats)

                # 最后一次重试不等待
                if attempt < self._config.retry_max_attempts:
                    wait = self._calculate_backoff(attempt)
                    logger.warning(
                        f"LLM call failed (attempt {attempt}/{self._config.retry_max_attempts}): {e}. "
                        f"Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)

        raise LLMClientError(
            request.model,
            f"All {self._config.retry_max_attempts} attempts failed. Last error: {last_error}",
        )

    def _call_once(self, request: LLMRequest, stats: LLMCallStats) -> LLMResponse:
        """执行单次调用"""
        client = self._get_client()
        kwargs = request.to_openai_kwargs()

        # 设置超时
        kwargs["timeout"] = self._config.timeout_read

        # 调用API
        raw_response = client.chat.completions.create(**kwargs)

        # 解析响应
        response = self._parse_response(raw_response)

        # 更新统计
        stats.end_time = time.time()
        stats.duration_ms = (stats.end_time - stats.start_time) * 1000
        if raw_response.usage:
            stats.prompt_tokens = raw_response.usage.prompt_tokens
            stats.completion_tokens = raw_response.usage.completion_tokens
            stats.total_tokens = raw_response.usage.total_tokens

        return response

    def _parse_response(self, raw_response: Any) -> LLMResponse:
        """
        解析OpenAI API响应为统一格式

        Args:
            raw_response: OpenAI SDK的ChatCompletion对象

        Returns:
            LLMResponse
        """
        choice = raw_response.choices[0]
        message = choice.message

        # 提取工具调用
        tool_calls = None
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

        # 提取Token使用
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        if hasattr(raw_response, "usage") and raw_response.usage:
            prompt_tokens = raw_response.usage.prompt_tokens or 0
            completion_tokens = raw_response.usage.completion_tokens or 0
            total_tokens = raw_response.usage.total_tokens or 0

        return LLMResponse(
            content=message.content or "",
            model=raw_response.model,
            finish_reason=choice.finish_reason or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tool_calls=tool_calls,
        )

    def _calculate_backoff(self, attempt: int) -> float:
        """
        计算指数退避等待时间

        Args:
            attempt: 当前尝试次数（从1开始）

        Returns:
            等待秒数
        """
        wait = self._config.retry_backoff_base ** attempt
        return min(wait, self._config.retry_max_backoff)

    # =========================================================================
    # 统计与查询
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """获取调用统计"""
        if not self._call_history:
            return {"total_calls": 0}

        total = len(self._call_history)
        success = sum(1 for s in self._call_history if s.success)
        failed = total - success

        total_duration = sum(s.duration_ms for s in self._call_history)
        avg_duration = total_duration / total if total > 0 else 0

        total_tokens = sum(s.total_tokens for s in self._call_history)

        return {
            "total_calls": total,
            "success_calls": success,
            "failed_calls": failed,
            "success_rate": success / total if total > 0 else 0,
            "total_duration_ms": round(total_duration, 2),
            "avg_duration_ms": round(avg_duration, 2),
            "total_tokens": total_tokens,
            "last_call": self._call_history[-1].to_dict() if self._call_history else None,
        }

    def get_last_call_stats(self) -> Optional[LLMCallStats]:
        """获取最近一次调用统计"""
        return self._call_history[-1] if self._call_history else None

    def clear_history(self) -> int:
        """清空调用历史"""
        count = len(self._call_history)
        self._call_history.clear()
        return count

    # =========================================================================
    # 便捷方法
    # =========================================================================

    def chat_simple(
        self,
        system_prompt: str,
        user_message: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        简单聊天（返回纯文本）

        Args:
            system_prompt: 系统提示
            user_message: 用户消息
            model: 模型名称
            **kwargs: 其他参数

        Returns:
            助手回复文本
        """
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_message),
        ]
        response = self.chat(messages, model=model, **kwargs)
        return response.content
