"""
无上限上下文Agent系统 - 响应解析器

统一解析多模型响应格式，提取内容、工具调用、Token使用等信息。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

import json
import re
from typing import Optional, Any

from src.infrastructure.llm_types import LLMResponse


class ResponseParser:
    """
    响应解析器

    将不同格式的LLM响应统一解析为LLMResponse。
    """

    @staticmethod
    def parse_openai(raw: Any) -> LLMResponse:
        """
        解析OpenAI格式响应

        Args:
            raw: OpenAI SDK的ChatCompletion对象

        Returns:
            LLMResponse
        """
        choice = raw.choices[0]
        message = choice.message

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

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        if hasattr(raw, "usage") and raw.usage:
            prompt_tokens = raw.usage.prompt_tokens or 0
            completion_tokens = raw.usage.completion_tokens or 0
            total_tokens = raw.usage.total_tokens or 0

        return LLMResponse(
            content=message.content or "",
            model=raw.model,
            finish_reason=choice.finish_reason or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tool_calls=tool_calls,
        )

    @staticmethod
    def parse_dict(data: dict[str, Any]) -> LLMResponse:
        """
        解析字典格式响应（用于测试和模拟）

        Args:
            data: 响应字典

        Returns:
            LLMResponse
        """
        return LLMResponse(
            content=data.get("content", ""),
            model=data.get("model", ""),
            finish_reason=data.get("finish_reason", "stop"),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            tool_calls=data.get("tool_calls"),
        )

    @staticmethod
    def extract_tool_calls(response: LLMResponse) -> list[dict[str, Any]]:
        """
        提取工具调用信息

        Args:
            response: LLM响应

        Returns:
            工具调用列表
        """
        if not response.is_tool_call or not response.tool_calls:
            return []
        return response.tool_calls

    @staticmethod
    def parse_tool_arguments(tool_calls: list[dict]) -> list[dict[str, Any]]:
        """
        解析工具调用参数（JSON字符串→字典）

        Args:
            tool_calls: 工具调用列表

        Returns:
            解析后的工具调用列表
        """
        parsed = []
        for tc in tool_calls:
            func = tc.get("function", {})
            args_str = func.get("arguments", "{}")
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {"raw_arguments": args_str}

            parsed.append({
                "id": tc.get("id", ""),
                "name": func.get("name", ""),
                "arguments": args,
            })
        return parsed

    @staticmethod
    def extract_json_from_content(content: str) -> Optional[dict[str, Any]]:
        """
        从响应内容中提取JSON块

        支持格式：
        - ```json ... ```
        - { ... }

        Args:
            content: 响应文本

        Returns:
            解析后的字典，提取失败返回None
        """
        # 尝试提取 ```json ... ``` 块
        json_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
        match = re.search(json_block_pattern, content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试提取 { ... } 块
        brace_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        match = re.search(brace_pattern, content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None
