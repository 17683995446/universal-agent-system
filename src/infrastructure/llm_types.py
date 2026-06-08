"""
无上限上下文Agent系统 - LLM调用类型定义

标准化LLM请求、响应、配置等数据结构。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

from typing import Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import time


# =============================================================================
# 枚举
# =============================================================================

class MessageRole(str, Enum):
    """消息角色（LLM API标准）"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    """完成原因"""
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"


# =============================================================================
# LLM请求
# =============================================================================

@dataclass
class LLMMessage:
    """LLM消息（轻量，用于API调用）"""
    role: str
    content: str = ""
    name: Optional[str] = None
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class LLMRequest:
    """
    LLM请求

    Attributes:
        model: 模型名称
        messages: 消息列表
        temperature: 温度（0-2）
        max_tokens: 最大输出Token数
        top_p: 核采样概率
        stream: 是否流式输出
        stop: 停止序列
        tools: 工具定义列表
        metadata: 请求元数据
    """
    model: str
    messages: list[LLMMessage] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[list[str]] = None
    tools: Optional[list[dict]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_openai_kwargs(self) -> dict[str, Any]:
        """转换为OpenAI SDK的参数字典"""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in self.messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": self.stream,
        }
        if self.stop:
            kwargs["stop"] = self.stop
        if self.tools:
            kwargs["tools"] = self.tools
        return kwargs


# =============================================================================
# LLM响应
# =============================================================================

@dataclass
class LLMResponse:
    """
    LLM响应（统一格式，屏蔽底层差异）

    Attributes:
        content: 生成的内容
        model: 使用的模型
        finish_reason: 完成原因
        prompt_tokens: 输入Token数
        completion_tokens: 输出Token数
        total_tokens: 总Token数
        tool_calls: 工具调用列表
        metadata: 响应元数据
    """
    content: str = ""
    model: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tool_calls: Optional[list[dict]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_tool_call(self) -> bool:
        return self.finish_reason == "tool_calls" and self.tool_calls is not None

    @property
    def is_stopped(self) -> bool:
        return self.finish_reason == "stop"

    @property
    def is_length_limited(self) -> bool:
        return self.finish_reason == "length"

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tool_calls": self.tool_calls,
        }


# =============================================================================
# LLM配置
# =============================================================================

@dataclass
class LLMConfig:
    """
    LLM客户端配置

    Attributes:
        model: 默认模型名称
        api_key: API密钥
        base_url: API基础URL（支持自定义端点）
        temperature: 默认温度
        max_tokens: 默认最大输出Token
        timeout_connect: 连接超时（秒）
        timeout_read: 读取超时（秒）
        retry_max_attempts: 最大重试次数
        retry_backoff_base: 重试退避基数
        retry_max_backoff: 最大退避时间（秒）
    """
    model: str = "gpt-4"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout_connect: float = 10.0
    timeout_read: float = 60.0
    retry_max_attempts: int = 3
    retry_backoff_base: float = 2.0
    retry_max_backoff: float = 30.0

    def to_dict(self) -> dict[str, Any]:
        d = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_connect": self.timeout_connect,
            "timeout_read": self.timeout_read,
            "retry_max_attempts": self.retry_max_attempts,
            "retry_backoff_base": self.retry_backoff_base,
        }
        if self.base_url:
            d["base_url"] = self.base_url
        return d

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """从环境变量创建配置"""
        import os
        return cls(
            model=os.getenv("DEFAULT_MODEL", "gpt-4"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            timeout_connect=float(os.getenv("API_TIMEOUT_CONNECT", "10.0")),
            timeout_read=float(os.getenv("API_TIMEOUT_READ", "60.0")),
            retry_max_attempts=int(os.getenv("API_RETRY_MAX_ATTEMPTS", "3")),
            retry_backoff_base=float(os.getenv("API_RETRY_BACKOFF_BASE", "2.0")),
        )


# =============================================================================
# 调用统计
# =============================================================================

@dataclass
class LLMCallStats:
    """单次调用统计"""
    model: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    attempt: int = 1
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "duration_ms": round(self.duration_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "attempt": self.attempt,
            "success": self.success,
            "error": self.error,
        }
