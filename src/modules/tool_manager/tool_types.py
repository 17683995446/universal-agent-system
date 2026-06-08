"""
无上限上下文Agent系统 - 工具类型定义

定义工具调用所需的数据结构：工具定义、执行结果、状态枚举。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-26
"""

from typing import Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time


# =============================================================================
# 枚举
# =============================================================================

class ToolStatus(str, Enum):
    """工具执行状态"""
    PENDING = "pending"       # 待执行
    RUNNING = "running"       # 执行中
    SUCCESS = "success"       # 成功
    FAILED = "failed"         # 失败
    TIMEOUT = "timeout"       # 超时
    CANCELLED = "cancelled"   # 取消


class ToolCategory(str, Enum):
    """工具类别"""
    SEARCH = "search"         # 搜索类（网络搜索、文档检索）
    COMPUTATION = "computation"  # 计算类（数学运算、代码执行）
    ACTION = "action"         # 动作类（API调用、文件操作）
    QUERY = "query"           # 查询类（数据库查询、状态检查）
    UTILITY = "utility"       # 工具类（格式转换、文本处理）


# =============================================================================
# 工具定义
# =============================================================================

@dataclass
class ToolParameter:
    """
    工具参数定义
    
    Attributes:
        name: 参数名称
        type: 参数类型（string/number/integer/boolean/array/object）
        description: 参数描述
        required: 是否必需
        default: 默认值
        enum: 可选值列表
    """
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[list[Any]] = None

    def to_schema(self) -> dict[str, Any]:
        """转换为OpenAI工具参数schema"""
        schema: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ToolDefinition:
    """
    工具定义
    
    定义一个可供LLM调用的工具，包含名称、描述、参数schema等。
    
    Attributes:
        name: 工具名称（唯一标识）
        description: 工具描述（供LLM理解用途）
        parameters: 参数列表
        category: 工具类别
        timeout: 执行超时（秒）
        handler: 执行函数（可选，运行时注入）
        metadata: 元数据
    """
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    category: ToolCategory = ToolCategory.UTILITY
    timeout: float = 30.0
    handler: Optional[Callable[..., Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_openai_schema(self) -> dict[str, Any]:
        """
        转换为OpenAI工具定义格式
        
        Returns:
            OpenAI tools格式的工具定义
        """
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        """
        校验参数
        
        Args:
            arguments: 待校验的参数字典
        
        Returns:
            错误信息列表，空列表表示校验通过
        """
        errors = []

        # 检查必需参数
        for param in self.parameters:
            if param.required and param.name not in arguments:
                if param.default is not None:
                    # 有默认值，不报错
                    continue
                errors.append(f"缺少必需参数: {param.name}")

        # 检查参数类型
        for key, value in arguments.items():
            param_def = None
            for p in self.parameters:
                if p.name == key:
                    param_def = p
                    break

            if param_def is None:
                # 未知参数，允许但不校验
                continue

            # 类型校验
            expected_type = param_def.type
            if expected_type == "string" and not isinstance(value, str):
                errors.append(f"参数 {key} 应为string类型")
            elif expected_type == "number" and not isinstance(value, (int, float)):
                errors.append(f"参数 {key} 应为number类型")
            elif expected_type == "integer" and not isinstance(value, int):
                errors.append(f"参数 {key} 应为integer类型")
            elif expected_type == "boolean" and not isinstance(value, bool):
                errors.append(f"参数 {key} 应为boolean类型")
            elif expected_type == "array" and not isinstance(value, list):
                errors.append(f"参数 {key} 应为array类型")
            elif expected_type == "object" and not isinstance(value, dict):
                errors.append(f"参数 {key} 应为object类型")

            # enum校验
            if param_def.enum and value not in param_def.enum:
                errors.append(f"参数 {key} 值 {value} 不在允许范围内: {param_def.enum}")

        return errors

    def get_default_arguments(self) -> dict[str, Any]:
        """获取默认参数值"""
        defaults = {}
        for param in self.parameters:
            if param.default is not None:
                defaults[param.name] = param.default
        return defaults


# =============================================================================
# 工具执行结果
# =============================================================================

@dataclass
class ToolResult:
    """
    工具执行结果
    
    Attributes:
        tool_call_id: 工具调用ID（对应LLM响应中的tool_call.id）
        tool_name: 工具名称
        status: 执行状态
        output: 执行输出（成功时的结果）
        error: 错误信息（失败时的错误）
        duration_ms: 执行耗时（毫秒）
        arguments: 调用参数
        metadata: 元数据
    """
    tool_call_id: str
    tool_name: str
    status: ToolStatus = ToolStatus.SUCCESS
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    arguments: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == ToolStatus.SUCCESS

    @property
    def is_failed(self) -> bool:
        return self.status in (ToolStatus.FAILED, ToolStatus.TIMEOUT, ToolStatus.CANCELLED)

    def to_tool_message_content(self) -> str:
        """
        转换为tool消息的content格式
        
        OpenAI要求tool消息的content为字符串。
        """
        if self.is_success:
            if isinstance(self.output, str):
                return self.output
            elif isinstance(self.output, (dict, list)):
                return json.dumps(self.output, ensure_ascii=False)
            else:
                return str(self.output)
        else:
            return f"Error: {self.error or 'Unknown error'}"

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
            "arguments": self.arguments,
        }


# =============================================================================
# 工具调用请求
# =============================================================================

@dataclass
class ToolCallRequest:
    """
    工具调用请求
    
    从LLM响应中提取的工具调用信息。
    
    Attributes:
        id: 工具调用ID
        name: 工具名称
        arguments: 参数（已解析为字典）
        arguments_raw: 原始参数字符串
    """
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    arguments_raw: str = ""

    @classmethod
    def from_openai_tool_call(cls, tool_call: dict[str, Any]) -> "ToolCallRequest":
        """
        从OpenAI tool_call格式创建
        
        Args:
            tool_call: OpenAI响应中的tool_call对象
        
        Returns:
            ToolCallRequest实例
        """
        func = tool_call.get("function", {})
        args_raw = func.get("arguments", "{}")

        # 解析参数
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            args = {"raw_arguments": args_raw}

        return cls(
            id=tool_call.get("id", ""),
            name=func.get("name", ""),
            arguments=args,
            arguments_raw=args_raw,
        )

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "arguments_raw": self.arguments_raw,
        }


# =============================================================================
# 工具调用统计
# =============================================================================

@dataclass
class ToolCallStats:
    """
    工具调用统计
    
    Attributes:
        tool_name: 工具名称
        call_count: 调用次数
        success_count: 成功次数
        failed_count: 失败次数
        total_duration_ms: 总耗时
        avg_duration_ms: 平均耗时
        last_call_time: 最后调用时间
    """
    tool_name: str = ""
    call_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    last_call_time: float = 0.0

    def record_call(self, duration_ms: float, success: bool) -> None:
        """记录一次调用"""
        self.call_count += 1
        self.total_duration_ms += duration_ms
        self.avg_duration_ms = self.total_duration_ms / self.call_count
        self.last_call_time = time.time()

        if success:
            self.success_count += 1
        else:
            self.failed_count += 1

    @property
    def success_rate(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.success_count / self.call_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "call_count": self.call_count,
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "success_rate": round(self.success_rate, 4),
            "total_duration_ms": round(self.total_duration_ms, 2),
            "avg_duration_ms": round(self.avg_duration_ms, 2),
        }


# =============================================================================
# 工具函数
# =============================================================================

def create_simple_tool(
    name: str,
    description: str,
    handler: Callable[..., Any],
    **param_kwargs,
) -> ToolDefinition:
    """
    快捷创建简单工具
    
    Args:
        name: 工具名称
        description: 工具描述
        handler: 执行函数
        **param_kwargs: 参数定义（name=type格式）
    
    Returns:
        ToolDefinition实例
    
    Example:
        tool = create_simple_tool(
            "get_weather",
            "Get weather for a city",
            weather_api,
            city="string",
            unit="string",
        )
    """
    parameters = []
    for param_name, param_type in param_kwargs.items():
        parameters.append(ToolParameter(
            name=param_name,
            type=param_type,
            description=f"{param_name} parameter",
            required=True,
        ))

    return ToolDefinition(
        name=name,
        description=description,
        parameters=parameters,
        handler=handler,
    )


def tools_to_openai_schema(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """
    将工具列表转换为OpenAI tools格式
    
    Args:
        tools: 工具定义列表
    
    Returns:
        OpenAI tools格式列表
    """
    return [t.to_openai_schema() for t in tools]