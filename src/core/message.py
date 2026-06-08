"""
无上限上下文Agent系统 - Message模块

标准化Message结构体的完整实现，包含序列化、验证、工具函数

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

from typing import Optional, Any
from dataclasses import dataclass, field, asdict
from typing_extensions import Self
from datetime import datetime
from enum import Enum
import json
import copy

from src.core.utils.id_generator import generate_msg_id


class Role(str, Enum):
    """消息角色枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MessageType(str, Enum):
    """消息类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


@dataclass
class Message:
    """
    标准化消息结构体
    
    核心字段（必须）：
        - role: 消息角色（system/user/assistant/tool）
        - content: 消息内容
    
    系统字段（自动生成）：
        - msg_id: 消息唯一标识
        - timestamp: 创建时间戳
    
    计算字段（后续填充）：
        - token_count: Token数量
    
    扩展字段（可选）：
        - msg_type: 消息类型
        - metadata: 元数据字典
        - parent_id: 父消息ID（用于回复链）
        - session_id: 所属会话ID
    
    Attributes:
        role: 消息角色
        content: 消息内容
        msg_id: 消息唯一标识
        timestamp: 时间戳
        token_count: Token数量（计算后填充）
        msg_type: 消息类型
        metadata: 元数据（图片URL、工具调用ID等）
        parent_id: 父消息ID
        session_id: 所属会话ID
    """
    # 核心字段
    role: Role
    content: str
    
    # 系统字段（自动生成）
    msg_id: str = field(default_factory=generate_msg_id)
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 计算字段
    token_count: int = 0
    
    # 扩展字段
    msg_type: MessageType = MessageType.TEXT
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保role是Role枚举
        if isinstance(self.role, str):
            self.role = Role(self.role)
        # 确保msg_type是MessageType枚举
        if isinstance(self.msg_type, str):
            self.msg_type = MessageType(self.msg_type)
    
    # =========================================================================
    # 序列化方法
    # =========================================================================
    
    def to_dict(self) -> dict[str, Any]:
        """
        转换为字典
        
        Returns:
            字典表示
        """
        return {
            "role": self.role.value,
            "content": self.content,
            "msg_id": self.msg_id,
            "timestamp": self.timestamp.isoformat(),
            "token_count": self.token_count,
            "msg_type": self.msg_type.value,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "session_id": self.session_id,
        }
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """
        转换为JSON字符串
        
        Args:
            indent: 缩进空格数
            
        Returns:
            JSON字符串
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """
        从字典创建Message
        
        Args:
            data: 字典数据
            
        Returns:
            Message实例
        """
        # 处理时间戳
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()
        
        return cls(
            role=Role(data.get("role", "user")),
            content=data.get("content", ""),
            msg_id=data.get("msg_id", generate_msg_id()),
            timestamp=timestamp,
            token_count=data.get("token_count", 0),
            msg_type=MessageType(data.get("msg_type", "text")),
            metadata=data.get("metadata", {}),
            parent_id=data.get("parent_id"),
            session_id=data.get("session_id"),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """
        从JSON字符串创建Message
        
        Args:
            json_str: JSON字符串
            
        Returns:
            Message实例
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    # =========================================================================
    # 验证方法
    # =========================================================================
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        验证消息有效性
        
        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []
        
        # 验证role
        if not self.role:
            errors.append("role不能为空")
        elif not isinstance(self.role, Role):
            errors.append(f"role必须是Role枚举，当前: {type(self.role)}")
        
        # 验证content
        if self.content is None:
            errors.append("content不能为None")
        elif not isinstance(self.content, str):
            errors.append(f"content必须是字符串，当前: {type(self.content)}")
        
        # 验证token_count
        if self.token_count < 0:
            errors.append(f"token_count不能为负数: {self.token_count}")
        
        # 验证msg_id
        if not self.msg_id:
            errors.append("msg_id不能为空")
        
        # 验证timestamp
        if not isinstance(self.timestamp, datetime):
            errors.append(f"timestamp必须是datetime类型")
        
        return len(errors) == 0, errors
    
    def is_valid(self) -> bool:
        """
        快速检查消息是否有效
        
        Returns:
            是否有效
        """
        valid, _ = self.validate()
        return valid
    
    # =========================================================================
    # 工具方法
    # =========================================================================
    
    def copy(self) -> Self:
        """
        创建消息的深拷贝
        
        Returns:
            新的Message实例
        """
        return copy.deepcopy(self)
    
    def update_token_count(self, count: int) -> Self:
        """
        更新Token计数
        
        Args:
            count: Token数量
            
        Returns:
            self（支持链式调用）
        """
        self.token_count = count
        return self
    
    def update_content(self, content: str) -> Self:
        """
        更新内容（重置token_count）
        
        Args:
            content: 新内容
            
        Returns:
            self（支持链式调用）
        """
        self.content = content
        self.token_count = 0  # 重置Token计数
        return self
    
    def add_metadata(self, key: str, value: Any) -> Self:
        """
        添加元数据
        
        Args:
            key: 键
            value: 值
            
        Returns:
            self（支持链式调用）
        """
        self.metadata[key] = value
        return self
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        获取元数据
        
        Args:
            key: 键
            default: 默认值
            
        Returns:
            元数据值
        """
        return self.metadata.get(key, default)
    
    def is_system(self) -> bool:
        """是否为系统消息"""
        return self.role == Role.SYSTEM
    
    def is_user(self) -> bool:
        """是否为用户消息"""
        return self.role == Role.USER
    
    def is_assistant(self) -> bool:
        """是否为助手消息"""
        return self.role == Role.ASSISTANT
    
    def is_tool(self) -> bool:
        """是否为工具消息"""
        return self.role == Role.TOOL
    
    def is_text(self) -> bool:
        """是否为文本类型"""
        return self.msg_type == MessageType.TEXT
    
    def is_image(self) -> bool:
        """是否为图片类型"""
        return self.msg_type == MessageType.IMAGE
    
    def is_tool_call(self) -> bool:
        """是否为工具调用"""
        return self.msg_type == MessageType.TOOL_CALL
    
    def is_tool_result(self) -> bool:
        """是否为工具结果"""
        return self.msg_type == MessageType.TOOL_RESULT
    
    def get_content_length(self) -> int:
        """获取内容长度（字符数）"""
        return len(self.content)
    
    def get_age_seconds(self) -> float:
        """
        获取消息年龄（秒）
        
        Returns:
            距离创建时间的秒数
        """
        return (datetime.now() - self.timestamp).total_seconds()
    
    def to_openai_format(self) -> dict[str, str]:
        """
        转换为OpenAI API格式
        
        Returns:
            OpenAI格式的字典
        """
        result = {
            "role": self.role.value,
            "content": self.content,
        }
        
        # 添加工具调用相关字段
        if self.is_tool_call():
            result["tool_calls"] = self.metadata.get("tool_calls", [])
        
        if self.is_tool():
            result["tool_call_id"] = self.metadata.get("tool_call_id", "")
        
        return result
    
    @classmethod
    def from_openai_format(cls, data: dict[str, Any]) -> Self:
        """
        从OpenAI API格式创建Message
        
        Args:
            data: OpenAI格式的字典
            
        Returns:
            Message实例
        """
        metadata = {}
        msg_type = MessageType.TEXT
        
        # 提取工具调用信息
        if "tool_calls" in data:
            metadata["tool_calls"] = data["tool_calls"]
            msg_type = MessageType.TOOL_CALL
        
        if "tool_call_id" in data:
            metadata["tool_call_id"] = data["tool_call_id"]
            msg_type = MessageType.TOOL_RESULT
        
        return cls(
            role=Role(data.get("role", "user")),
            content=data.get("content", ""),
            msg_type=msg_type,
            metadata=metadata,
        )
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"[{self.role.value}] {self.content[:50]}{'...' if len(self.content) > 50 else ''}"
    
    def __repr__(self) -> str:
        """详细表示"""
        return f"Message(role={self.role.value}, content={self.content[:30]}..., msg_id={self.msg_id})"
    
    def __hash__(self) -> int:
        """哈希值（基于msg_id）"""
        return hash(self.msg_id)
    
    def __eq__(self, other: object) -> bool:
        """相等比较（基于msg_id）"""
        if not isinstance(other, Message):
            return False
        return self.msg_id == other.msg_id


# =========================================================================
# 消息列表工具函数
# =========================================================================

def create_system_message(content: str, **kwargs) -> Message:
    """
    创建系统消息
    
    Args:
        content: 内容
        **kwargs: 其他字段
        
    Returns:
        系统消息
    """
    return Message(role=Role.SYSTEM, content=content, **kwargs)


def create_user_message(content: str, **kwargs) -> Message:
    """
    创建用户消息
    
    Args:
        content: 内容
        **kwargs: 其他字段
        
    Returns:
        用户消息
    """
    return Message(role=Role.USER, content=content, **kwargs)


def create_assistant_message(content: str, **kwargs) -> Message:
    """
    创建助手消息
    
    Args:
        content: 内容
        **kwargs: 其他字段
        
    Returns:
        助手消息
    """
    return Message(role=Role.ASSISTANT, content=content, **kwargs)


def create_tool_message(content: str, tool_call_id: str, **kwargs) -> Message:
    """
    创建工具消息
    
    Args:
        content: 内容
        tool_call_id: 工具调用ID
        **kwargs: 其他字段
        
    Returns:
        工具消息
    """
    metadata = kwargs.pop("metadata", {})
    metadata["tool_call_id"] = tool_call_id
    return Message(
        role=Role.TOOL,
        content=content,
        msg_type=MessageType.TOOL_RESULT,
        metadata=metadata,
        **kwargs
    )


def filter_messages_by_role(messages: list[Message], role: Role) -> list[Message]:
    """
    按角色过滤消息
    
    Args:
        messages: 消息列表
        role: 目标角色
        
    Returns:
        过滤后的消息列表
    """
    return [msg for msg in messages if msg.role == role]


def filter_messages_by_type(messages: list[Message], msg_type: MessageType) -> list[Message]:
    """
    按类型过滤消息
    
    Args:
        messages: 消息列表
        msg_type: 目标类型
        
    Returns:
        过滤后的消息列表
    """
    return [msg for msg in messages if msg.msg_type == msg_type]


def sort_messages_by_timestamp(messages: list[Message], reverse: bool = False) -> list[Message]:
    """
    按时间戳排序消息
    
    Args:
        messages: 消息列表
        reverse: 是否倒序
        
    Returns:
        排序后的消息列表
    """
    return sorted(messages, key=lambda m: m.timestamp, reverse=reverse)


def get_total_token_count(messages: list[Message]) -> int:
    """
    获取消息列表的总Token数
    
    Args:
        messages: 消息列表
        
    Returns:
        总Token数
    """
    return sum(msg.token_count for msg in messages)


def validate_message_list(messages: list[Message]) -> tuple[bool, list[str]]:
    """
    验证消息列表
    
    Args:
        messages: 消息列表
        
    Returns:
        (是否全部有效, 错误信息列表)
    """
    all_errors = []
    for i, msg in enumerate(messages):
        valid, errors = msg.validate()
        if not valid:
            all_errors.extend([f"Message[{i}]: {e}" for e in errors])
    
    return len(all_errors) == 0, all_errors


def messages_to_openai_format(messages: list[Message]) -> list[dict[str, str]]:
    """
    将消息列表转换为OpenAI格式
    
    Args:
        messages: 消息列表
        
    Returns:
        OpenAI格式的消息列表
    """
    return [msg.to_openai_format() for msg in messages]


def messages_to_json(messages: list[Message], indent: Optional[int] = None) -> str:
    """
    将消息列表转换为JSON字符串
    
    Args:
        messages: 消息列表
        indent: 缩进空格数
        
    Returns:
        JSON字符串
    """
    data = [msg.to_dict() for msg in messages]
    return json.dumps(data, indent=indent, ensure_ascii=False)


def messages_from_json(json_str: str) -> list[Message]:
    """
    从JSON字符串创建消息列表
    
    Args:
        json_str: JSON字符串
        
    Returns:
        消息列表
    """
    data = json.loads(json_str)
    return [Message.from_dict(item) for item in data]
