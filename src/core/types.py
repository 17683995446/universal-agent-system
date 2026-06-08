"""
无上限上下文Agent系统 - 核心类型定义

本文件定义系统所有核心数据结构，作为各模块间交互的契约。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

from typing import TypedDict, Optional, Literal, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# =============================================================================
# 基础类型定义
# =============================================================================

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


# =============================================================================
# 消息相关类型
# =============================================================================

@dataclass
class Message:
    """
    标准化消息结构体
    
    Attributes:
        role: 消息角色
        content: 消息内容
        msg_id: 消息唯一标识
        timestamp: 时间戳
        token_count: Token数量（计算后填充）
        metadata: 元数据（图片URL、工具调用ID等）
    """
    role: Role
    content: str
    msg_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.msg_id:
            from src.core.utils.id_generator import generate_msg_id
            self.msg_id = generate_msg_id()


@dataclass
class MessageBatch:
    """消息批次，用于批量计算Token"""
    messages: list[Message]
    total_tokens: int = 0
    
    def calculate_total(self) -> int:
        """计算批次总Token数"""
        self.total_tokens = sum(msg.token_count for msg in self.messages)
        return self.total_tokens


# =============================================================================
# Token计算相关类型
# =============================================================================

class TokenizerBackend(str, Enum):
    """Tokenizer后端枚举"""
    TIKTOKEN = "tiktoken"
    TRANSFORMERS = "transformers"
    CL100K_BASE = "cl100k_base"  # GPT-4, GPT-3.5-turbo
    P50K_BASE = "p50k_base"      # GPT-3 (text-davinci-003)
    R50K_BASE = "r50k_base"      # GPT-3 (text-davinci-001/002)


@dataclass
class TokenCountResult:
    """
    Token计算结果
    
    Attributes:
        text: 原始文本
        token_count: Token数量
        model: 使用的模型
        backend: 使用的Tokenizer后端
        error: 错误信息（如有）
    """
    text: str
    token_count: int
    model: str
    backend: TokenizerBackend
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """计算是否成功"""
        return self.error is None


@dataclass
class TokenBudget:
    """
    Token预算分配
    
    根据窗口分区策略计算各区域可用Token数
    """
    window_size: int                    # 模型窗口总大小
    system_zone: int                    # 系统指令区配额
    session_zone: int                   # 有效会话区配额
    reserve_zone: int                   # 预留推理区配额
    
    @property
    def total(self) -> int:
        """总配额"""
        return self.system_zone + self.session_zone + self.reserve_zone
    
    @property
    def available_for_messages(self) -> int:
        """可用于消息的Token数（系统区+会话区）"""
        return self.system_zone + self.session_zone


# =============================================================================
# 记忆相关类型
# =============================================================================

class MemoryLevel(str, Enum):
    """记忆层级枚举"""
    INSTANT = "instant"      # 瞬时记忆
    SHORT_TERM = "short"     # 短期记忆
    LONG_TERM = "long"       # 长期记忆


@dataclass
class MemoryEntry:
    """
    记忆条目结构
    
    Attributes:
        content: 记忆内容
        memory_id: 记忆唯一标识
        level: 记忆层级
        created_at: 创建时间
        accessed_at: 最后访问时间
        access_count: 访问次数
        relevance_score: 相关性评分（0-1）
        embedding: 向量嵌入（长期记忆）
        metadata: 元数据（任务标签、来源等）
    """
    content: str
    memory_id: str = ""
    level: MemoryLevel = MemoryLevel.SHORT_TERM
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    relevance_score: float = 0.0
    embedding: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.memory_id:
            from src.core.utils.id_generator import generate_memory_id
            self.memory_id = generate_memory_id()


@dataclass
class MemoryRecallResult:
    """记忆召回结果"""
    query: str                          # 查询内容
    recalled_memories: list[MemoryEntry]  # 召回的记忆
    total_score: float                  # 综合评分
    semantic_score: float = 0.0         # 语义相似度评分
    time_score: float = 0.0             # 时间衰减评分
    task_score: float = 0.0             # 任务标签评分
    behavior_score: float = 0.0         # 行为权重评分


# =============================================================================
# 任务快照相关类型
# =============================================================================

@dataclass
class TaskStep:
    """任务步骤"""
    step_id: str
    description: str
    status: Literal["pending", "in_progress", "completed", "failed"]
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class TaskSnapshot:
    """
    任务状态快照
    
    Attributes:
        task_id: 任务唯一标识
        session_id: 会话标识
        goal: 任务目标
        completed_steps: 已完成步骤
        pending_steps: 待执行步骤
        variables: 关键变量
        constraints: 约束条件
        error_log: 错误记录
        progress_ratio: 进度比例（0-1）
        created_at: 创建时间
        updated_at: 更新时间
    """
    task_id: str
    session_id: str
    goal: str
    completed_steps: list[TaskStep] = field(default_factory=list)
    pending_steps: list[TaskStep] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    error_log: list[dict[str, Any]] = field(default_factory=list)
    progress_ratio: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def update_progress(self) -> None:
        """更新进度比例"""
        total = len(self.completed_steps) + len(self.pending_steps)
        if total > 0:
            self.progress_ratio = len(self.completed_steps) / total
        self.updated_at = datetime.now()


# =============================================================================
# 压缩与净化相关类型
# =============================================================================

class PurificationLevel(int, Enum):
    """净化级别枚举"""
    NONE = 0           # 无净化
    RULE_BASED = 1     # 一级：规则去噪
    STRUCTURED = 2     # 二级：结构化转换
    SUMMARIZED = 3     # 三级：小模型摘要


@dataclass
class PurificationResult:
    """
    净化结果
    
    Attributes:
        original_text: 原始文本
        purified_text: 净化后文本
        level: 应用的净化级别
        original_tokens: 原始Token数
        purified_tokens: 净化后Token数
        compression_ratio: 压缩比例
        integrity_score: 完整性评分（0-1）
        preserved_keys: 保留的关键信息
    """
    original_text: str
    purified_text: str
    level: PurificationLevel
    original_tokens: int
    purified_tokens: int
    compression_ratio: float = 0.0
    integrity_score: float = 1.0
    preserved_keys: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.original_tokens > 0:
            self.compression_ratio = self.purified_tokens / self.original_tokens


# =============================================================================
# 分片相关类型
# =============================================================================

@dataclass
class TextChunk:
    """文本分片"""
    chunk_id: str
    content: str
    start_pos: int          # 在原文中的起始位置
    end_pos: int            # 在原文中的结束位置
    token_count: int
    overlap_prev: str = ""  # 与前一片的重叠区
    overlap_next: str = ""  # 与后一片的重叠区


@dataclass
class ChunkingResult:
    """分片结果"""
    original_text: str
    chunks: list[TextChunk]
    total_chunks: int
    total_tokens: int
    overlap_tokens: int


# =============================================================================
# 上下文组装相关类型
# =============================================================================

@dataclass
class ContextAssembly:
    """
    上下文组装结果
    
    按配额将系统指令、历史消息、召回记忆组装成最终上下文
    """
    system_messages: list[Message]      # 系统指令区消息
    session_messages: list[Message]     # 有效会话区消息
    recalled_memories: list[MemoryEntry]  # 召回的长期记忆
    total_tokens: int                   # 总Token数
    budget: TokenBudget                 # 使用的预算配置
    
    def to_messages(self) -> list[Message]:
        """转换为消息列表（按正确顺序）"""
        return self.system_messages + self.session_messages


# =============================================================================
# 异常类型
# =============================================================================

class BaseAgentException(Exception):
    """基础异常类"""
    def __init__(self, message: str, context: Optional[dict] = None):
        super().__init__(message)
        self.context = context or {}


class TokenLimitExceeded(BaseAgentException):
    """Token限制超限异常"""
    def __init__(self, current: int, limit: int, message: Optional[str] = None):
        msg = message or f"Token limit exceeded: {current} > {limit}"
        super().__init__(msg, {"current": current, "limit": limit})
        self.current = current
        self.limit = limit


class TokenizerNotFound(BaseAgentException):
    """Tokenizer未找到异常"""
    def __init__(self, model: str):
        super().__init__(f"Tokenizer not found for model: {model}", {"model": model})
        self.model = model


class MemoryNotFound(BaseAgentException):
    """记忆未找到异常"""
    def __init__(self, memory_id: str):
        super().__init__(f"Memory not found: {memory_id}", {"memory_id": memory_id})
        self.memory_id = memory_id


class SnapshotNotFound(BaseAgentException):
    """快照未找到异常"""
    def __init__(self, task_id: str):
        super().__init__(f"Snapshot not found: {task_id}", {"task_id": task_id})
        self.task_id = task_id
