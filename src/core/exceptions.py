"""
无上限上下文Agent系统 - 异常定义

所有自定义异常类

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

from typing import Optional, Any


class BaseAgentException(Exception):
    """
    基础异常类
    
    所有自定义异常的基类
    """
    
    def __init__(self, message: str, context: Optional[dict[str, Any]] = None):
        """
        初始化异常
        
        Args:
            message: 异常信息
            context: 异常上下文数据
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
    
    def __str__(self) -> str:
        if self.context:
            return f"{self.message} | context: {self.context}"
        return self.message
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context
        }


# =============================================================================
# Token相关异常
# =============================================================================

class TokenLimitExceeded(BaseAgentException):
    """
    Token限制超限异常
    
    当Token数量超过模型窗口限制时抛出
    """
    
    def __init__(
        self,
        current: int,
        limit: int,
        message: Optional[str] = None
    ):
        msg = message or f"Token limit exceeded: {current} > {limit}"
        super().__init__(msg, {"current": current, "limit": limit})
        self.current = current
        self.limit = limit


class TokenizerNotFound(BaseAgentException):
    """
    Tokenizer未找到异常
    
    当无法找到对应模型的Tokenizer时抛出
    """
    
    def __init__(self, model: str, message: Optional[str] = None):
        msg = message or f"Tokenizer not found for model: {model}"
        super().__init__(msg, {"model": model})
        self.model = model


class TokenCalculationError(BaseAgentException):
    """Token计算错误"""
    
    def __init__(self, text: str, error: str):
        super().__init__(
            f"Token calculation failed: {error}",
            {"text_preview": text[:100], "error": error}
        )


# =============================================================================
# 记忆相关异常
# =============================================================================

class MemoryNotFound(BaseAgentException):
    """
    记忆未找到异常
    
    当尝试访问不存在的记忆时抛出
    """
    
    def __init__(self, memory_id: str):
        super().__init__(f"Memory not found: {memory_id}", {"memory_id": memory_id})
        self.memory_id = memory_id


class MemoryStorageError(BaseAgentException):
    """记忆存储错误"""
    
    def __init__(self, operation: str, error: str):
        super().__init__(
            f"Memory storage error during {operation}: {error}",
            {"operation": operation, "error": error}
        )


class MemoryRecallError(BaseAgentException):
    """记忆召回错误"""
    
    def __init__(self, query: str, error: str):
        super().__init__(
            f"Memory recall failed: {error}",
            {"query": query[:100], "error": error}
        )


# =============================================================================
# 任务快照相关异常
# =============================================================================

class SnapshotNotFound(BaseAgentException):
    """
    快照未找到异常
    
    当尝试恢复不存在的任务快照时抛出
    """
    
    def __init__(self, task_id: str):
        super().__init__(f"Snapshot not found: {task_id}", {"task_id": task_id})
        self.task_id = task_id


class SnapshotSaveError(BaseAgentException):
    """快照保存错误"""
    
    def __init__(self, task_id: str, error: str):
        super().__init__(
            f"Failed to save snapshot for task {task_id}: {error}",
            {"task_id": task_id, "error": error}
        )


class SnapshotLoadError(BaseAgentException):
    """快照加载错误"""
    
    def __init__(self, task_id: str, error: str):
        super().__init__(
            f"Failed to load snapshot for task {task_id}: {error}",
            {"task_id": task_id, "error": error}
        )


class TaskDriftDetected(BaseAgentException):
    """
    任务漂移检测异常
    
    当检测到任务目标发生显著偏离时抛出
    """
    
    def __init__(
        self,
        task_id: str,
        original_goal: str,
        current_intent: str,
        similarity: float
    ):
        super().__init__(
            f"Task drift detected for {task_id}: similarity={similarity:.2f}",
            {
                "task_id": task_id,
                "original_goal": original_goal,
                "current_intent": current_intent,
                "similarity": similarity
            }
        )
        self.task_id = task_id
        self.similarity = similarity


# =============================================================================
# 压缩与净化相关异常
# =============================================================================

class PurificationError(BaseAgentException):
    """净化处理错误"""
    
    def __init__(self, text: str, level: str, error: str):
        super().__init__(
            f"Purification failed at level {level}: {error}",
            {"text_preview": text[:100], "level": level, "error": error}
        )


class IntegrityCheckFailed(BaseAgentException):
    """
    完整性校验失败异常
    
    当压缩后的内容丢失关键信息时抛出
    """
    
    def __init__(
        self,
        original_tokens: int,
        preserved_ratio: float,
        required_ratio: float
    ):
        super().__init__(
            f"Integrity check failed: {preserved_ratio:.2%} < {required_ratio:.2%}",
            {
                "original_tokens": original_tokens,
                "preserved_ratio": preserved_ratio,
                "required_ratio": required_ratio
            }
        )


# =============================================================================
# 分片相关异常
# =============================================================================

class ChunkingError(BaseAgentException):
    """分片处理错误"""
    
    def __init__(self, text_length: int, chunk_size: int, error: str):
        super().__init__(
            f"Chunking failed: {error}",
            {"text_length": text_length, "chunk_size": chunk_size, "error": error}
        )


class AggregationError(BaseAgentException):
    """分片聚合错误"""
    
    def __init__(self, chunk_count: int, error: str):
        super().__init__(
            f"Chunk aggregation failed: {error}",
            {"chunk_count": chunk_count, "error": error}
        )


# =============================================================================
# 基础设施相关异常
# =============================================================================

class LLMClientError(BaseAgentException):
    """LLM客户端错误"""
    
    def __init__(self, model: str, error: str, status_code: Optional[int] = None):
        context: dict[str, Any] = {"model": model, "error": error}
        if status_code:
            context["status_code"] = status_code
        super().__init__(f"LLM client error: {error}", context)
        self.status_code = status_code


class VectorDBError(BaseAgentException):
    """向量数据库错误"""
    
    def __init__(self, operation: str, error: str):
        super().__init__(
            f"Vector DB error during {operation}: {error}",
            {"operation": operation, "error": error}
        )


class CacheError(BaseAgentException):
    """缓存错误"""
    
    def __init__(self, operation: str, key: str, error: str):
        super().__init__(
            f"Cache error during {operation}: {error}",
            {"operation": operation, "key": key, "error": error}
        )


# =============================================================================
# 配置相关异常
# =============================================================================

class ConfigurationError(BaseAgentException):
    """配置错误"""
    
    def __init__(self, key: str, value: Any, error: str):
        super().__init__(
            f"Configuration error for {key}: {error}",
            {"key": key, "value": value, "error": error}
        )


class ValidationError(BaseAgentException):
    """校验错误"""
    
    def __init__(self, errors: list[str]):
        super().__init__(
            f"Validation failed with {len(errors)} error(s)",
            {"errors": errors}
        )
        self.errors = errors


# =============================================================================
# 工具函数
# =============================================================================

def is_agent_exception(error: Exception) -> bool:
    """检查是否为Agent自定义异常"""
    return isinstance(error, BaseAgentException)


def get_error_context(error: Exception) -> dict:
    """获取异常的上下文信息"""
    if isinstance(error, BaseAgentException):
        return error.context
    return {}
