"""
无上限上下文Agent系统 - ID生成器

生成各类唯一标识符

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

import uuid
import time
from typing import Final

# ID前缀定义
PREFIX_MSG: Final[str] = "msg"           # 消息ID前缀
PREFIX_MEMORY: Final[str] = "mem"        # 记忆ID前缀
PREFIX_TASK: Final[str] = "task"         # 任务ID前缀
PREFIX_SESSION: Final[str] = "sess"      # 会话ID前缀
PREFIX_CHUNK: Final[str] = "chunk"       # 分片ID前缀
PREFIX_SNAPSHOT: Final[str] = "snap"     # 快照ID前缀


def generate_uuid() -> str:
    """生成标准UUID"""
    return str(uuid.uuid4())


def generate_short_uuid() -> str:
    """生成短UUID（前8位）"""
    return uuid.uuid4().hex[:8]


def generate_id(prefix: str) -> str:
    """
    生成带前缀的唯一ID
    
    格式: {prefix}_{timestamp}_{short_uuid}
    
    Args:
        prefix: ID前缀
        
    Returns:
        唯一ID字符串
    """
    timestamp = int(time.time() * 1000)  # 毫秒级时间戳
    short_id = generate_short_uuid()
    return f"{prefix}_{timestamp}_{short_id}"


def generate_msg_id() -> str:
    """生成消息ID"""
    return generate_id(PREFIX_MSG)


def generate_memory_id() -> str:
    """生成记忆ID"""
    return generate_id(PREFIX_MEMORY)


def generate_task_id() -> str:
    """生成任务ID"""
    return generate_id(PREFIX_TASK)


def generate_session_id() -> str:
    """生成会话ID"""
    return generate_id(PREFIX_SESSION)


def generate_chunk_id() -> str:
    """生成分片ID"""
    return generate_id(PREFIX_CHUNK)


def generate_snapshot_id() -> str:
    """生成快照ID"""
    return generate_id(PREFIX_SNAPSHOT)


def parse_id(id_str: str) -> dict:
    """
    解析ID字符串
    
    Args:
        id_str: ID字符串
        
    Returns:
        解析结果字典，包含prefix、timestamp、uuid
    """
    parts = id_str.split("_")
    if len(parts) >= 3:
        return {
            "prefix": parts[0],
            "timestamp": int(parts[1]) if parts[1].isdigit() else None,
            "uuid": parts[2],
            "raw": id_str
        }
    return {"prefix": None, "timestamp": None, "uuid": None, "raw": id_str}


def get_id_timestamp(id_str: str) -> float:
    """
    从ID中提取时间戳
    
    Args:
        id_str: ID字符串
        
    Returns:
        时间戳（秒），解析失败返回0
    """
    parsed = parse_id(id_str)
    if parsed["timestamp"]:
        return parsed["timestamp"] / 1000.0
    return 0.0
