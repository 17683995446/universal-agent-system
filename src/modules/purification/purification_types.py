"""
无上限上下文Agent系统 - 净化类型定义

标准化净化相关的枚举、结构体和工具函数。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

from typing import Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import copy


# =============================================================================
# 枚举定义
# =============================================================================

class PurificationLevel(int, Enum):
    """净化级别枚举"""
    NONE = 0           # 无净化
    RULE_BASED = 1     # 一级：规则去噪
    STRUCTURED = 2     # 二级：结构化转换
    SUMMARIZED = 3     # 三级：小模型摘要


# =============================================================================
# PurificationResult 结构体
# =============================================================================

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
        compression_ratio: 压缩比例（purified/original）
        integrity_score: 完整性评分（0-1）
        preserved_keys: 保留的关键信息列表
        metadata: 元数据（各级别处理的详细信息）
    """
    original_text: str
    purified_text: str
    level: PurificationLevel
    original_tokens: int
    purified_tokens: int
    compression_ratio: float = 0.0
    integrity_score: float = 1.0
    preserved_keys: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理"""
        if isinstance(self.level, str):
            self.level = PurificationLevel(int(self.level))
        # 自动计算压缩比例
        if self.compression_ratio == 0.0 and self.original_tokens > 0:
            self.compression_ratio = self.purified_tokens / self.original_tokens

    # =========================================================================
    # 序列化
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "original_text": self.original_text,
            "purified_text": self.purified_text,
            "level": int(self.level),
            "level_name": self.level.name,
            "original_tokens": self.original_tokens,
            "purified_tokens": self.purified_tokens,
            "compression_ratio": self.compression_ratio,
            "integrity_score": self.integrity_score,
            "preserved_keys": self.preserved_keys,
            "metadata": self.metadata,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """转换为JSON"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PurificationResult":
        """从字典创建"""
        return cls(
            original_text=data.get("original_text", ""),
            purified_text=data.get("purified_text", ""),
            level=PurificationLevel(data.get("level", 0)),
            original_tokens=data.get("original_tokens", 0),
            purified_tokens=data.get("purified_tokens", 0),
            compression_ratio=data.get("compression_ratio", 0.0),
            integrity_score=data.get("integrity_score", 1.0),
            preserved_keys=data.get("preserved_keys", []),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "PurificationResult":
        """从JSON创建"""
        return cls.from_dict(json.loads(json_str))

    # =========================================================================
    # 验证
    # =========================================================================

    def validate(self) -> tuple[bool, list[str]]:
        """验证结果有效性"""
        errors: list[str] = []

        if self.original_tokens < 0:
            errors.append(f"original_tokens不能为负数: {self.original_tokens}")
        if self.purified_tokens < 0:
            errors.append(f"purified_tokens不能为负数: {self.purified_tokens}")
        if not (0.0 <= self.integrity_score <= 1.0):
            errors.append(f"integrity_score必须在0-1之间: {self.integrity_score}")
        if not (0.0 <= self.compression_ratio):
            errors.append(f"compression_ratio不能为负数: {self.compression_ratio}")

        return len(errors) == 0, errors

    def is_valid(self) -> bool:
        """快速检查"""
        valid, _ = self.validate()
        return valid

    # =========================================================================
    # 工具方法
    # =========================================================================

    def copy(self) -> "PurificationResult":
        """深拷贝"""
        return copy.deepcopy(self)

    def tokens_saved(self) -> int:
        """节省的Token数"""
        return self.original_tokens - self.purified_tokens

    def compression_percentage(self) -> float:
        """压缩百分比（0-100），50表示压缩了50%"""
        if self.original_tokens == 0:
            return 0.0
        return (1.0 - self.compression_ratio) * 100

    def is_compressed(self) -> bool:
        """是否发生了压缩"""
        return self.purified_tokens < self.original_tokens

    def meets_integrity_threshold(self, threshold: float) -> bool:
        """是否满足完整性阈值"""
        return self.integrity_score >= threshold

    def add_metadata(self, key: str, value: Any) -> "PurificationResult":
        """添加元数据（链式调用）"""
        self.metadata[key] = value
        return self

    def add_preserved_key(self, key: str) -> "PurificationResult":
        """添加保留的关键信息（链式调用）"""
        if key not in self.preserved_keys:
            self.preserved_keys.append(key)
        return self

    # =========================================================================
    # 魔术方法
    # =========================================================================

    def __str__(self) -> str:
        return (
            f"PurificationResult(level={self.level.name}, "
            f"tokens={self.original_tokens}→{self.purified_tokens}, "
            f"integrity={self.integrity_score:.2f})"
        )

    def __repr__(self) -> str:
        return (
            f"PurificationResult(level={self.level.name}, "
            f"ratio={self.compression_ratio:.2f}, "
            f"integrity={self.integrity_score:.2f})"
        )
