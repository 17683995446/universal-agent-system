"""
无上限上下文Agent系统 - 上下文压缩器

渐进式压缩策略，根据Token预算动态调整压缩级别。

核心思想：
- 轻度压力（使用率<60%）：不压缩
- 中度压力（60%-80%）：规则去噪（L1）
- 重度压力（80%-95%）：结构化压缩（L2）
- 极限压力（>95%）：摘要压缩（L3）

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-26
"""

import logging
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional, Any

from src.modules.purification.pipeline import PurificationPipeline
from src.modules.purification.purification_types import (
    PurificationLevel,
    PurificationResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 压缩策略枚举
# =============================================================================

class CompressionStrategy(IntEnum):
    """压缩策略枚举"""
    NONE = 0             # 不压缩
    CONSERVATIVE = 1     # 保守策略：仅在极限压力时压缩
    BALANCED = 2         # 均衡策略：中度压力开始压缩（默认）
    AGGRESSIVE = 3       # 激进策略：轻度压力就开始压缩


# =============================================================================
# 压缩配置
# =============================================================================

@dataclass
class CompressionConfig:
    """
    压缩配置

    定义压缩触发的Token压力阈值和策略。

    Attributes:
        strategy: 压缩策略
        enabled: 是否启用压缩
        light_threshold: 轻度压力阈值（使用率比例，0-1）
        medium_threshold: 中度压力阈值
        heavy_threshold: 重度压力阈值
        critical_threshold: 极限压力阈值
        min_tokens_to_compress: 最小压缩Token数（低于此不压缩）
        preserve_recent_n: 始终保留最近N条消息不压缩
        integrity_threshold: 完整性最低阈值
    """
    strategy: CompressionStrategy = CompressionStrategy.BALANCED
    enabled: bool = True
    light_threshold: float = 0.60
    medium_threshold: float = 0.80
    heavy_threshold: float = 0.95
    critical_threshold: float = 1.0
    min_tokens_to_compress: int = 200
    preserve_recent_n: int = 2
    integrity_threshold: float = 0.7

    def get_thresholds(self) -> tuple[float, float, float, float]:
        """获取当前策略对应的阈值"""
        if self.strategy == CompressionStrategy.NONE:
            # 不压缩：阈值设为极高
            return (1.0, 1.0, 1.0, 1.0)
        elif self.strategy == CompressionStrategy.CONSERVATIVE:
            # 保守：阈值后移
            return (0.80, 0.90, 0.97, 1.0)
        elif self.strategy == CompressionStrategy.BALANCED:
            # 均衡（默认）
            return (
                self.light_threshold,
                self.medium_threshold,
                self.heavy_threshold,
                self.critical_threshold,
            )
        elif self.strategy == CompressionStrategy.AGGRESSIVE:
            # 激进：阈值前移
            return (0.40, 0.60, 0.80, 0.95)
        else:
            return (
                self.light_threshold,
                self.medium_threshold,
                self.heavy_threshold,
                self.critical_threshold,
            )

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "strategy": int(self.strategy),
            "strategy_name": self.strategy.name,
            "enabled": self.enabled,
            "light_threshold": self.light_threshold,
            "medium_threshold": self.medium_threshold,
            "heavy_threshold": self.heavy_threshold,
            "critical_threshold": self.critical_threshold,
            "min_tokens_to_compress": self.min_tokens_to_compress,
            "preserve_recent_n": self.preserve_recent_n,
            "integrity_threshold": self.integrity_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompressionConfig":
        """从字典创建"""
        config = cls()
        if "strategy" in data:
            strategy_val = data["strategy"]
            if isinstance(strategy_val, str):
                config.strategy = CompressionStrategy[strategy_val]
            else:
                config.strategy = CompressionStrategy(int(strategy_val))
        config.enabled = data.get("enabled", config.enabled)
        config.light_threshold = data.get("light_threshold", config.light_threshold)
        config.medium_threshold = data.get("medium_threshold", config.medium_threshold)
        config.heavy_threshold = data.get("heavy_threshold", config.heavy_threshold)
        config.critical_threshold = data.get("critical_threshold", config.critical_threshold)
        config.min_tokens_to_compress = data.get("min_tokens_to_compress", config.min_tokens_to_compress)
        config.preserve_recent_n = data.get("preserve_recent_n", config.preserve_recent_n)
        config.integrity_threshold = data.get("integrity_threshold", config.integrity_threshold)
        return config


# =============================================================================
# 压缩结果
# =============================================================================

@dataclass
class CompressionResult:
    """
    压缩结果

    Attributes:
        original_tokens: 压缩前Token数
        compressed_tokens: 压缩后Token数
        compression_level: 使用的压缩级别
        messages_compressed: 被压缩的消息数量
        messages_preserved: 保留的消息数量
        pressure_level: 压力级别（light/medium/heavy/critical/none）
        compression_ratio: 压缩比例
        details: 各消息的压缩详情
    """
    original_tokens: int
    compressed_tokens: int
    compression_level: PurificationLevel
    messages_compressed: int
    messages_preserved: int
    pressure_level: str
    compression_ratio: float = 0.0
    details: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.compression_ratio == 0.0 and self.original_tokens > 0:
            self.compression_ratio = self.compressed_tokens / self.original_tokens

    @property
    def tokens_saved(self) -> int:
        """节省的Token数"""
        return self.original_tokens - self.compressed_tokens

    @property
    def compression_percentage(self) -> float:
        """压缩百分比"""
        if self.original_tokens == 0:
            return 0.0
        return (1.0 - self.compression_ratio) * 100

    @property
    def was_compressed(self) -> bool:
        """是否发生了压缩"""
        return self.messages_compressed > 0

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "tokens_saved": self.tokens_saved,
            "compression_level": int(self.compression_level),
            "compression_level_name": self.compression_level.name,
            "messages_compressed": self.messages_compressed,
            "messages_preserved": self.messages_preserved,
            "pressure_level": self.pressure_level,
            "compression_ratio": round(self.compression_ratio, 4),
            "compression_percentage": round(self.compression_percentage, 2),
        }


# =============================================================================
# 上下文压缩器
# =============================================================================

class ContextCompressor:
    """
    上下文压缩器

    根据Token预算压力，对历史消息进行渐进式压缩。

    压缩流程：
    1. 评估当前Token压力级别
    2. 根据压力级别选择压缩策略
    3. 对可压缩消息执行净化
    4. 保留最近N条消息不压缩
    5. 返回压缩结果

    Attributes:
        config: 压缩配置
        pipeline: 净化管道
    """

    def __init__(
        self,
        config: Optional[CompressionConfig] = None,
        pipeline: Optional[PurificationPipeline] = None,
    ):
        """
        Args:
            config: 压缩配置（None则使用默认配置）
            pipeline: 净化管道（None则自动创建）
        """
        self._config = config or CompressionConfig()

        if pipeline:
            self._pipeline = pipeline
        else:
            from src.modules.purification.rule_denoiser import RuleDenoiser
            from src.modules.purification.structure_converter import StructureConverter
            from src.modules.purification.summarizer import Summarizer

            self._pipeline = PurificationPipeline(
                denoiser=RuleDenoiser(),
                converter=StructureConverter(),
                summarizer=Summarizer(),
            )

        # 统计
        self._total_compressions: int = 0
        self._total_tokens_saved: int = 0
        self._total_messages_compressed: int = 0

    # =========================================================================
    # 属性
    # =========================================================================

    @property
    def config(self) -> CompressionConfig:
        """压缩配置"""
        return self._config

    @property
    def enabled(self) -> bool:
        """是否启用"""
        return self._config.enabled

    # =========================================================================
    # 核心方法
    # =========================================================================

    def assess_pressure(
        self,
        used_tokens: int,
        window_size: int,
    ) -> tuple[str, float]:
        """
        评估Token压力级别

        Args:
            used_tokens: 已使用Token数
            window_size: 窗口大小

        Returns:
            (压力级别, 使用率比例)
            压力级别: "none" | "light" | "medium" | "heavy" | "critical"
        """
        if window_size <= 0:
            return ("critical", 1.0)

        ratio = used_tokens / window_size
        light_t, medium_t, heavy_t, critical_t = self._config.get_thresholds()

        if ratio >= critical_t:
            return ("critical", ratio)
        elif ratio >= heavy_t:
            return ("heavy", ratio)
        elif ratio >= medium_t:
            return ("medium", ratio)
        elif ratio >= light_t:
            return ("light", ratio)
        else:
            return ("none", ratio)

    def determine_compression_level(self, pressure_level: str) -> PurificationLevel:
        """
        根据压力级别确定压缩级别

        Args:
            pressure_level: 压力级别

        Returns:
            对应的净化级别
        """
        level_map = {
            "none": PurificationLevel.NONE,
            "light": PurificationLevel.NONE,
            "medium": PurificationLevel.RULE_BASED,
            "heavy": PurificationLevel.STRUCTURED,
            "critical": PurificationLevel.SUMMARIZED,
        }
        return level_map.get(pressure_level, PurificationLevel.NONE)

    def compress_messages(
        self,
        messages: list[dict[str, Any]],
        used_tokens: int,
        window_size: int,
    ) -> tuple[list[dict[str, Any]], CompressionResult]:
        """
        压缩消息列表

        根据Token压力对历史消息进行压缩，保留最近N条不压缩。

        Args:
            messages: 消息列表（OpenAI格式）
            used_tokens: 当前已使用Token数
            window_size: Token窗口大小

        Returns:
            (压缩后的消息列表, 压缩结果)
        """
        if not self._config.enabled or not messages:
            result = CompressionResult(
                original_tokens=used_tokens,
                compressed_tokens=used_tokens,
                compression_level=PurificationLevel.NONE,
                messages_compressed=0,
                messages_preserved=len(messages),
                pressure_level="none",
            )
            return messages, result

        # 评估压力
        pressure_level, usage_ratio = self.assess_pressure(used_tokens, window_size)

        # 确定压缩级别
        compression_level = self.determine_compression_level(pressure_level)

        if compression_level == PurificationLevel.NONE:
            result = CompressionResult(
                original_tokens=used_tokens,
                compressed_tokens=used_tokens,
                compression_level=PurificationLevel.NONE,
                messages_compressed=0,
                messages_preserved=len(messages),
                pressure_level=pressure_level,
            )
            return messages, result

        # 分离可压缩和保留消息
        preserve_n = self._config.preserve_recent_n
        if len(messages) <= preserve_n:
            # 消息数少于保留数，不压缩
            result = CompressionResult(
                original_tokens=used_tokens,
                compressed_tokens=used_tokens,
                compression_level=PurificationLevel.NONE,
                messages_compressed=0,
                messages_preserved=len(messages),
                pressure_level=pressure_level,
            )
            return messages, result

        preserved = messages[-preserve_n:]
        to_compress = messages[:-preserve_n]

        # 执行压缩
        compressed_msgs = []
        total_saved = 0
        details = []

        for i, msg in enumerate(to_compress):
            content = msg.get("content", "")
            if not content or len(content) < self._config.min_tokens_to_compress:
                # 内容太短，不压缩
                compressed_msgs.append(msg)
                details.append({
                    "index": i,
                    "role": msg.get("role", "unknown"),
                    "compressed": False,
                    "reason": "too_short",
                })
                continue

            try:
                purify_result = self._pipeline.purify_to_level(
                    text=content,
                    target_level=compression_level,
                    original_tokens=max(1, len(content) // 4),
                )

                saved = purify_result.tokens_saved()
                total_saved += max(0, saved)

                compressed_msg = {**msg, "content": purify_result.purified_text}
                compressed_msgs.append(compressed_msg)

                details.append({
                    "index": i,
                    "role": msg.get("role", "unknown"),
                    "compressed": True,
                    "original_tokens": purify_result.original_tokens,
                    "compressed_tokens": purify_result.purified_tokens,
                    "level": purify_result.level.name,
                    "integrity": purify_result.integrity_score,
                })

            except Exception as e:
                # 压缩失败，保留原消息
                logger.warning(f"Failed to compress message {i}: {e}")
                compressed_msgs.append(msg)
                details.append({
                    "index": i,
                    "role": msg.get("role", "unknown"),
                    "compressed": False,
                    "reason": f"error: {e}",
                })

        # 合并保留消息
        final_messages = compressed_msgs + preserved

        compressed_tokens = max(1, used_tokens - total_saved)
        messages_compressed = sum(1 for d in details if d.get("compressed"))

        result = CompressionResult(
            original_tokens=used_tokens,
            compressed_tokens=compressed_tokens,
            compression_level=compression_level,
            messages_compressed=messages_compressed,
            messages_preserved=len(preserved),
            pressure_level=pressure_level,
            details=details,
        )

        # 更新统计
        self._total_compressions += 1
        self._total_tokens_saved += total_saved
        self._total_messages_compressed += messages_compressed

        logger.info(
            f"Context compressed: pressure={pressure_level}, "
            f"level={compression_level.name}, "
            f"saved={total_saved} tokens, "
            f"messages_compressed={messages_compressed}/{len(to_compress)}"
        )

        return final_messages, result

    def compress_text(
        self,
        text: str,
        target_level: Optional[PurificationLevel] = None,
        original_tokens: int = 0,
    ) -> PurificationResult:
        """
        压缩单段文本

        Args:
            text: 原始文本
            target_level: 目标压缩级别（None则使用RULE_BASED）
            original_tokens: 原始Token数

        Returns:
            PurificationResult
        """
        if not self._config.enabled or not text:
            if original_tokens == 0:
                original_tokens = max(1, len(text) // 4) if text else 0
            return PurificationResult(
                original_text=text,
                purified_text=text,
                level=PurificationLevel.NONE,
                original_tokens=original_tokens,
                purified_tokens=original_tokens,
            )

        level = target_level or PurificationLevel.RULE_BASED

        return self._pipeline.purify_to_level(
            text=text,
            target_level=level,
            original_tokens=original_tokens,
        )

    # =========================================================================
    # 统计
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """获取压缩统计"""
        return {
            "enabled": self._config.enabled,
            "strategy": self._config.strategy.name,
            "total_compressions": self._total_compressions,
            "total_tokens_saved": self._total_tokens_saved,
            "total_messages_compressed": self._total_messages_compressed,
            "avg_tokens_saved_per_compression": (
                self._total_tokens_saved / self._total_compressions
                if self._total_compressions > 0
                else 0
            ),
        }

    def reset_stats(self) -> None:
        """重置统计"""
        self._total_compressions = 0
        self._total_tokens_saved = 0
        self._total_messages_compressed = 0
