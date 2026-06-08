"""
无上限上下文Agent系统 - 净化管道

统一调度三级净化器，实现逐级触发和完整性校验。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

from typing import Optional

from src.modules.purification.purification_types import (
    PurificationResult, PurificationLevel,
)
from src.modules.purification.rule_denoiser import RuleDenoiser
from src.modules.purification.structure_converter import StructureConverter
from src.modules.purification.summarizer import Summarizer
from src.core.constants import INTEGRITY_MIN_PRESERVE_RATIO
from src.core.exceptions import IntegrityCheckFailed


class PurificationPipeline:
    """
    净化管道

    统一调度三级净化：
    1. 始终执行：一级规则去噪
    2. Token > 2000：二级结构化转换
    3. Token > 4000：三级LLM摘要

    每级净化后执行完整性校验。
    """

    def __init__(
        self,
        denoiser: Optional[RuleDenoiser] = None,
        converter: Optional[StructureConverter] = None,
        summarizer: Optional[Summarizer] = None,
        integrity_threshold: float = INTEGRITY_MIN_PRESERVE_RATIO,
        max_level: PurificationLevel = PurificationLevel.SUMMARIZED,
    ):
        """
        Args:
            denoiser: 一级净化器
            converter: 二级净化器
            summarizer: 三级净化器
            integrity_threshold: 完整性最低阈值
            max_level: 最大净化级别
        """
        self._denoiser = denoiser or RuleDenoiser()
        self._converter = converter or StructureConverter()
        self._summarizer = summarizer or Summarizer()
        self._integrity_threshold = integrity_threshold
        self._max_level = max_level

    def purify(
        self,
        text: str,
        original_tokens: int = 0,
        max_level: Optional[PurificationLevel] = None,
    ) -> PurificationResult:
        """
        执行净化管道

        Args:
            text: 原始文本
            original_tokens: 原始Token数（0则估算）
            max_level: 最大净化级别（None则使用默认值）

        Returns:
            最终净化结果

        Raises:
            IntegrityCheckFailed: 完整性校验失败
        """
        if not text or not text.strip():
            return PurificationResult(
                original_text=text,
                purified_text=text,
                level=PurificationLevel.NONE,
                original_tokens=original_tokens,
                purified_tokens=original_tokens,
            )

        effective_max = max_level if max_level is not None else self._max_level

        # 估算Token数
        if original_tokens == 0:
            original_tokens = self._estimate_tokens(text)

        # ===== 第一级：规则去噪（始终执行）=====
        result = self._denoiser.denoise(text, original_tokens)
        current_text = result.purified_text
        current_tokens = result.purified_tokens
        applied_level = PurificationLevel.RULE_BASED

        # ===== 第二级：结构化转换（Token > 2000）=====
        if (effective_max.value >= PurificationLevel.STRUCTURED.value
                and self._converter.should_convert(current_tokens)):
            result = self._converter.convert(current_text, current_tokens)
            current_text = result.purified_text
            current_tokens = result.purified_tokens
            applied_level = PurificationLevel.STRUCTURED

            # 完整性校验
            self._check_integrity(result)

        # ===== 第三级：LLM摘要（Token > 4000）=====
        if (effective_max.value >= PurificationLevel.SUMMARIZED.value
                and self._summarizer.should_summarize(current_tokens)):
            result = self._summarizer.summarize(current_text, current_tokens)
            current_text = result.purified_text
            current_tokens = result.purified_tokens
            applied_level = PurificationLevel.SUMMARIZED

            # 完整性校验
            self._check_integrity(result)

        # 更新最终结果的元数据
        result.metadata["pipeline_max_level"] = effective_max.name
        result.metadata["pipeline_applied_level"] = applied_level.name

        return result

    def purify_to_level(
        self,
        text: str,
        target_level: PurificationLevel,
        original_tokens: int = 0,
    ) -> PurificationResult:
        """
        净化到指定级别

        Args:
            text: 原始文本
            target_level: 目标净化级别
            original_tokens: 原始Token数

        Returns:
            净化结果
        """
        if target_level == PurificationLevel.NONE:
            if original_tokens == 0:
                original_tokens = self._estimate_tokens(text)
            return PurificationResult(
                original_text=text,
                purified_text=text,
                level=PurificationLevel.NONE,
                original_tokens=original_tokens,
                purified_tokens=original_tokens,
            )

        return self.purify(text, original_tokens, max_level=target_level)

    def _check_integrity(self, result: PurificationResult) -> None:
        """完整性校验"""
        if result.integrity_score < self._integrity_threshold:
            raise IntegrityCheckFailed(
                original_tokens=result.original_tokens,
                preserved_ratio=result.integrity_score,
                required_ratio=self._integrity_threshold,
            )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """简单估算Token数"""
        return max(1, len(text) // 4)
