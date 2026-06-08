"""
无上限上下文Agent系统 - 一级净化器（规则去噪）

通过正则和规则移除冗余空白、重复内容、格式噪声。
始终执行，无阈值门槛。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

import re
from typing import Optional

from src.modules.purification.purification_types import PurificationResult, PurificationLevel
from src.core.constants import DENOISE_MAX_EMPTY_LINES, DENOISE_REPEAT_WINDOW
from src.core.exceptions import PurificationError


class RuleDenoiser:
    """
    一级净化器：规则去噪

    处理规则：
    1. 合并连续空行（超过阈值时合并为单行）
    2. 移除行首行尾空白
    3. 移除重复段落（滑动窗口检测）
    4. 统一换行符
    5. 移除多余空格
    """

    def __init__(
        self,
        max_empty_lines: int = DENOISE_MAX_EMPTY_LINES,
        repeat_window: int = DENOISE_REPEAT_WINDOW,
    ):
        self._max_empty_lines = max_empty_lines
        self._repeat_window = repeat_window

    def denoise(self, text: str, original_tokens: int = 0) -> PurificationResult:
        """
        执行规则去噪

        Args:
            text: 原始文本
            original_tokens: 原始Token数（0则估算）

        Returns:
            PurificationResult
        """
        if not text or not text.strip():
            return PurificationResult(
                original_text=text,
                purified_text=text,
                level=PurificationLevel.RULE_BASED,
                original_tokens=original_tokens,
                purified_tokens=original_tokens,
                integrity_score=1.0,
            )

        try:
            purified = text

            # 1. 统一换行符
            purified = self._normalize_newlines(purified)

            # 2. 移除行首行尾空白
            purified = self._strip_lines(purified)

            # 3. 合并连续空行
            purified = self._merge_empty_lines(purified)

            # 4. 移除重复段落
            purified = self._remove_repeated_paragraphs(purified)

            # 5. 移除多余空格
            purified = self._collapse_spaces(purified)

            # 6. 清理首尾
            purified = purified.strip()

            # 估算净化后Token数
            purified_tokens = self._estimate_tokens(purified) if original_tokens == 0 else self._estimate_tokens(purified)

            if original_tokens == 0:
                original_tokens = self._estimate_tokens(text)

            return PurificationResult(
                original_text=text,
                purified_text=purified,
                level=PurificationLevel.RULE_BASED,
                original_tokens=original_tokens,
                purified_tokens=purified_tokens,
                integrity_score=1.0,  # 规则去噪不丢失语义信息
                metadata={"rules_applied": [
                    "normalize_newlines", "strip_lines",
                    "merge_empty_lines", "remove_repeated",
                    "collapse_spaces",
                ]},
            )
        except Exception as e:
            raise PurificationError(text, "RULE_BASED", str(e))

    def _normalize_newlines(self, text: str) -> str:
        """统一换行符为\\n"""
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _strip_lines(self, text: str) -> str:
        """移除每行首尾空白"""
        lines = text.split("\n")
        return "\n".join(line.strip() for line in lines)

    def _merge_empty_lines(self, text: str) -> str:
        """合并连续空行"""
        lines = text.split("\n")
        result = []
        empty_count = 0

        for line in lines:
            if not line.strip():
                empty_count += 1
                if empty_count <= self._max_empty_lines:
                    result.append("")
            else:
                empty_count = 0
                result.append(line)

        return "\n".join(result)

    def _remove_repeated_paragraphs(self, text: str) -> str:
        """移除重复段落（滑动窗口）"""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) <= 1:
            return text

        seen = []
        result = []

        for para in paragraphs:
            # 简单相似度：完全匹配或归一化后匹配
            normalized = re.sub(r"\s+", " ", para.lower().strip())
            is_duplicate = False

            for prev in seen[-self._repeat_window:]:
                if normalized == prev:
                    is_duplicate = True
                    break

            if not is_duplicate:
                seen.append(normalized)
                result.append(para)

        return "\n\n".join(result)

    def _collapse_spaces(self, text: str) -> str:
        """移除多余空格（保留换行）"""
        lines = text.split("\n")
        result = []
        for line in lines:
            # 将多个空格合并为一个
            collapsed = re.sub(r" {2,}", " ", line)
            result.append(collapsed)
        return "\n".join(result)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """简单估算Token数（约4字符/token）"""
        return max(1, len(text) // 4)
