"""
无上限上下文Agent系统 - 三级净化器（LLM摘要）

使用小模型对文本进行摘要压缩。
Token > 4000 时触发，压缩到原文的30%，至少保留500 Token。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

import re
from typing import Optional, Callable

from src.modules.purification.purification_types import PurificationResult, PurificationLevel
from src.core.constants import (
    SUMMARIZE_THRESHOLD,
    SUMMARIZE_TARGET_RATIO,
    SUMMARIZE_MIN_TOKENS,
    CRITICAL_INFO_PATTERNS,
)
from src.core.exceptions import PurificationError


class Summarizer:
    """
    三级净化器：LLM摘要

    使用小模型对文本进行摘要压缩。
    当前实现为提取式摘要（规则基），生产环境替换为LLM调用。

    特点：
    - 提取关键句子组成摘要
    - 保留关键信息模式匹配的句子
    - 控制输出长度在目标范围内
    """

    def __init__(
        self,
        threshold: int = SUMMARIZE_THRESHOLD,
        target_ratio: float = SUMMARIZE_TARGET_RATIO,
        min_tokens: int = SUMMARIZE_MIN_TOKENS,
        llm_fn: Optional[Callable] = None,
    ):
        """
        Args:
            threshold: 触发摘要的Token阈值
            target_ratio: 摘要目标比例（保留原文的多少）
            min_tokens: 最小保留Token数
            llm_fn: LLM摘要函数（可选，用于生产环境）
        """
        self._threshold = threshold
        self._target_ratio = target_ratio
        self._min_tokens = min_tokens
        self._llm_fn = llm_fn
        self._critical_patterns = [
            re.compile(p) for p in CRITICAL_INFO_PATTERNS
        ]

    def should_summarize(self, token_count: int) -> bool:
        """判断是否需要摘要"""
        return token_count > self._threshold

    def summarize(
        self,
        text: str,
        original_tokens: int = 0,
    ) -> PurificationResult:
        """
        执行摘要压缩

        Args:
            text: 经二级转换后的文本
            original_tokens: 原始Token数

        Returns:
            PurificationResult
        """
        if not text or not text.strip():
            return PurificationResult(
                original_text=text,
                purified_text=text,
                level=PurificationLevel.SUMMARIZED,
                original_tokens=original_tokens,
                purified_tokens=original_tokens,
                integrity_score=1.0,
            )

        try:
            if original_tokens == 0:
                original_tokens = self._estimate_tokens(text)

            # 计算目标Token数
            target_tokens = max(
                self._min_tokens,
                int(original_tokens * self._target_ratio),
            )

            # 如果有LLM函数，使用LLM摘要
            if self._llm_fn is not None:
                purified = self._llm_fn(text, target_tokens)
            else:
                # 降级：提取式摘要
                purified = self._extractive_summarize(text, target_tokens)

            purified_tokens = self._estimate_tokens(purified)

            # 提取保留的关键信息
            preserved_keys = self._extract_preserved_keys(text, purified)

            # 计算完整性
            integrity = self._calculate_integrity(text, purified)

            return PurificationResult(
                original_text=text,
                purified_text=purified,
                level=PurificationLevel.SUMMARIZED,
                original_tokens=original_tokens,
                purified_tokens=purified_tokens,
                integrity_score=integrity,
                preserved_keys=preserved_keys,
                metadata={
                    "summarization_method": "llm" if self._llm_fn else "extractive",
                    "target_tokens": target_tokens,
                },
            )
        except Exception as e:
            raise PurificationError(text, "SUMMARIZED", str(e))

    def _extractive_summarize(self, text: str, target_tokens: int) -> str:
        """
        提取式摘要

        策略：
        1. 按句子分割
        2. 优先保留关键信息句子
        3. 按位置权重排序（首尾句子权重高）
        4. 截取到目标Token数
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= 3:
            return text  # 太短不需要摘要

        # 为每个句子评分
        scored = []
        for i, sentence in enumerate(sentences):
            score = 0.0

            # 关键信息加分
            if any(p.search(sentence) for p in self._critical_patterns):
                score += 10.0

            # 位置权重：首尾句子更重要
            position_ratio = i / max(1, len(sentences) - 1)
            if position_ratio < 0.2:  # 前20%
                score += 3.0
            elif position_ratio > 0.8:  # 后20%
                score += 2.0

            # 长度适中的句子优先
            sent_len = len(sentence)
            if 20 < sent_len < 200:
                score += 1.0

            scored.append((sentence, score))

        # 按评分排序
        scored.sort(key=lambda x: x[1], reverse=True)

        # 选择句子直到达到目标Token数
        selected = []
        current_tokens = 0

        for sentence, score in scored:
            sent_tokens = self._estimate_tokens(sentence)
            if current_tokens + sent_tokens > target_tokens:
                continue
            selected.append((sentence, score))
            current_tokens += sent_tokens

        # 按原始顺序排列
        selected_set = {s for s, _ in selected}
        result_sentences = [s for s in sentences if s in selected_set]

        return "".join(result_sentences)

    def _split_sentences(self, text: str) -> list[str]:
        """按句子分割文本"""
        # 中文句号、问号、感叹号、英文句号
        parts = re.split(r"(?<=[。！？.!?\n])", text)
        return [p for p in parts if p.strip()]

    def _extract_preserved_keys(
        self, original: str, summary: str
    ) -> list[str]:
        """提取保留的关键信息"""
        keys = []
        for pattern in self._critical_patterns:
            matches = pattern.findall(original)
            for match in matches:
                if match in summary:
                    keys.append(match)
        return list(set(keys))

    def _calculate_integrity(self, original: str, summary: str) -> float:
        """计算完整性评分"""
        # 基于关键信息保留率
        all_keys = []
        for pattern in self._critical_patterns:
            matches = pattern.findall(original)
            all_keys.extend(matches)

        if not all_keys:
            return 0.9  # 没有关键信息，给一个较高的默认值

        preserved = sum(1 for k in all_keys if k in summary)
        return preserved / len(all_keys)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """简单估算Token数"""
        return max(1, len(text) // 4)
