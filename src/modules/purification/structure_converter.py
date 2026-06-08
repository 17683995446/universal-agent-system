"""
无上限上下文Agent系统 - 二级净化器（结构化转换）

将非结构化文本转为结构化格式以减少Token消耗。
Token > 2000 时触发。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

import re
from typing import Optional

from src.modules.purification.purification_types import PurificationResult, PurificationLevel
from src.core.constants import STRUCTURE_CONVERT_THRESHOLD, CRITICAL_INFO_PATTERNS
from src.core.exceptions import PurificationError


class StructureConverter:
    """
    二级净化器：结构化转换

    处理规则：
    1. 将冗长的描述性文本转为要点列表
    2. 提取关键信息（参数、结论、步骤等）
    3. 移除填充词和冗余表达
    4. 保留关键信息模式匹配的内容
    """

    def __init__(self, threshold: int = STRUCTURE_CONVERT_THRESHOLD):
        self._threshold = threshold
        # 编译关键信息正则
        self._critical_patterns = [
            re.compile(p) for p in CRITICAL_INFO_PATTERNS
        ]

    def should_convert(self, token_count: int) -> bool:
        """判断是否需要结构化转换"""
        return token_count > self._threshold

    def convert(self, text: str, original_tokens: int = 0) -> PurificationResult:
        """
        执行结构化转换

        Args:
            text: 经一级去噪后的文本
            original_tokens: 原始Token数

        Returns:
            PurificationResult
        """
        if not text or not text.strip():
            return PurificationResult(
                original_text=text,
                purified_text=text,
                level=PurificationLevel.STRUCTURED,
                original_tokens=original_tokens,
                purified_tokens=original_tokens,
                integrity_score=1.0,
            )

        try:
            # 提取关键信息
            preserved_keys = self._extract_critical_info(text)

            # 执行转换
            purified = self._convert_to_structured(text)

            # 估算Token数
            if original_tokens == 0:
                original_tokens = self._estimate_tokens(text)
            purified_tokens = self._estimate_tokens(purified)

            # 计算完整性评分
            integrity = self._calculate_integrity(text, purified, preserved_keys)

            return PurificationResult(
                original_text=text,
                purified_text=purified,
                level=PurificationLevel.STRUCTURED,
                original_tokens=original_tokens,
                purified_tokens=purified_tokens,
                integrity_score=integrity,
                preserved_keys=preserved_keys,
                metadata={"conversion_applied": True},
            )
        except Exception as e:
            raise PurificationError(text, "STRUCTURED", str(e))

    def _extract_critical_info(self, text: str) -> list[str]:
        """提取关键信息标识"""
        keys = []
        for pattern in self._critical_patterns:
            matches = pattern.findall(text)
            keys.extend(matches)
        return list(set(keys))

    def _convert_to_structured(self, text: str) -> str:
        """
        将文本转为结构化格式

        策略：
        - 将长段落拆分为要点
        - 移除填充词
        - 保留关键信息行
        """
        lines = text.split("\n")
        result = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                result.append("")
                continue

            # 检查是否为关键信息行（必须保留）
            is_critical = any(p.search(stripped) for p in self._critical_patterns)

            if is_critical:
                result.append(stripped)
            else:
                # 对非关键行进行精简
                simplified = self._simplify_line(stripped)
                result.append(simplified)

        purified = "\n".join(result)
        return purified.strip()

    def _simplify_line(self, line: str) -> str:
        """
        精简单行文本

        移除常见填充词和冗余表达
        """
        # 移除常见填充词前缀
        fillers = [
            r"^(好的|当然|没问题|明白了|了解了|我知道了)[，,。.、\s]+",
            r"^(那么|然后|接下来|另外|此外)[，,、\s]+",
            r"^(简单来说|总而言之|综上所述|换句话说)[，,：:\s]+",
            r"^(我认为|我觉得|在我看来|个人认为)[，,：:\s]+",
        ]
        simplified = line
        for filler in fillers:
            simplified = re.sub(filler, "", simplified, flags=re.IGNORECASE)

        # 移除句末填充
        simplified = re.sub(
            r"[，,]?(希望这有帮助|希望对你有用|如有问题请告知|谢谢)[。.！!]*$",
            "", simplified
        )

        return simplified.strip() if simplified.strip() else line

    def _calculate_integrity(
        self, original: str, purified: str, preserved_keys: list[str]
    ) -> float:
        """计算完整性评分"""
        if not preserved_keys:
            return 1.0  # 没有关键信息需要保留

        preserved_count = 0
        for key in preserved_keys:
            if key in purified:
                preserved_count += 1

        return preserved_count / len(preserved_keys)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """简单估算Token数"""
        return max(1, len(text) // 4)
