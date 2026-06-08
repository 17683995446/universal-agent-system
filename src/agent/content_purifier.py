"""
无上限上下文Agent系统 - 内容净化器

统一净化入口，管理输入净化和输出净化。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-26
"""

import logging
from typing import Optional, Any, Callable

from src.modules.purification.pipeline import PurificationPipeline
from src.modules.purification.purification_types import PurificationLevel, PurificationResult
from src.modules.purification.rule_denoiser import RuleDenoiser
from src.modules.purification.structure_converter import StructureConverter
from src.modules.purification.summarizer import Summarizer

logger = logging.getLogger(__name__)


class ContentPurifier:
    """
    内容净化器
    
    统一管理输入净化和输出净化：
    - 输入净化：净化用户输入和召回的记忆
    - 输出净化：净化LLM响应和工具结果
    
    Attributes:
        pipeline: 净化管道
        enable_input_purification: 是否启用输入净化
        enable_output_purification: 是否启用输出净化
        input_max_level: 输入最大净化级别
        output_max_level: 输出最大净化级别
    """

    def __init__(
        self,
        pipeline: Optional[PurificationPipeline] = None,
        enable_input_purification: bool = True,
        enable_output_purification: bool = False,
        input_max_level: PurificationLevel = PurificationLevel.STRUCTURED,
        output_max_level: PurificationLevel = PurificationLevel.RULE_BASED,
        llm_fn: Optional[Callable[[str], str]] = None,
    ):
        """
        Args:
            pipeline: 净化管道（None则自动创建）
            enable_input_purification: 是否净化输入
            enable_output_purification: 是否净化输出
            input_max_level: 输入最大净化级别
            output_max_level: 输出最大净化级别
            llm_fn: LLM摘要函数（用于三级净化）
        """
        if pipeline:
            self._pipeline = pipeline
        else:
            # 自动创建管道
            summarizer = Summarizer(llm_fn=llm_fn) if llm_fn else Summarizer()
            self._pipeline = PurificationPipeline(
                denoiser=RuleDenoiser(),
                converter=StructureConverter(),
                summarizer=summarizer,
            )

        self._enable_input = enable_input_purification
        self._enable_output = enable_output_purification
        self._input_max_level = input_max_level
        self._output_max_level = output_max_level

        # 统计
        self._input_purifications: int = 0
        self._output_purifications: int = 0
        self._total_tokens_saved: int = 0

    # =========================================================================
    # 属性访问
    # =========================================================================

    @property
    def enable_input_purification(self) -> bool:
        """是否启用输入净化"""
        return self._enable_input

    @property
    def enable_output_purification(self) -> bool:
        """是否启用输出净化"""
        return self._enable_output

    @property
    def input_max_level(self) -> PurificationLevel:
        """输入最大净化级别"""
        return self._input_max_level

    @property
    def output_max_level(self) -> PurificationLevel:
        """输出最大净化级别"""
        return self._output_max_level

    # =========================================================================
    # 输入净化
    # =========================================================================

    def purify_input(
        self,
        text: str,
        original_tokens: int = 0,
        max_level: Optional[PurificationLevel] = None,
    ) -> PurificationResult:
        """
        净化输入内容
        
        Args:
            text: 输入文本
            original_tokens: 原始Token数
            max_level: 最大净化级别
        
        Returns:
            PurificationResult
        """
        if not self._enable_input:
            # 不净化，返回原始结果
            if original_tokens == 0:
                original_tokens = max(1, len(text) // 4)
            return PurificationResult(
                original_text=text,
                purified_text=text,
                level=PurificationLevel.NONE,
                original_tokens=original_tokens,
                purified_tokens=original_tokens,
            )

        effective_level = max_level if max_level is not None else self._input_max_level

        try:
            result = self._pipeline.purify_to_level(
                text=text,
                target_level=effective_level,
                original_tokens=original_tokens,
            )

            self._input_purifications += 1
            self._total_tokens_saved += result.tokens_saved()

            if result.is_compressed():
                logger.debug(
                    f"Input purified: level={result.level.name}, "
                    f"tokens={result.original_tokens}→{result.purified_tokens}"
                )

            return result

        except Exception as e:
            logger.warning(f"Input purification failed: {e}")
            # 返回原始文本
            if original_tokens == 0:
                original_tokens = max(1, len(text) // 4)
            return PurificationResult(
                original_text=text,
                purified_text=text,
                level=PurificationLevel.NONE,
                original_tokens=original_tokens,
                purified_tokens=original_tokens,
                metadata={"error": str(e)},
            )

    def purify_user_input(self, user_input: str) -> str:
        """
        净化用户输入（便捷方法）
        
        Args:
            user_input: 用户输入
        
        Returns:
            净化后的文本
        """
        result = self.purify_input(user_input)
        return result.purified_text

    def purify_memory_context(self, memory_context: str) -> str:
        """
        净化记忆上下文
        
        Args:
            memory_context: 记忆上下文
        
        Returns:
            净化后的文本
        """
        if not memory_context:
            return memory_context

        result = self.purify_input(
            memory_context,
            max_level=PurificationLevel.RULE_BASED,  # 记忆只做基础净化
        )
        return result.purified_text

    # =========================================================================
    # 输出净化
    # =========================================================================

    def purify_output(
        self,
        text: str,
        original_tokens: int = 0,
        max_level: Optional[PurificationLevel] = None,
    ) -> PurificationResult:
        """
        净化输出内容
        
        Args:
            text: 输出文本
            original_tokens: 原始Token数
            max_level: 最大净化级别
        
        Returns:
            PurificationResult
        """
        if not self._enable_output:
            if original_tokens == 0:
                original_tokens = max(1, len(text) // 4)
            return PurificationResult(
                original_text=text,
                purified_text=text,
                level=PurificationLevel.NONE,
                original_tokens=original_tokens,
                purified_tokens=original_tokens,
            )

        effective_level = max_level if max_level is not None else self._output_max_level

        try:
            result = self._pipeline.purify_to_level(
                text=text,
                target_level=effective_level,
                original_tokens=original_tokens,
            )

            self._output_purifications += 1
            self._total_tokens_saved += result.tokens_saved()

            if result.is_compressed():
                logger.debug(
                    f"Output purified: level={result.level.name}, "
                    f"tokens={result.original_tokens}→{result.purified_tokens}"
                )

            return result

        except Exception as e:
            logger.warning(f"Output purification failed: {e}")
            if original_tokens == 0:
                original_tokens = max(1, len(text) // 4)
            return PurificationResult(
                original_text=text,
                purified_text=text,
                level=PurificationLevel.NONE,
                original_tokens=original_tokens,
                purified_tokens=original_tokens,
                metadata={"error": str(e)},
            )

    def purify_assistant_reply(self, reply: str) -> str:
        """
        净化助手回复（便捷方法）
        
        Args:
            reply: 助手回复
        
        Returns:
            净化后的文本
        """
        result = self.purify_output(reply)
        return result.purified_text

    def purify_tool_result(self, result: Any) -> str:
        """
        净化工具结果
        
        Args:
            result: 工具执行结果
        
        Returns:
            净化后的文本
        """
        if result is None:
            return ""

        # 转换为字符串
        if isinstance(result, str):
            text = result
        elif isinstance(result, (dict, list)):
            import json
            text = json.dumps(result, ensure_ascii=False)
        else:
            text = str(result)

        # 净化（工具结果只做基础净化）
        purification_result = self.purify_output(
            text,
            max_level=PurificationLevel.RULE_BASED,
        )
        return purification_result.purified_text

    # =========================================================================
    # 批量净化
    # =========================================================================

    def purify_messages(
        self,
        messages: list[dict[str, Any]],
        purify_role: str = "user",
    ) -> list[dict[str, Any]]:
        """
        批量净化消息
        
        Args:
            messages: 消息列表
            purify_role: 要净化的角色
        
        Returns:
            净化后的消息列表
        """
        purified = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == purify_role and content:
                purified_content = self.purify_input(content)
                purified.append({
                    **msg,
                    "content": purified_content,
                })
            else:
                purified.append(msg)

        return purified

    # =========================================================================
    # 配置
    # =========================================================================

    def set_enable_input(self, enable: bool) -> None:
        """设置是否启用输入净化"""
        self._enable_input = enable

    def set_enable_output(self, enable: bool) -> None:
        """设置是否启用输出净化"""
        self._enable_output = enable

    def set_input_max_level(self, level: PurificationLevel) -> None:
        """设置输入最大净化级别"""
        self._input_max_level = level

    def set_output_max_level(self, level: PurificationLevel) -> None:
        """设置输出最大净化级别"""
        self._output_max_level = level

    # =========================================================================
    # 统计
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        return {
            "input_purifications": self._input_purifications,
            "output_purifications": self._output_purifications,
            "total_purifications": self._input_purifications + self._output_purifications,
            "total_tokens_saved": self._total_tokens_saved,
            "enable_input": self._enable_input,
            "enable_output": self._enable_output,
            "input_max_level": self._input_max_level.name,
            "output_max_level": self._output_max_level.name,
        }

    def reset_stats(self) -> None:
        """重置统计"""
        self._input_purifications = 0
        self._output_purifications = 0
        self._total_tokens_saved = 0