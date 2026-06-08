"""
无上限上下文Agent系统 - 上下文构建器

统一构建LLM上下文，整合Token预算、消息组装、记忆注入、截断策略。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-26
"""

import logging
from typing import Optional, Any, TYPE_CHECKING

from src.core.message import Message, Role, create_system_message, create_user_message, create_assistant_message
from src.modules.context_manager.context_assembly import ContextAssembly, TokenBudget
from src.modules.context_manager.partitioner import Partitioner, StrategyPartitioner, PartitionStrategy
from src.modules.context_manager.assembler import ContextAssembler, AssemblyConfig, AssemblyResult
from src.modules.memory_manager.memory_entry import MemoryEntry

if TYPE_CHECKING:
    from src.modules.memory_manager.coordinator import MemoryCoordinator
    from src.agent.context_compressor import ContextCompressor

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    上下文构建器
    
    统一管理上下文构建流程：
    1. 分配Token预算
    2. 构建系统消息
    3. 构建会话消息（历史+当前输入）
    4. 注入召回记忆
    5. 执行截断（如超预算）
    6. 转换为LLM格式
    
    Attributes:
        assembler: 上下文组装器
        partitioner: 分区器
        strategy_partitioner: 策略分区器
        window_size: 默认窗口大小
        strategy: 默认分区策略
    """

    def __init__(
        self,
        window_size: int = 4096,
        strategy: str = PartitionStrategy.DEFAULT,
        truncation_strategy: str = "oldest_first",
        preserve_system: bool = True,
        min_messages: int = 1,
        memory_max_ratio: float = 0.30,
        compressor: Optional["ContextCompressor"] = None,
    ):
        """
        Args:
            window_size: 默认Token窗口大小
            strategy: 分区策略
            truncation_strategy: 截断策略
            preserve_system: 是否保留系统消息
            min_messages: 最小保留消息数
            memory_max_ratio: 记忆最大Token占比
            compressor: 上下文压缩器（可选）
        """
        self._window_size = window_size
        self._strategy = strategy
        self._compressor = compressor
        
        # 组装配置
        self._assembly_config = AssemblyConfig(
            truncation_strategy=truncation_strategy,
            preserve_system=preserve_system,
            min_messages=min_messages,
            memory_max_ratio=memory_max_ratio,
        )
        
        # 组装器和分区器
        self._assembler = ContextAssembler(config=self._assembly_config)
        self._partitioner = Partitioner()
        self._strategy_partitioner = StrategyPartitioner()
        
        # 统计
        self._total_builds: int = 0
        self._total_truncations: int = 0
        self._total_tokens_used: int = 0
        self._last_compression_result: Optional[Any] = None

    # =========================================================================
    # 核心构建方法
    # =========================================================================

    def build(
        self,
        system_prompt: str,
        history: list[dict[str, Any]],
        user_input: str,
        memory_context: Optional[str] = None,
        window_size: Optional[int] = None,
        strategy: Optional[str] = None,
    ) -> AssemblyResult:
        """
        构建完整上下文
        
        Args:
            system_prompt: 系统提示词
            history: 历史消息列表（OpenAI格式）
            user_input: 当前用户输入
            memory_context: 记忆上下文（可选）
            window_size: 窗口大小（None则使用默认值）
            strategy: 分区策略（None则使用默认值）
        
        Returns:
            AssemblyResult 组装结果
        """
        window_size = window_size if window_size is not None else self._window_size
        strategy = strategy if strategy is not None else self._strategy
        
        # 1. 分配预算
        budget = self._strategy_partitioner.allocate(window_size, strategy)
        
        # 2. 构建系统消息
        system_messages = self._build_system_messages(system_prompt, memory_context)
        
        # 3. 构建会话消息（可能经过压缩）
        effective_history = self._maybe_compress_history(history, window_size)
        session_messages = self._build_session_messages(effective_history, user_input)
        
        # 4. 组装上下文
        result = self._assembler.assemble(
            system_messages=system_messages,
            session_messages=session_messages,
            budget=budget,
            window_size=window_size,
        )
        
        # 5. 更新统计
        self._update_stats(result)
        
        return result

    def build_with_adaptive_budget(
        self,
        system_prompt: str,
        history: list[dict[str, Any]],
        user_input: str,
        memory_context: Optional[str] = None,
        window_size: Optional[int] = None,
    ) -> AssemblyResult:
        """
        使用自适应预算构建上下文
        
        根据系统消息实际Token需求弹性调整预算。
        
        Args:
            system_prompt: 系统提示词
            history: 历史消息列表
            user_input: 用户输入
            memory_context: 记忆上下文
            window_size: 窗口大小
        
        Returns:
            AssemblyResult 组装结果
        """
        window_size = window_size if window_size is not None else self._window_size
        
        # 构建系统消息
        system_messages = self._build_system_messages(system_prompt, memory_context)
        
        # 计算系统消息Token数
        system_tokens = sum(m.token_count for m in system_messages)
        
        # 自适应分配预算
        budget = self._partitioner.allocate_adaptive(
            window_size,
            system_tokens_needed=system_tokens,
        )
        
        # 构建会话消息
        session_messages = self._build_session_messages(history, user_input)
        
        # 组装
        result = self._assembler.assemble(
            system_messages=system_messages,
            session_messages=session_messages,
            budget=budget,
            window_size=window_size,
        )
        
        self._update_stats(result)
        return result

    def build_with_memories(
        self,
        system_prompt: str,
        history: list[dict[str, Any]],
        user_input: str,
        memories: list[MemoryEntry],
        window_size: Optional[int] = None,
        strategy: Optional[str] = None,
    ) -> AssemblyResult:
        """
        带记忆注入的上下文构建
        
        Args:
            system_prompt: 系统提示词
            history: 历史消息列表
            user_input: 用户输入
            memories: 召回的记忆列表
            window_size: 窗口大小
            strategy: 分区策略
        
        Returns:
            AssemblyResult 组装结果
        """
        result = self.build(
            system_prompt=system_prompt,
            history=history,
            user_input=user_input,
            window_size=window_size,
            strategy=strategy,
        )
        
        # 注入记忆
        if memories:
            added = self._assembler.add_memories_to_context(
                assembly=result.assembly,
                memories=memories,
            )
            logger.debug(f"Added {added} memories to context")
        
        return result

    # =========================================================================
    # 压缩集成
    # =========================================================================

    @property
    def compressor(self) -> Optional["ContextCompressor"]:
        """上下文压缩器"""
        return self._compressor

    @property
    def last_compression_result(self) -> Optional[Any]:
        """最近一次压缩结果"""
        return self._last_compression_result

    def set_compressor(self, compressor: Optional["ContextCompressor"]) -> None:
        """设置压缩器"""
        self._compressor = compressor

    def _maybe_compress_history(
        self,
        history: list[dict[str, Any]],
        window_size: int,
    ) -> list[dict[str, Any]]:
        """
        根据Token压力决定是否压缩历史消息

        Args:
            history: 原始历史消息
            window_size: Token窗口大小

        Returns:
            压缩后的历史消息（或原始消息）
        """
        self._last_compression_result = None

        if self._compressor is None or not history:
            return history

        # 估算历史消息Token数
        estimated_tokens = sum(
            max(1, len(msg.get("content", "")) // 4)
            for msg in history
        )

        try:
            compressed_history, result = self._compressor.compress_messages(
                messages=history,
                used_tokens=estimated_tokens,
                window_size=window_size,
            )
            self._last_compression_result = result

            if result.was_compressed:
                logger.info(
                    f"History compressed: {result.messages_compressed} messages, "
                    f"saved ~{result.tokens_saved} tokens "
                    f"(pressure={result.pressure_level})"
                )

            return compressed_history

        except Exception as e:
            logger.warning(f"History compression failed, using original: {e}")
            return history

    # =========================================================================
    # 消息构建辅助
    # =========================================================================

    def _build_system_messages(
        self,
        system_prompt: str,
        memory_context: Optional[str] = None,
    ) -> list[Message]:
        """构建系统消息列表"""
        messages = []
        
        # 主系统提示
        if system_prompt:
            messages.append(create_system_message(system_prompt))
        
        # 记忆上下文（作为额外系统消息）
        if memory_context:
            messages.append(create_system_message(
                memory_context,
                metadata={"source": "memory_recall"}
            ))
        
        return messages

    def _build_session_messages(
        self,
        history: list[dict[str, Any]],
        user_input: str,
    ) -> list[Message]:
        """构建会话消息列表"""
        messages = []
        
        # 历史消息
        for msg in history:
            role_str = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role_str == "user":
                messages.append(create_user_message(content))
            elif role_str == "assistant":
                messages.append(create_assistant_message(content))
            elif role_str == "system":
                messages.append(create_system_message(content))
            elif role_str == "tool":
                # tool消息作为系统消息处理
                tool_call_id = msg.get("tool_call_id", "")
                messages.append(create_system_message(
                    f"[Tool Result {tool_call_id}] {content}",
                    metadata={"role": "tool", "tool_call_id": tool_call_id}
                ))
        
        # 当前用户输入
        messages.append(create_user_message(user_input))
        
        return messages

    # =========================================================================
    # 统计
    # =========================================================================

    def _update_stats(self, result: AssemblyResult) -> None:
        """更新统计"""
        self._total_builds += 1
        self._total_tokens_used += result.assembly.total_tokens
        if result.truncated:
            self._total_truncations += 1

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        return {
            "total_builds": self._total_builds,
            "total_truncations": self._total_truncations,
            "total_tokens_used": self._total_tokens_used,
            "avg_tokens_per_build": (
                self._total_tokens_used / self._total_builds
                if self._total_builds > 0 else 0
            ),
            "truncation_rate": (
                self._total_truncations / self._total_builds
                if self._total_builds > 0 else 0
            ),
        }

    def reset_stats(self) -> None:
        """重置统计"""
        self._total_builds = 0
        self._total_truncations = 0
        self._total_tokens_used = 0

    # =========================================================================
    # 配置更新
    # =========================================================================

    def set_window_size(self, size: int) -> None:
        """设置窗口大小"""
        self._window_size = size

    def set_strategy(self, strategy: str) -> None:
        """设置分区策略"""
        self._strategy = strategy

    def get_available_strategies(self) -> list[str]:
        """获取可用策略列表"""
        return self._strategy_partitioner.list_strategies()

    # =========================================================================
    # 便捷方法
    # =========================================================================

    def to_llm_messages(self, result: AssemblyResult) -> list[dict[str, str]]:
        """
        将组装结果转换为LLM消息格式
        
        Args:
            result: 组装结果
        
        Returns:
            OpenAI格式的消息列表
        """
        return result.assembly.to_openai_format()

    def get_context_stats(self, result: AssemblyResult) -> dict[str, Any]:
        """
        获取上下文统计
        
        Args:
            result: 组装结果
        
        Returns:
            统计信息
        """
        return result.assembly.get_stats()