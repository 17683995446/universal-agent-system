"""
无上限上下文Agent系统 - 上下文组装器

实现消息组装、智能截断、格式转换。
将系统消息、会话消息、召回记忆组装成最终上下文。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

from typing import Optional, Any
from dataclasses import dataclass, field

from src.core.message import Message, Role
from src.modules.memory_manager.memory_entry import MemoryEntry
from src.modules.context_manager.context_assembly import ContextAssembly, TokenBudget
from src.modules.context_manager.partitioner import Partitioner


# =============================================================================
# 组装配置
# =============================================================================

@dataclass
class AssemblyConfig:
    """
    组装配置

    定义上下文组装的行为参数
    """
    # 截断策略
    truncation_strategy: str = "oldest_first"  # oldest_first, priority_based

    # 是否保留系统消息（截断时）
    preserve_system: bool = True

    # 最小保留消息数
    min_messages: int = 1

    # 是否包含记忆
    include_memories: bool = True

    # 记忆最大Token占比（相对于会话区）
    memory_max_ratio: float = 0.30

    # 元数据
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 组装结果
# =============================================================================

@dataclass
class AssemblyResult:
    """
    组装结果

    包含组装后的上下文和统计信息
    """
    assembly: ContextAssembly
    success: bool
    truncated: bool
    messages_removed: int = 0
    tokens_saved: int = 0
    error: Optional[str] = None


# =============================================================================
# 上下文组装器
# =============================================================================

class ContextAssembler:
    """
    上下文组装器

    负责将系统消息、会话消息、召回记忆组装成最终上下文。
    支持智能截断、格式转换。
    """

    def __init__(self, config: Optional[AssemblyConfig] = None):
        """
        Args:
            config: 组装配置
        """
        self._config = config or AssemblyConfig()
        self._partitioner = Partitioner()

    def assemble(
        self,
        system_messages: list[Message],
        session_messages: list[Message],
        recalled_memories: Optional[list[MemoryEntry]] = None,
        budget: Optional[TokenBudget] = None,
        window_size: int = 4096,
    ) -> AssemblyResult:
        """
        组装上下文

        Args:
            system_messages: 系统消息列表
            session_messages: 会话消息列表
            recalled_memories: 召回的记忆（可选）
            budget: Token预算（None则自动分配）
            window_size: 窗口大小（budget为None时使用）

        Returns:
            AssemblyResult 组装结果
        """
        # 1. 分配预算
        if budget is None:
            budget = self._partitioner.allocate(window_size)

        # 2. 创建组装对象
        assembly = ContextAssembly(
            system_messages=list(system_messages),
            session_messages=list(session_messages),
            recalled_memories=list(recalled_memories) if recalled_memories else [],
            budget=budget,
        )

        # 3. 更新Token计数
        assembly.update_total_tokens()

        # 4. 检查是否需要截断
        if assembly.is_within_budget():
            return AssemblyResult(
                assembly=assembly,
                success=True,
                truncated=False,
            )

        # 5. 执行截断
        return self._truncate(assembly)

    def _truncate(self, assembly: ContextAssembly) -> AssemblyResult:
        """
        执行截断

        当消息超出预算时，按策略截断消息。

        Args:
            assembly: 当前组装

        Returns:
            AssemblyResult 截断结果
        """
        original_count = assembly.get_message_count()
        original_tokens = assembly.total_tokens

        # 计算需要释放的Token数
        excess_tokens = assembly.total_tokens - assembly.budget.available_for_messages

        if excess_tokens <= 0:
            return AssemblyResult(
                assembly=assembly,
                success=True,
                truncated=False,
            )

        # 执行截断策略
        if self._config.truncation_strategy == "oldest_first":
            self._truncate_oldest_first(assembly, excess_tokens)
        else:
            self._truncate_oldest_first(assembly, excess_tokens)  # 默认策略

        # 更新统计
        assembly.update_total_tokens()

        messages_removed = original_count - assembly.get_message_count()
        tokens_saved = original_tokens - assembly.total_tokens

        return AssemblyResult(
            assembly=assembly,
            success=assembly.is_within_budget(),
            truncated=messages_removed > 0,
            messages_removed=messages_removed,
            tokens_saved=tokens_saved,
            error=None if assembly.is_within_budget() else "截断后仍超预算",
        )

    def _truncate_oldest_first(
        self,
        assembly: ContextAssembly,
        excess_tokens: int,
    ) -> None:
        """
        最旧优先截断策略

        保留系统消息，从最早的会话消息开始截断。

        Args:
            assembly: 当前组装
            excess_tokens: 需要释放的Token数
        """
        # 优先截断会话消息（保留系统消息）
        session_msgs = assembly.session_messages

        # 从最早的非系统消息开始移除
        tokens_freed = 0
        to_remove = []

        for msg in session_msgs:
            if tokens_freed >= excess_tokens:
                break

            # 保留最小消息数
            remaining = len(session_msgs) - len(to_remove)
            if remaining <= self._config.min_messages:
                break

            tokens_freed += msg.token_count
            to_remove.append(msg)

        # 移除标记的消息
        for msg in to_remove:
            assembly.session_messages.remove(msg)

        # 记录截断信息
        assembly.add_metadata("truncated", True)
        assembly.add_metadata("truncation_strategy", "oldest_first")
        assembly.add_metadata("messages_removed", len(to_remove))
        assembly.add_metadata("tokens_freed", tokens_freed)

    def assemble_with_adaptive_budget(
        self,
        system_messages: list[Message],
        session_messages: list[Message],
        window_size: int = 4096,
    ) -> AssemblyResult:
        """
        使用自适应预算组装上下文

        根据系统消息实际所需Token数，弹性调整系统区比例。

        Args:
            system_messages: 系统消息列表
            session_messages: 会话消息列表
            window_size: 窗口大小

        Returns:
            AssemblyResult 组装结果
        """
        # 计算系统消息所需Token
        system_tokens = sum(m.token_count for m in system_messages)

        # 自适应分配预算
        budget = self._partitioner.allocate_adaptive(
            window_size,
            system_tokens_needed=system_tokens,
        )

        return self.assemble(
            system_messages=system_messages,
            session_messages=session_messages,
            budget=budget,
        )

    def add_memories_to_context(
        self,
        assembly: ContextAssembly,
        memories: list[MemoryEntry],
        max_tokens: Optional[int] = None,
    ) -> int:
        """
        将记忆添加到上下文

        将召回的长期记忆转换为系统消息添加到上下文。

        Args:
            assembly: 当前组装
            memories: 记忆列表
            max_tokens: 记忆最大Token数（None则使用配置比例）

        Returns:
            添加的记忆数量
        """
        if not self._config.include_memories or not memories:
            return 0

        # 计算记忆Token上限
        if max_tokens is None:
            max_tokens = int(
                assembly.budget.session_zone * self._config.memory_max_ratio
            )

        added = 0
        tokens_used = 0

        for memory in memories:
            # 检查是否超上限
            if tokens_used + memory.token_count > max_tokens:
                break

            # 将记忆转换为系统消息
            memory_msg = Message(
                role=Role.SYSTEM,
                content=f"[Memory] {memory.content}",
                metadata={
                    "source": "memory",
                    "memory_id": memory.memory_id,
                    "relevance": memory.relevance_score,
                },
            )

            assembly.system_messages.append(memory_msg)
            assembly.recalled_memories.append(memory)
            tokens_used += memory.token_count
            added += 1

        assembly.update_total_tokens()
        assembly.add_metadata("memories_added", added)
        assembly.add_metadata("memory_tokens_used", tokens_used)

        return added

    def to_openai_format(
        self,
        assembly: ContextAssembly,
        include_metadata: bool = False,
    ) -> list[dict[str, Any]]:
        """
        转换为OpenAI API格式

        Args:
            assembly: 上下文组装
            include_metadata: 是否包含元数据

        Returns:
            OpenAI格式的消息列表
        """
        messages = assembly.to_openai_format()

        if include_metadata:
            return [
                {
                    **msg,
                    "_metadata": {
                        "budget_usage": assembly.get_budget_usage_ratio(),
                        "truncated": assembly.get_metadata("truncated", False),
                    }
                }
                for msg in messages
            ]

        return messages

    def validate_assembly(self, assembly: ContextAssembly) -> tuple[bool, list[str]]:
        """
        验证组装结果

        Args:
            assembly: 上下文组装

        Returns:
            (是否有效, 错误列表)
        """
        errors = []

        # 基础验证
        valid, base_errors = assembly.validate()
        if not valid:
            errors.extend(base_errors)

        # 检查预算
        if not assembly.is_within_budget():
            errors.append(
                f"超出预算: {assembly.total_tokens} > "
                f"{assembly.budget.available_for_messages}"
            )

        # 检查消息顺序（系统消息应在会话消息之前）
        all_msgs = assembly.to_messages()
        seen_non_system = False
        for msg in all_msgs:
            if msg.role != Role.SYSTEM:
                seen_non_system = True
            elif seen_non_system and msg.role == Role.SYSTEM:
                errors.append("系统消息不应出现在非系统消息之后")
                break

        return len(errors) == 0, errors


# =============================================================================
# 便捷函数
# =============================================================================

def quick_assemble(
    system_prompt: str,
    user_message: str,
    history: Optional[list[Message]] = None,
    window_size: int = 4096,
) -> ContextAssembly:
    """
    快速组装上下文

    便捷函数，用于简单的单轮对话场景。

    Args:
        system_prompt: 系统提示
        user_message: 用户消息
        history: 历史消息（可选）
        window_size: 窗口大小

    Returns:
        ContextAssembly 组装结果
    """
    from src.core.message import create_system_message, create_user_message

    assembler = ContextAssembler()

    system_msgs = [create_system_message(system_prompt)]
    session_msgs = list(history) if history else []
    session_msgs.append(create_user_message(user_message))

    result = assembler.assemble(
        system_messages=system_msgs,
        session_messages=session_msgs,
        window_size=window_size,
    )

    return result.assembly
