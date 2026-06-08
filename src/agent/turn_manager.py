"""
无上限上下文Agent系统 - 回合管理器

管理单轮对话的完整流程：接收输入、调用LLM、处理工具调用。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-26
"""

import time
import logging
from typing import Optional, Any, TYPE_CHECKING

from src.agent.agent_types import TurnResult, TurnStatus
from src.infrastructure.llm_types import LLMMessage, LLMResponse
from src.infrastructure.llm_client import LLMClient
from src.modules.tool_manager.tool_executor import ToolExecutor
from src.modules.tool_manager.tool_types import ToolResult, ToolStatus
from src.core.exceptions import TaskDriftDetected

if TYPE_CHECKING:
    from src.modules.memory_manager.coordinator import MemoryCoordinator
    from src.agent.task_context import TaskContextManager
    from src.agent.context_builder import ContextBuilder
    from src.agent.content_purifier import ContentPurifier

logger = logging.getLogger(__name__)


class TurnManager:
    """
    回合管理器
    
    负责处理单轮对话的完整生命周期：
    1. 输入净化（可选）
    2. 构建上下文（系统提示 + 召回记忆 + 历史 + 用户输入）
    3. Token预算管理与截断
    4. 调用LLM获取响应
    5. 输出净化（可选）
    6. 处理工具调用（如有）
    7. 返回结果
    
    Attributes:
        llm_client: LLM客户端
        tool_executor: 工具执行器（可选）
        memory_coordinator: 记忆协调器（可选）
        task_context: 任务上下文管理器（可选）
        context_builder: 上下文构建器（可选）
        content_purifier: 内容净化器（可选）
        max_tool_calls: 每轮最大工具调用次数
        window_size: Token窗口大小
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_executor: Optional[ToolExecutor] = None,
        memory_coordinator: Optional["MemoryCoordinator"] = None,
        task_context: Optional["TaskContextManager"] = None,
        context_builder: Optional["ContextBuilder"] = None,
        content_purifier: Optional["ContentPurifier"] = None,
        max_tool_calls: int = 10,
        enable_drift_detection: bool = True,
        window_size: int = 4096,
    ):
        """
        Args:
            llm_client: LLM客户端
            tool_executor: 工具执行器（None则禁用工具）
            memory_coordinator: 记忆协调器（None则禁用记忆召回）
            task_context: 任务上下文管理器（None则禁用漂移检测）
            context_builder: 上下文构建器（None则使用简单构建）
            content_purifier: 内容净化器（None则禁用净化）
            max_tool_calls: 每轮最大工具调用次数
            enable_drift_detection: 是否启用漂移检测
            window_size: Token窗口大小
        """
        self._llm_client = llm_client
        self._tool_executor = tool_executor
        self._memory_coordinator = memory_coordinator
        self._task_context = task_context
        self._context_builder = context_builder
        self._content_purifier = content_purifier
        self._max_tool_calls = max_tool_calls
        self._enable_drift_detection = enable_drift_detection
        self._window_size = window_size

    def process_turn(
        self,
        turn_id: str,
        user_input: str,
        system_prompt: str,
        history: list[dict[str, Any]],
        tools_schema: Optional[list[dict[str, Any]]] = None,
    ) -> TurnResult:
        """
        处理单轮对话
        
        Args:
            turn_id: 回合ID
            user_input: 用户输入
            system_prompt: 系统提示词
            history: 历史消息列表（OpenAI格式）
            tools_schema: 工具schema列表（None则禁用工具）
        
        Returns:
            TurnResult 回合结果
        """
        start_time = time.time()

        # 输入净化
        purified_input = self._purify_input(user_input)

        # 漂移检测
        if self._enable_drift_detection and self._task_context:
            try:
                drift_result = self._task_context.check_drift_periodic(purified_input)
                if drift_result and drift_result.has_drift:
                    logger.warning(f"Task drift detected in turn {turn_id}")
                    duration_ms = (time.time() - start_time) * 1000
                    return TurnResult(
                        turn_id=turn_id,
                        user_input=user_input,
                        status=TurnStatus.ERROR,
                        duration_ms=duration_ms,
                        metadata={
                            "error": "Task drift detected",
                            "drift_info": {
                                "similarity": drift_result.similarity,
                                "threshold": drift_result.threshold,
                                "original_goal": drift_result.original_goal,
                                "current_intent": drift_result.current_intent,
                            }
                        },
                    )
            except Exception as e:
                logger.warning(f"Drift detection failed: {e}")

        # 构建消息列表（使用净化后的输入）
        messages = self._build_messages(system_prompt, history, purified_input)

        # 调用LLM
        try:
            response = self._call_llm(messages, tools_schema)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            duration_ms = (time.time() - start_time) * 1000
            return TurnResult(
                turn_id=turn_id,
                user_input=user_input,
                status=TurnStatus.ERROR,
                duration_ms=duration_ms,
                metadata={"error": str(e)},
            )

        # 输出净化
        purified_reply = self._purify_output(response.content)

        # 检查是否需要工具调用
        if response.is_tool_call and self._tool_executor and tools_schema:
            return self._handle_tool_calls(
                turn_id=turn_id,
                user_input=user_input,
                messages=messages,
                response=response,
                start_time=start_time,
            )

        # 纯文本回复（使用净化后的回复）
        duration_ms = (time.time() - start_time) * 1000
        return TurnResult(
            turn_id=turn_id,
            user_input=user_input,
            assistant_reply=purified_reply,
            status=TurnStatus.COMPLETED,
            tokens_used={
                "prompt": response.prompt_tokens,
                "completion": response.completion_tokens,
                "total": response.total_tokens,
            },
            duration_ms=duration_ms,
        )

    def _purify_input(self, text: str) -> str:
        """净化输入"""
        if self._content_purifier:
            return self._content_purifier.purify_user_input(text)
        return text

    def _purify_output(self, text: str) -> str:
        """净化输出"""
        if self._content_purifier:
            return self._content_purifier.purify_assistant_reply(text)
        return text

    def _build_messages(
        self,
        system_prompt: str,
        history: list[dict[str, Any]],
        user_input: str,
    ) -> list[LLMMessage]:
        """
        构建消息列表
        
        使用ContextBuilder进行Token预算管理和截断（如可用），
        否则使用简单构建方式。
        
        Args:
            system_prompt: 系统提示词
            history: 历史消息
            user_input: 用户输入
        
        Returns:
            LLMMessage列表
        """
        # 如果有ContextBuilder，使用完整构建流程
        if self._context_builder:
            return self._build_with_context_builder(system_prompt, history, user_input)
        
        # 否则使用简单构建
        return self._build_simple(system_prompt, history, user_input)

    def _build_with_context_builder(
        self,
        system_prompt: str,
        history: list[dict[str, Any]],
        user_input: str,
    ) -> list[LLMMessage]:
        """使用ContextBuilder构建消息"""
        # 召回记忆上下文
        memory_context = self._recall_memory(user_input)
        
        # 使用ContextBuilder构建
        result = self._context_builder.build(
            system_prompt=system_prompt,
            history=history,
            user_input=user_input,
            memory_context=memory_context,
            window_size=self._window_size,
        )
        
        # 记录截断信息
        if result.truncated:
            logger.info(
                f"Context truncated: removed {result.messages_removed} messages, "
                f"saved {result.tokens_saved} tokens"
            )
        
        # 转换为LLMMessage格式
        llm_messages = []
        for msg_dict in result.assembly.to_openai_format():
            llm_messages.append(LLMMessage(
                role=msg_dict.get("role", "user"),
                content=msg_dict.get("content", ""),
            ))
        
        return llm_messages

    def _build_simple(
        self,
        system_prompt: str,
        history: list[dict[str, Any]],
        user_input: str,
    ) -> list[LLMMessage]:
        """简单构建消息列表（无Token管理）"""
        messages: list[LLMMessage] = []

        # 系统提示
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))

        # 召回相关记忆（注入为system消息）
        memory_context = self._recall_memory(user_input)
        if memory_context:
            messages.append(LLMMessage(role="system", content=memory_context))

        # 历史消息
        for msg in history:
            messages.append(LLMMessage(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
            ))

        # 用户输入
        messages.append(LLMMessage(role="user", content=user_input))

        return messages

    def _recall_memory(self, query: str) -> str:
        """
        根据查询召回相关记忆
        
        Args:
            query: 用户输入
        
        Returns:
            格式化的记忆上下文，空字符串表示无相关记忆
        """
        if self._memory_coordinator is None:
            return ""

        try:
            return self._memory_coordinator.recall_as_context(query)
        except Exception as e:
            logger.warning(f"Memory recall failed: {e}")
            return ""

    def _call_llm(
        self,
        messages: list[LLMMessage],
        tools_schema: Optional[list[dict[str, Any]]],
    ) -> LLMResponse:
        """
        调用LLM
        
        Args:
            messages: 消息列表
            tools_schema: 工具schema
        
        Returns:
            LLMResponse
        """
        kwargs: dict[str, Any] = {}
        if tools_schema:
            kwargs["tools"] = tools_schema

        return self._llm_client.chat(messages, **kwargs)

    def _handle_tool_calls(
        self,
        turn_id: str,
        user_input: str,
        messages: list[LLMMessage],
        response: LLMResponse,
        start_time: float,
    ) -> TurnResult:
        """
        处理工具调用
        
        执行工具调用并将结果回传给LLM获取最终回复。
        
        Args:
            turn_id: 回合ID
            user_input: 用户输入
            messages: 当前消息列表
            response: LLM响应（包含tool_calls）
            start_time: 开始时间
        
        Returns:
            TurnResult
        """
        if not self._tool_executor or not response.tool_calls:
            # 不应该走到这里，但做防御
            duration_ms = (time.time() - start_time) * 1000
            return TurnResult(
                turn_id=turn_id,
                user_input=user_input,
                assistant_reply=response.content,
                status=TurnStatus.COMPLETED,
                duration_ms=duration_ms,
            )

        tool_calls = response.tool_calls

        # 限制工具调用次数
        if len(tool_calls) > self._max_tool_calls:
            logger.warning(f"Tool calls exceed limit: {len(tool_calls)} > {self._max_tool_calls}")
            tool_calls = tool_calls[:self._max_tool_calls]

        # 执行工具调用
        tool_results = self._tool_executor.execute_from_llm_response(tool_calls)

        # 构建工具调用消息
        assistant_msg = {
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": tool_calls,
        }

        # 构建工具结果消息
        tool_messages = self._tool_executor.results_to_tool_messages(tool_results)

        # 追加到消息列表
        messages.append(LLMMessage(
            role="assistant",
            content=response.content or "",
            tool_calls=tool_calls,
        ))

        for tm in tool_messages:
            messages.append(LLMMessage(
                role="tool",
                content=tm["content"],
                tool_call_id=tm["tool_call_id"],
            ))

        # 再次调用LLM获取最终回复
        try:
            final_response = self._llm_client.chat(messages)
        except Exception as e:
            logger.error(f"LLM call after tool execution failed: {e}")
            duration_ms = (time.time() - start_time) * 1000
            return TurnResult(
                turn_id=turn_id,
                user_input=user_input,
                status=TurnStatus.ERROR,
                tool_calls=tool_calls,
                tool_results=[r.to_dict() for r in tool_results],
                duration_ms=duration_ms,
                metadata={"error": str(e)},
            )

        duration_ms = (time.time() - start_time) * 1000

        # 统计Token
        total_tokens = response.total_tokens + final_response.total_tokens

        return TurnResult(
            turn_id=turn_id,
            user_input=user_input,
            assistant_reply=final_response.content,
            status=TurnStatus.COMPLETED,
            tool_calls=tool_calls,
            tool_results=[r.to_dict() for r in tool_results],
            tokens_used={
                "prompt": response.prompt_tokens + final_response.prompt_tokens,
                "completion": response.completion_tokens + final_response.completion_tokens,
                "total": total_tokens,
            },
            duration_ms=duration_ms,
        )
