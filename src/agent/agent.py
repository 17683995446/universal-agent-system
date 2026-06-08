"""
无上限上下文Agent系统 - Agent核心

Agent主循环，整合所有模块提供完整对话能力。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-26
"""

import time
import logging
import uuid
import os
from typing import Optional, Any

from src.agent.agent_types import AgentState, AgentStats, TurnResult, TurnStatus, ConversationEntry, AgentSnapshot
from src.agent.agent_config import AgentConfig
from src.agent.turn_manager import TurnManager
from src.agent.task_context import TaskContextManager
from src.agent.context_builder import ContextBuilder
from src.agent.content_purifier import ContentPurifier
from src.agent.context_compressor import ContextCompressor, CompressionConfig, CompressionStrategy
from src.agent.error_handler import ErrorHandler, ErrorCategory
from src.agent.event_bus import EventBus, AgentEventType, AgentEvent
from src.agent.config_manager import ConfigManager, ConfigUpdateResult
from src.agent.performance_monitor import PerformanceMonitor, MetricType, PerformanceReport
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.llm_types import LLMMessage
from src.modules.tool_manager.tool_registry import ToolRegistry
from src.modules.tool_manager.tool_executor import ToolExecutor
from src.modules.memory_manager.coordinator import MemoryCoordinator

logger = logging.getLogger(__name__)


class Agent:
    """
    Agent核心类
    
    整合所有模块，提供完整的对话能力：
    - 对话管理（多轮对话）
    - 工具调用（自动执行并回传）
    - 记忆管理（自动存储对话历史）
    - 统计追踪（Token、耗时、成功率）
    
    Attributes:
        config: Agent配置
        state: 当前状态
        stats: 运行统计
        llm_client: LLM客户端
        tool_registry: 工具注册器
        tool_executor: 工具执行器
        turn_manager: 回合管理器
        conversation_history: 对话历史
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Args:
            config: Agent配置（None则使用默认配置）
        """
        self._config = config or AgentConfig()
        self._state = AgentState.IDLE
        self._stats = AgentStats()
        self._stats.start_time = time.time()

        # 初始化组件
        self._llm_client = LLMClient(config=self._config.llm_config)
        self._tool_registry: Optional[ToolRegistry] = None
        self._tool_executor: Optional[ToolExecutor] = None
        self._turn_manager: Optional[TurnManager] = None
        self._memory_coordinator: Optional[MemoryCoordinator] = None
        self._task_context: Optional[TaskContextManager] = None
        self._context_builder: Optional[ContextBuilder] = None
        self._content_purifier: Optional[ContentPurifier] = None
        self._context_compressor: Optional[ContextCompressor] = None
        self._error_handler: Optional[ErrorHandler] = None
        self._event_bus: Optional[EventBus] = None
        self._config_manager: Optional[ConfigManager] = None
        self._performance_monitor: Optional[PerformanceMonitor] = None

        # 对话历史（简化格式，用于上下文构建）
        self._conversation_history: list[dict[str, Any]] = []

        # 初始化任务上下文（使用系统提示作为初始任务目标）
        self._init_task_context()

        # 初始化上下文构建器
        self._init_context_builder()

        # 初始化上下文压缩器
        self._init_context_compressor()

        # 初始化内容净化器
        self._init_content_purifier()

        # 初始化错误处理器
        self._init_error_handler()

        # 初始化事件总线
        self._init_event_bus()

        # 初始化配置管理器
        self._init_config_manager()

        # 初始化性能监控
        self._init_performance_monitor()

        # 初始化记忆（如启用）
        if self._config.enable_memory:
            self._init_memory()

        # 初始化工具（如启用）
        if self._config.enable_tools:
            self._init_tools()

        # 初始化回合管理器
        self._init_turn_manager()

        logger.info(f"Agent initialized: {self._config.name} (id={self._config.agent_id})")

    def _init_tools(self) -> None:
        """初始化工具组件"""
        self._tool_registry = ToolRegistry()
        self._tool_executor = ToolExecutor(
            registry=self._tool_registry,
            default_timeout=30.0,
        )
        logger.debug("Tools initialized")

    def _init_memory(self) -> None:
        """初始化记忆组件"""
        self._memory_coordinator = MemoryCoordinator(
            config=self._config.memory_config,
        )
        logger.debug("Memory initialized")

    def _init_task_context(self) -> None:
        """初始化任务上下文"""
        # 使用系统提示作为初始任务目标
        task_goal = self._config.system_prompt or "General conversation"
        self._task_context = TaskContextManager(
            task_goal=task_goal,
            auto_save=True,
        )
        logger.debug("Task context initialized")

    def _init_context_builder(self) -> None:
        """初始化上下文构建器"""
        # 使用LLM配置中的max_tokens作为窗口大小
        window_size = self._config.llm_config.max_tokens or 4096
        self._context_builder = ContextBuilder(
            window_size=window_size,
            truncation_strategy="oldest_first",
            preserve_system=True,
        )
        logger.debug(f"Context builder initialized: window_size={window_size}")

    def _init_context_compressor(self) -> None:
        """初始化上下文压缩器"""
        # 解析策略名称
        strategy_name = self._config.compression_strategy.upper()
        try:
            strategy = CompressionStrategy[strategy_name]
        except KeyError:
            strategy = CompressionStrategy.BALANCED

        compression_config = CompressionConfig(
            strategy=strategy,
            enabled=self._config.enable_context_compression,
        )
        self._context_compressor = ContextCompressor(config=compression_config)

        # 将压缩器注入到上下文构建器
        if self._context_builder is not None:
            self._context_builder.set_compressor(self._context_compressor)

        logger.debug(
            f"Context compressor initialized: "
            f"enabled={self._config.enable_context_compression}, "
            f"strategy={strategy.name}"
        )

    def _init_content_purifier(self) -> None:
        """初始化内容净化器"""
        # 从阶段9导入净化管道
        from src.modules.purification.pipeline import PurificationPipeline
        
        pipeline = PurificationPipeline()
        self._content_purifier = ContentPurifier(
            pipeline=pipeline,
            enable_input_purification=self._config.enable_input_purification,
            enable_output_purification=self._config.enable_output_purification,
        )
        logger.debug(
            f"Content purifier initialized: "
            f"input={self._config.enable_input_purification}, "
            f"output={self._config.enable_output_purification}"
        )

    def _init_error_handler(self) -> None:
        """初始化错误处理器"""
        self._error_handler = ErrorHandler()
        logger.debug("Error handler initialized")

    def _init_event_bus(self) -> None:
        """初始化事件总线"""
        self._event_bus = EventBus()
        logger.debug("Event bus initialized")

    def _init_config_manager(self) -> None:
        """初始化配置管理器"""
        self._config_manager = ConfigManager(self._config)
        logger.debug("Config manager initialized")

    def _init_performance_monitor(self) -> None:
        """初始化性能监控"""
        self._performance_monitor = PerformanceMonitor()
        logger.debug("Performance monitor initialized")

    def _init_turn_manager(self) -> None:
        """初始化回合管理器"""
        window_size = self._config.llm_config.max_tokens or 4096
        self._turn_manager = TurnManager(
            llm_client=self._llm_client,
            tool_executor=self._tool_executor,
            memory_coordinator=self._memory_coordinator,
            task_context=self._task_context,
            context_builder=self._context_builder,
            content_purifier=self._content_purifier,
            max_tool_calls=self._config.max_tool_calls_per_turn,
            window_size=window_size,
        )
        logger.debug("Turn manager initialized")

    # =========================================================================
    # 属性访问
    # =========================================================================

    @property
    def config(self) -> AgentConfig:
        """Agent配置"""
        return self._config

    @property
    def state(self) -> AgentState:
        """当前状态"""
        return self._state

    @property
    def stats(self) -> AgentStats:
        """运行统计"""
        return self._stats

    @property
    def tool_registry(self) -> Optional[ToolRegistry]:
        """工具注册器"""
        return self._tool_registry

    @property
    def memory_coordinator(self) -> Optional[MemoryCoordinator]:
        """记忆协调器"""
        return self._memory_coordinator

    @property
    def context_builder(self) -> Optional[ContextBuilder]:
        """上下文构建器"""
        return self._context_builder

    @property
    def task_context(self) -> Optional[TaskContextManager]:
        """任务上下文管理器"""
        return self._task_context

    @property
    def content_purifier(self) -> Optional[ContentPurifier]:
        """内容净化器"""
        return self._content_purifier

    @property
    def context_compressor(self) -> Optional[ContextCompressor]:
        """上下文压缩器"""
        return self._context_compressor

    @property
    def error_handler(self) -> Optional[ErrorHandler]:
        """错误处理器"""
        return self._error_handler

    @property
    def event_bus(self) -> Optional[EventBus]:
        """事件总线"""
        return self._event_bus

    @property
    def config_manager(self) -> Optional[ConfigManager]:
        """配置管理器"""
        return self._config_manager

    @property
    def performance_monitor(self) -> Optional[PerformanceMonitor]:
        """性能监控器"""
        return self._performance_monitor

    # =========================================================================
    # 性能监控
    # =========================================================================

    def get_performance_report(self) -> Optional[PerformanceReport]:
        """
        获取性能报告

        Returns:
            PerformanceReport 或 None
        """
        if self._performance_monitor:
            return self._performance_monitor.generate_report()
        return None

    def get_performance_stats(self) -> dict[str, Any]:
        """获取性能统计"""
        if self._performance_monitor:
            return self._performance_monitor.get_stats()
        return {}

    # =========================================================================
    # 配置热更新
    # =========================================================================

    def update_config(
        self,
        new_config: AgentConfig,
        validate: bool = True,
    ) -> ConfigUpdateResult:
        """
        热更新配置

        Args:
            new_config: 新配置
            validate: 是否验证

        Returns:
            ConfigUpdateResult
        """
        if self._config_manager is None:
            return ConfigUpdateResult(
                success=False,
                errors=["Config manager not initialized"],
            )

        # 执行更新
        result = self._config_manager.update(new_config, validate)

        if result.success:
            # 更新Agent内部配置引用
            self._config = self._config_manager.current_config

            # 发射配置变更事件
            if self._event_bus:
                self._event_bus.emit(
                    AgentEventType.CONFIG_CHANGED,
                    agent_id=self._config.agent_id,
                    data={
                        "changes": [c.to_dict() for c in result.changes],
                        "requires_restart": result.requires_restart,
                    },
                    source="Agent.update_config",
                )

            logger.info(
                f"Config updated: {len(result.changes)} changes, "
                f"restart_required={result.requires_restart}"
            )

            # 如果需要重启，发出警告
            if result.requires_restart:
                logger.warning(
                    "Some configuration changes require agent restart to take full effect"
                )

        return result

    def update_config_partial(
        self,
        updates: dict[str, Any],
        validate: bool = True,
    ) -> ConfigUpdateResult:
        """
        部分更新配置

        Args:
            updates: 部分配置更新（扁平字典）
            validate: 是否验证

        Returns:
            ConfigUpdateResult
        """
        if self._config_manager is None:
            return ConfigUpdateResult(
                success=False,
                errors=["Config manager not initialized"],
            )

        result = self._config_manager.update_partial(updates, validate)

        if result.success:
            self._config = self._config_manager.current_config

            if self._event_bus:
                self._event_bus.emit(
                    AgentEventType.CONFIG_CHANGED,
                    agent_id=self._config.agent_id,
                    data={
                        "partial_update": True,
                        "changes": [c.to_dict() for c in result.changes],
                    },
                    source="Agent.update_config_partial",
                )

        return result

    def rollback_config(self) -> ConfigUpdateResult:
        """
        回滚配置到上一个版本

        Returns:
            ConfigUpdateResult
        """
        if self._config_manager is None:
            return ConfigUpdateResult(
                success=False,
                errors=["Config manager not initialized"],
            )

        result = self._config_manager.rollback()

        if result.success:
            self._config = self._config_manager.current_config

            if self._event_bus:
                self._event_bus.emit(
                    AgentEventType.CONFIG_CHANGED,
                    agent_id=self._config.agent_id,
                    data={
                        "rollback": True,
                        "changes": [c.to_dict() for c in result.changes],
                    },
                    source="Agent.rollback_config",
                )

            logger.info("Config rolled back successfully")

        return result

    # =========================================================================
    # 工具管理
    # =========================================================================

    def register_tool(self, tool: Any) -> None:
        """
        注册工具
        
        Args:
            tool: 工具定义
        """
        if self._tool_registry is None:
            logger.warning("Tools not enabled, cannot register tool")
            return

        self._tool_registry.register(tool)
        logger.info(f"Tool registered: {tool.name}")

    def register_tools(self, tools: list[Any]) -> int:
        """
        批量注册工具
        
        Args:
            tools: 工具定义列表
        
        Returns:
            成功注册的数量
        """
        if self._tool_registry is None:
            logger.warning("Tools not enabled, cannot register tools")
            return 0

        count = self._tool_registry.register_batch(tools)
        logger.info(f"Registered {count} tools")
        return count

    def get_tools_schema(self) -> Optional[list[dict[str, Any]]]:
        """
        获取工具schema
        
        Returns:
            OpenAI tools格式列表，或None（工具未启用）
        """
        if self._tool_registry is None or not self._config.enable_tools:
            return None

        return self._tool_registry.get_openai_schema()

    # =========================================================================
    # 对话核心
    # =========================================================================

    def chat(self, user_input: str) -> TurnResult:
        """
        对话入口
        
        处理单轮对话，返回结果。
        
        Args:
            user_input: 用户输入
        
        Returns:
            TurnResult 回合结果
        """
        if self._state == AgentState.STOPPED:
            return TurnResult(
                turn_id=self._generate_turn_id(),
                user_input=user_input,
                status=TurnStatus.ERROR,
                metadata={"error": "Agent is stopped"},
            )

        # 检查回合数限制
        if self._config.max_turns > 0 and self._stats.total_turns >= self._config.max_turns:
            return TurnResult(
                turn_id=self._generate_turn_id(),
                user_input=user_input,
                status=TurnStatus.ERROR,
                metadata={"error": f"Max turns ({self._config.max_turns}) reached"},
            )

        # 生成回合ID
        turn_id = self._generate_turn_id()

        # 记录回合开始时间
        turn_start_time = time.time()

        # 发射回合开始事件
        if self._event_bus:
            self._event_bus.emit(
                AgentEventType.TURN_STARTED,
                agent_id=self._config.agent_id,
                data={"user_input": user_input},
                source="Agent.chat",
                turn_id=turn_id,
            )

        # 设置状态
        self._state = AgentState.THINKING

        try:
            result = self._process_turn(turn_id, user_input)

            # 更新统计
            self._update_stats(result)

            # 更新历史
            self._update_history(result)

            # 存储到记忆
            self._store_to_memory(result)

            # 发射回合完成事件
            if self._event_bus:
                self._event_bus.emit(
                    AgentEventType.TURN_COMPLETED,
                    agent_id=self._config.agent_id,
                    data={
                        "status": result.status.value,
                        "tokens_used": result.tokens_used,
                        "duration_ms": result.duration_ms,
                    },
                    source="Agent.chat",
                    turn_id=turn_id,
                )

            # 记录性能指标
            if self._performance_monitor:
                turn_duration = (time.time() - turn_start_time) * 1000
                self._performance_monitor.record_timing(
                    MetricType.TURN_DURATION,
                    turn_duration,
                )
                if result.tokens_used:
                    total_tokens = sum(result.tokens_used.values())
                    self._performance_monitor.record(
                        MetricType.TOTAL_TOKENS,
                        total_tokens,
                    )

            self._state = AgentState.IDLE
            return result

        except Exception as e:
            # 使用错误处理器记录
            if self._error_handler:
                error_record = self._error_handler.handle_error(
                    error=e,
                    component="Agent.chat",
                    turn_id=turn_id,
                )
                error_category = error_record.category.name
            else:
                error_category = "UNKNOWN"

            logger.exception(f"Unexpected error during chat [{error_category}]")
            self._state = AgentState.ERROR

            # 发射回合错误事件
            if self._event_bus:
                self._event_bus.emit(
                    AgentEventType.TURN_ERROR,
                    agent_id=self._config.agent_id,
                    data={
                        "error": str(e),
                        "error_category": error_category,
                    },
                    source="Agent.chat",
                    turn_id=turn_id,
                )

            # 记录错误统计
            self._stats.record_turn(0, 0, error=True)

            return TurnResult(
                turn_id=turn_id,
                user_input=user_input,
                status=TurnStatus.ERROR,
                metadata={
                    "error": str(e),
                    "error_category": error_category,
                },
            )

    def _process_turn(self, turn_id: str, user_input: str) -> TurnResult:
        """
        处理单轮对话（内部）
        
        Args:
            turn_id: 回合ID
            user_input: 用户输入
        
        Returns:
            TurnResult
        """
        if self._turn_manager is None:
            raise RuntimeError("Turn manager not initialized")

        # 获取工具schema
        tools_schema = self.get_tools_schema()

        # 处理回合
        result = self._turn_manager.process_turn(
            turn_id=turn_id,
            user_input=user_input,
            system_prompt=self._config.system_prompt,
            history=self._conversation_history.copy(),
            tools_schema=tools_schema,
        )

        return result

    def _update_stats(self, result: TurnResult) -> None:
        """更新统计"""
        tokens = result.tokens_used.get("total", 0)
        tool_calls = len(result.tool_calls)
        error = result.is_error

        self._stats.record_turn(
            duration_ms=result.duration_ms,
            tokens=tokens,
            tool_calls=tool_calls,
            error=error,
        )

    def _update_history(self, result: TurnResult) -> None:
        """更新对话历史"""
        # 用户输入
        self._conversation_history.append({
            "role": "user",
            "content": result.user_input,
        })

        # 助手回复（包含工具调用）
        if result.tool_calls:
            self._conversation_history.append({
                "role": "assistant",
                "content": result.assistant_reply,
                "tool_calls": result.tool_calls,
            })

            # 工具结果
            for tr in result.tool_results:
                self._conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tr.get("tool_call_id", ""),
                    "content": tr.get("output", ""),
                })
        else:
            self._conversation_history.append({
                "role": "assistant",
                "content": result.assistant_reply,
            })

    def _store_to_memory(self, result: TurnResult) -> None:
        """将对话结果存储到记忆"""
        if self._memory_coordinator is None:
            return

        try:
            # 存储用户输入和助手回复
            self._memory_coordinator.store_conversation(
                user_input=result.user_input,
                assistant_reply=result.assistant_reply,
            )

            # 存储工具结果
            for tr in result.tool_results:
                tool_name = tr.get("tool_name", "unknown")
                output = tr.get("output")
                self._memory_coordinator.store_tool_result(
                    tool_name=tool_name,
                    result=output,
                )
        except Exception as e:
            logger.warning(f"Failed to store to memory: {e}")

    def _generate_turn_id(self) -> str:
        """生成回合ID"""
        return f"turn_{self._config.agent_id}_{int(time.time() * 1000)}"

    # =========================================================================
    # 历史管理
    # =========================================================================

    def get_history(self) -> list[dict[str, Any]]:
        """获取对话历史"""
        return self._conversation_history.copy()

    def clear_history(self) -> None:
        """清空对话历史"""
        self._conversation_history.clear()
        logger.info("Conversation history cleared")

    def get_conversation_summary(self) -> dict[str, Any]:
        """获取对话摘要"""
        return {
            "total_messages": len(self._conversation_history),
            "user_messages": sum(1 for m in self._conversation_history if m.get("role") == "user"),
            "assistant_messages": sum(1 for m in self._conversation_history if m.get("role") == "assistant"),
            "tool_calls": sum(1 for m in self._conversation_history if m.get("role") == "tool"),
        }

    # =========================================================================
    # 生命周期 - 状态管理
    # =========================================================================

    def _transition_state(self, new_state: AgentState) -> bool:
        """
        转换Agent状态
        
        Args:
            new_state: 目标状态
            
        Returns:
            是否成功转换
        """
        if AgentState.can_transition(self._state, new_state):
            old_state = self._state
            self._state = new_state
            logger.debug(f"Agent state: {old_state.value} -> {new_state.value}")

            # 发射状态变更事件
            if self._event_bus:
                self._event_bus.emit(
                    AgentEventType.STATE_CHANGED,
                    agent_id=self._config.agent_id,
                    data={
                        "old_state": old_state.value,
                        "new_state": new_state.value,
                    },
                    source="Agent._transition_state",
                )

            return True
        else:
            logger.warning(
                f"Invalid state transition: {self._state.value} -> {new_state.value}"
            )
            return False

    def reset(self) -> None:
        """重置Agent状态"""
        # 重置允许从任何状态强制转换到IDLE
        self._state = AgentState.IDLE

        self._stats = AgentStats()
        self._stats.start_time = time.time()
        self._conversation_history.clear()
        if self._memory_coordinator:
            self._memory_coordinator.clear_all()
        if self._task_context:
            self._task_context.clear_task()
            self._task_context.reset_stats()
        logger.info("Agent reset")

    def stop(self) -> None:
        """停止Agent"""
        if self._transition_state(AgentState.STOPPED):
            logger.info("Agent stopped")

    def pause(self) -> bool:
        """
        暂停Agent
        
        Returns:
            是否成功暂停
        """
        return self._transition_state(AgentState.PAUSED)

    def resume(self) -> bool:
        """
        恢复Agent
        
        Returns:
            是否成功恢复
        """
        return self._transition_state(AgentState.IDLE)

    def restart(self) -> bool:
        """
        重启Agent
        
        从STOPPED状态恢复到IDLE状态。
        
        Returns:
            是否成功重启
        """
        if self._state == AgentState.STOPPED:
            self._state = AgentState.IDLE
            self._stats.start_time = time.time()
            logger.info("Agent restarted")
            return True
        return False

    # =========================================================================
    # 生命周期 - 资源清理
    # =========================================================================

    def cleanup(self) -> None:
        """
        清理资源
        
        优雅关闭，释放所有资源。
        """
        logger.info("Starting Agent cleanup...")

        # 停止接受新请求
        self.stop()

        # 清理各组件
        self._cleanup_components()

        logger.info("Agent cleanup completed")

    def _cleanup_components(self) -> None:
        """清理各组件资源"""
        # 清理记忆
        if self._memory_coordinator:
            try:
                self._memory_coordinator.clear_all()
                logger.debug("Memory coordinator cleaned up")
            except Exception as e:
                logger.warning(f"Failed to cleanup memory: {e}")

        # 清理任务上下文
        if self._task_context:
            try:
                self._task_context.clear_task()
                logger.debug("Task context cleaned up")
            except Exception as e:
                logger.warning(f"Failed to cleanup task context: {e}")

        # 清理工具执行器
        if self._tool_executor:
            try:
                self._tool_executor.clear_history()
                logger.debug("Tool executor cleaned up")
            except Exception as e:
                logger.warning(f"Failed to cleanup tool executor: {e}")

        # 清理上下文构建器统计
        if self._context_builder:
            try:
                self._context_builder.reset_stats()
                logger.debug("Context builder stats reset")
            except Exception as e:
                logger.warning(f"Failed to reset context builder: {e}")

        # 清理压缩器统计
        if self._context_compressor:
            try:
                self._context_compressor.reset_stats()
                logger.debug("Context compressor stats reset")
            except Exception as e:
                logger.warning(f"Failed to reset context compressor: {e}")

    # =========================================================================
    # 生命周期 - 快照管理
    # =========================================================================

    def create_snapshot(self) -> AgentSnapshot:
        """
        创建当前状态快照
        
        Returns:
            AgentSnapshot 快照对象
        """
        snapshot = AgentSnapshot(
            snapshot_id=f"snap_{self._config.agent_id}_{int(time.time() * 1000)}",
            agent_id=self._config.agent_id,
            agent_name=self._config.name,
            state=self._state.value,
            config_dict=self._config.to_dict(),
            conversation_history=self._conversation_history.copy(),
            stats_dict=self._stats.to_dict(),
            task_context=self._task_context.get_stats() if self._task_context else {},
            compression_stats=self._context_compressor.get_stats() if self._context_compressor else {},
            metadata={
                "created_by": "agent.create_snapshot",
                "version": "1.0",
            },
        )
        logger.info(f"Snapshot created: {snapshot.snapshot_id}")
        return snapshot

    def save_snapshot(self, filepath: Optional[str] = None) -> str:
        """
        保存快照到文件
        
        Args:
            filepath: 保存路径（None则使用默认路径）
            
        Returns:
            保存的文件路径
        """
        snapshot = self.create_snapshot()

        if filepath is None:
            # 默认保存路径
            snapshot_dir = "/workspace/snapshots"
            os.makedirs(snapshot_dir, exist_ok=True)
            filepath = f"{snapshot_dir}/{snapshot.snapshot_id}.json"

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(snapshot.to_json(indent=2))
            logger.info(f"Snapshot saved to: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            raise

    def restore_from_snapshot(self, snapshot: AgentSnapshot) -> bool:
        """
        从快照恢复状态
        
        Args:
            snapshot: 快照对象
            
        Returns:
            是否成功恢复
        """
        if not snapshot.is_valid():
            logger.error("Invalid snapshot, cannot restore")
            return False

        try:
            # 恢复状态
            self._state = AgentState(snapshot.state)

            # 恢复历史
            self._conversation_history = snapshot.conversation_history.copy()

            # 恢复统计
            self._stats = AgentStats.from_dict(snapshot.stats_dict)

            logger.info(f"Agent restored from snapshot: {snapshot.snapshot_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore from snapshot: {e}")
            return False

    def load_snapshot(self, filepath: str) -> bool:
        """
        从文件加载快照并恢复
        
        Args:
            filepath: 快照文件路径
            
        Returns:
            是否成功加载并恢复
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                snapshot = AgentSnapshot.from_json(f.read())
            return self.restore_from_snapshot(snapshot)
        except Exception as e:
            logger.error(f"Failed to load snapshot from {filepath}: {e}")
            return False

    # =========================================================================
    # 摘要信息
    # =========================================================================

    def get_summary(self) -> dict[str, Any]:
        """获取Agent摘要"""
        summary = {
            "agent_id": self._config.agent_id,
            "name": self._config.name,
            "state": self._state.value,
            "stats": self._stats.to_dict(),
            "conversation": self.get_conversation_summary(),
            "tools_enabled": self._config.enable_tools,
            "tools_count": len(self._tool_registry.list_all()) if self._tool_registry else 0,
            "memory_enabled": self._config.enable_memory,
        }
        if self._memory_coordinator:
            summary["memory"] = self._memory_coordinator.get_stats()
        if self._task_context:
            summary["task"] = self._task_context.get_stats()
        if self._error_handler:
            summary["errors"] = self._error_handler.get_stats()
        if self._event_bus:
            summary["events"] = self._event_bus.get_stats()
        return summary
