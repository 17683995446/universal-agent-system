"""
无上限上下文Agent系统 - Agent配置

Agent初始化配置，整合各模块配置。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-26
"""

from typing import Optional, Any
from dataclasses import dataclass, field

from src.infrastructure.llm_types import LLMConfig
from src.modules.memory_manager.factory import MemoryConfig


@dataclass
class AgentConfig:
    """
    Agent配置
    
    整合LLM配置、记忆配置、工具配置等。
    
    Attributes:
        # 基础配置
        agent_id: Agent唯一标识
        name: Agent名称
        description: Agent描述
        
        # LLM配置
        llm_config: LLM客户端配置
        system_prompt: 系统提示词
        max_turns: 最大回合数（0=无限制）
        
        # 记忆配置
        memory_config: 记忆管理配置
        enable_memory: 是否启用记忆
        
        # 工具配置
        enable_tools: 是否启用工具
        max_tool_calls_per_turn: 每轮最大工具调用次数
        
        # 行为配置
        auto_save: 是否自动保存快照
        verbose: 是否详细输出
        
        # 元数据
        metadata: 额外元数据
    """
    # 基础配置
    agent_id: str = "default"
    name: str = "Agent"
    description: str = ""

    # LLM配置
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    system_prompt: str = "You are a helpful assistant."
    max_turns: int = 0  # 0 = 无限制

    # 记忆配置
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
    enable_memory: bool = True

    # 工具配置
    enable_tools: bool = True
    max_tool_calls_per_turn: int = 10

    # 内容净化配置
    enable_input_purification: bool = True
    enable_output_purification: bool = False

    # 上下文压缩配置
    enable_context_compression: bool = True
    compression_strategy: str = "balanced"  # none/conservative/balanced/aggressive

    # 行为配置
    auto_save: bool = True
    verbose: bool = False

    # 元数据
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "max_turns": self.max_turns,
            "enable_memory": self.enable_memory,
            "enable_tools": self.enable_tools,
            "max_tool_calls_per_turn": self.max_tool_calls_per_turn,
            "enable_input_purification": self.enable_input_purification,
            "enable_output_purification": self.enable_output_purification,
            "enable_context_compression": self.enable_context_compression,
            "compression_strategy": self.compression_strategy,
            "auto_save": self.auto_save,
            "verbose": self.verbose,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        """从字典创建配置"""
        config = cls()

        # 基础配置
        config.agent_id = data.get("agent_id", config.agent_id)
        config.name = data.get("name", config.name)
        config.description = data.get("description", config.description)

        # LLM配置
        if "llm_config" in data:
            config.llm_config = LLMConfig(**data["llm_config"])
        config.system_prompt = data.get("system_prompt", config.system_prompt)
        config.max_turns = data.get("max_turns", config.max_turns)

        # 记忆配置
        if "memory_config" in data:
            config.memory_config = MemoryConfig(**data["memory_config"])
        config.enable_memory = data.get("enable_memory", config.enable_memory)

        # 工具配置
        config.enable_tools = data.get("enable_tools", config.enable_tools)
        config.max_tool_calls_per_turn = data.get(
            "max_tool_calls_per_turn", config.max_tool_calls_per_turn
        )

        # 内容净化配置
        config.enable_input_purification = data.get(
            "enable_input_purification", config.enable_input_purification
        )
        config.enable_output_purification = data.get(
            "enable_output_purification", config.enable_output_purification
        )

        # 上下文压缩配置
        config.enable_context_compression = data.get(
            "enable_context_compression", config.enable_context_compression
        )
        config.compression_strategy = data.get(
            "compression_strategy", config.compression_strategy
        )

        # 行为配置
        config.auto_save = data.get("auto_save", config.auto_save)
        config.verbose = data.get("verbose", config.verbose)

        # 元数据
        config.metadata = data.get("metadata", {})

        return config
