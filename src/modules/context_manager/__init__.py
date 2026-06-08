"""
无上限上下文Agent系统 - 上下文管理器模块

提供上下文组装、分区、截断功能。

主要组件：
- ContextAssembly: 上下文组装结果
- TokenBudget: Token预算分配
- Partitioner: 上下文分区器
- StrategyPartitioner: 策略分区器
- ContextAssembler: 上下文组装器
"""

from src.modules.context_manager.context_assembly import (
    ContextAssembly,
    TokenBudget,
)

from src.modules.context_manager.partitioner import (
    Partitioner,
    StrategyPartitioner,
    PartitionStrategy,
    PartitionConfig,
)

from src.modules.context_manager.assembler import (
    ContextAssembler,
    AssemblyConfig,
    AssemblyResult,
    quick_assemble,
)

__all__ = [
    # ContextAssembly
    "ContextAssembly",
    "TokenBudget",
    # Partitioner
    "Partitioner",
    "StrategyPartitioner",
    "PartitionStrategy",
    "PartitionConfig",
    # Assembler
    "ContextAssembler",
    "AssemblyConfig",
    "AssemblyResult",
    "quick_assemble",
]
