"""
无上限上下文Agent系统 - 上下文分区器

实现Token配额的弹性分配策略。
支持系统区/会话区/预留区的动态调整。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

from typing import Optional
from dataclasses import dataclass

from src.modules.context_manager.context_assembly import TokenBudget
from src.core.constants import (
    SYSTEM_ZONE_DEFAULT_RATIO,
    SYSTEM_ZONE_MIN_RATIO,
    SYSTEM_ZONE_MAX_RATIO,
    SESSION_ZONE_RATIO,
    RESERVE_ZONE_RATIO,
    RESERVE_ZONE_MIN_RATIO,
)


# =============================================================================
# 分区配置
# =============================================================================

@dataclass
class PartitionConfig:
    """
    分区配置

    定义各区域的Token配额比例
    """
    system_ratio: float = SYSTEM_ZONE_DEFAULT_RATIO
    session_ratio: float = SESSION_ZONE_RATIO
    reserve_ratio: float = RESERVE_ZONE_RATIO

    def __post_init__(self):
        """验证配置有效性"""
        total = self.system_ratio + self.session_ratio + self.reserve_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"分区比例总和必须等于1.0，当前为{total}")


# =============================================================================
# 分区器
# =============================================================================

class Partitioner:
    """
    上下文分区器

    根据窗口大小和分区策略计算各区域Token配额。
    支持弹性调整：系统区可在min-max范围内自适应。
    """

    def __init__(
        self,
        system_min_ratio: float = SYSTEM_ZONE_MIN_RATIO,
        system_max_ratio: float = SYSTEM_ZONE_MAX_RATIO,
        reserve_min_ratio: float = RESERVE_ZONE_MIN_RATIO,
    ):
        """
        Args:
            system_min_ratio: 系统区最小比例
            system_max_ratio: 系统区最大比例
            reserve_min_ratio: 预留区最小比例
        """
        self._system_min = system_min_ratio
        self._system_max = system_max_ratio
        self._reserve_min = reserve_min_ratio

    def allocate(
        self,
        window_size: int,
        system_ratio: Optional[float] = None,
        session_ratio: Optional[float] = None,
        reserve_ratio: Optional[float] = None,
    ) -> TokenBudget:
        """
        分配Token预算

        Args:
            window_size: 窗口总大小
            system_ratio: 系统区比例（None则使用默认值）
            session_ratio: 会话区比例（None则使用默认值）
            reserve_ratio: 预留区比例（None则使用默认值）

        Returns:
            Token预算分配
        """
        system_ratio = system_ratio if system_ratio is not None else SYSTEM_ZONE_DEFAULT_RATIO
        session_ratio = session_ratio if session_ratio is not None else SESSION_ZONE_RATIO
        reserve_ratio = reserve_ratio if reserve_ratio is not None else RESERVE_ZONE_RATIO

        # 验证比例
        total_ratio = system_ratio + session_ratio + reserve_ratio
        if abs(total_ratio - 1.0) > 0.001:
            # 自动归一化
            system_ratio /= total_ratio
            session_ratio /= total_ratio
            reserve_ratio /= total_ratio

        system_zone = int(window_size * system_ratio)
        session_zone = int(window_size * session_ratio)
        reserve_zone = int(window_size * reserve_ratio)

        return TokenBudget(
            window_size=window_size,
            system_zone=system_zone,
            session_zone=session_zone,
            reserve_zone=reserve_zone,
        )

    def allocate_adaptive(
        self,
        window_size: int,
        system_tokens_needed: int,
        min_session_ratio: float = 0.40,
    ) -> TokenBudget:
        """
        自适应分配Token预算

        根据实际系统消息所需Token数，在允许范围内弹性调整系统区比例。

        Args:
            window_size: 窗口总大小
            system_tokens_needed: 系统消息实际所需Token数
            min_session_ratio: 会话区最小比例（不能低于此值）

        Returns:
            自适应调整后的Token预算
        """
        # 计算系统区所需最小比例
        system_needed_ratio = system_tokens_needed / window_size

        # 确保在允许范围内
        system_ratio = max(
            self._system_min,
            min(self._system_max, system_needed_ratio)
        )

        # 确保预留区不低于最小值
        reserve_ratio = max(self._reserve_min, RESERVE_ZONE_RATIO)

        # 剩余给会话区
        session_ratio = 1.0 - system_ratio - reserve_ratio

        # 如果会话区太小，压缩系统区
        if session_ratio < min_session_ratio:
            session_ratio = min_session_ratio
            remaining = 1.0 - session_ratio - reserve_ratio
            system_ratio = max(self._system_min, remaining)

        return self.allocate(window_size, system_ratio, session_ratio, reserve_ratio)

    def calculate_for_model(
        self,
        model_name: str,
        custom_system_ratio: Optional[float] = None,
    ) -> TokenBudget:
        """
        为指定模型计算Token预算

        Args:
            model_name: 模型名称（如"gpt-4", "claude-3-opus"）
            custom_system_ratio: 自定义系统区比例

        Returns:
            Token预算分配
        """
        from src.core.constants import MODEL_CONTEXT_WINDOWS

        window_size = MODEL_CONTEXT_WINDOWS.get(model_name, 4096)
        return self.allocate(
            window_size,
            system_ratio=custom_system_ratio,
        )


# =============================================================================
# 分区策略枚举
# =============================================================================

class PartitionStrategy:
    """分区策略常量"""

    # 默认策略：系统20% + 会话60% + 预留20%
    DEFAULT = "default"

    # 系统重载策略：系统30% + 会话50% + 预留20%
    # 适用于工具多、系统提示复杂的场景
    SYSTEM_HEAVY = "system_heavy"

    # 会话优先策略：系统15% + 会话70% + 预留15%
    # 适用于需要长对话历史的场景
    SESSION_PRIORITY = "session_priority"

    # 平衡策略：系统20% + 会话65% + 预留15%
    # 降低预留区，增加会话区
    BALANCED = "balanced"


# =============================================================================
# 策略分区器
# =============================================================================

class StrategyPartitioner:
    """
    策略分区器

    根据预定义策略快速生成分区配置
    """

    STRATEGIES = {
        PartitionStrategy.DEFAULT: (0.20, 0.60, 0.20),
        PartitionStrategy.SYSTEM_HEAVY: (0.30, 0.50, 0.20),
        PartitionStrategy.SESSION_PRIORITY: (0.15, 0.70, 0.15),
        PartitionStrategy.BALANCED: (0.20, 0.65, 0.15),
    }

    def __init__(self):
        self._partitioner = Partitioner()

    def allocate(
        self,
        window_size: int,
        strategy: str = PartitionStrategy.DEFAULT,
    ) -> TokenBudget:
        """
        根据策略分配Token预算

        Args:
            window_size: 窗口总大小
            strategy: 分区策略名称

        Returns:
            Token预算分配
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"未知策略: {strategy}，可用: {list(self.STRATEGIES.keys())}")

        system_ratio, session_ratio, reserve_ratio = self.STRATEGIES[strategy]
        return self._partitioner.allocate(
            window_size, system_ratio, session_ratio, reserve_ratio
        )

    def get_strategy_ratios(self, strategy: str) -> tuple[float, float, float]:
        """获取策略的比例配置"""
        if strategy not in self.STRATEGIES:
            raise ValueError(f"未知策略: {strategy}")
        return self.STRATEGIES[strategy]

    def list_strategies(self) -> list[str]:
        """列出所有可用策略"""
        return list(self.STRATEGIES.keys())
