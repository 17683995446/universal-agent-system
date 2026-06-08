"""
无上限上下文Agent系统 - 配置管理器

实现运行时配置热更新、配置验证、变更检测和回滚。

核心设计：
- ConfigManager: 配置管理器
- ConfigChange: 配置变更记录
- ConfigValidator: 配置验证器

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-27
"""

import time
import logging
import copy
from dataclasses import dataclass, field
from typing import Optional, Any, Callable
from enum import Enum

from src.agent.agent_config import AgentConfig

logger = logging.getLogger(__name__)


# =============================================================================
# 配置变更类型
# =============================================================================

class ConfigChangeType(str, Enum):
    """配置变更类型"""
    ADDED = "added"           # 新增配置项
    REMOVED = "removed"       # 删除配置项
    MODIFIED = "modified"     # 修改配置项
    UNCHANGED = "unchanged"   # 未变更


# =============================================================================
# 配置变更记录
# =============================================================================

@dataclass
class ConfigChange:
    """
    单个配置项变更记录

    Attributes:
        key: 配置键路径（如 "llm_config.max_tokens"）
        change_type: 变更类型
        old_value: 旧值
        new_value: 新值
        requires_restart: 是否需要重启生效
    """
    key: str
    change_type: ConfigChangeType
    old_value: Any = None
    new_value: Any = None
    requires_restart: bool = False

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "key": self.key,
            "change_type": self.change_type.value,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "requires_restart": self.requires_restart,
        }


# =============================================================================
# 配置变更结果
# =============================================================================

@dataclass
class ConfigUpdateResult:
    """
    配置更新结果

    Attributes:
        success: 是否成功
        changes: 变更列表
        errors: 错误信息列表
        timestamp: 更新时间
        rollback_available: 是否可回滚
    """
    success: bool
    changes: list[ConfigChange] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    rollback_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "success": self.success,
            "changes": [c.to_dict() for c in self.changes],
            "errors": self.errors,
            "timestamp": self.timestamp,
            "rollback_available": self.rollback_available,
        }

    @property
    def has_changes(self) -> bool:
        """是否有有效变更"""
        return len(self.changes) > 0

    @property
    def requires_restart(self) -> bool:
        """是否需要重启"""
        return any(c.requires_restart for c in self.changes)


# =============================================================================
# 配置验证器
# =============================================================================

class ConfigValidator:
    """配置验证器"""

    # 需要重启才能生效的配置项
    _RESTART_REQUIRED_KEYS: set[str] = {
        "agent_id",
        "llm_config.model",
        "llm_config.base_url",
        "memory_config.vector_db_url",
    }

    @classmethod
    def validate(cls, config: AgentConfig) -> list[str]:
        """
        验证配置有效性

        Args:
            config: 配置对象

        Returns:
            错误信息列表（空列表表示验证通过）
        """
        errors: list[str] = []

        # 验证基本字段
        if not config.agent_id:
            errors.append("agent_id cannot be empty")

        if config.max_turns < 0:
            errors.append("max_turns must be non-negative")

        if config.max_tool_calls_per_turn < 0:
            errors.append("max_tool_calls_per_turn must be non-negative")

        # 验证LLM配置
        if config.llm_config.max_tokens < 1:
            errors.append("llm_config.max_tokens must be at least 1")

        return errors

    @classmethod
    def requires_restart(cls, key: str) -> bool:
        """检查配置项变更是否需要重启"""
        return key in cls._RESTART_REQUIRED_KEYS


# =============================================================================
# 配置管理器
# =============================================================================

class ConfigManager:
    """
    配置管理器

    管理Agent配置的运行时热更新，支持：
    - 配置验证
    - 变更检测
    - 原子更新
    - 回滚
    - 变更历史

    Attributes:
        current_config: 当前配置
        config_history: 配置历史
    """

    def __init__(
        self,
        initial_config: AgentConfig,
        max_history: int = 10,
    ):
        """
        Args:
            initial_config: 初始配置
            max_history: 最大历史记录数
        """
        self._config = copy.deepcopy(initial_config)
        self._history: list[tuple[AgentConfig, float]] = []
        self._max_history = max_history
        self._update_count: int = 0
        self._last_update_time: float = time.time()

        # 保存初始配置
        self._save_to_history()

    # =========================================================================
    # 属性
    # =========================================================================

    @property
    def current_config(self) -> AgentConfig:
        """当前配置（深拷贝）"""
        return copy.deepcopy(self._config)

    @property
    def update_count(self) -> int:
        """更新次数"""
        return self._update_count

    @property
    def last_update_time(self) -> float:
        """最后更新时间"""
        return self._last_update_time

    # =========================================================================
    # 核心方法
    # =========================================================================

    def update(
        self,
        new_config: AgentConfig,
        validate: bool = True,
    ) -> ConfigUpdateResult:
        """
        更新配置

        Args:
            new_config: 新配置
            validate: 是否验证

        Returns:
            ConfigUpdateResult
        """
        # 验证新配置
        if validate:
            errors = ConfigValidator.validate(new_config)
            if errors:
                return ConfigUpdateResult(
                    success=False,
                    errors=errors,
                )

        # 检测变更
        changes = self._detect_changes(self._config, new_config)

        if not changes:
            logger.debug("No configuration changes detected")
            return ConfigUpdateResult(
                success=True,
                changes=[],
            )

        # 保存当前配置到历史
        self._save_to_history()

        # 应用新配置
        self._config = copy.deepcopy(new_config)
        self._update_count += 1
        self._last_update_time = time.time()

        # 标记可回滚
        result = ConfigUpdateResult(
            success=True,
            changes=changes,
            rollback_available=len(self._history) > 0,
        )

        logger.info(
            f"Configuration updated: {len(changes)} changes, "
            f"requires_restart={result.requires_restart}"
        )

        return result

    def update_partial(
        self,
        updates: dict[str, Any],
        validate: bool = True,
    ) -> ConfigUpdateResult:
        """
        部分更新配置

        Args:
            updates: 部分配置更新（扁平字典，如 {"llm_config.max_tokens": 2048}）
            validate: 是否验证

        Returns:
            ConfigUpdateResult
        """
        # 创建新配置的深拷贝
        new_config = copy.deepcopy(self._config)

        # 应用更新
        for key, value in updates.items():
            self._set_nested_attr(new_config, key, value)

        return self.update(new_config, validate)

    def rollback(self) -> ConfigUpdateResult:
        """
        回滚到上一个配置

        Returns:
            ConfigUpdateResult
        """
        if len(self._history) < 2:
            return ConfigUpdateResult(
                success=False,
                errors=["No previous configuration to rollback to"],
            )

        # 移除当前配置，恢复到上一个
        self._history.pop()  # 移除当前
        previous_config, _ = self._history[-1]  # 获取上一个

        # 检测变更（当前 → 上一个）
        changes = self._detect_changes(self._config, previous_config)

        # 应用回滚
        self._config = copy.deepcopy(previous_config)
        self._update_count += 1
        self._last_update_time = time.time()

        logger.info(f"Configuration rolled back: {len(changes)} changes")

        return ConfigUpdateResult(
            success=True,
            changes=changes,
            rollback_available=len(self._history) > 1,
        )

    def can_rollback(self) -> bool:
        """是否可以回滚"""
        return len(self._history) > 1

    # =========================================================================
    # 变更检测
    # =========================================================================

    def _detect_changes(
        self,
        old_config: AgentConfig,
        new_config: AgentConfig,
    ) -> list[ConfigChange]:
        """检测配置变更"""
        changes: list[ConfigChange] = []

        old_dict = old_config.to_dict()
        new_dict = new_config.to_dict()

        # 扁平化字典进行比较
        old_flat = self._flatten_dict(old_dict)
        new_flat = self._flatten_dict(new_dict)

        all_keys = set(old_flat.keys()) | set(new_flat.keys())

        for key in all_keys:
            old_val = old_flat.get(key)
            new_val = new_flat.get(key)

            if key not in old_flat:
                changes.append(ConfigChange(
                    key=key,
                    change_type=ConfigChangeType.ADDED,
                    new_value=new_val,
                    requires_restart=ConfigValidator.requires_restart(key),
                ))
            elif key not in new_flat:
                changes.append(ConfigChange(
                    key=key,
                    change_type=ConfigChangeType.REMOVED,
                    old_value=old_val,
                    requires_restart=ConfigValidator.requires_restart(key),
                ))
            elif old_val != new_val:
                changes.append(ConfigChange(
                    key=key,
                    change_type=ConfigChangeType.MODIFIED,
                    old_value=old_val,
                    new_value=new_val,
                    requires_restart=ConfigValidator.requires_restart(key),
                ))

        return changes

    def _flatten_dict(
        self,
        d: dict[str, Any],
        parent_key: str = "",
        sep: str = ".",
    ) -> dict[str, Any]:
        """扁平化嵌套字典"""
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _set_nested_attr(self, obj: Any, key: str, value: Any) -> None:
        """设置嵌套属性"""
        parts = key.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    # =========================================================================
    # 历史管理
    # =========================================================================

    def _save_to_history(self) -> None:
        """保存当前配置到历史"""
        self._history.append((copy.deepcopy(self._config), time.time()))

        # 限制历史长度
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def get_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """获取配置历史"""
        return [
            {
                "timestamp": ts,
                "config_summary": {
                    "agent_id": cfg.agent_id,
                    "name": cfg.name,
                    "max_turns": cfg.max_turns,
                },
            }
            for cfg, ts in reversed(self._history[-limit:])
        ]

    # =========================================================================
    # 统计与查询
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """获取配置管理器统计"""
        return {
            "update_count": self._update_count,
            "last_update_time": self._last_update_time,
            "history_size": len(self._history),
            "can_rollback": self.can_rollback(),
        }

    def get_change_summary(self) -> dict[str, int]:
        """获取变更摘要"""
        return {
            "total_updates": self._update_count,
            "history_entries": len(self._history),
        }
