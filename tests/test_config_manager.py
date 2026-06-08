"""
阶段21测试：Agent配置热更新

测试ConfigManager、ConfigChange、ConfigValidator及Agent集成。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-27
"""

import pytest
from unittest.mock import Mock, patch

from src.agent.config_manager import (
    ConfigChangeType,
    ConfigChange,
    ConfigUpdateResult,
    ConfigValidator,
    ConfigManager,
)
from src.agent.agent_config import AgentConfig


class TestConfigChangeType:
    """ConfigChangeType枚举测试"""

    def test_enum_values(self):
        assert ConfigChangeType.ADDED == "added"
        assert ConfigChangeType.REMOVED == "removed"
        assert ConfigChangeType.MODIFIED == "modified"
        assert ConfigChangeType.UNCHANGED == "unchanged"


class TestConfigChange:
    """ConfigChange测试"""

    def test_creation(self):
        change = ConfigChange(
            key="llm_config.max_tokens",
            change_type=ConfigChangeType.MODIFIED,
            old_value=1024,
            new_value=2048,
            requires_restart=False,
        )
        assert change.key == "llm_config.max_tokens"
        assert change.old_value == 1024
        assert change.new_value == 2048

    def test_serialization(self):
        change = ConfigChange(
            key="name",
            change_type=ConfigChangeType.MODIFIED,
            old_value="Old",
            new_value="New",
        )
        data = change.to_dict()
        assert data["key"] == "name"
        assert data["change_type"] == "modified"


class TestConfigUpdateResult:
    """ConfigUpdateResult测试"""

    def test_success_result(self):
        result = ConfigUpdateResult(
            success=True,
            changes=[ConfigChange(key="k", change_type=ConfigChangeType.MODIFIED)],
        )
        assert result.success is True
        assert result.has_changes is True
        assert result.requires_restart is False

    def test_requires_restart(self):
        result = ConfigUpdateResult(
            success=True,
            changes=[
                ConfigChange(
                    key="agent_id",
                    change_type=ConfigChangeType.MODIFIED,
                    requires_restart=True,
                )
            ],
        )
        assert result.requires_restart is True

    def test_no_changes(self):
        result = ConfigUpdateResult(success=True, changes=[])
        assert result.has_changes is False


class TestConfigValidator:
    """ConfigValidator测试"""

    def test_valid_config(self):
        config = AgentConfig()
        errors = ConfigValidator.validate(config)
        assert len(errors) == 0

    def test_empty_agent_id(self):
        config = AgentConfig(agent_id="")
        errors = ConfigValidator.validate(config)
        assert "agent_id cannot be empty" in errors

    def test_negative_max_turns(self):
        config = AgentConfig(max_turns=-1)
        errors = ConfigValidator.validate(config)
        assert "max_turns must be non-negative" in errors

    def test_llm_config_validation(self):
        config = AgentConfig()
        config.llm_config.max_tokens = 0
        errors = ConfigValidator.validate(config)
        assert "llm_config.max_tokens must be at least 1" in errors

    def test_requires_restart(self):
        assert ConfigValidator.requires_restart("agent_id") is True
        assert ConfigValidator.requires_restart("llm_config.model") is True
        assert ConfigValidator.requires_restart("name") is False


class TestConfigManager:
    """ConfigManager核心功能测试"""

    def test_creation(self):
        config = AgentConfig(agent_id="test")
        manager = ConfigManager(config)
        assert manager.current_config.agent_id == "test"
        assert manager.update_count == 0

    def test_update_no_changes(self):
        config = AgentConfig()
        manager = ConfigManager(config)

        result = manager.update(config)
        assert result.success is True
        assert result.has_changes is False

    def test_update_with_changes(self):
        config = AgentConfig(name="Original")
        manager = ConfigManager(config)

        new_config = AgentConfig(name="Updated")
        result = manager.update(new_config)

        assert result.success is True
        assert result.has_changes is True
        assert manager.current_config.name == "Updated"
        assert manager.update_count == 1

    def test_update_validation_failure(self):
        config = AgentConfig()
        manager = ConfigManager(config)

        invalid_config = AgentConfig(agent_id="", max_turns=-1)
        result = manager.update(invalid_config)

        assert result.success is False
        assert len(result.errors) > 0

    def test_update_partial(self):
        config = AgentConfig(max_turns=10)
        manager = ConfigManager(config)

        result = manager.update_partial({"max_turns": 20})

        assert result.success is True
        assert manager.current_config.max_turns == 20

    def test_rollback(self):
        config = AgentConfig(name="V1")
        manager = ConfigManager(config)

        # 更新到V2
        manager.update(AgentConfig(name="V2"))
        assert manager.current_config.name == "V2"

        # 回滚到V1
        result = manager.rollback()
        assert result.success is True
        assert manager.current_config.name == "V1"

    def test_cannot_rollback_initial(self):
        config = AgentConfig()
        manager = ConfigManager(config)

        result = manager.rollback()
        assert result.success is False
        assert "No previous configuration" in result.errors[0]

    def test_can_rollback(self):
        config = AgentConfig()
        manager = ConfigManager(config)

        assert manager.can_rollback() is False

        manager.update(AgentConfig(name="Updated"))
        assert manager.can_rollback() is True

    def test_history_limit(self):
        config = AgentConfig()
        manager = ConfigManager(config, max_history=3)

        for i in range(5):
            manager.update(AgentConfig(name=f"V{i}"))

        history = manager.get_history()
        assert len(history) <= 3

    def test_get_stats(self):
        config = AgentConfig()
        manager = ConfigManager(config)

        manager.update(AgentConfig(name="V1"))
        manager.update(AgentConfig(name="V2"))

        stats = manager.get_stats()
        assert stats["update_count"] == 2
        assert stats["can_rollback"] is True

    def test_detect_top_level_changes(self):
        """测试顶层属性变更检测"""
        config = AgentConfig(name="Original")
        manager = ConfigManager(config)

        new_config = AgentConfig(name="Updated")
        result = manager.update(new_config)

        # 检查是否检测到顶层变更
        change_keys = [c.key for c in result.changes]
        print(f"Change keys: {change_keys}")
        assert "name" in change_keys
        assert len(result.changes) > 0


class TestAgentConfigIntegration:
    """Agent配置集成测试"""

    @patch('src.infrastructure.llm_client.LLMClient')
    @patch('src.modules.purification.pipeline.PurificationPipeline')
    def test_agent_has_config_manager(self, mock_pipeline, mock_llm):
        mock_pipeline.return_value = Mock()

        from src.agent.agent import Agent
        from src.agent.agent_config import AgentConfig

        config = AgentConfig()
        with patch('src.agent.agent.LLMClient', mock_llm):
            agent = Agent(config=config)

        assert agent.config_manager is not None

    @patch('src.infrastructure.llm_client.LLMClient')
    @patch('src.modules.purification.pipeline.PurificationPipeline')
    def test_agent_update_config(self, mock_pipeline, mock_llm):
        mock_pipeline.return_value = Mock()

        from src.agent.agent import Agent
        from src.agent.agent_config import AgentConfig

        config = AgentConfig(name="Original")
        with patch('src.agent.agent.LLMClient', mock_llm):
            agent = Agent(config=config)

        new_config = AgentConfig(name="Updated")
        result = agent.update_config(new_config)

        assert result.success is True
        assert agent.config.name == "Updated"

    @patch('src.infrastructure.llm_client.LLMClient')
    @patch('src.modules.purification.pipeline.PurificationPipeline')
    def test_agent_update_config_partial(self, mock_pipeline, mock_llm):
        mock_pipeline.return_value = Mock()

        from src.agent.agent import Agent
        from src.agent.agent_config import AgentConfig

        config = AgentConfig(max_turns=10)
        with patch('src.agent.agent.LLMClient', mock_llm):
            agent = Agent(config=config)

        result = agent.update_config_partial({"max_turns": 50})

        assert result.success is True
        assert agent.config.max_turns == 50

    @patch('src.infrastructure.llm_client.LLMClient')
    @patch('src.modules.purification.pipeline.PurificationPipeline')
    def test_agent_rollback_config(self, mock_pipeline, mock_llm):
        mock_pipeline.return_value = Mock()

        from src.agent.agent import Agent
        from src.agent.agent_config import AgentConfig

        config = AgentConfig(name="V1")
        with patch('src.agent.agent.LLMClient', mock_llm):
            agent = Agent(config=config)

        agent.update_config(AgentConfig(name="V2"))
        assert agent.config.name == "V2"

        result = agent.rollback_config()
        assert result.success is True
        assert agent.config.name == "V1"


class TestExportConfigManager:
    """导出测试"""

    def test_config_manager_exported(self):
        from src.agent import ConfigManager
        assert ConfigManager is not None

    def test_config_change_exported(self):
        from src.agent import ConfigChange
        change = ConfigChange(key="test", change_type=ConfigChangeType.MODIFIED)
        assert isinstance(change, ConfigChange)

    def test_config_validator_exported(self):
        from src.agent import ConfigValidator
        assert ConfigValidator is not None


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
