"""
阶段17测试：上下文压缩策略

测试ContextCompressor、CompressionConfig、CompressionStrategy
以及与ContextBuilder和Agent的集成。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-26
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from src.agent.context_compressor import (
    ContextCompressor,
    CompressionConfig,
    CompressionStrategy,
    CompressionResult,
)
from src.modules.purification.purification_types import PurificationLevel


class TestCompressionStrategy:
    """CompressionStrategy枚举测试"""

    def test_enum_values(self):
        """测试枚举值"""
        assert CompressionStrategy.NONE == 0
        assert CompressionStrategy.CONSERVATIVE == 1
        assert CompressionStrategy.BALANCED == 2
        assert CompressionStrategy.AGGRESSIVE == 3

    def test_enum_count(self):
        """测试枚举数量"""
        assert len(CompressionStrategy) == 4

    def test_enum_comparison(self):
        """测试枚举比较"""
        assert CompressionStrategy.AGGRESSIVE > CompressionStrategy.BALANCED
        assert CompressionStrategy.BALANCED > CompressionStrategy.CONSERVATIVE


class TestCompressionConfig:
    """CompressionConfig测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = CompressionConfig()
        assert config.strategy == CompressionStrategy.BALANCED
        assert config.enabled == True
        assert config.light_threshold == 0.60
        assert config.medium_threshold == 0.80
        assert config.heavy_threshold == 0.95
        assert config.preserve_recent_n == 2

    def test_custom_config(self):
        """测试自定义配置"""
        config = CompressionConfig(
            strategy=CompressionStrategy.AGGRESSIVE,
            enabled=False,
            preserve_recent_n=5,
        )
        assert config.strategy == CompressionStrategy.AGGRESSIVE
        assert config.enabled == False
        assert config.preserve_recent_n == 5

    def test_get_thresholds_balanced(self):
        """测试均衡策略阈值"""
        config = CompressionConfig(strategy=CompressionStrategy.BALANCED)
        light, medium, heavy, critical = config.get_thresholds()
        assert light == 0.60
        assert medium == 0.80
        assert heavy == 0.95
        assert critical == 1.0

    def test_get_thresholds_none(self):
        """测试不压缩策略阈值"""
        config = CompressionConfig(strategy=CompressionStrategy.NONE)
        light, medium, heavy, critical = config.get_thresholds()
        assert light == 1.0
        assert medium == 1.0
        assert heavy == 1.0
        assert critical == 1.0

    def test_get_thresholds_conservative(self):
        """测试保守策略阈值"""
        config = CompressionConfig(strategy=CompressionStrategy.CONSERVATIVE)
        light, medium, heavy, critical = config.get_thresholds()
        assert light == 0.80
        assert medium == 0.90
        assert heavy == 0.97
        assert critical == 1.0

    def test_get_thresholds_aggressive(self):
        """测试激进策略阈值"""
        config = CompressionConfig(strategy=CompressionStrategy.AGGRESSIVE)
        light, medium, heavy, critical = config.get_thresholds()
        assert light == 0.40
        assert medium == 0.60
        assert heavy == 0.80
        assert critical == 0.95

    def test_serialization(self):
        """测试序列化"""
        config = CompressionConfig(
            strategy=CompressionStrategy.BALANCED,
            enabled=True,
        )
        data = config.to_dict()
        assert data["strategy"] == 2
        assert data["strategy_name"] == "BALANCED"
        assert data["enabled"] == True

    def test_deserialization(self):
        """测试反序列化"""
        data = {
            "strategy": "AGGRESSIVE",
            "enabled": False,
            "preserve_recent_n": 3,
        }
        config = CompressionConfig.from_dict(data)
        assert config.strategy == CompressionStrategy.AGGRESSIVE
        assert config.enabled == False
        assert config.preserve_recent_n == 3

    def test_deserialization_int_strategy(self):
        """测试整数策略反序列化"""
        data = {"strategy": 1}
        config = CompressionConfig.from_dict(data)
        assert config.strategy == CompressionStrategy.CONSERVATIVE


class TestCompressionResult:
    """CompressionResult测试"""

    def test_basic_result(self):
        """测试基本结果"""
        result = CompressionResult(
            original_tokens=1000,
            compressed_tokens=800,
            compression_level=PurificationLevel.RULE_BASED,
            messages_compressed=5,
            messages_preserved=2,
            pressure_level="medium",
        )
        assert result.tokens_saved == 200
        assert result.compression_ratio == 0.8
        assert result.compression_percentage == pytest.approx(20.0)
        assert result.was_compressed == True

    def test_no_compression(self):
        """测试无压缩结果"""
        result = CompressionResult(
            original_tokens=500,
            compressed_tokens=500,
            compression_level=PurificationLevel.NONE,
            messages_compressed=0,
            messages_preserved=3,
            pressure_level="none",
        )
        assert result.tokens_saved == 0
        assert result.was_compressed == False

    def test_serialization(self):
        """测试结果序列化"""
        result = CompressionResult(
            original_tokens=1000,
            compressed_tokens=700,
            compression_level=PurificationLevel.STRUCTURED,
            messages_compressed=4,
            messages_preserved=2,
            pressure_level="heavy",
        )
        data = result.to_dict()
        assert data["original_tokens"] == 1000
        assert data["compressed_tokens"] == 700
        assert data["tokens_saved"] == 300
        assert data["compression_level_name"] == "STRUCTURED"


class TestContextCompressor:
    """ContextCompressor核心功能测试"""

    def test_creation(self):
        """测试创建"""
        compressor = ContextCompressor()
        assert compressor.enabled == True
        assert compressor.config.strategy == CompressionStrategy.BALANCED

    def test_creation_disabled(self):
        """测试创建禁用"""
        config = CompressionConfig(enabled=False)
        compressor = ContextCompressor(config=config)
        assert compressor.enabled == False

    def test_assess_pressure_none(self):
        """测试压力评估 - 无压力"""
        compressor = ContextCompressor()
        pressure, ratio = compressor.assess_pressure(1000, 4096)
        assert pressure == "none"
        assert ratio < 0.60

    def test_assess_pressure_light(self):
        """测试压力评估 - 轻度"""
        config = CompressionConfig(strategy=CompressionStrategy.BALANCED)
        compressor = ContextCompressor(config=config)
        # 60% = 2457.6
        pressure, ratio = compressor.assess_pressure(2500, 4096)
        assert pressure == "light"

    def test_assess_pressure_medium(self):
        """测试压力评估 - 中度"""
        config = CompressionConfig(strategy=CompressionStrategy.BALANCED)
        compressor = ContextCompressor(config=config)
        # 80% = 3276.8
        pressure, ratio = compressor.assess_pressure(3300, 4096)
        assert pressure == "medium"

    def test_assess_pressure_heavy(self):
        """测试压力评估 - 重度"""
        config = CompressionConfig(strategy=CompressionStrategy.BALANCED)
        compressor = ContextCompressor(config=config)
        # 95% = 3891.2
        pressure, ratio = compressor.assess_pressure(3900, 4096)
        assert pressure == "heavy"

    def test_assess_pressure_critical(self):
        """测试压力评估 - 极限"""
        compressor = ContextCompressor()
        pressure, ratio = compressor.assess_pressure(5000, 4096)
        assert pressure == "critical"

    def test_determine_compression_level(self):
        """测试压缩级别确定"""
        compressor = ContextCompressor()
        assert compressor.determine_compression_level("none") == PurificationLevel.NONE
        assert compressor.determine_compression_level("light") == PurificationLevel.NONE
        assert compressor.determine_compression_level("medium") == PurificationLevel.RULE_BASED
        assert compressor.determine_compression_level("heavy") == PurificationLevel.STRUCTURED
        assert compressor.determine_compression_level("critical") == PurificationLevel.SUMMARIZED

    def test_compress_messages_no_pressure(self):
        """测试无压力时不压缩"""
        compressor = ContextCompressor()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result_msgs, result = compressor.compress_messages(
            messages=messages,
            used_tokens=100,
            window_size=4096,
        )
        assert result.messages_compressed == 0
        assert result.pressure_level == "none"
        assert len(result_msgs) == 2

    def test_compress_messages_disabled(self):
        """测试禁用时不压缩"""
        config = CompressionConfig(enabled=False)
        compressor = ContextCompressor(config=config)
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        result_msgs, result = compressor.compress_messages(
            messages=messages,
            used_tokens=5000,
            window_size=4096,
        )
        assert result.messages_compressed == 0

    def test_compress_messages_empty(self):
        """测试空消息列表"""
        compressor = ContextCompressor()
        result_msgs, result = compressor.compress_messages(
            messages=[],
            used_tokens=0,
            window_size=4096,
        )
        assert len(result_msgs) == 0
        assert result.messages_compressed == 0

    def test_compress_messages_preserve_recent(self):
        """测试保留最近N条消息"""
        compressor = ContextCompressor()
        # 3条消息，preserve_recent_n=2，只有1条可压缩
        messages = [
            {"role": "user", "content": "A" * 500},  # 可压缩
            {"role": "assistant", "content": "B" * 500},  # 保留
            {"role": "user", "content": "C" * 500},  # 保留
        ]
        # 模拟高压
        result_msgs, result = compressor.compress_messages(
            messages=messages,
            used_tokens=3500,
            window_size=4096,
        )
        # 最近2条应保留
        assert result.messages_preserved == 2

    def test_compress_text(self):
        """测试单文本压缩"""
        compressor = ContextCompressor()
        result = compressor.compress_text(
            text="This is a test message with some content.",
            target_level=PurificationLevel.RULE_BASED,
        )
        assert isinstance(result, type(result))  # PurificationResult
        assert result.original_text == "This is a test message with some content."

    def test_compress_text_disabled(self):
        """测试禁用文本压缩"""
        config = CompressionConfig(enabled=False)
        compressor = ContextCompressor(config=config)
        result = compressor.compress_text("Hello world")
        assert result.level == PurificationLevel.NONE

    def test_get_stats(self):
        """测试统计信息"""
        compressor = ContextCompressor()
        stats = compressor.get_stats()
        assert stats["enabled"] == True
        assert stats["strategy"] == "BALANCED"
        assert stats["total_compressions"] == 0

    def test_reset_stats(self):
        """测试重置统计"""
        compressor = ContextCompressor()
        compressor.reset_stats()
        stats = compressor.get_stats()
        assert stats["total_compressions"] == 0
        assert stats["total_tokens_saved"] == 0


class TestContextBuilderCompression:
    """ContextBuilder压缩集成测试"""

    def test_builder_without_compressor(self):
        """测试无压缩器的构建器"""
        from src.agent.context_builder import ContextBuilder

        builder = ContextBuilder(window_size=4096)
        assert builder.compressor is None

    def test_builder_with_compressor(self):
        """测试带压缩器的构建器"""
        from src.agent.context_builder import ContextBuilder

        compressor = ContextCompressor()
        builder = ContextBuilder(window_size=4096, compressor=compressor)
        assert builder.compressor is not None
        assert builder.compressor is compressor

    def test_set_compressor(self):
        """测试动态设置压缩器"""
        from src.agent.context_builder import ContextBuilder

        builder = ContextBuilder(window_size=4096)
        assert builder.compressor is None

        compressor = ContextCompressor()
        builder.set_compressor(compressor)
        assert builder.compressor is compressor

    def test_set_compressor_none(self):
        """测试移除压缩器"""
        from src.agent.context_builder import ContextBuilder

        compressor = ContextCompressor()
        builder = ContextBuilder(window_size=4096, compressor=compressor)
        builder.set_compressor(None)
        assert builder.compressor is None

    def test_build_without_compression(self):
        """测试无压缩构建"""
        from src.agent.context_builder import ContextBuilder

        builder = ContextBuilder(window_size=4096)
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = builder.build(
            system_prompt="You are helpful.",
            history=history,
            user_input="How are you?",
        )
        assert result is not None
        assert builder.last_compression_result is None

    def test_build_with_compression_low_pressure(self):
        """测试低压构建（不触发压缩）"""
        from src.agent.context_builder import ContextBuilder

        compressor = ContextCompressor()
        builder = ContextBuilder(window_size=4096, compressor=compressor)

        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = builder.build(
            system_prompt="You are helpful.",
            history=history,
            user_input="How are you?",
        )
        assert result is not None
        # 低压不压缩
        assert builder.last_compression_result is not None
        assert builder.last_compression_result.pressure_level == "none"


class TestAgentCompressionIntegration:
    """Agent压缩集成测试"""

    def test_agent_config_compression_fields(self):
        """测试AgentConfig压缩配置字段"""
        from src.agent.agent_config import AgentConfig

        config = AgentConfig()
        assert hasattr(config, 'enable_context_compression')
        assert hasattr(config, 'compression_strategy')
        assert config.enable_context_compression == True
        assert config.compression_strategy == "balanced"

    def test_agent_config_custom_compression(self):
        """测试AgentConfig自定义压缩配置"""
        from src.agent.agent_config import AgentConfig

        config = AgentConfig(
            enable_context_compression=False,
            compression_strategy="aggressive",
        )
        assert config.enable_context_compression == False
        assert config.compression_strategy == "aggressive"

    def test_agent_config_serialization(self):
        """测试AgentConfig序列化包含压缩配置"""
        from src.agent.agent_config import AgentConfig

        config = AgentConfig(
            enable_context_compression=True,
            compression_strategy="conservative",
        )
        data = config.to_dict()
        assert "enable_context_compression" in data
        assert "compression_strategy" in data
        assert data["compression_strategy"] == "conservative"

    def test_agent_config_deserialization(self):
        """测试AgentConfig反序列化"""
        from src.agent.agent_config import AgentConfig

        data = {
            "enable_context_compression": False,
            "compression_strategy": "none",
        }
        config = AgentConfig.from_dict(data)
        assert config.enable_context_compression == False
        assert config.compression_strategy == "none"

    @patch('src.infrastructure.llm_client.LLMClient')
    @patch('src.modules.purification.pipeline.PurificationPipeline')
    def test_agent_initializes_compressor(self, mock_pipeline, mock_llm):
        """测试Agent初始化压缩器"""
        mock_pipeline.return_value = Mock()

        from src.agent.agent_config import AgentConfig
        from src.agent.agent import Agent

        config = AgentConfig(enable_context_compression=True)

        with patch('src.agent.agent.LLMClient', mock_llm):
            agent = Agent(config=config)

        assert agent.context_compressor is not None
        assert isinstance(agent.context_compressor, ContextCompressor)

    @patch('src.infrastructure.llm_client.LLMClient')
    @patch('src.modules.purification.pipeline.PurificationPipeline')
    def test_agent_compressor_disabled(self, mock_pipeline, mock_llm):
        """测试Agent禁用压缩器"""
        mock_pipeline.return_value = Mock()

        from src.agent.agent_config import AgentConfig
        from src.agent.agent import Agent

        config = AgentConfig(enable_context_compression=False)

        with patch('src.agent.agent.LLMClient', mock_llm):
            agent = Agent(config=config)

        # 压缩器存在但禁用
        assert agent.context_compressor is not None
        assert agent.context_compressor.enabled == False


class TestExportCompression:
    """导出测试"""

    def test_context_compressor_exported(self):
        """测试ContextCompressor导出"""
        from src.agent import ContextCompressor
        assert ContextCompressor is not None

    def test_compression_config_exported(self):
        """测试CompressionConfig导出"""
        from src.agent import CompressionConfig
        assert CompressionConfig is not None

    def test_compression_strategy_exported(self):
        """测试CompressionStrategy导出"""
        from src.agent import CompressionStrategy
        assert CompressionStrategy is not None
        assert CompressionStrategy.BALANCED == 2


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])