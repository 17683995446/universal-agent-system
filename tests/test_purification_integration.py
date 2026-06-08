"""
阶段16测试：净化管道集成层

测试ContentPurifier与Agent系统的集成。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-26
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from src.agent.content_purifier import ContentPurifier
from src.agent.agent_config import AgentConfig
from src.modules.purification.purification_types import PurificationLevel, PurificationResult


class TestContentPurifierBasic:
    """ContentPurifier基础功能测试"""

    def test_content_purifier_creation(self):
        """测试ContentPurifier创建"""
        purifier = ContentPurifier()
        assert purifier is not None
        assert purifier.enable_input_purification == True
        assert purifier.enable_output_purification == False

    def test_content_purifier_with_custom_settings(self):
        """测试自定义配置创建"""
        purifier = ContentPurifier(
            enable_input_purification=False,
            enable_output_purification=True,
            input_max_level=PurificationLevel.RULE_BASED,
            output_max_level=PurificationLevel.STRUCTURED,
        )
        assert purifier.enable_input_purification == False
        assert purifier.enable_output_purification == True

    def test_purify_user_input_enabled(self):
        """测试用户输入净化（启用）"""
        purifier = ContentPurifier(enable_input_purification=True)
        
        # 简单输入应该通过规则净化
        result = purifier.purify_user_input("Hello, how are you?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_purify_user_input_disabled(self):
        """测试用户输入净化（禁用）"""
        purifier = ContentPurifier(enable_input_purification=False)
        
        input_text = "Hello, how are you?"
        result = purifier.purify_user_input(input_text)
        assert result == input_text  # 禁用时返回原文本

    def test_purify_memory_context(self):
        """测试记忆上下文净化"""
        purifier = ContentPurifier(enable_input_purification=True)
        
        context = "Previous conversation: User asked about weather."
        result = purifier.purify_memory_context(context)
        assert isinstance(result, str)

    def test_purify_assistant_reply_enabled(self):
        """测试助手回复净化（启用）"""
        purifier = ContentPurifier(enable_output_purification=True)
        
        reply = "I can help you with that. Let me check the weather."
        result = purifier.purify_assistant_reply(reply)
        assert isinstance(result, str)

    def test_purify_assistant_reply_disabled(self):
        """测试助手回复净化（禁用）"""
        purifier = ContentPurifier(enable_output_purification=False)
        
        reply = "I can help you with that."
        result = purifier.purify_assistant_reply(reply)
        assert result == reply  # 禁用时返回原文本

    def test_purify_tool_result(self):
        """测试工具结果净化"""
        purifier = ContentPurifier(enable_output_purification=True)
        
        # 字符串结果
        result = purifier.purify_tool_result("Tool output: success")
        assert isinstance(result, str)
        
        # 非字符串结果（转为JSON）
        result = purifier.purify_tool_result({"status": "ok", "data": [1, 2, 3]})
        assert isinstance(result, str)
        assert "status" in result or "ok" in result

    def test_get_stats(self):
        """测试统计信息获取"""
        purifier = ContentPurifier()
        
        stats = purifier.get_stats()
        assert isinstance(stats, dict)
        assert "input_purifications" in stats
        assert "output_purifications" in stats


class TestContentPurifierWithPipeline:
    """ContentPurifier与Pipeline集成测试"""

    def test_purify_input_returns_result(self):
        """测试输入净化返回PurificationResult"""
        purifier = ContentPurifier(enable_input_purification=True)
        
        result = purifier.purify_input("Test input", original_tokens=100)
        assert isinstance(result, PurificationResult)
        assert result.original_text == "Test input"
        assert result.level >= PurificationLevel.NONE

    def test_purify_output_returns_result(self):
        """测试输出净化返回PurificationResult"""
        purifier = ContentPurifier(enable_output_purification=True)
        
        result = purifier.purify_output("Test output", original_tokens=50)
        assert isinstance(result, PurificationResult)
        assert result.original_text == "Test output"

    def test_max_level_limiting(self):
        """测试最大级别限制"""
        purifier = ContentPurifier(
            enable_input_purification=True,
            input_max_level=PurificationLevel.RULE_BASED,
        )
        
        # 即使文本很长，也不应超过RULE_BASED级别
        long_text = "This is a very long text " * 100  # 约2000+ tokens
        result = purifier.purify_input(long_text, original_tokens=3000)
        assert result.level <= PurificationLevel.RULE_BASED


class TestAgentPurificationIntegration:
    """Agent与净化集成测试"""

    def test_agent_config_purification_fields(self):
        """测试AgentConfig净化配置字段"""
        config = AgentConfig()
        assert hasattr(config, 'enable_input_purification')
        assert hasattr(config, 'enable_output_purification')
        assert config.enable_input_purification == True
        assert config.enable_output_purification == False

    def test_agent_config_custom_purification(self):
        """测试AgentConfig自定义净化配置"""
        config = AgentConfig(
            enable_input_purification=False,
            enable_output_purification=True,
        )
        assert config.enable_input_purification == False
        assert config.enable_output_purification == True

    def test_agent_config_serialization_with_purification(self):
        """测试AgentConfig序列化包含净化配置"""
        config = AgentConfig(
            enable_input_purification=True,
            enable_output_purification=True,
        )
        
        data = config.to_dict()
        assert "enable_input_purification" in data
        assert "enable_output_purification" in data
        assert data["enable_input_purification"] == True
        assert data["enable_output_purification"] == True

    def test_agent_config_deserialization_with_purification(self):
        """测试AgentConfig反序列化包含净化配置"""
        data = {
            "enable_input_purification": False,
            "enable_output_purification": True,
        }
        
        config = AgentConfig.from_dict(data)
        assert config.enable_input_purification == False
        assert config.enable_output_purification == True

    @patch('src.infrastructure.llm_client.LLMClient')
    @patch('src.modules.purification.pipeline.PurificationPipeline')
    def test_agent_initializes_content_purifier(self, mock_pipeline, mock_llm):
        """测试Agent初始化ContentPurifier"""
        # 配置mock
        mock_pipeline.return_value = Mock()
        
        config = AgentConfig(enable_input_purification=True)
        
        with patch('src.agent.agent.LLMClient', mock_llm):
            from src.agent.agent import Agent
            agent = Agent(config=config)
        
        # 检查净化器已初始化
        assert agent.content_purifier is not None
        assert isinstance(agent.content_purifier, ContentPurifier)

    @patch('src.infrastructure.llm_client.LLMClient')
    @patch('src.modules.purification.pipeline.PurificationPipeline')
    def test_agent_purifier_disabled(self, mock_pipeline, mock_llm):
        """测试Agent禁用净化器"""
        mock_pipeline.return_value = Mock()
        
        config = AgentConfig(
            enable_input_purification=False,
            enable_output_purification=False,
        )
        
        with patch('src.agent.agent.LLMClient', mock_llm):
            from src.agent.agent import Agent
            agent = Agent(config=config)
        
        # 净化器存在但功能禁用
        assert agent.content_purifier is not None
        assert agent.content_purifier.enable_input_purification == False
        assert agent.content_purifier.enable_output_purification == False


class TestTurnManagerPurificationFlow:
    """TurnManager净化流程测试"""

    def test_purify_input_helper(self):
        """测试TurnManager输入净化辅助方法"""
        from src.agent.turn_manager import TurnManager
        
        # 创建mock purifier
        mock_purifier = Mock(spec=ContentPurifier)
        mock_purifier.purify_user_input.return_value = "purified input"
        
        # 创建TurnManager（简化参数）
        turn_manager = TurnManager(
            llm_client=Mock(),
            content_purifier=mock_purifier,
        )
        
        # 测试输入净化
        result = turn_manager._purify_input("original input")
        assert result == "purified input"
        mock_purifier.purify_user_input.assert_called_once_with("original input")

    def test_purify_input_no_purifier(self):
        """测试TurnManager无净化器时的输入处理"""
        from src.agent.turn_manager import TurnManager
        
        turn_manager = TurnManager(
            llm_client=Mock(),
            content_purifier=None,
        )
        
        # 无净化器时返回原文本
        result = turn_manager._purify_input("original input")
        assert result == "original input"

    def test_purify_output_helper(self):
        """测试TurnManager输出净化辅助方法"""
        from src.agent.turn_manager import TurnManager
        
        # 创建mock purifier
        mock_purifier = Mock(spec=ContentPurifier)
        mock_purifier.purify_assistant_reply.return_value = "purified reply"
        
        turn_manager = TurnManager(
            llm_client=Mock(),
            content_purifier=mock_purifier,
        )
        
        result = turn_manager._purify_output("original reply")
        assert result == "purified reply"
        mock_purifier.purify_assistant_reply.assert_called_once_with("original reply")

    def test_purify_output_no_purifier(self):
        """测试TurnManager无净化器时的输出处理"""
        from src.agent.turn_manager import TurnManager
        
        turn_manager = TurnManager(
            llm_client=Mock(),
            content_purifier=None,
        )
        
        result = turn_manager._purify_output("original reply")
        assert result == "original reply"


class TestExportContentPurifier:
    """导出测试"""

    def test_content_purifier_exported(self):
        """测试ContentPurifier从agent模块导出"""
        from src.agent import ContentPurifier
        
        assert ContentPurifier is not None
        
        # 可以创建实例
        purifier = ContentPurifier()
        assert isinstance(purifier, ContentPurifier)


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])