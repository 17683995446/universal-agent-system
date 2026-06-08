"""
阶段18测试：Agent生命周期管理

测试Agent状态机、资源清理、快照保存/恢复功能。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-26
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch

from src.agent.agent_types import AgentState, AgentStats, AgentSnapshot


class TestAgentStateTransitions:
    """AgentState状态转换测试"""

    def test_can_transition_valid(self):
        """测试合法状态转换"""
        assert AgentState.can_transition(AgentState.IDLE, AgentState.THINKING) == True
        assert AgentState.can_transition(AgentState.IDLE, AgentState.STOPPED) == True
        assert AgentState.can_transition(AgentState.THINKING, AgentState.EXECUTING) == True
        assert AgentState.can_transition(AgentState.THINKING, AgentState.IDLE) == True
        assert AgentState.can_transition(AgentState.EXECUTING, AgentState.IDLE) == True
        assert AgentState.can_transition(AgentState.ERROR, AgentState.IDLE) == True
        assert AgentState.can_transition(AgentState.STOPPED, AgentState.IDLE) == True

    def test_can_transition_invalid(self):
        """测试非法状态转换"""
        assert AgentState.can_transition(AgentState.STOPPED, AgentState.THINKING) == False
        assert AgentState.can_transition(AgentState.STOPPED, AgentState.EXECUTING) == False
        assert AgentState.can_transition(AgentState.ERROR, AgentState.THINKING) == False

    def test_is_active(self):
        """测试活跃状态判断"""
        assert AgentState.IDLE.is_active == True
        assert AgentState.THINKING.is_active == True
        assert AgentState.EXECUTING.is_active == True
        assert AgentState.PAUSED.is_active == False
        assert AgentState.ERROR.is_active == False
        assert AgentState.STOPPED.is_active == False

    def test_is_terminal(self):
        """测试终止状态判断"""
        assert AgentState.STOPPED.is_terminal == True
        assert AgentState.IDLE.is_terminal == False
        assert AgentState.ERROR.is_terminal == False


class TestAgentSnapshot:
    """AgentSnapshot快照测试"""

    def test_creation(self):
        """测试创建快照"""
        snapshot = AgentSnapshot(
            snapshot_id="snap_001",
            agent_id="agent_1",
            agent_name="TestAgent",
            state="idle",
        )
        assert snapshot.snapshot_id == "snap_001"
        assert snapshot.agent_id == "agent_1"
        assert snapshot.agent_name == "TestAgent"
        assert snapshot.state == "idle"

    def test_serialization(self):
        """测试序列化"""
        snapshot = AgentSnapshot(
            snapshot_id="snap_001",
            agent_id="agent_1",
            conversation_history=[{"role": "user", "content": "hello"}],
            stats_dict={"total_turns": 5},
        )
        data = snapshot.to_dict()
        assert data["snapshot_id"] == "snap_001"
        assert data["agent_id"] == "agent_1"
        assert len(data["conversation_history"]) == 1

    def test_deserialization(self):
        """测试反序列化"""
        data = {
            "snapshot_id": "snap_002",
            "agent_id": "agent_2",
            "state": "thinking",
            "conversation_history": [],
            "stats": {"total_turns": 3},
        }
        snapshot = AgentSnapshot.from_dict(data)
        assert snapshot.snapshot_id == "snap_002"
        assert snapshot.state == "thinking"
        assert snapshot.stats_dict["total_turns"] == 3

    def test_json_roundtrip(self):
        """测试JSON序列化往返"""
        original = AgentSnapshot(
            snapshot_id="snap_003",
            agent_id="agent_3",
            agent_name="Test",
            state="idle",
            conversation_history=[{"role": "user", "content": "test"}],
        )
        json_str = original.to_json()
        restored = AgentSnapshot.from_json(json_str)
        assert restored.snapshot_id == original.snapshot_id
        assert restored.agent_name == original.agent_name

    def test_is_valid(self):
        """测试有效性检查"""
        valid = AgentSnapshot(
            snapshot_id="snap_001",
            agent_id="agent_1",
            version="1.0",
        )
        assert valid.is_valid() == True

        invalid = AgentSnapshot()
        assert invalid.is_valid() == False

    def test_get_summary(self):
        """测试摘要获取"""
        snapshot = AgentSnapshot(
            snapshot_id="snap_001",
            agent_id="agent_1",
            agent_name="Test",
            state="idle",
            conversation_history=[{}, {}],
            stats_dict={"total_turns": 10},
        )
        summary = snapshot.get_summary()
        assert summary["snapshot_id"] == "snap_001"
        assert summary["history_length"] == 2
        assert summary["total_turns"] == 10


class TestAgentStatsSerialization:
    """AgentStats序列化测试"""

    def test_to_dict(self):
        """测试序列化为字典"""
        stats = AgentStats()
        stats.total_turns = 10
        stats.total_tokens = 1000
        data = stats.to_dict()
        assert data["total_turns"] == 10
        assert data["total_tokens"] == 1000

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "total_turns": 5,
            "total_tokens": 500,
            "errors_count": 2,
        }
        stats = AgentStats.from_dict(data)
        assert stats.total_turns == 5
        assert stats.total_tokens == 500
        assert stats.errors_count == 2


class TestAgentLifecycle:
    """Agent生命周期测试"""

    @patch('src.infrastructure.llm_client.LLMClient')
    @patch('src.modules.purification.pipeline.PurificationPipeline')
    def test_state_transition(self, mock_pipeline, mock_llm):
        """测试状态转换"""
        mock_pipeline.return_value = Mock()

        from src.agent.agent import Agent
        from src.agent.agent_config import AgentConfig

        config = AgentConfig()
        with patch('src.agent.agent.LLMClient', mock_llm):
            agent = Agent(config=config)

        assert agent.state == AgentState.IDLE

    @patch('src.infrastructure.llm_client.LLMClient')
    @patch('src.modules.purification.pipeline.PurificationPipeline')
    def test_pause_resume(self, mock_pipeline, mock_llm):
        """测试暂停和恢复"""
        mock_pipeline.return_value = Mock()

        from src.agent.agent import Agent
        from src.agent.agent_config import AgentConfig

        config = AgentConfig()
        with patch('src.agent.agent.LLMClient', mock_llm):
            agent = Agent(config=config)

        # 暂停
        assert agent.pause() == True
        assert agent.state == AgentState.PAUSED

        # 恢复
        assert agent.resume() == True
        assert agent.state == AgentState.IDLE

    @patch('src.infrastructure.llm_client.LLMClient')
    @patch('src.modules.purification.pipeline.PurificationPipeline')
    def test_stop_restart(self, mock_pipeline, mock_llm):
        """测试停止和重启"""
        mock_pipeline.return_value = Mock()

        from src.agent.agent import Agent
        from src.agent.agent_config import AgentConfig

        config = AgentConfig()
        with patch('src.agent.agent.LLMClient', mock_llm):
            agent = Agent(config=config)

        # 停止
        agent.stop()
        assert agent.state == AgentState.STOPPED

        # 重启
        assert agent.restart() == True
        assert agent.state == AgentState.IDLE

    @patch('src.infrastructure.llm_client.LLMClient')
    @patch('src.modules.purification.pipeline.PurificationPipeline')
    def test_invalid_transition(self, mock_pipeline, mock_llm):
        """测试非法状态转换"""
        mock_pipeline.return_value = Mock()

        from src.agent.agent import Agent
        from src.agent.agent_config import AgentConfig

        config = AgentConfig()
        with patch('src.agent.agent.LLMClient', mock_llm):
            agent = Agent(config=config)

        # 停止后不能直接到THINKING
        agent.stop()
        # pause()内部使用_transition_state，应该返回False
        result = agent.pause()
        assert result == False
        assert agent.state == AgentState.STOPPED

    @patch('src.infrastructure.llm_client.LLMClient')
    @patch('src.modules.purification.pipeline.PurificationPipeline')
    def test_create_snapshot(self, mock_pipeline, mock_llm):
        """测试创建快照"""
        mock_pipeline.return_value = Mock()

        from src.agent.agent import Agent
        from src.agent.agent_config import AgentConfig

        config = AgentConfig(agent_id="test_agent", name="Test")
        with patch('src.agent.agent.LLMClient', mock_llm):
            agent = Agent(config=config)

        snapshot = agent.create_snapshot()
        assert snapshot is not None
        assert snapshot.agent_id == "test_agent"
        assert snapshot.agent_name == "Test"
        assert snapshot.state == "idle"
        assert snapshot.is_valid() == True

    @patch('src.infrastructure.llm_client.LLMClient')
    @patch('src.modules.purification.pipeline.PurificationPipeline')
    def test_save_and_load_snapshot(self, mock_pipeline, mock_llm):
        """测试保存和加载快照"""
        mock_pipeline.return_value = Mock()

        from src.agent.agent import Agent
        from src.agent.agent_config import AgentConfig

        config = AgentConfig(agent_id="test_agent")
        with patch('src.agent.agent.LLMClient', mock_llm):
            agent = Agent(config=config)

        # 添加一些历史
        agent._conversation_history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

        # 保存到临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            filepath = agent.save_snapshot(temp_path)
            assert os.path.exists(filepath)

            # 创建新Agent并加载
            with patch('src.agent.agent.LLMClient', mock_llm):
                new_agent = Agent(config=config)

            result = new_agent.load_snapshot(filepath)
            assert result == True
            assert len(new_agent.get_history()) == 2
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @patch('src.infrastructure.llm_client.LLMClient')
    @patch('src.modules.purification.pipeline.PurificationPipeline')
    def test_restore_from_snapshot(self, mock_pipeline, mock_llm):
        """测试从快照恢复"""
        mock_pipeline.return_value = Mock()

        from src.agent.agent import Agent
        from src.agent.agent_config import AgentConfig

        config = AgentConfig(agent_id="test_agent")
        with patch('src.agent.agent.LLMClient', mock_llm):
            agent = Agent(config=config)

        # 创建快照
        snapshot = agent.create_snapshot()

        # 修改Agent状态
        agent._conversation_history = [{"role": "user", "content": "new"}]

        # 恢复
        result = agent.restore_from_snapshot(snapshot)
        assert result == True
        # 历史应该恢复为空（因为快照创建时为空）
        assert len(agent.get_history()) == 0

    @patch('src.infrastructure.llm_client.LLMClient')
    @patch('src.modules.purification.pipeline.PurificationPipeline')
    def test_invalid_snapshot_restore(self, mock_pipeline, mock_llm):
        """测试无效快照恢复失败"""
        mock_pipeline.return_value = Mock()

        from src.agent.agent import Agent
        from src.agent.agent_config import AgentConfig

        config = AgentConfig()
        with patch('src.agent.agent.LLMClient', mock_llm):
            agent = Agent(config=config)

        invalid_snapshot = AgentSnapshot()  # 无效快照
        result = agent.restore_from_snapshot(invalid_snapshot)
        assert result == False

    @patch('src.infrastructure.llm_client.LLMClient')
    @patch('src.modules.purification.pipeline.PurificationPipeline')
    def test_cleanup(self, mock_pipeline, mock_llm):
        """测试资源清理"""
        mock_pipeline.return_value = Mock()

        from src.agent.agent import Agent
        from src.agent.agent_config import AgentConfig

        config = AgentConfig()
        with patch('src.agent.agent.LLMClient', mock_llm):
            agent = Agent(config=config)

        # 清理不应抛出异常
        agent.cleanup()
        assert agent.state == AgentState.STOPPED


class TestAgentSnapshotExport:
    """导出测试"""

    def test_agent_snapshot_exported(self):
        """测试AgentSnapshot导出"""
        from src.agent import AgentSnapshot
        assert AgentSnapshot is not None

        snapshot = AgentSnapshot(snapshot_id="test", agent_id="agent")
        assert isinstance(snapshot, AgentSnapshot)


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
