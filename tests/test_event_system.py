"""
阶段20测试：Agent事件系统

测试AgentEventType、AgentEvent、EventBus及Agent集成。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-26
"""

import pytest
from unittest.mock import Mock, patch

from src.agent.event_bus import (
    AgentEventType,
    AgentEvent,
    EventBus,
)


class TestAgentEventType:
    """AgentEventType枚举测试"""

    def test_enum_values_exist(self):
        """测试关键事件类型存在"""
        assert AgentEventType.STATE_CHANGED.value == "state_changed"
        assert AgentEventType.TURN_STARTED.value == "turn_started"
        assert AgentEventType.TURN_COMPLETED.value == "turn_completed"
        assert AgentEventType.TURN_ERROR.value == "turn_error"
        assert AgentEventType.ERROR_OCCURRED.value == "error_occurred"
        assert AgentEventType.SNAPSHOT_CREATED.value == "snapshot_created"

    def test_enum_count(self):
        """测试事件类型数量"""
        assert len(AgentEventType) >= 26

    def test_lifecycle_events(self):
        """测试生命周期事件"""
        lifecycle = [
            AgentEventType.STATE_CHANGED,
            AgentEventType.AGENT_STARTED,
            AgentEventType.AGENT_STOPPED,
            AgentEventType.AGENT_RESET,
            AgentEventType.AGENT_PAUSED,
            AgentEventType.AGENT_RESUMED,
        ]
        for event in lifecycle:
            assert event in AgentEventType


class TestAgentEvent:
    """AgentEvent事件数据测试"""

    def test_creation(self):
        """测试创建事件"""
        event = AgentEvent(
            event_type=AgentEventType.STATE_CHANGED,
            agent_id="agent_1",
            data={"old_state": "idle", "new_state": "thinking"},
        )
        assert event.event_type == AgentEventType.STATE_CHANGED
        assert event.agent_id == "agent_1"
        assert event.data["old_state"] == "idle"

    def test_create_factory(self):
        """测试便捷创建方法"""
        event = AgentEvent.create(
            event_type=AgentEventType.TURN_STARTED,
            agent_id="agent_1",
            data={"user_input": "hello"},
            turn_id="turn_001",
        )
        assert event.event_type == AgentEventType.TURN_STARTED
        assert event.turn_id == "turn_001"

    def test_get_data(self):
        """测试数据获取"""
        event = AgentEvent(
            event_type=AgentEventType.ERROR_OCCURRED,
            data={"error": "timeout", "category": "RECOVERABLE"},
        )
        assert event.get("error") == "timeout"
        assert event.get("missing", "default") == "default"

    def test_serialization(self):
        """测试序列化"""
        event = AgentEvent(
            event_type=AgentEventType.TURN_COMPLETED,
            agent_id="agent_1",
            data={"status": "completed"},
            turn_id="turn_001",
        )
        data = event.to_dict()
        assert data["event_type"] == "turn_completed"
        assert data["agent_id"] == "agent_1"
        assert data["turn_id"] == "turn_001"
        assert data["data"]["status"] == "completed"


class TestEventBus:
    """EventBus事件总线测试"""

    def test_creation(self):
        """测试创建"""
        bus = EventBus()
        assert bus.subscriber_count == 0
        assert len(bus.event_history) == 0

    def test_subscribe_and_publish(self):
        """测试订阅和发布"""
        bus = EventBus()
        handler = Mock()

        bus.subscribe(AgentEventType.STATE_CHANGED, handler)
        bus.emit(AgentEventType.STATE_CHANGED, data={"old": "idle", "new": "thinking"})

        handler.assert_called_once()
        event = handler.call_args[0][0]
        assert event.event_type == AgentEventType.STATE_CHANGED
        assert event.data["old"] == "idle"

    def test_multiple_subscribers(self):
        """测试多个订阅者"""
        bus = EventBus()
        handler1 = Mock()
        handler2 = Mock()

        bus.subscribe(AgentEventType.TURN_STARTED, handler1)
        bus.subscribe(AgentEventType.TURN_STARTED, handler2)

        triggered = bus.emit(AgentEventType.TURN_STARTED)
        assert triggered == 2
        handler1.assert_called_once()
        handler2.assert_called_once()

    def test_unsubscribe(self):
        """测试取消订阅"""
        bus = EventBus()
        handler = Mock()

        bus.subscribe(AgentEventType.TURN_STARTED, handler)
        assert bus.subscriber_count == 1

        bus.unsubscribe(AgentEventType.TURN_STARTED, handler)
        assert bus.subscriber_count == 0

        bus.emit(AgentEventType.TURN_STARTED)
        handler.assert_not_called()

    def test_wildcard_subscribe(self):
        """测试通配符订阅"""
        bus = EventBus()
        handler = Mock()

        bus.subscribe_all(handler)

        bus.emit(AgentEventType.STATE_CHANGED)
        bus.emit(AgentEventType.TURN_STARTED)
        bus.emit(AgentEventType.ERROR_OCCURRED)

        assert handler.call_count == 3

    def test_unsubscribe_wildcard(self):
        """测试取消通配符订阅"""
        bus = EventBus()
        handler = Mock()

        bus.subscribe_all(handler)
        assert bus.subscriber_count == 1

        bus.unsubscribe_all(handler)
        assert bus.subscriber_count == 0

    def test_once_subscription(self):
        """测试一次性订阅"""
        bus = EventBus()
        handler = Mock()

        bus.subscribe(AgentEventType.TURN_STARTED, handler, once=True)

        bus.emit(AgentEventType.TURN_STARTED)
        assert handler.call_count == 1

        # 第二次不应触发
        bus.emit(AgentEventType.TURN_STARTED)
        assert handler.call_count == 1

    def test_once_wildcard(self):
        """测试一次性通配符订阅"""
        bus = EventBus()
        handler = Mock()

        bus.subscribe_all(handler, once=True)

        bus.emit(AgentEventType.STATE_CHANGED)
        assert handler.call_count == 1

        bus.emit(AgentEventType.TURN_STARTED)
        assert handler.call_count == 1  # 仍然只触发一次

    def test_handler_error_does_not_stop(self):
        """测试处理器错误不影响其他处理器"""
        bus = EventBus()

        bad_handler = Mock(side_effect=RuntimeError("handler error"))
        good_handler = Mock()

        bus.subscribe(AgentEventType.TURN_STARTED, bad_handler)
        bus.subscribe(AgentEventType.TURN_STARTED, good_handler)

        triggered = bus.emit(AgentEventType.TURN_STARTED)
        # bad_handler抛出异常不计入triggered，但good_handler正常执行
        assert triggered >= 1
        good_handler.assert_called_once()

    def test_no_subscribers(self):
        """测试无订阅者时发布"""
        bus = EventBus()
        triggered = bus.emit(AgentEventType.TURN_STARTED)
        assert triggered == 0

    def test_event_history(self):
        """测试事件历史记录"""
        bus = EventBus(max_history=10)

        for i in range(15):
            bus.emit(AgentEventType.TURN_COMPLETED)

        assert len(bus.event_history) == 10  # 被max_history限制

    def test_get_history(self):
        """测试获取历史"""
        bus = EventBus()

        bus.emit(AgentEventType.STATE_CHANGED)
        bus.emit(AgentEventType.TURN_STARTED)
        bus.emit(AgentEventType.STATE_CHANGED)

        # 按类型筛选
        state_events = bus.get_history(event_type=AgentEventType.STATE_CHANGED)
        assert len(state_events) == 2

        # 全部
        all_events = bus.get_history(limit=10)
        assert len(all_events) == 3

    def test_get_stats(self):
        """测试统计"""
        bus = EventBus()
        handler = Mock()
        bus.subscribe(AgentEventType.TURN_STARTED, handler)

        bus.emit(AgentEventType.TURN_STARTED)
        bus.emit(AgentEventType.TURN_COMPLETED)

        stats = bus.get_stats()
        assert stats["total_published"] == 2
        assert stats["total_handled"] == 1
        assert stats["subscriber_count"] == 1

    def test_clear_subscribers(self):
        """测试清除订阅者"""
        bus = EventBus()
        bus.subscribe(AgentEventType.TURN_STARTED, Mock())
        bus.subscribe(AgentEventType.TURN_COMPLETED, Mock())

        bus.clear_subscribers()
        assert bus.subscriber_count == 0

    def test_clear_subscribers_by_type(self):
        """测试按类型清除订阅者"""
        bus = EventBus()
        bus.subscribe(AgentEventType.TURN_STARTED, Mock())
        bus.subscribe(AgentEventType.TURN_COMPLETED, Mock())

        bus.clear_subscribers(AgentEventType.TURN_STARTED)
        assert bus.subscriber_count == 1

    def test_clear_history(self):
        """测试清空历史"""
        bus = EventBus()
        bus.emit(AgentEventType.TURN_STARTED)
        bus.emit(AgentEventType.TURN_COMPLETED)

        bus.clear_history()
        assert len(bus.event_history) == 0

    def test_reset_stats(self):
        """测试重置统计"""
        bus = EventBus()
        bus.emit(AgentEventType.TURN_STARTED)

        bus.reset_stats()
        stats = bus.get_stats()
        assert stats["total_published"] == 0
        assert stats["history_size"] == 0

    def test_publish_returns_triggered_count(self):
        """测试发布返回触发数"""
        bus = EventBus()
        h1, h2, h3 = Mock(), Mock(), Mock()

        bus.subscribe(AgentEventType.TURN_STARTED, h1)
        bus.subscribe(AgentEventType.TURN_STARTED, h2)
        bus.subscribe_all(h3)

        count = bus.emit(AgentEventType.TURN_STARTED)
        assert count == 3


class TestAgentEventIntegration:
    """Agent事件集成测试"""

    @patch('src.infrastructure.llm_client.LLMClient')
    @patch('src.modules.purification.pipeline.PurificationPipeline')
    def test_agent_has_event_bus(self, mock_pipeline, mock_llm):
        """测试Agent拥有事件总线"""
        mock_pipeline.return_value = Mock()

        from src.agent.agent import Agent
        from src.agent.agent_config import AgentConfig

        config = AgentConfig()
        with patch('src.agent.agent.LLMClient', mock_llm):
            agent = Agent(config=config)

        assert agent.event_bus is not None
        assert isinstance(agent.event_bus, EventBus)

    @patch('src.infrastructure.llm_client.LLMClient')
    @patch('src.modules.purification.pipeline.PurificationPipeline')
    def test_state_change_emits_event(self, mock_pipeline, mock_llm):
        """测试状态变更发射事件"""
        mock_pipeline.return_value = Mock()

        from src.agent.agent import Agent
        from src.agent.agent_config import AgentConfig

        config = AgentConfig(agent_id="test")
        with patch('src.agent.agent.LLMClient', mock_llm):
            agent = Agent(config=config)

        handler = Mock()
        agent.event_bus.subscribe(AgentEventType.STATE_CHANGED, handler)

        agent.pause()
        handler.assert_called_once()
        event = handler.call_args[0][0]
        assert event.data["old_state"] == "idle"
        assert event.data["new_state"] == "paused"

    @patch('src.infrastructure.llm_client.LLMClient')
    @patch('src.modules.purification.pipeline.PurificationPipeline')
    def test_summary_includes_events(self, mock_pipeline, mock_llm):
        """测试摘要包含事件统计"""
        mock_pipeline.return_value = Mock()

        from src.agent.agent import Agent
        from src.agent.agent_config import AgentConfig

        config = AgentConfig()
        with patch('src.agent.agent.LLMClient', mock_llm):
            agent = Agent(config=config)

        summary = agent.get_summary()
        assert "events" in summary
        assert "total_published" in summary["events"]


class TestExportEventBus:
    """导出测试"""

    def test_event_bus_exported(self):
        from src.agent import EventBus
        assert EventBus is not None

    def test_event_type_exported(self):
        from src.agent import AgentEventType
        assert AgentEventType.STATE_CHANGED.value == "state_changed"

    def test_agent_event_exported(self):
        from src.agent import AgentEvent
        event = AgentEvent(event_type=AgentEventType.TURN_STARTED)
        assert isinstance(event, AgentEvent)


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
