"""
阶段23测试：Agent多实例管理

测试AgentPool、AgentInstance、AgentMessage、LoadBalancer。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-27
"""

import pytest
from unittest.mock import Mock, MagicMock

from src.agent.agent_pool import (
    PoolAgentStatus,
    AgentInstance,
    AgentMessage,
    LoadBalancer,
    AgentPool,
)


class TestPoolAgentStatus:
    """PoolAgentStatus枚举测试"""

    def test_enum_values(self):
        assert PoolAgentStatus.IDLE.value == "idle"
        assert PoolAgentStatus.BUSY.value == "busy"
        assert PoolAgentStatus.PAUSED.value == "paused"
        assert PoolAgentStatus.ERROR.value == "error"
        assert PoolAgentStatus.STOPPED.value == "stopped"


class TestAgentInstance:
    """AgentInstance测试"""

    def test_creation(self):
        mock_agent = Mock()
        instance = AgentInstance(
            instance_id="test_001",
            agent=mock_agent,
            status=PoolAgentStatus.IDLE,
        )
        assert instance.instance_id == "test_001"
        assert instance.status == PoolAgentStatus.IDLE

    def test_mark_busy(self):
        mock_agent = Mock()
        instance = AgentInstance(instance_id="test_001", agent=mock_agent)
        instance.mark_busy()

        assert instance.status == PoolAgentStatus.BUSY
        assert instance.total_requests == 1

    def test_mark_idle(self):
        mock_agent = Mock()
        instance = AgentInstance(instance_id="test_001", agent=mock_agent)
        instance.mark_busy()
        instance.mark_idle()

        assert instance.status == PoolAgentStatus.IDLE

    def test_mark_error(self):
        mock_agent = Mock()
        instance = AgentInstance(instance_id="test_001", agent=mock_agent)
        instance.mark_error()

        assert instance.status == PoolAgentStatus.ERROR
        assert instance.total_errors == 1

    def test_serialization(self):
        mock_agent = Mock()
        instance = AgentInstance(instance_id="test_001", agent=mock_agent)
        data = instance.to_dict()

        assert data["instance_id"] == "test_001"
        assert data["status"] == "idle"


class TestAgentMessage:
    """AgentMessage测试"""

    def test_creation(self):
        msg = AgentMessage(
            message_id="msg_001",
            sender_id="agent_1",
            receiver_id="agent_2",
            message_type="test",
            payload={"data": "hello"},
        )
        assert msg.message_id == "msg_001"
        assert msg.sender_id == "agent_1"
        assert msg.payload["data"] == "hello"

    def test_broadcast_message(self):
        msg = AgentMessage(
            message_id="msg_002",
            sender_id="agent_1",
            receiver_id=None,  # 广播
            message_type="broadcast",
        )
        assert msg.receiver_id is None

    def test_serialization(self):
        msg = AgentMessage(
            message_id="msg_003",
            sender_id="agent_1",
            message_type="test",
        )
        data = msg.to_dict()
        assert data["message_id"] == "msg_003"
        assert data["message_type"] == "test"


class TestLoadBalancer:
    """LoadBalancer测试"""

    def test_round_robin(self):
        balancer = LoadBalancer(strategy="round_robin")
        instances = [
            AgentInstance(instance_id="a1", agent=Mock(), status=PoolAgentStatus.IDLE),
            AgentInstance(instance_id="a2", agent=Mock(), status=PoolAgentStatus.IDLE),
            AgentInstance(instance_id="a3", agent=Mock(), status=PoolAgentStatus.IDLE),
        ]

        selected_ids = [balancer.select(instances).instance_id for _ in range(5)]
        # 轮询应该循环选择
        assert selected_ids[0] == "a1"
        assert selected_ids[1] == "a2"
        assert selected_ids[2] == "a3"
        assert selected_ids[3] == "a1"

    def test_least_busy(self):
        balancer = LoadBalancer(strategy="least_busy")
        instances = [
            AgentInstance(instance_id="a1", agent=Mock(), status=PoolAgentStatus.IDLE, total_requests=10),
            AgentInstance(instance_id="a2", agent=Mock(), status=PoolAgentStatus.IDLE, total_requests=5),
            AgentInstance(instance_id="a3", agent=Mock(), status=PoolAgentStatus.IDLE, total_requests=15),
        ]

        selected = balancer.select(instances)
        assert selected.instance_id == "a2"  # 最少请求

    def test_no_idle_instances(self):
        balancer = LoadBalancer(strategy="round_robin")
        instances = [
            AgentInstance(instance_id="a1", agent=Mock(), status=PoolAgentStatus.BUSY),
            AgentInstance(instance_id="a2", agent=Mock(), status=PoolAgentStatus.ERROR),
        ]

        selected = balancer.select(instances)
        assert selected is None

    def test_empty_instances(self):
        balancer = LoadBalancer(strategy="round_robin")
        selected = balancer.select([])
        assert selected is None


class TestAgentPool:
    """AgentPool核心功能测试"""

    def test_creation(self):
        pool = AgentPool(pool_id="test_pool", max_size=5, min_size=1)
        assert pool.pool_id == "test_pool"
        assert pool.size == 0
        assert pool.available_count == 0

    def test_create_instance(self):
        pool = AgentPool(pool_id="test_pool", max_size=2)
        mock_factory = Mock(return_value=Mock())

        instance = pool.create_instance(mock_factory)

        assert instance is not None
        assert pool.size == 1
        mock_factory.assert_called_once()

    def test_create_instance_max_limit(self):
        pool = AgentPool(pool_id="test_pool", max_size=1)
        mock_factory = Mock(return_value=Mock())

        pool.create_instance(mock_factory)
        second = pool.create_instance(mock_factory)

        assert second is None  # 达到上限

    def test_get_instance(self):
        pool = AgentPool(pool_id="test_pool", max_size=2)
        mock_factory = Mock(return_value=Mock())

        pool.create_instance(mock_factory)
        instance = pool.get_instance()

        assert instance is not None
        assert instance.status == PoolAgentStatus.BUSY

    def test_release_instance(self):
        pool = AgentPool(pool_id="test_pool", max_size=2)
        mock_factory = Mock(return_value=Mock())

        instance = pool.create_instance(mock_factory)
        pool.get_instance()  # 标记为busy
        result = pool.release_instance(instance.instance_id)

        assert result is True
        assert instance.status == PoolAgentStatus.IDLE

    def test_destroy_instance(self):
        pool = AgentPool(pool_id="test_pool", max_size=2)
        mock_agent = Mock()
        mock_factory = Mock(return_value=mock_agent)

        instance = pool.create_instance(mock_factory)
        result = pool.destroy_instance(instance.instance_id)

        assert result is True
        assert pool.size == 0
        mock_agent.cleanup.assert_called_once()

    def test_get_instance_by_id(self):
        pool = AgentPool(pool_id="test_pool", max_size=2)
        mock_factory = Mock(return_value=Mock())

        instance = pool.create_instance(mock_factory)
        found = pool.get_instance_by_id(instance.instance_id)

        assert found is not None
        assert found.instance_id == instance.instance_id

    def test_initialize_pool(self):
        pool = AgentPool(pool_id="test_pool", max_size=5, min_size=3)
        mock_factory = Mock(return_value=Mock())

        created = pool.initialize_pool(mock_factory)

        assert created == 3
        assert pool.size == 3

    def test_scale_up(self):
        pool = AgentPool(pool_id="test_pool", max_size=5, min_size=1)
        mock_factory = Mock(return_value=Mock())

        pool.initialize_pool(mock_factory, count=1)
        changed = pool.scale_to(3, mock_factory)

        assert changed == 2
        assert pool.size == 3

    def test_scale_down(self):
        pool = AgentPool(pool_id="test_pool", max_size=5, min_size=1)
        mock_factory = Mock(return_value=Mock())

        pool.initialize_pool(mock_factory, count=3)
        # 先释放一个实例使其变为idle
        for instance_id in list(pool._instances.keys())[:1]:
            pool.release_instance(instance_id)

        changed = pool.scale_to(1, mock_factory)

        assert changed < 0  # 负数表示销毁数量

    def test_clear(self):
        pool = AgentPool(pool_id="test_pool", max_size=5)
        mock_factory = Mock(return_value=Mock())

        pool.initialize_pool(mock_factory, count=3)
        pool.clear()

        assert pool.size == 0

    def test_send_message(self):
        pool = AgentPool(pool_id="test_pool")
        mock_handler = Mock()
        pool.subscribe_message("test_type", mock_handler)

        msg_id = pool.send_message(
            sender_id="agent_1",
            receiver_id="agent_2",
            message_type="test_type",
            payload={"data": "hello"},
        )

        assert msg_id != ""
        mock_handler.assert_called_once()

    def test_broadcast(self):
        pool = AgentPool(pool_id="test_pool")
        mock_handler = Mock()
        pool.subscribe_message("broadcast_type", mock_handler)

        msg_id = pool.broadcast(
            sender_id="agent_1",
            message_type="broadcast_type",
            payload={"data": "all"},
        )

        assert msg_id != ""
        mock_handler.assert_called_once()

    def test_get_stats(self):
        pool = AgentPool(pool_id="test_pool", max_size=5)
        mock_factory = Mock(return_value=Mock())

        pool.initialize_pool(mock_factory, count=2)
        stats = pool.get_stats()

        assert stats["pool_id"] == "test_pool"
        assert stats["size"] == 2
        assert stats["max_size"] == 5

    def test_get_summary(self):
        pool = AgentPool(pool_id="test_pool", max_size=5)
        mock_factory = Mock(return_value=Mock())

        pool.initialize_pool(mock_factory, count=2)
        summary = pool.get_summary()

        assert summary["pool_id"] == "test_pool"
        assert "utilization" in summary

    def test_get_instances_info(self):
        pool = AgentPool(pool_id="test_pool", max_size=5)
        mock_factory = Mock(return_value=Mock())

        pool.initialize_pool(mock_factory, count=2)
        infos = pool.get_instances_info()

        assert len(infos) == 2
        assert infos[0]["instance_id"] is not None


class TestAgentPoolIntegration:
    """Agent池集成测试"""

    def test_full_lifecycle(self):
        pool = AgentPool(pool_id="integration_test", max_size=3)
        mock_factory = Mock(return_value=Mock())

        # 初始化
        pool.initialize_pool(mock_factory, count=2)
        assert pool.size == 2

        # 获取实例
        instance = pool.get_instance()
        assert instance.status == PoolAgentStatus.BUSY

        # 释放实例
        pool.release_instance(instance.instance_id)
        assert instance.status == PoolAgentStatus.IDLE

        # 销毁
        pool.destroy_instance(instance.instance_id)
        assert pool.size == 1

        # 清空
        pool.clear()
        assert pool.size == 0


class TestExportAgentPool:
    """导出测试"""

    def test_agent_pool_exported(self):
        from src.agent import AgentPool
        assert AgentPool is not None

    def test_agent_instance_exported(self):
        from src.agent import AgentInstance
        mock_agent = Mock()
        instance = AgentInstance(instance_id="test", agent=mock_agent)
        assert isinstance(instance, AgentInstance)

    def test_agent_message_exported(self):
        from src.agent import AgentMessage
        msg = AgentMessage(message_id="test", sender_id="agent_1")
        assert isinstance(msg, AgentMessage)

    def test_pool_agent_status_exported(self):
        from src.agent import PoolAgentStatus
        assert PoolAgentStatus.IDLE.value == "idle"

    def test_load_balancer_exported(self):
        from src.agent import LoadBalancer
        balancer = LoadBalancer(strategy="round_robin")
        assert balancer is not None


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
