# =============================================================================
# 集群协调器测试 - 阶段24测试
# =============================================================================

import time
import pytest
from unittest.mock import Mock, MagicMock

from src.agent.cluster_coordinator import (
    ClusterCoordinator,
    ClusterNode,
    DistributedTask,
    TaskShard,
    NodeStatus,
    ShardStrategy,
    ResultAggregator,
)
from src.agent.agent_pool import AgentPool, PoolAgentStatus


class TestNodeStatus:
    """节点状态枚举测试"""

    def test_enum_values(self):
        assert NodeStatus.HEALTHY.value == "healthy"
        assert NodeStatus.BUSY.value == "busy"
        assert NodeStatus.DEGRADED.value == "degraded"
        assert NodeStatus.OFFLINE.value == "offline"
        assert NodeStatus.JOINING.value == "joining"
        assert NodeStatus.LEAVING.value == "leaving"


class TestShardStrategy:
    """分片策略枚举测试"""

    def test_enum_values(self):
        assert ShardStrategy.HASH.value == "hash"
        assert ShardStrategy.RANGE.value == "range"
        assert ShardStrategy.ROUND_ROBIN.value == "round_robin"
        assert ShardStrategy.ADAPTIVE.value == "adaptive"


class TestClusterNode:
    """集群节点测试"""

    def test_creation(self):
        pool = Mock(spec=AgentPool)
        node = ClusterNode(
            node_id="node_1",
            pool=pool,
            status=NodeStatus.HEALTHY,
            capacity=10,
        )

        assert node.node_id == "node_1"
        assert node.status == NodeStatus.HEALTHY
        assert node.capacity == 10
        assert node.load_factor == 0.0

    def test_update_heartbeat(self):
        pool = Mock(spec=AgentPool)
        node = ClusterNode(node_id="node_1", pool=pool)

        old_time = node.last_heartbeat
        time.sleep(0.01)
        node.update_heartbeat()

        assert node.last_heartbeat > old_time

    def test_is_healthy(self):
        pool = Mock(spec=AgentPool)

        healthy_node = ClusterNode(node_id="n1", pool=pool, status=NodeStatus.HEALTHY)
        busy_node = ClusterNode(node_id="n2", pool=pool, status=NodeStatus.BUSY)
        offline_node = ClusterNode(node_id="n3", pool=pool, status=NodeStatus.OFFLINE)

        assert healthy_node.is_healthy() is True
        assert busy_node.is_healthy() is True
        assert offline_node.is_healthy() is False

    def test_calculate_load(self):
        pool = Mock(spec=AgentPool)
        pool.get_stats.return_value = {"size": 10, "available": 3}

        node = ClusterNode(node_id="node_1", pool=pool)
        load = node.calculate_load()

        assert load == 0.7  # (10-3)/10
        assert node.load_factor == 0.7

    def test_calculate_load_empty_pool(self):
        pool = Mock(spec=AgentPool)
        pool.get_stats.return_value = {"size": 0, "available": 0}

        node = ClusterNode(node_id="node_1", pool=pool)
        load = node.calculate_load()

        assert load == 0.0

    def test_to_dict(self):
        pool = Mock(spec=AgentPool)
        pool.get_summary.return_value = {"size": 5}

        node = ClusterNode(
            node_id="node_1",
            pool=pool,
            status=NodeStatus.HEALTHY,
            capacity=10,
            metadata={"region": "us-east"},
        )

        result = node.to_dict()

        assert result["node_id"] == "node_1"
        assert result["status"] == "healthy"
        assert result["capacity"] == 10
        assert result["metadata"]["region"] == "us-east"
        assert result["pool_stats"]["size"] == 5


class TestTaskShard:
    """任务分片测试"""

    def test_creation(self):
        shard = TaskShard(
            shard_id="shard_1",
            task_id="task_1",
            node_id="node_1",
            payload={"data": "test"},
            index=0,
            total=3,
        )

        assert shard.shard_id == "shard_1"
        assert shard.status == "pending"
        assert shard.index == 0
        assert shard.total == 3

    def test_mark_running(self):
        shard = TaskShard(
            shard_id="shard_1",
            task_id="task_1",
            node_id=None,
            payload={},
            index=0,
            total=1,
        )

        shard.mark_running()
        assert shard.status == "running"

    def test_mark_completed(self):
        shard = TaskShard(
            shard_id="shard_1",
            task_id="task_1",
            node_id=None,
            payload={},
            index=0,
            total=1,
        )

        shard.mark_completed(result=["item1", "item2"])

        assert shard.status == "completed"
        assert shard.result == ["item1", "item2"]
        assert shard.completed_at is not None

    def test_mark_failed(self):
        shard = TaskShard(
            shard_id="shard_1",
            task_id="task_1",
            node_id=None,
            payload={},
            index=0,
            total=1,
        )

        shard.mark_failed("Connection timeout")

        assert shard.status == "failed"
        assert shard.error == "Connection timeout"
        assert shard.completed_at is not None

    def test_to_dict(self):
        shard = TaskShard(
            shard_id="shard_1",
            task_id="task_1",
            node_id="node_1",
            payload={"data": "test"},
            index=0,
            total=3,
            status="completed",
            result=[1, 2, 3],
        )

        result = shard.to_dict()

        assert result["shard_id"] == "shard_1"
        assert result["status"] == "completed"
        assert result["result"] == [1, 2, 3]


class TestDistributedTask:
    """分布式任务测试"""

    def test_creation_with_shards(self):
        task = DistributedTask(
            task_id="task_1",
            payload=[1, 2, 3, 4, 5, 6],
            shard_count=3,
            strategy=ShardStrategy.ROUND_ROBIN,
        )

        assert task.task_id == "task_1"
        assert len(task.shards) == 3
        assert task.strategy == ShardStrategy.ROUND_ROBIN

    def test_list_payload_sharding(self):
        task = DistributedTask(
            task_id="task_1",
            payload=[1, 2, 3, 4, 5, 6],
            shard_count=3,
        )

        # 验证分片数据
        assert len(task.shards[0].payload) == 2  # [1, 2]
        assert len(task.shards[1].payload) == 2  # [3, 4]
        assert len(task.shards[2].payload) == 2  # [5, 6]

    def test_dict_payload_sharding(self):
        task = DistributedTask(
            task_id="task_1",
            payload={"items": [1, 2, 3, 4], "meta": "data"},
            shard_count=2,
        )

        # 验证分片包含items
        assert "items" in task.shards[0].payload
        assert task.shards[0].payload["shard_index"] == 0
        assert task.shards[1].payload["shard_index"] == 1

    def test_other_payload_sharding(self):
        task = DistributedTask(
            task_id="task_1",
            payload="simple_string",
            shard_count=2,
        )

        # 验证包装为字典
        assert task.shards[0].payload["data"] == "simple_string"
        assert task.shards[0].payload["shard_index"] == 0

    def test_assign_shard(self):
        task = DistributedTask(
            task_id="task_1",
            payload=[1, 2, 3],
            shard_count=3,
        )

        assert task.assign_shard(0, "node_1") is True
        assert task.shards[0].node_id == "node_1"

        assert task.assign_shard(10, "node_1") is False  # 无效索引

    def test_update_status_all_completed(self):
        task = DistributedTask(
            task_id="task_1",
            payload=[1, 2],
            shard_count=2,
        )

        task.shards[0].mark_completed([1])
        task.shards[1].mark_completed([2])
        task.update_status()

        assert task.status == "completed"
        assert task.completed_at is not None

    def test_update_status_partial_failed(self):
        task = DistributedTask(
            task_id="task_1",
            payload=[1, 2],
            shard_count=2,
        )

        task.shards[0].mark_completed([1])
        task.shards[1].mark_failed("error")
        task.update_status()

        assert task.status == "failed"

    def test_update_status_running(self):
        task = DistributedTask(
            task_id="task_1",
            payload=[1, 2],
            shard_count=2,
        )

        task.shards[0].mark_running()
        task.update_status()

        assert task.status == "running"

    def test_get_progress(self):
        task = DistributedTask(
            task_id="task_1",
            payload=[1, 2, 3, 4],
            shard_count=4,
        )

        task.shards[0].mark_completed([1])
        task.shards[1].mark_failed("err")
        task.shards[2].mark_running()
        # shard[3] pending

        progress = task.get_progress()

        assert progress["total"] == 4
        assert progress["completed"] == 1
        assert progress["failed"] == 1
        assert progress["running"] == 1
        assert progress["pending"] == 1
        assert progress["progress_percent"] == 50.0

    def test_to_dict(self):
        task = DistributedTask(
            task_id="task_1",
            payload=[1, 2],
            shard_count=2,
            strategy=ShardStrategy.HASH,
        )

        result = task.to_dict()

        assert result["task_id"] == "task_1"
        assert result["shard_count"] == 2
        assert result["strategy"] == "hash"
        assert len(result["shards"]) == 2


class TestResultAggregator:
    """结果聚合器测试"""

    def test_aggregate_list(self):
        shards = [
            TaskShard("s1", "t1", None, [1, 2], 0, 3),
            TaskShard("s2", "t1", None, [3, 4], 1, 3),
            TaskShard("s3", "t1", None, [5, 6], 2, 3),
        ]

        for s in shards:
            s.mark_completed(s.payload)

        result = ResultAggregator.aggregate_list(shards)
        assert result == [1, 2, 3, 4, 5, 6]

    def test_aggregate_dict(self):
        shards = [
            TaskShard("s1", "t1", None, {"a": 1}, 0, 2),
            TaskShard("s2", "t1", None, {"b": 2}, 1, 2),
        ]

        for s in shards:
            s.mark_completed(s.payload)

        result = ResultAggregator.aggregate_dict(shards)
        assert result == {"a": 1, "b": 2}

    def test_aggregate_count(self):
        shards = [
            TaskShard("s1", "t1", None, {}, 0, 3),
            TaskShard("s2", "t1", None, {}, 1, 3),
            TaskShard("s3", "t1", None, {}, 2, 3),
        ]

        shards[0].mark_completed(10)
        shards[1].mark_completed(20)
        shards[2].mark_completed(30)

        result = ResultAggregator.aggregate_count(shards)
        assert result == 60

    def test_aggregate_first_success(self):
        shards = [
            TaskShard("s1", "t1", None, {}, 0, 3),
            TaskShard("s2", "t1", None, {}, 1, 3),
            TaskShard("s3", "t1", None, {}, 2, 3),
        ]

        shards[0].mark_failed("error")
        shards[1].mark_completed("success_result")
        shards[2].mark_completed("another_result")

        result = ResultAggregator.aggregate_first_success(shards)
        assert result == "success_result"


class TestClusterCoordinator:
    """集群协调器测试"""

    def test_creation(self):
        coordinator = ClusterCoordinator(cluster_id="test_cluster")

        assert coordinator.cluster_id == "test_cluster"
        assert coordinator.node_count == 0

    def test_register_node(self):
        coordinator = ClusterCoordinator()
        pool = Mock(spec=AgentPool)

        node = coordinator.register_node(
            node_id="node_1",
            pool=pool,
            capacity=10,
            metadata={"region": "us-east"},
        )

        assert node.node_id == "node_1"
        assert coordinator.node_count == 1
        assert node.capacity == 10

    def test_unregister_node(self):
        coordinator = ClusterCoordinator()
        pool = Mock(spec=AgentPool)

        coordinator.register_node("node_1", pool)
        result = coordinator.unregister_node("node_1")

        assert result is True
        assert coordinator.node_count == 0

    def test_unregister_nonexistent_node(self):
        coordinator = ClusterCoordinator()
        result = coordinator.unregister_node("nonexistent")

        assert result is False

    def test_update_node_status(self):
        coordinator = ClusterCoordinator()
        pool = Mock(spec=AgentPool)

        coordinator.register_node("node_1", pool)
        result = coordinator.update_node_status("node_1", NodeStatus.BUSY)

        assert result is True

        node_info = coordinator.get_node_info("node_1")
        assert node_info["status"] == "busy"

    def test_heartbeat(self):
        coordinator = ClusterCoordinator()
        pool = Mock(spec=AgentPool)

        coordinator.register_node("node_1", pool)
        result = coordinator.heartbeat("node_1")

        assert result is True

        node_info = coordinator.get_node_info("node_1")
        assert node_info["last_heartbeat"] > 0

    def test_heartbeat_nonexistent(self):
        coordinator = ClusterCoordinator()
        result = coordinator.heartbeat("nonexistent")

        assert result is False

    def test_submit_task_no_nodes(self):
        coordinator = ClusterCoordinator()

        task_id = coordinator.submit_task(
            payload=[1, 2, 3],
            shard_count=2,
        )

        assert task_id.startswith("task_")
        # 没有节点，任务创建但不会分配

    def test_submit_task_with_nodes(self):
        coordinator = ClusterCoordinator()
        pool = Mock(spec=AgentPool)

        coordinator.register_node("node_1", pool, capacity=10)

        task_id = coordinator.submit_task(
            payload=[1, 2, 3, 4],
            shard_count=2,
            strategy=ShardStrategy.ROUND_ROBIN,
        )

        task = coordinator.get_task(task_id)
        assert task is not None
        assert len(task.shards) == 2

        # 验证广播被调用
        assert pool.broadcast.called

    def test_get_task_result_not_completed(self):
        coordinator = ClusterCoordinator()
        pool = Mock(spec=AgentPool)

        coordinator.register_node("node_1", pool)
        task_id = coordinator.submit_task(payload=[1, 2], shard_count=1)

        result = coordinator.get_task_result(task_id)
        assert result is None  # 任务未完成

    def test_get_cluster_stats(self):
        coordinator = ClusterCoordinator(cluster_id="test_cluster")
        pool = Mock(spec=AgentPool)
        pool.get_summary.return_value = {"size": 5}

        coordinator.register_node("node_1", pool, capacity=10)
        coordinator.register_node("node_2", pool, capacity=20)

        stats = coordinator.get_cluster_stats()

        assert stats["cluster_id"] == "test_cluster"
        assert stats["node_count"] == 2
        assert stats["healthy_nodes"] == 2
        assert stats["total_capacity"] == 30
        assert len(stats["nodes"]) == 2

    def test_get_node_info(self):
        coordinator = ClusterCoordinator()
        pool = Mock(spec=AgentPool)
        pool.get_summary.return_value = {"size": 5}

        coordinator.register_node("node_1", pool)
        info = coordinator.get_node_info("node_1")

        assert info is not None
        assert info["node_id"] == "node_1"

    def test_get_node_info_nonexistent(self):
        coordinator = ClusterCoordinator()
        info = coordinator.get_node_info("nonexistent")

        assert info is None

    def test_start_stop(self):
        coordinator = ClusterCoordinator()

        coordinator.start()
        assert coordinator._running is True

        coordinator.stop()
        assert coordinator._running is False

    def test_healthy_nodes_property(self):
        coordinator = ClusterCoordinator()
        pool = Mock(spec=AgentPool)

        node1 = coordinator.register_node("node_1", pool)
        node1.status = NodeStatus.HEALTHY
        node2 = coordinator.register_node("node_2", pool)
        node2.status = NodeStatus.OFFLINE

        healthy = coordinator.healthy_nodes
        assert len(healthy) == 1
        assert healthy[0].node_id == "node_1"

    def test_register_task_handler(self):
        coordinator = ClusterCoordinator()
        handler = Mock()

        coordinator.register_task_handler("my_task", handler)
        assert coordinator._task_handlers["my_task"] == handler


class TestClusterCoordinatorIntegration:
    """集群协调器集成测试"""

    def test_full_lifecycle(self):
        coordinator = ClusterCoordinator()
        pool = Mock(spec=AgentPool)

        # 注册节点
        coordinator.register_node("node_1", pool, capacity=10)
        assert coordinator.node_count == 1

        # 提交任务
        task_id = coordinator.submit_task(
            payload=[1, 2, 3, 4],
            shard_count=2,
        )

        task = coordinator.get_task(task_id)
        assert task is not None

        # 获取进度
        progress = task.get_progress()
        assert progress["total"] == 2

        # 获取集群统计
        stats = coordinator.get_cluster_stats()
        assert stats["node_count"] == 1

        # 注销节点
        coordinator.unregister_node("node_1")
        assert coordinator.node_count == 0

    def test_node_failure_migration(self):
        coordinator = ClusterCoordinator(node_timeout=0.1)
        pool = Mock(spec=AgentPool)

        # 注册节点
        coordinator.register_node("node_1", pool)

        # 提交任务
        task_id = coordinator.submit_task(payload=[1, 2], shard_count=1)
        task = coordinator.get_task(task_id)

        # 模拟分片正在运行
        task.shards[0].status = "running"
        task.shards[0].node_id = "node_1"

        # 模拟节点超时（手动触发迁移）
        coordinator._migrate_node_tasks("node_1")

        # 验证分片被重置
        assert task.shards[0].status == "pending"
        assert task.shards[0].node_id is None


class TestExportClusterCoordinator:
    """导出测试"""

    def test_cluster_coordinator_exported(self):
        from src.agent import ClusterCoordinator
        assert ClusterCoordinator is not None

    def test_cluster_node_exported(self):
        from src.agent import ClusterNode
        assert ClusterNode is not None

    def test_distributed_task_exported(self):
        from src.agent import DistributedTask
        assert DistributedTask is not None

    def test_task_shard_exported(self):
        from src.agent import TaskShard
        assert TaskShard is not None

    def test_node_status_exported(self):
        from src.agent import NodeStatus
        assert NodeStatus.HEALTHY is not None

    def test_shard_strategy_exported(self):
        from src.agent import ShardStrategy
        assert ShardStrategy.HASH is not None

    def test_result_aggregator_exported(self):
        from src.agent import ResultAggregator
        assert ResultAggregator is not None


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
