"""阶段62: Agent数据复制系统 - 单元测试"""
import time
import pytest
from src.agent.data_replication import (
    ReplicaRole, ReplicationMode, ReplicaStatus, Replica, ReplicationLog,
    ReplicationManager, create_replication_manager,
)

class TestReplicationManager:
    def test_set_primary(self):
        m = ReplicationManager()
        p = m.set_primary("r1")
        assert p.role == ReplicaRole.PRIMARY

    def test_sync_write(self):
        m = ReplicationManager(mode=ReplicationMode.SYNC)
        m.add_replica("r1")
        m.add_replica("r2")
        m.set_primary("r1")
        assert m.write("key", "value")
        assert m.read("key", "r1") == "value"
        assert m.read("key", "r2") == "value"

    def test_async_write(self):
        m = ReplicationManager(mode=ReplicationMode.ASYNC)
        m.add_replica("r1")
        m.add_replica("r2")
        m.set_primary("r1")
        assert m.write("key", "value")
        time.sleep(0.1)
        assert m.read("key", "r2") == "value"

    def test_delete(self):
        m = ReplicationManager(mode=ReplicationMode.SYNC)
        m.add_replica("r1")
        m.add_replica("r2")
        m.set_primary("r1")
        m.write("key", "value")
        assert m.delete("key")
        assert m.read("key") is None

    def test_promote_replica(self):
        m = ReplicationManager()
        m.add_replica("r1")
        m.add_replica("r2")
        m.set_primary("r1")
        assert m.promote_replica("r2")
        assert m._primary.replica_id == "r2"
        assert m._replicas["r1"].role == ReplicaRole.SECONDARY

    def test_replica_status(self):
        m = ReplicationManager(mode=ReplicationMode.SYNC)
        m.add_replica("r1")
        m.add_replica("r2")
        m.set_primary("r1")
        m.write("key", "value")
        status = m.get_replica_status()
        assert status["r1"]["role"] == "primary"
        assert status["r2"]["role"] == "secondary"

    def test_stats(self):
        m = ReplicationManager()
        m.add_replica("r1")
        m.set_primary("r1")
        m.write("a", 1)
        m.write("b", 2)
        stats = m.get_stats()
        assert stats["primary"] == "r1"
        assert stats["log_entries"] == 2

    def test_create_replication_manager(self):
        m = create_replication_manager()
        assert isinstance(m, ReplicationManager)
