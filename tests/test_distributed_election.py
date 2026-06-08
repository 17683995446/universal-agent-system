"""阶段56: Agent分布式选举 - 单元测试"""
import time
import pytest
from src.agent.distributed_election import (
    ElectionRole, ElectionState, ElectionTerm, ElectionNode,
    DistributedElection, ElectionManager, create_election_manager,
)

class TestDistributedElection:
    def test_add_node(self):
        e = DistributedElection("n1")
        e.add_node("n2")
        assert "n2" in e._nodes

    def test_start_stop(self):
        e = DistributedElection("n1", heartbeat_interval=0.1)
        e.start()
        time.sleep(0.05)
        assert e._state == ElectionState.RUNNING
        e.stop()
        assert e._state == ElectionState.STOPPED

    def test_election_becomes_leader(self):
        e = DistributedElection("n1")
        e.add_node("n2")
        e.add_node("n3")
        e._start_election()
        assert e.is_leader
        assert e.current_leader == "n1"

    def test_vote(self):
        e = DistributedElection("n1")
        assert e.vote("n2", 1)
        assert e._term.voted_for == "n2"

    def test_vote_lower_term(self):
        e = DistributedElection("n1")
        e._term.term_id = 5
        assert not e.vote("n2", 1)

    def test_receive_heartbeat(self):
        e = DistributedElection("n1")
        e.receive_heartbeat("n2", 1)
        assert e.current_leader == "n2"
        assert e.role == ElectionRole.FOLLOWER

    def test_callback(self):
        changes = []
        e = DistributedElection("n1")
        e.register_callback(lambda nid, role: changes.append(role.value))
        e.add_node("n2")
        e._start_election()
        assert "candidate" in changes

    def test_stats(self):
        e = DistributedElection("n1")
        e.add_node("n2")
        stats = e.get_stats()
        assert stats["node_id"] == "n1"
        assert stats["nodes"] == 1

class TestElectionManager:
    def test_create_election(self):
        m = ElectionManager()
        e = m.create_election("e1", "n1")
        assert isinstance(e, DistributedElection)

    def test_remove_election(self):
        m = ElectionManager()
        m.create_election("e1", "n1")
        assert m.remove_election("e1")

    def test_create_election_manager(self):
        m = create_election_manager()
        assert isinstance(m, ElectionManager)
