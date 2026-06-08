"""阶段59: Agent分布式事务 - 单元测试"""
import pytest
from src.agent.distributed_transaction import (
    TransactionStatus, ParticipantStatus, TransactionParticipant, Transaction,
    DistributedTransactionManager, create_transaction_manager,
)

class TestDistributedTransactionManager:
    def test_begin(self):
        m = DistributedTransactionManager()
        tx = m.begin()
        assert len(tx.tx_id) == 12
        assert tx.status == TransactionStatus.ACTIVE

    def test_commit_success(self):
        m = DistributedTransactionManager()
        tx = m.begin()
        m.add_participant(tx.tx_id, TransactionParticipant(
            "p1", lambda: True, lambda: True, lambda: True,
        ))
        assert m.commit(tx.tx_id)
        assert m.get_transaction(tx.tx_id).status == TransactionStatus.COMMITTED

    def test_commit_prepare_fail(self):
        m = DistributedTransactionManager()
        tx = m.begin()
        m.add_participant(tx.tx_id, TransactionParticipant(
            "p1", lambda: False, lambda: True, lambda: True,
        ))
        assert not m.commit(tx.tx_id)
        assert m.get_transaction(tx.tx_id).status == TransactionStatus.ABORTED

    def test_abort(self):
        m = DistributedTransactionManager()
        tx = m.begin()
        m.add_participant(tx.tx_id, TransactionParticipant(
            "p1", lambda: True, lambda: True, lambda: True,
        ))
        assert m.abort(tx.tx_id)
        assert m.get_transaction(tx.tx_id).status == TransactionStatus.ABORTED

    def test_multiple_participants(self):
        m = DistributedTransactionManager()
        tx = m.begin()
        m.add_participant(tx.tx_id, TransactionParticipant("p1", lambda: True, lambda: True, lambda: True))
        m.add_participant(tx.tx_id, TransactionParticipant("p2", lambda: True, lambda: True, lambda: True))
        assert m.commit(tx.tx_id)
        assert len(m.get_transaction(tx.tx_id).participants) == 2

    def test_stats(self):
        m = DistributedTransactionManager()
        m.begin()
        tx2 = m.begin()
        m.add_participant(tx2.tx_id, TransactionParticipant("p1", lambda: True, lambda: True, lambda: True))
        m.commit(tx2.tx_id)
        stats = m.get_stats()
        assert stats["total"] == 2
        assert stats["committed"] == 1

    def test_create_transaction_manager(self):
        m = create_transaction_manager()
        assert isinstance(m, DistributedTransactionManager)
