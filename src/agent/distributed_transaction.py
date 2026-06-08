"""
阶段59: Agent分布式事务

遵循"图难于其易，为大于其细"的原则，实现事务管理器、参与者、两阶段提交、补偿机制。
"""
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

class TransactionStatus(str, Enum):
    ACTIVE = "active"
    PREPARING = "preparing"
    PREPARED = "prepared"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ABORTING = "aborting"
    ABORTED = "aborted"

class ParticipantStatus(str, Enum):
    ACTIVE = "active"
    PREPARED = "prepared"
    COMMITTED = "committed"
    ABORTED = "aborted"
    FAILED = "failed"

@dataclass
class TransactionParticipant:
    participant_id: str
    prepare: Callable[[], bool]
    commit: Callable[[], bool]
    abort: Callable[[], bool]
    status: ParticipantStatus = ParticipantStatus.ACTIVE
    error: str = ""

@dataclass
class Transaction:
    tx_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    status: TransactionStatus = TransactionStatus.ACTIVE
    participants: List[TransactionParticipant] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tx_id": self.tx_id,
            "status": self.status.value,
            "participants": len(self.participants),
            "created_at": self.created_at,
        }

class DistributedTransactionManager:
    """分布式事务管理器（本地模拟两阶段提交）"""
    def __init__(self):
        self._transactions: Dict[str, Transaction] = {}
        self._lock = threading.RLock()

    def begin(self, tx_id: Optional[str] = None) -> Transaction:
        with self._lock:
            tx = Transaction(tx_id=tx_id or str(uuid.uuid4())[:12])
            self._transactions[tx.tx_id] = tx
            return tx

    def add_participant(self, tx_id: str, participant: TransactionParticipant) -> bool:
        with self._lock:
            tx = self._transactions.get(tx_id)
            if not tx or tx.status != TransactionStatus.ACTIVE:
                return False
            tx.participants.append(participant)
            return True

    def commit(self, tx_id: str) -> bool:
        with self._lock:
            tx = self._transactions.get(tx_id)
            if not tx:
                return False
            if tx.status not in (TransactionStatus.ACTIVE,):
                return False

            # Phase 1: Prepare
            tx.status = TransactionStatus.PREPARING
            all_prepared = True
            for p in tx.participants:
                try:
                    if p.prepare():
                        p.status = ParticipantStatus.PREPARED
                    else:
                        p.status = ParticipantStatus.FAILED
                        all_prepared = False
                        p.error = "Prepare failed"
                except Exception as e:
                    p.status = ParticipantStatus.FAILED
                    p.error = str(e)
                    all_prepared = False

            if not all_prepared:
                # Abort all
                tx.status = TransactionStatus.ABORTING
                for p in tx.participants:
                    if p.status == ParticipantStatus.PREPARED:
                        try:
                            p.abort()
                            p.status = ParticipantStatus.ABORTED
                        except Exception as e:
                            p.error = str(e)
                tx.status = TransactionStatus.ABORTED
                tx.completed_at = time.time()
                return False

            # Phase 2: Commit
            tx.status = TransactionStatus.COMMITTING
            all_committed = True
            for p in tx.participants:
                try:
                    if p.commit():
                        p.status = ParticipantStatus.COMMITTED
                    else:
                        p.status = ParticipantStatus.FAILED
                        all_committed = False
                except Exception as e:
                    p.status = ParticipantStatus.FAILED
                    p.error = str(e)
                    all_committed = False

            tx.status = TransactionStatus.COMMITTED if all_committed else TransactionStatus.ABORTED
            tx.completed_at = time.time()
            return all_committed

    def abort(self, tx_id: str) -> bool:
        with self._lock:
            tx = self._transactions.get(tx_id)
            if not tx:
                return False
            tx.status = TransactionStatus.ABORTING
            for p in tx.participants:
                try:
                    p.abort()
                    p.status = ParticipantStatus.ABORTED
                except Exception as e:
                    p.error = str(e)
            tx.status = TransactionStatus.ABORTED
            tx.completed_at = time.time()
            return True

    def get_transaction(self, tx_id: str) -> Optional[Transaction]:
        return self._transactions.get(tx_id)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._transactions)
            committed = sum(1 for t in self._transactions.values() if t.status == TransactionStatus.COMMITTED)
            aborted = sum(1 for t in self._transactions.values() if t.status == TransactionStatus.ABORTED)
            return {
                "total": total,
                "committed": committed,
                "aborted": aborted,
                "active": total - committed - aborted,
            }

def create_transaction_manager() -> DistributedTransactionManager:
    return DistributedTransactionManager()
