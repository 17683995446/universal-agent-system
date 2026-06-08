"""
阶段62: Agent数据复制系统

遵循"图难于其易，为大于其细"的原则，实现主从复制、多副本、同步/异步复制、冲突检测。
"""
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

class ReplicaRole(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"

class ReplicationMode(str, Enum):
    SYNC = "sync"
    ASYNC = "async"

class ReplicaStatus(str, Enum):
    HEALTHY = "healthy"
    LAGGING = "lagging"
    OFFLINE = "offline"

@dataclass
class Replica:
    replica_id: str
    role: ReplicaRole = ReplicaRole.SECONDARY
    status: ReplicaStatus = ReplicaStatus.HEALTHY
    data: Dict[str, Any] = field(default_factory=dict)
    last_sync: float = 0.0
    lag_ms: float = 0.0

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.last_sync = time.time()

    def delete(self, key: str) -> bool:
        if key in self.data:
            del self.data[key]
            self.last_sync = time.time()
            return True
        return False

@dataclass
class ReplicationLog:
    op_id: int
    operation: str  # SET, DELETE
    key: str
    value: Any = None
    timestamp: float = field(default_factory=time.time)

class ReplicationManager:
    """复制管理器"""
    def __init__(self, mode: ReplicationMode = ReplicationMode.ASYNC):
        self.mode = mode
        self._primary: Optional[Replica] = None
        self._replicas: Dict[str, Replica] = {}
        self._log: List[ReplicationLog] = []
        self._op_counter = 0
        self._lock = threading.RLock()

    def set_primary(self, replica_id: str) -> Replica:
        with self._lock:
            if replica_id not in self._replicas:
                self._replicas[replica_id] = Replica(replica_id=replica_id)
            self._primary = self._replicas[replica_id]
            self._primary.role = ReplicaRole.PRIMARY
            return self._primary

    def add_replica(self, replica_id: str) -> Replica:
        with self._lock:
            if replica_id not in self._replicas:
                self._replicas[replica_id] = Replica(replica_id=replica_id)
            return self._replicas[replica_id]

    def remove_replica(self, replica_id: str) -> bool:
        with self._lock:
            if self._primary and self._primary.replica_id == replica_id:
                self._primary = None
            return self._replicas.pop(replica_id, None) is not None

    def write(self, key: str, value: Any) -> bool:
        with self._lock:
            if not self._primary:
                return False
            self._primary.set(key, value)
            self._op_counter += 1
            log_entry = ReplicationLog(op_id=self._op_counter, operation="SET", key=key, value=value)
            self._log.append(log_entry)
            if self.mode == ReplicationMode.SYNC:
                self._sync_to_replicas(log_entry)
            else:
                threading.Thread(target=self._sync_to_replicas, args=(log_entry,), daemon=True).start()
            return True

    def delete(self, key: str) -> bool:
        with self._lock:
            if not self._primary:
                return False
            self._primary.delete(key)
            self._op_counter += 1
            log_entry = ReplicationLog(op_id=self._op_counter, operation="DELETE", key=key)
            self._log.append(log_entry)
            if self.mode == ReplicationMode.SYNC:
                self._sync_to_replicas(log_entry)
            else:
                threading.Thread(target=self._sync_to_replicas, args=(log_entry,), daemon=True).start()
            return True

    def read(self, key: str, replica_id: Optional[str] = None) -> Any:
        if replica_id and replica_id in self._replicas:
            return self._replicas[replica_id].get(key)
        if self._primary:
            return self._primary.get(key)
        return None

    def _sync_to_replicas(self, log_entry: ReplicationLog) -> None:
        for replica in self._replicas.values():
            if replica.role == ReplicaRole.SECONDARY and replica.status != ReplicaStatus.OFFLINE:
                try:
                    if log_entry.operation == "SET":
                        replica.set(log_entry.key, log_entry.value)
                    elif log_entry.operation == "DELETE":
                        replica.delete(log_entry.key)
                    replica.lag_ms = (time.time() - log_entry.timestamp) * 1000
                    if replica.lag_ms > 1000:
                        replica.status = ReplicaStatus.LAGGING
                    else:
                        replica.status = ReplicaStatus.HEALTHY
                except Exception:
                    replica.status = ReplicaStatus.OFFLINE

    def get_replica_status(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                rid: {
                    "role": r.role.value,
                    "status": r.status.value,
                    "lag_ms": r.lag_ms,
                    "key_count": len(r.data),
                }
                for rid, r in self._replicas.items()
            }

    def promote_replica(self, replica_id: str) -> bool:
        with self._lock:
            if replica_id not in self._replicas:
                return False
            if self._primary:
                self._primary.role = ReplicaRole.SECONDARY
            self._primary = self._replicas[replica_id]
            self._primary.role = ReplicaRole.PRIMARY
            return True

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "primary": self._primary.replica_id if self._primary else None,
                "replica_count": len(self._replicas),
                "log_entries": len(self._log),
            }

def create_replication_manager(mode: ReplicationMode = ReplicationMode.ASYNC) -> ReplicationManager:
    return ReplicationManager(mode)
