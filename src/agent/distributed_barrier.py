"""
阶段57: Agent分布式屏障

遵循"图难于其易，为大于其细"的原则，实现同步屏障、循环屏障、超时屏障。
"""
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

class BarrierStatus(str, Enum):
    WAITING = "waiting"
    BROKEN = "broken"
    TRIGGERED = "triggered"

@dataclass
class BarrierParticipant:
    participant_id: str
    arrived: bool = False
    arrival_time: float = 0.0

class DistributedBarrier:
    """分布式屏障（本地模拟）"""
    def __init__(self, barrier_id: str, parties: int, timeout: Optional[float] = None):
        self.barrier_id = barrier_id
        self.parties = parties
        self.timeout = timeout
        self._participants: Dict[str, BarrierParticipant] = {}
        self._arrived_count = 0
        self._status = BarrierStatus.WAITING
        self._generation = 0
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._action: Optional[Callable[[], None]] = None

    def set_action(self, action: Callable[[], None]) -> None:
        self._action = action

    def register(self, participant_id: str) -> bool:
        with self._lock:
            if len(self._participants) >= self.parties:
                return False
            self._participants[participant_id] = BarrierParticipant(participant_id=participant_id)
            return True

    def arrive(self, participant_id: str) -> bool:
        with self._condition:
            participant = self._participants.get(participant_id)
            if not participant:
                return False
            if participant.arrived:
                return True
            participant.arrived = True
            participant.arrival_time = time.time()
            self._arrived_count += 1
            if self._arrived_count >= self.parties:
                self._trigger()
                return True
            # 等待
            deadline = time.time() + self.timeout if self.timeout else None
            while self._status == BarrierStatus.WAITING:
                remaining = None
                if deadline:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        self._status = BarrierStatus.BROKEN
                        self._condition.notify_all()
                        return False
                self._condition.wait(timeout=remaining)
                if self._status == BarrierStatus.TRIGGERED:
                    return True
                if self._status == BarrierStatus.BROKEN:
                    return False
            return self._status == BarrierStatus.TRIGGERED

    def _trigger(self) -> None:
        self._status = BarrierStatus.TRIGGERED
        if self._action:
            try:
                self._action()
            except Exception:
                pass
        self._condition.notify_all()

    def reset(self) -> None:
        with self._lock:
            self._generation += 1
            self._status = BarrierStatus.WAITING
            self._arrived_count = 0
            for p in self._participants.values():
                p.arrived = False
                p.arrival_time = 0.0

    def get_status(self) -> BarrierStatus:
        return self._status

    def get_waiting_count(self) -> int:
        with self._lock:
            return self._arrived_count

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "barrier_id": self.barrier_id,
                "parties": self.parties,
                "arrived": self._arrived_count,
                "status": self._status.value,
                "generation": self._generation,
            }

class CyclicBarrier(DistributedBarrier):
    """循环屏障"""
    def arrive(self, participant_id: str) -> bool:
        result = super().arrive(participant_id)
        if result:
            # 自动重置
            with self._lock:
                if self._arrived_count >= self.parties:
                    self.reset()
        return result

class BarrierManager:
    """屏障管理器"""
    def __init__(self):
        self._barriers: Dict[str, DistributedBarrier] = {}
        self._lock = threading.RLock()

    def create_barrier(self, barrier_id: str, parties: int,
                       timeout: Optional[float] = None) -> DistributedBarrier:
        with self._lock:
            barrier = DistributedBarrier(barrier_id, parties, timeout)
            self._barriers[barrier_id] = barrier
            return barrier

    def create_cyclic_barrier(self, barrier_id: str, parties: int,
                              timeout: Optional[float] = None) -> CyclicBarrier:
        with self._lock:
            barrier = CyclicBarrier(barrier_id, parties, timeout)
            self._barriers[barrier_id] = barrier
            return barrier

    def get_barrier(self, barrier_id: str) -> Optional[DistributedBarrier]:
        return self._barriers.get(barrier_id)

    def remove_barrier(self, barrier_id: str) -> bool:
        with self._lock:
            return self._barriers.pop(barrier_id, None) is not None

def create_barrier_manager() -> BarrierManager:
    return BarrierManager()
