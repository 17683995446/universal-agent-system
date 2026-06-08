"""
阶段56: Agent分布式选举

遵循"图难于其易，为大于其细"的原则，实现Leader选举、任期管理、心跳检测、自动切换。
"""
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

class ElectionRole(str, Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"

class ElectionState(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class ElectionTerm:
    term_id: int = 0
    leader_id: str = ""
    voted_for: str = ""
    start_time: float = field(default_factory=time.time)

@dataclass
class ElectionNode:
    node_id: str
    role: ElectionRole = ElectionRole.FOLLOWER
    term: ElectionTerm = field(default_factory=ElectionTerm)
    last_heartbeat: float = field(default_factory=time.time)
    is_alive: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class DistributedElection:
    """分布式选举（本地模拟）"""
    def __init__(self, node_id: str, heartbeat_interval: float = 1.0,
                 election_timeout: float = 3.0):
        self.node_id = node_id
        self._heartbeat_interval = heartbeat_interval
        self._election_timeout = election_timeout
        self._role = ElectionRole.FOLLOWER
        self._term = ElectionTerm()
        self._leader_id: Optional[str] = None
        self._nodes: Dict[str, ElectionNode] = {}
        self._votes: Dict[str, str] = {}  # node_id -> voted_for
        self._callbacks: List[Callable[[str, ElectionRole], None]] = []
        self._lock = threading.RLock()
        self._state = ElectionState.STOPPED
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def register_callback(self, callback: Callable[[str, ElectionRole], None]) -> None:
        self._callbacks.append(callback)

    def add_node(self, node_id: str) -> None:
        with self._lock:
            self._nodes[node_id] = ElectionNode(node_id=node_id)

    def remove_node(self, node_id: str) -> bool:
        with self._lock:
            return self._nodes.pop(node_id, None) is not None

    def start(self) -> None:
        with self._lock:
            if self._state == ElectionState.RUNNING:
                return
            self._state = ElectionState.RUNNING
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._state = ElectionState.STOPPED
            self._stop_event.set()
            if self._thread:
                self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._tick()
            time.sleep(self._heartbeat_interval)

    def _tick(self) -> None:
        with self._lock:
            if self._role == ElectionRole.LEADER:
                self._send_heartbeat()
            else:
                self._check_election_timeout()

    def _send_heartbeat(self) -> None:
        now = time.time()
        for node in self._nodes.values():
            if node.node_id != self.node_id:
                node.last_heartbeat = now

    def _check_election_timeout(self) -> None:
        if not self._leader_id:
            self._start_election()
            return
        leader = self._nodes.get(self._leader_id)
        if not leader or time.time() - leader.last_heartbeat > self._election_timeout:
            self._start_election()

    def _start_election(self) -> None:
        self._role = ElectionRole.CANDIDATE
        self._term.term_id += 1
        self._term.voted_for = self.node_id
        self._votes = {self.node_id: self.node_id}
        self._leader_id = None
        self._notify_callbacks()

        # 模拟投票
        for node in self._nodes.values():
            if node.node_id != self.node_id and node.is_alive:
                self._votes[node.node_id] = self.node_id

        if len(self._votes) > (len(self._nodes) + 1) // 2:
            self._become_leader()

    def _become_leader(self) -> None:
        self._role = ElectionRole.LEADER
        self._leader_id = self.node_id
        self._term.leader_id = self.node_id
        self._term.start_time = time.time()
        self._notify_callbacks()

    def _notify_callbacks(self) -> None:
        for cb in self._callbacks:
            try:
                cb(self.node_id, self._role)
            except Exception:
                pass

    def vote(self, candidate_id: str, term_id: int) -> bool:
        with self._lock:
            if term_id > self._term.term_id:
                self._term.term_id = term_id
                self._term.voted_for = candidate_id
                self._role = ElectionRole.FOLLOWER
                self._leader_id = None
                return True
            if term_id == self._term.term_id and not self._term.voted_for:
                self._term.voted_for = candidate_id
                return True
            return False

    def receive_heartbeat(self, leader_id: str, term_id: int) -> None:
        with self._lock:
            if term_id >= self._term.term_id:
                self._term.term_id = term_id
                self._term.leader_id = leader_id
                self._leader_id = leader_id
                self._role = ElectionRole.FOLLOWER
                self._term.voted_for = ""
                if leader_id in self._nodes:
                    self._nodes[leader_id].last_heartbeat = time.time()

    @property
    def is_leader(self) -> bool:
        return self._role == ElectionRole.LEADER

    @property
    def current_leader(self) -> Optional[str]:
        return self._leader_id

    @property
    def role(self) -> ElectionRole:
        return self._role

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "node_id": self.node_id,
                "role": self._role.value,
                "term": self._term.term_id,
                "leader": self._leader_id,
                "nodes": len(self._nodes),
            }

class ElectionManager:
    """选举管理器"""
    def __init__(self):
        self._elections: Dict[str, DistributedElection] = {}
        self._lock = threading.RLock()

    def create_election(self, election_id: str, node_id: str) -> DistributedElection:
        with self._lock:
            election = DistributedElection(node_id)
            self._elections[election_id] = election
            return election

    def get_election(self, election_id: str) -> Optional[DistributedElection]:
        return self._elections.get(election_id)

    def remove_election(self, election_id: str) -> bool:
        with self._lock:
            e = self._elections.pop(election_id, None)
            if e:
                e.stop()
                return True
            return False

def create_election_manager() -> ElectionManager:
    return ElectionManager()
