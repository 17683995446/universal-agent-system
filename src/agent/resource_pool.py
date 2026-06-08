"""
阶段44: Agent资源池管理系统

遵循"图难于其易，为大于其细"的原则，实现资源池化、分配、回收、限额管理。
"""
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

class ResourceType(str, Enum):
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"
    CUSTOM = "custom"

class PoolStatus(str, Enum):
    AVAILABLE = "available"
    ALLOCATED = "allocated"
    EXHAUSTED = "exhausted"
    CLOSED = "closed"

@dataclass
class ResourceUnit:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    resource_type: ResourceType = ResourceType.CUSTOM
    capacity: float = 1.0
    allocated: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> float:
        return self.capacity - self.allocated

    @property
    def utilization(self) -> float:
        return self.allocated / self.capacity if self.capacity > 0 else 0

    def can_allocate(self, amount: float) -> bool:
        return self.available >= amount

@dataclass
class Allocation:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    resource_id: str = ""
    owner: str = ""
    amount: float = 0.0
    allocated_at: float = field(default_factory=time.time)
    released_at: Optional[float] = None
    active: bool = True

@dataclass
class ResourceQuota:
    owner: str
    max_cpu: float = 4.0
    max_memory: float = 8192.0  # MB
    max_gpu: float = 1.0
    max_custom: float = 100.0

class ResourcePool:
    """资源池"""
    def __init__(self, name: str, resource_type: ResourceType, total_capacity: float):
        self.name = name
        self.resource_type = resource_type
        self._total_capacity = total_capacity
        self._units: Dict[str, ResourceUnit] = {}
        self._allocations: Dict[str, Allocation] = {}
        self._lock = threading.RLock()
        self._status = PoolStatus.AVAILABLE

    def add_unit(self, unit: ResourceUnit) -> None:
        with self._lock:
            self._units[unit.id] = unit
            self._total_capacity += unit.capacity

    def allocate(self, amount: float, owner: str, labels: Optional[Dict] = None) -> Optional[Allocation]:
        with self._lock:
            if self._status == PoolStatus.CLOSED:
                return None
            if self._total_capacity - self._used() < amount:
                self._status = PoolStatus.EXHAUSTED
                return None
            # 找到合适的单元
            for unit in self._units.values():
                if unit.can_allocate(amount):
                    if labels:
                        for k, v in labels.items():
                            if unit.labels.get(k) != v:
                                continue
                    unit.allocated += amount
                    alloc = Allocation(resource_id=unit.id, owner=owner, amount=amount)
                    self._allocations[alloc.id] = alloc
                    self._status = PoolStatus.AVAILABLE
                    return alloc
            # 从总量分配
            if self._total_capacity - self._used() >= amount:
                alloc = Allocation(owner=owner, amount=amount)
                self._allocations[alloc.id] = alloc
                return alloc
            return None

    def release(self, allocation_id: str) -> bool:
        with self._lock:
            alloc = self._allocations.get(allocation_id)
            if not alloc or not alloc.active:
                return False
            alloc.active = False
            alloc.released_at = time.time()
            unit = self._units.get(alloc.resource_id)
            if unit:
                unit.allocated = max(0, unit.allocated - alloc.amount)
            self._status = PoolStatus.AVAILABLE
            return True

    def _used(self) -> float:
        return sum(a.amount for a in self._allocations.values() if a.active)

    @property
    def available(self) -> float:
        with self._lock:
            return self._total_capacity - self._used()

    @property
    def utilization(self) -> float:
        with self._lock:
            used = self._used()
            return used / self._total_capacity if self._total_capacity > 0 else 0

    @property
    def status(self) -> PoolStatus:
        return self._status

    def get_allocations(self, owner: Optional[str] = None) -> List[Allocation]:
        with self._lock:
            allocs = [a for a in self._allocations.values() if a.active]
            if owner:
                allocs = [a for a in allocs if a.owner == owner]
            return allocs

    def close(self) -> None:
        with self._lock:
            self._status = PoolStatus.CLOSED

class ResourceManager:
    """资源管理器"""
    def __init__(self):
        self._pools: Dict[str, ResourcePool] = {}
        self._quotas: Dict[str, ResourceQuota] = {}
        self._lock = threading.RLock()

    def create_pool(self, name: str, rtype: ResourceType, capacity: float) -> ResourcePool:
        with self._lock:
            pool = ResourcePool(name, rtype, capacity)
            self._pools[name] = pool
            return pool

    def get_pool(self, name: str) -> Optional[ResourcePool]:
        return self._pools.get(name)

    def allocate(self, pool_name: str, amount: float, owner: str) -> Optional[Allocation]:
        pool = self._pools.get(pool_name)
        if not pool:
            return None
        # 检查配额
        quota = self._quotas.get(owner)
        if quota:
            current = sum(a.amount for a in pool.get_allocations(owner))
            if pool.resource_type == ResourceType.CPU and current + amount > quota.max_cpu:
                return None
            if pool.resource_type == ResourceType.MEMORY and current + amount > quota.max_memory:
                return None
        return pool.allocate(amount, owner)

    def release(self, pool_name: str, allocation_id: str) -> bool:
        pool = self._pools.get(pool_name)
        return pool.release(allocation_id) if pool else False

    def set_quota(self, owner: str, quota: ResourceQuota) -> None:
        with self._lock:
            self._quotas[owner] = quota

    def get_utilization(self, pool_name: str) -> float:
        pool = self._pools.get(pool_name)
        return pool.utilization if pool else 0

    def list_pools(self) -> List[str]:
        return list(self._pools.keys())

def create_resource_manager() -> ResourceManager:
    return ResourceManager()
